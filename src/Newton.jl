export newton_polytope

"""
    newton_polytope(method, objective; verbose=false, preprocess=true)

Calculates the Newton polytope for the sum of squares optimization of a given objective. This requires the availability of a
linear solver. Currently, `:Mosek` is the only supported method (which is also the default). If preprocessing is turned on,
the monomials in the objective will first be checked for redundancies, leading to a possibly smaller extremal set.
For large initial sets of monomials (≥ 10⁵), this function will use all available threads to speed up the process.
"""
newton_polytope(method::Symbol, objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_polytope(Val(method), objective; kwargs...)

newton_polytope(objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_polytope(Val(:Mosek), objective; kwargs...)

function newton_polytope(::Val{:Mosek}, objective::P; verbose::Bool=false, preprocess::Bool=true,
    parameters...) where {P<:AbstractPolynomialLike}
    coeffs = exponents.(monomials(objective))
    nc = length(coeffs)
    nv = nvariables(objective)
    if preprocess
        required_coeffs = fill(true, nc)
        # first eliminate all the coefficients that are redundant themselves to make the linear system smaller
        @verbose_info("Removing redundancies from the convex hull")
        preproc_time = @elapsed begin
            Mosek.maketask() do task
                # verbose && Mosek.putstreamfunc(task, Mosek.MSK_STREAM_LOG, printstream)
                for (k, v) in parameters
                    Mosek.putparam(task, string(k), v)
                end
                # basic initialization: every point will initially get a variable
                Mosek.appendvars(task, nc)
                Mosek.putvarboundsliceconst(task, 1, nc +1, Mosek.MSK_BK_LO, 0., Inf)
                Mosek.appendcons(task, nv +1)
                Mosek.putconboundsliceconst(task, 1, nv +2, Mosek.MSK_BK_FX, 0.0, 0.0)
                # ^ since we always fix the point in question to be -1, the sum of all points must be zero (condition nv +1)
                idxs = collect(1:nv)
                for (i, cf) in enumerate(coeffs)
                    Mosek.putacol(task, i, idxs, cf)
                end
                Mosek.putarow(task, nv +1, collect(1:nc), fill(1., nc))
                # and then we start to iterate through the points and try to express one in terms of the others
                lastinfo = time_ns()
                for i in 1:nc
                    # first enforce this variable to be fixed: all others must add up to this point
                    Mosek.putvarbound(task, i, Mosek.MSK_BK_FX, -1., -1.)
                    # then try to find a solution
                    Mosek.optimize(task)
                    if Mosek.getsolsta(task, Mosek.MSK_SOL_BAS) == Mosek.MSK_SOL_STA_OPTIMAL
                        # this was indeed possible, so our point is redundant; remove it!
                        Mosek.putvarbound(task, i, Mosek.MSK_BK_FX, 0., 0.)
                        @inbounds required_coeffs[i] = false
                    else
                        # it was not possible, we must keep this point
                        Mosek.putvarbound(task, i, Mosek.MSK_BK_LO, 0., Inf)
                    end
                    if verbose
                        nextinfo = time_ns()
                        if nextinfo - lastinfo > 10_000_000_000
                            print("Status update: ", i, " of ", nc, "\r")
                            flush(stdout)
                            lastinfo = nextinfo
                        end
                    end
                end
                # we now have the points that make up the linear hull
            end
        end
        @inbounds coeffs = coeffs[required_coeffs]
        nc = length(coeffs)
        @verbose_info("Found ", length(coeffs), " extremal points of the convex hull in ", preproc_time, " seconds")
    else
        @verbose_info("Skipping preprocessing, there are ", nc, " possibly non-extremal points")
    end

    # now we rebuild a task with this minimal number of extremal points and try to find every possible part of the Newton
    # polytope for SOS polynomials
    candidates = FastVec{monomial_type(objective)}()
    newton_time = @elapsed begin
        task = Mosek.Task(Mosek.msk_global_env)
        try
            # verbose && Mosek.putstreamfunc(task, Mosek.MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                Mosek.putparam(task, string(k), v)
            end
            Mosek.appendvars(task, nc)
            Mosek.putvarboundsliceconst(task, 1, nc +1, Mosek.MSK_BK_LO, 0., Inf)
            Mosek.appendcons(task, nv +1)
            idxs = collect(1:nv)
            for (i, cf) in enumerate(coeffs)
                Mosek.putacol(task, i, idxs, cf)
            end
            Mosek.putarow(task, nv +1, collect(1:nc), fill(1., nc))
            Mosek.putconbound(task, nv +1, Mosek.MSK_BK_FX, 1.0, 1.0)
            deg = maxdegree(objective) ÷ 2
            num = binomial(nv + deg, nv)
            sizehint!(candidates, num)
            bk = fill(Mosek.MSK_BK_FX, nv)
            @verbose_info("Starting point selection among ", num, " possible monomials")
            if num < 100_000
                # single threading is better
                verbose && (lastinfo = time_ns())
                for (i, candidate) in enumerate(monomials(variables(objective), 0:deg))
                    exps = exponents(candidate) .<< 1 # half the polytope
                    Mosek.putconboundslice(task, 1, nv +1, bk, exps, exps)
                    Mosek.optimize(task)
                    if Mosek.getsolsta(task, Mosek.MSK_SOL_BAS) == Mosek.MSK_SOL_STA_OPTIMAL
                        # this candidate is part of the Newton polytope
                        unsafe_push!(candidates, candidate)
                    end
                    if verbose
                        nextinfo = time_ns()
                        if nextinfo - lastinfo > 10_000_000_000
                            print("Status update: ", floor(Int, 100*i/num), "%\r")
                            flush(stdout)
                            lastinfo = nextinfo
                        end
                    end
                end
            else
                # we do multithreading, but note that we need to copy the task for every thread
                ccall(:jl_enter_threaded_region, Cvoid, ())
                try
                    nthreads = Threads.nthreads()
                    mons = monomials(variables(objective), 0:deg)
                    local orddiv, remdiv
                    while true
                        orddiv, remdiv = divrem(length(mons), nthreads)
                        remdiv < orddiv && break
                        # avoid distribution on too many cores that hardly do something on supercomputers
                        nthreads >>= 1
                    end
                    @verbose_info("Preparing to determine Newton polytope using ", nthreads, " threads, each checking about ",
                        orddiv, " candidates")
                    Mosek.putintparam(task, Mosek.MSK_IPAR_NUM_THREADS, 1) # single-threaded for Mosek itself
                    tasks = [Mosek.Task(task) for _ in 1:nthreads]
                    nums = fill(0, nthreads)
                    itemcounts = fill(orddiv, nthreads)
                    itemcounts[1:remdiv] .= orddiv +1
                    items = let i = 1;
                        [(i += ic; @view(mons[i-ic:i-1])) for ic in itemcounts]
                    end
                    check = Base.Event()
                    local taskfun
                    function taskfun(tid)
                        _task = tasks[tid]
                        _candidates = similar(candidates, 0, buffer=itemcounts[tid])
                        @inbounds for candidate in items[tid]
                            exps = exponents(candidate) .<< 1 # half the polytope
                            Mosek.putconboundslice(_task, 1, nv +1, bk, exps, exps)
                            Mosek.optimize(_task)
                            if Mosek.getsolsta(_task, Mosek.MSK_SOL_BAS) == Mosek.MSK_SOL_STA_OPTIMAL
                                unsafe_push!(candidates, candidate)
                            end
                            nums[tid] += 1
                            verbose && tid == 1 && yield()
                        end
                        notify(check)
                        finish!(_candidates)
                    end
                    # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
                    threads = Vector{Task}(undef, nthreads)
                    for tid in 1:nthreads
                        t = Task(() -> taskfun(tid))
                        t.sticky = true # do not allow migration of the task between different threads!
                        ccall(:jl_set_task_tid, Cint, (Any, Cint), t, tid-1)
                        @inbounds threads[tid] = t
                        schedule(t)
                    end
                    # Note that these are not "worker threads", but the first one is actually the main thread - which means
                    # that as soon as this function yields and the task takes over (which will never yield), the notification
                    # mechanism will never be triggered until the first thread finished. This is why the first thread yields
                    # after every optimization step (if the verbosity requires it; else we will collect all the candidates at
                    # the end).
                    @verbose_info("All tasks are running")
                    timer = Timer(10; interval=10) do _
                        notify(check)
                    end
                    for thread in threads
                        while !istaskdone(thread)
                            wait(check)
                            reset(check)
                            istaskdone(thread) && unsafe_append!(candidates, fetch(thread))
                            if verbose
                                print("Status update: ", floor(Int, 100*sum(nums)/num), "%\r")
                                flush(stdout)
                            end
                        end
                    end
                    close(timer)
                    @verbose_info("All tasks have finished")
                finally
                    ccall(:jl_exit_threaded_region, Cvoid, ())
                end
            end
        finally
            Mosek.deletetask(task)
        end
    end
    @verbose_info("Found ", length(candidates), " elements in the Newton polytope superset in ", newton_time, " seconds")
    return finish!(candidates)
end