export newton_polytope

"""
    newton_polytope(method, objective; verbose=false, preprocess_quick=false, preprocess_fine=true, preprocess=nothing,
        parameters...)

Calculates the Newton polytope for the sum of squares optimization of a given objective. This requires the availability of a
linear solver. Currently, `:Mosek` is the only supported method (which is also the default). There are two preprocessing
methods:
- `preprocess_quick` is the Akl-Toussaint heuristic based. It is only helpful if the objective is very unbalanced with regard
  to the occurring powers, and therefore disabled by default. Every monomial will be checked against a linear program that
  scales as the number of variables in the objective.
- `preprocess_fine` performs an extensive reduction of the possible number of monomials that comprise the convex hull. Every
  monomial will be checked against a linear program that scales as the number of monomials in the objective (though it might
  become more efficient when monomials are ruled out).
After preprocessing is done, the monomials in the Newton polytope are filtered by performing linear programs that scale at most
in the number of monomials in the objective, or less if they were filtered by preprocessing.
For large initial sets of monomials (≥ 10⁵), the final function will use all available threads to speed up the process.
The parameters will be passed on to the linear solver.
"""
newton_polytope(method::Symbol, objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_polytope(Val(method), objective; kwargs...)

newton_polytope(objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_polytope(Val(:Mosek), objective; kwargs...)

function newton_polytope(::Val{:Mosek}, objective::P; verbose::Bool=false, preprocess_quick::Bool=false,
    preprocess_fine::Bool=true, preprocess::Union{Nothing,Bool}=nothing, parameters...) where {P<:AbstractPolynomialLike}
    coeffs = [exponents(mon) for mon in monomials(objective)]
    nc = length(coeffs)
    nv = nvariables(objective)
    @verbose_info("Removing reduncancies from the convex hull - ", nc, " initial candidates")
    if !isnothing(preprocess)
        preprocess_quick = preprocess_fine = preprocess
    end
    preprocess_quick && let
        # eliminate all the coefficients that by the Akl-Toussaint heuristic cannot be part of the convex hull anyway
        # For typical polynomial problems which tend to have powers occurring relatively evenly distributed, this heuristic
        # will not lead to any reduction, so the quick preprocessing is disabled by default.
        @verbose_info("Removing reduncancies from the convex hull - quick heuristic")
        vertices = Matrix{Int}(undef, nv, 2nv)
        lowestpoints = @view(vertices[:, 1:nv])
        highestpoints = @view(vertices[:, nv+1:2nv])
        for col in eachcol(lowestpoints)
            copyto!(col, first(coeffs))
        end
        copyto!(highestpoints, lowestpoints)
        # we might also add the sum of all coordinates, or differences (but there are 2^nv ways to combine, so let's skip it)
        @inbounds for coeff in coeffs
            for (i, coeffᵢ) in enumerate(coeff)
                if lowestpoints[i, i] > coeffᵢ
                    copyto!(@view(lowestpoints[:, i]), coeff)
                end
                if highestpoints[i, i] < coeffᵢ
                    copyto!(@view(highestpoints[:, i]), coeff)
                end
            end
        end
        required_coeffs = Vector{Bool}(undef, nc)
        # now every point that is not a member of the convex polytope determined by vertices can be dropped immediately
        lastinfo = time_ns()
        preproc_time = @elapsed let task=Mosek.Task(Mosek.msk_global_env)
            try
                # verbose && Mosek.putstreamfunc(task, Mosek.MSK_STREAM_LOG, printstream)
                for (k, v) in parameters
                    Mosek.putparam(task, string(k), v)
                end
                Mosek.appendvars(task, size(vertices, 2))
                Mosek.putvarboundsliceconst(task, 1, size(vertices, 2) +1, Mosek.MSK_BK_LO, 0., Inf)
                Mosek.appendcons(task, nv)
                idxs = collect(1:nv)
                for (i, vert) in enumerate(eachcol(vertices))
                    Mosek.putacol(task, i, idxs, vert)
                end
                fx = fill(Mosek.MSK_BK_FX, nv)
                for (i, coeff) in enumerate(coeffs)
                    Mosek.putconboundslice(task, 1, nv +1, fx, coeff, coeff)
                    Mosek.optimize(task)
                    @inbounds required_coeffs[i] = Mosek.getsolsta(task, Mosek.MSK_SOL_BAS) == Mosek.MSK_SOL_STA_OPTIMAL
                    if verbose
                        nextinfo = time_ns()
                        if nextinfo - lastinfo > 10_000_000_000
                            print("Status update: ", i, " of ", nc, "\r")
                            flush(stdout)
                            lastinfo = nextinfo
                        end
                    end
                end
            finally
                Mosek.deletetask(task)
            end
        end
        keepat!(coeffs, required_coeffs)
        nc = length(coeffs)
        @verbose_info("Found ", length(coeffs), " potential extremal points of the convex hull in ", preproc_time, " seconds")
    end
    if preprocess_fine let
        # eliminate all the coefficients that are redundant themselves to make the linear system smaller
        @verbose_info("Removing redundancies from the convex hull - more detailed")
        required_coeffs = fill(true, nc)
        preproc_time = @elapsed let task=Mosek.Task(Mosek.msk_global_env)
            try
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
            finally
                Mosek.deletetask(task)
            end
        end
        keepat!(coeffs, required_coeffs)
        nc = length(coeffs)
        @verbose_info("Found ", length(coeffs), " extremal points of the convex hull in ", preproc_time, " seconds")
    end else
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
            maxdeg, mindeg = 0, typemax(Int)
            maxmultideg, minmultideg = fill(0, nv), fill(typemax(Int), nv)
            # do some hopefully quick preprocessing (on top of the previous preprocessing)
            for coeff in coeffs
                deg = 0
                @inbounds for (i, coeffᵢ) in enumerate(coeff)
                    deg += coeffᵢ
                    if coeffᵢ > maxmultideg[i]
                        maxmultideg[i] = coeffᵢ
                    end
                    if coeffᵢ < minmultideg[i]
                        minmultideg[i] = coeffᵢ
                    end
                end
                if deg > maxdeg
                    maxdeg = deg
                end
                if deg < mindeg
                    mindeg = deg
                end
            end
            maxmultideg .= div.(maxmultideg, 2, RoundDown)
            minmultideg .= div.(minmultideg, 2, RoundUp)
            maxdeg = div(maxdeg, 2, RoundDown)
            mindeg = div(mindeg, 2, RoundUp)
            mons = monomials(variables(objective), mindeg:maxdeg,
                m -> all(x -> x[1] ≤ x[2] ≤ x[3], zip(minmultideg, exponents(m), maxmultideg)))
            num = length(mons)
            sizehint!(candidates, num)
            bk = fill(Mosek.MSK_BK_FX, nv)
            nthreads = Threads.nthreads()
            @verbose_info("Starting point selection among ", num, " possible monomials")
            if num < 100_000 || isone(nthreads)
                # single threading is better
                verbose && (lastinfo = time_ns())
                for (i, candidate) in enumerate(mons)
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
                        [(i += ic; @view(mons[i-ic:i-1])) for ic in itemcounts]::Vector{typeof(@view(mons[begin:end]))}
                    end
                    check = Base.Event()
                    local taskfun
                    function taskfun(tid)
                        _task = tasks[tid]
                        _candidates = similar(candidates, 0, buffer=itemcounts[tid])
                        @inbounds for candidate in items[tid]
                            exps = exponents(candidate)
                            all(x -> x[1] ≤ x[2] ≤ x[3], zip(minmultideg, exps, maxmultideg)) || continue
                            exps = exps .<< 1 # half the polytope
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
                            istaskdone(thread) && unsafe_append!(candidates, fetch(thread)::typeof(candidates))
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