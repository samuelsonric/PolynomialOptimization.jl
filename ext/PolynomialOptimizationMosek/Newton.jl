function PolynomialOptimization.newton_polytope_preproc_quick(::Val{:Mosek}, coeffs, vertexindices, verbose; parameters...)
    nv, nc = size(coeffs)
    nvertices = length(vertexindices)
    required_coeffs = Vector{Bool}(undef, nc)
    # now every point that is not a member of the convex polytope determined by vertices can be dropped immediately
    lastinfo = time_ns()
    task = Mosek.Task(msk_global_env::Env)
    try
        # verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
        for (k, v) in parameters
            putparam(task, string(k), v)
        end
        appendvars(task, nvertices)
        putvarboundsliceconst(task, 1, nvertices +1, MSK_BK_LO, 0., Inf)
        appendcons(task, nv +1)
        tmp = Vector{Float64}(undef, max(nv, nvertices))
        let
            idxs = collect(Int32(0):Int32(max(nv, nvertices) -1))
            for (i, vert) in zip(Iterators.countfrom(zero(Int32)), vertexindices)
                @inbounds copyto!(tmp, @view(coeffs[:, vert]))
                Mosek.@MSK_putacol(task.task, i, nv, idxs, tmp)
            end
            @inbounds fill!(@view(tmp[1:nvertices]), 1.)
            Mosek.@MSK_putarow(task.task, nv, nvertices, idxs, tmp)
            putconbound(task, nv +1, MSK_BK_FX, 1.0, 1.0)
        end
        fx = fill(MSK_BK_FX.value, nv)
        for (i, coeff) in enumerate(eachcol(coeffs))
            if insorted(i, vertexindices)
                @inbounds required_coeffs[i] = true
                continue
            end
            @inbounds copyto!(tmp, coeff)
            Mosek.@MSK_putconboundslice(task.task, 0, nv, fx, tmp, tmp)
            optimize(task)
            @inbounds required_coeffs[i] = getsolsta(task, MSK_SOL_BAS) != MSK_SOL_STA_OPTIMAL
            if verbose
                nextinfo = time_ns()
                if nextinfo - lastinfo > 1_000_000_000
                    print("Status update: ", i, " of ", nc, "\r")
                    flush(stdout)
                    lastinfo = nextinfo
                end
            end
        end
    finally
        deletetask(task)
    end
    return required_coeffs
end

function PolynomialOptimization.newton_polytope_preproc_remove(::Val{:Mosek}, nv, nc, getvarcon, verbose, singlethread;
    parameters...)
    task = Mosek.Task(msk_global_env::Env)
    singlethread && putintparam(task, MSK_IPAR_NUM_THREADS, 1)
    # verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
    for (k, v) in parameters
        putparam(task, string(k), v)
    end
    # basic initialization: every point will get a variable
    appendvars(task, nc)
    putvarboundsliceconst(task, 1, nc +1, MSK_BK_LO, 0., Inf)
    appendcons(task, nv +1)
    putconboundsliceconst(task, 1, nv +2, MSK_BK_FX, 0., 0.)
    # ^ since we always fix the point in question to be -1, the sum of all points must be zero (condition nv +1)
    let
        idxs = collect(Int32(0):Int32(max(nv, nc) -1))
        tmp = Vector{Float64}(undef, max(nv, nc))
        for i in 1:nc
            copyto!(tmp, @inline(getvarcon(i)))
            Mosek.@MSK_putacol(task.task, i -1, nv, idxs, tmp)
        end
        @inbounds fill!(@view(tmp[1:nc]), 1.)
        Mosek.@MSK_putarow(task.task, nv, nc, idxs, tmp)
    end

    required_coeffs = fill(true, nc)
    removed = 0
    lastremoved = 0
    varupto = nc
    varnum = nc
    lastinfo = time_ns()
    # and then we start to iterate through the points and try to express one in terms of the others
    for i in nc:-1:1
        # first enforce this variable to be fixed: all others must add up to this point
        putvarbound(task, i, MSK_BK_FX, -1., -1.)
        # then try to find a solution
        optimize(task)
        if getsolsta(task, MSK_SOL_BAS) == MSK_SOL_STA_OPTIMAL
            # this was indeed possible, so our point is redundant; remove it!
            putvarbound(task, i, MSK_BK_FX, 0., 0.)
            @inbounds required_coeffs[i] = false
            lastremoved += 1
        else
            # it was not possible, we must keep this point
            putvarbound(task, i, MSK_BK_LO, 0., Inf)
        end
        if verbose
            nextinfo = time_ns()
            if nextinfo - lastinfo > 1_000_000_000
                if verbose
                    print("\33[2KStatus update: ", nc - i, " of ", nc, " (removed ", 100(removed + lastremoved) ÷ (nc - i +1),
                        "% so far)\r")
                    flush(stdout)
                    lastinfo = nextinfo
                end
            end
        end
        # Deleting Mosek variables is expensive, but every once in a while, it may be worth the effort
        if lastremoved > 20 && 10lastremoved ≥ varnum
            drops = FastVec{Int32}(buffer=lastremoved)
            for j in i:varupto
                @inbounds if !required_coeffs[j]
                    unsafe_push!(drops, j -1)
                end
            end
            @assert(lastremoved == length(drops))
            let lastremoved=lastremoved # the macro contains a closure which would box lastremoved
                Mosek.@MSK_removevars(task.task, lastremoved, finish!(drops))
            end
            removed += lastremoved
            varnum -= lastremoved
            lastremoved = 0
            varupto = i -1
        end
    end
    deletetask(task)
    return required_coeffs
end

PolynomialOptimization.newton_halfpolytope_alloc_global(::Val{:Mosek}, nv) = fill(MSK_BK_FX.value, nv)
PolynomialOptimization.newton_halfpolytope_alloc_local(::Val{:Mosek}, nv) = Vector{Float64}(undef, nv)
PolynomialOptimization.newton_halfpolytope_clonetask(t::Mosek.Task) = Mosek.Task(t)

@inline function PolynomialOptimization.newton_polytope_do_worker(::Val{:Mosek}, task, bk, tmp, moniter, Δprogress,
    Δacceptance, add_callback, iteration_callback)
    for powers in moniter
        # check the previous power in the linear program and add it if possible
        copyto!(tmp, powers)
        Mosek.@MSK_putconboundslice(task.task, 0, length(bk), bk, tmp, tmp)
        optimize(task)
        if getsolsta(task, MSK_SOL_BAS) == MSK_SOL_STA_OPTIMAL
            # this candidate is part of the Newton polytope
            @inline add_callback(powers)
            Δacceptance[] += 1
        end
        Δprogress[] += 1
        isnothing(iteration_callback) || @inline iteration_callback(powers)
    end
    return
end

function PolynomialOptimization.newton_halfpolytope_do_prepare(::Val{:Mosek}, coeffs, num, verbose; parameters...)
    nv, nc = size(coeffs)
    # now we build a task with the minimal number of extremal points and try to find every possible part of the Newton polytope
    # for SOS polynomials
    task = Mosek.Task(msk_global_env::Env)
    # verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
    for (k, v) in parameters
        putparam(task, string(k), v)
    end
    appendvars(task, nc)
    putvarboundsliceconst(task, 1, nc +1, MSK_BK_LO, 0., Inf)
    appendcons(task, nv +1)
    let
        idxs = collect(Int32(0):Int32(max(nv, nc) -1))
        tmp = Vector{Float64}(undef, max(nv, nc))
        for (i, cf) in zip(Iterators.countfrom(zero(Int32)), eachcol(coeffs))
            @inbounds @view(tmp[1:nv]) .= 0.5 .* cf
            Mosek.@MSK_putacol(task.task, i, nv, idxs, tmp)
        end
        @inbounds fill!(@view(tmp[1:nc]), 1.)
        Mosek.@MSK_putarow(task.task, nv, nc, idxs, tmp)
    end
    putconbound(task, nv +1, MSK_BK_FX, 1.0, 1.0)
    if num < 10_000 || isone(nv)
        nthreads = 1
        secondtask = nothing
    else
        nthreads = Threads.nthreads()
        if nthreads > 1
            # we need to figure out if we have enough memory for all the threads. Unfortunately, getmemusage() seems to
            # always return 1081, "No available information about the space usage." - so we need to fetch the information
            # ourselves. Let's assume that at least a second task can be created.
            mem = @allocdiff begin
                secondtask = Mosek.Task(task)
                optimize(secondtask)
            end
            if mem ≤ 0
                @verbose_info("Memory requirements of a single thread could not be determined, using all available threads")
            else
                @verbose_info("Memory requirements of a single thread: ", div(mem, 1024*1024, RoundUp), " MiB")
                nthreads = min(nthreads, Int(Sys.free_memory() ÷ mem +2))
            end
            # Note that this is potentially still an underestimation, as our candidates list will also grow. But this is
            # something that can potentially be swapped, so if swap space is available beyond the free_memory limit, then we
            # are still fine.
        else
            secondtask = nothing
        end
    end
    isone(nthreads) || putintparam(task, MSK_IPAR_NUM_THREADS, 1) # single-threaded for Mosek itself
    return nthreads, task, secondtask
end