module PolynomialOptimizationMPI

using PolynomialOptimization, MultivariatePolynomials, Printf
import MPI, Random, Mosek
import PolynomialOptimization: @verbose_info, FastVec, prepare_push!, unsafe_push!, finish!

__init__() = PolynomialOptimization.haveMPI[] = true

abstract type MPIRank end

struct RootRank <: MPIRank end
struct OtherRank <: MPIRank
    rank::Int
end

const root = 0

Base.convert(::Type{<:Integer}, ::RootRank) = root
Base.convert(::Type{<:Integer}, rank::OtherRank) = rank.rank
Base.:(==)(rank::MPIRank, with::Integer) = convert(typeof(with), rank) == with
Base.:(==)(with::Integer, rank::MPIRank) = with == convert(typeof(with), rank)
isroot(::MPIRank) = false
isroot(::RootRank) = true
MPIRank(rank::Integer) = iszero(rank) ? RootRank() : OtherRank(rank)

function PolynomialOptimization.newton_polytope_do_taskfun(V::Val{:Mosek}, task, ranges, nv, mindeg, maxdeg, bk, cond,
    allprogress, allacceptance, allcandidates, notifier, init_time, num, rank::MPIRank)
    # notifier: 0 - no notification; 1 - the next to get becomes the notifier; 2 - notifier is taken
    verbose = notifier[] != 0
    lastappend = time_ns()
    isnotifier = Ref(false) # necessary due to the capturing/boxing bug
    Δprogress_acceptance = Vector{Int}(undef, 2)
    if verbose
        lastinfo = Ref{Int}(lastappend)
        if isroot(rank)
            req = Ref(MPI.Irecv!(Δprogress_acceptance, comm, tag=1))
        end
    end
    powers = Vector{Int}(undef, nv)
    tmp = Vector{Float64}(undef, nv)
    candidates = similar(allcandidates)
    try
        while true
            local curminrange, curmaxrange
            try
                curminrange, curmaxrange = take!(ranges)
            catch e
                e isa InvalidStateException && break
                rethrow(e)
            end
            if isroot(rank)
                PolynomialOptimization.newton_polytope_do_worker(V, task, bk,
                    MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers), tmp,
                    let candidates=candidates; p -> push!(candidates, copy(p)) end,
                    !verbose ? nothing : let isnotifier=isnotifier, lastinfo=lastinfo,
                                             Δprogress_acceptance=Δprogress_acceptance
                    (Δprogress, Δacceptance) -> let
                        nextinfo = time_ns()
                        if nextinfo - lastinfo[] > 10_000_000_000 && trylock(cond)
                            # no reason to block, we can also just retry in the next iteration
                            progress = allprogress[] += Δprogress
                            acceptance = allacceptance[] += Δacceptance
                            if !isnotifier[] && notifier[] == 1
                                isnotifier[] = true
                                notifier[] = 2
                            end
                            @inbounds while MPI.Test(req[])
                                progress += Δprogress_acceptance[1]
                                acceptance += Δprogress_acceptance[2]
                                req[] = MPI.Irecv!(Δprogress_acceptance, comm, tag=1)
                            end
                            allprogress[] = progress
                            allacceptance[] = acceptance
                            unlock(cond)
                            lastinfo[] = nextinfo
                            if isnotifier[]
                                rem_sec = round(Int, ((nextinfo - init_time) / 1_000_000_000progress) * (num - progress))
                                @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                                    100progress / num, 100acceptance / progress, rem_sec ÷ 60, rem_sec % 60)
                                flush(stdout)
                            end
                            return 0, 0
                        else
                            return Δprogress, Δacceptance
                        end
                    end
                end)
            else
                PolynomialOptimization.newton_polytope_do_worker(V, task, bk,
                    MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers), tmp,
                    let candidates=candidates; p -> append!(candidates, p) end,
                    !verbose ? nothing : let isnotifier=isnotifier, lastinfo=lastinfo,
                                             Δprogress_acceptance=Δprogress_acceptance
                    (Δprogress, Δacceptance) -> let
                        nextinfo = time_ns()
                        @inbounds if nextinfo - lastinfo[] > 10_000_000_000 && trylock(cond)
                            # no reason to block, we can also just retry in the next iteration
                            progress = allprogress[] += Δprogress
                            acceptance = allacceptance[] += Δacceptance
                            if !isnotifier[] && notifier[] == 1
                                isnotifier[] = true
                                notifier[] = 2
                            end
                            allprogress[] = progress
                            allacceptance[] = acceptance
                            unlock(cond)
                            Δprogress_acceptance[1] = progress
                            Δprogress_acceptance[2] = acceptance
                            MPI.Send!(Δprogress_acceptance, comm, dest=root, tag=1)
                            lastinfo[] = nextinfo
                            return 0, 0
                        else
                            return Δprogress, Δacceptance
                        end
                    end
                end)
            end
            # make sure that we update the main list regularly, but not ridiculously often
            nextappend = time_ns()
            if nextappend - lastappend > 10_000_000_000 && trylock(cond)
                try
                    append!(allcandidates, candidates)
                finally
                    unlock(cond)
                end
                empty!(candidates)
                lastappend = nextappend
            end
        end
        if isnotifier[]
            notifier[] = 1
        end
        lock(cond)
        try
            append!(allcandidates, candidates)
        finally
            unlock(cond)
        end
    finally
        Mosek.deletetask(task)
    end
end

function PolynomialOptimization.newton_halfpolytope(V::Val{:Mosek}, objective::P, comm, rank::MPIRank; verbose::Bool=false,
    kwargs...) where {P<:AbstractPolynomialLike}
    nworkers = MPI.Comm_size(comm) # we don't need a master, everyone can do the same work
    isone(nworkers) && return PolynomialOptimization.newton_halfpolytope(V, objective, Val(false); verbose, kwargs...)

    MPI.Barrier(comm)

    if haskey(kwargs, :preprocess_randomized)
        # We want to get the same results of the randomized preprocessing on all workers
        seed = isroot(rank) ? Ref(rand(Int)) : Ref{Int}()
        MPI.Bcast!(seed, root, comm)

        Random.seed!(seed[])
    end
    parameters, coeffs = PolynomialOptimization.newton_polytope_preproc(V, objective; verbose=verbose && isroot(rank),
        warn_disable_randomization=isroot(rank), kwargs...)
    newton_time = @elapsed begin
        nv = size(coeffs, 1)

        maxdeg, mindeg = 0, typemax(Int)
        maxmultideg, minmultideg = fill(0, nv), fill(typemax(Int), nv)
        # do some hopefully quick preprocessing (on top of the previous preprocessing)
        for coeff in eachcol(coeffs)
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

        _, num, nthreads, task, secondtask = PolynomialOptimization.newton_halfpolytope_do_prepare(V, coeffs, mindeg, maxdeg,
            minmultideg, maxmultideg, verbose && isroot(rank); parameters...)
        workersize = div(num, nworkers, RoundUp)
        bk = fill(Mosek.MSK_BK_FX, nv)
        allcandidates = isroot(rank) ? FastVec{Vector{Int}}() : FastVec{Int}()
        if !isone(nthreads)
            threadsize = div(workersize, nthreads, RoundUp)
            cutat = PolynomialOptimization.monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
            cutat_thread = PolynomialOptimization.monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
            if cutat_thread ≥ cutat # why would > even happen?
                nthreads = 1
            end
        end
        if isone(nthreads)
            isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers,
                " single-threaded workers, each checking about ", workersize, " candidates")
            let
                tmp = Vector{Float64}(undef, nv)
                powers = Vector{Int}(undef, nv)
                init_time = time_ns()
                Δprogress_acceptance = Vector{Int}(undef, 2)
                if verbose
                    lastinfo = Ref(init_time)
                    if isroot(rank)
                        allprogress = Ref(0)
                        allacceptance = Ref(0)
                        req = Ref(MPI.Irecv!(Δprogress_acceptance, comm, tag=1))
                    end
                end
                cutat = PolynomialOptimization.monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
                cutlen = nv - cutat
                cutat += 1
                whichworker = 0
                @inbounds for workertask in MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end],
                                                                               maxmultideg[cutat:end],
                                                                               @view(minmultideg[cutat:end]))
                    if rank == whichworker
                        copyto!(maxmultideg, cutat, workertask, 1, cutlen)
                        # instead of putting the multidegs into a Channel, we now directly act on them ring-a-ring
                        if isroot(rank)
                            PolynomialOptimization.newton_polytope_do_worker(V, task, bk,
                                MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, powers), tmp,
                                let allcandidates=allcandidates; p -> push!(allcandidates, copy(p)) end,
                                !verbose ? nothing : let comm=comm, lastinfo=lastinfo, init_time=init_time, req=req,
                                                         allprogress=allprogress, allacceptance=allacceptance, num=num,
                                                         Δprogress_acceptance=Δprogress_acceptance
                                    (Δprogress, Δacceptance) -> let
                                        nextinfo = time_ns()
                                        if nextinfo - lastinfo[] > 10_000_000_000
                                            progress = allprogress[] + Δprogress
                                            acceptance = allacceptance[] + Δacceptance
                                            @inbounds while MPI.Test(req[])
                                                progress += Δprogress_acceptance[1]
                                                acceptance += Δprogress_acceptance[2]
                                                req[] = MPI.Irecv!(Δprogress_acceptance, comm, tag=1)
                                            end
                                            allprogress[] = progress
                                            allacceptance[] = acceptance
                                            rem_sec = round(Int, ((time_ns() - init_time) / 1_000_000_000progress) *
                                                (num - progress))
                                            @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                                                100progress / num, 100acceptance / progress, rem_sec ÷ 60, rem_sec % 60)
                                            flush(stdout)
                                            lastinfo[] = nextinfo
                                            return 0, 0
                                        else
                                            return Δprogress, Δacceptance
                                        end
                                    end
                                end)
                        else
                            PolynomialOptimization.newton_polytope_do_worker(V, task, bk,
                                MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, powers), tmp,
                                let allcandidates=allcandidates; p -> append!(allcandidates, p) end,
                                !verbose ? nothing : let comm=comm, lastinfo=lastinfo,
                                    Δprogress_acceptance=Δprogress_acceptance
                                    (Δprogress, Δacceptance) -> let
                                        nextinfo = time_ns()
                                        @inbounds if nextinfo - lastinfo[] > 10_000_000_000
                                            Δprogress_acceptance[1] = Δprogress
                                            Δprogress_acceptance[2] = Δacceptance
                                            MPI.Send(Δprogress_acceptance, comm, dest=root, tag=1)
                                            lastinfo[] = nextinfo
                                            return 0, 0
                                        else
                                            return Δprogress, Δacceptance
                                        end
                                    end
                                end)
                        end
                    end
                    if (whichworker += 1) == nworkers
                        whichworker = 0
                    end
                end
            end
            Mosek.deletetask(task)
        else
            isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers, " workers, each with ",
                nthreads, " threads checking about ", threadsize, " candidates")
            ranges = Base.Channel{NTuple{2,Vector{Int}}}(typemax(Int))
            cond = Threads.SpinLock()
            allprogress = Ref(0)
            allacceptance = Ref(0)
            # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
            ccall(:jl_enter_threaded_region, Cvoid, ())
            try
                threads = Vector{Task}(undef, nthreads)
                # We start the task with tid 1 only after we kicked off all the rest, as it will run in the main thread. In
                # this way, all the other threads can already start working while we feed them data.
                init_time = time_ns()
                notifier = Ref(verbose ? 1 : 0)
                for tid in nthreads:-1:3
                    # secondtask has a solution, so we just use task (better than deletesolution).
                    # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
                    # parallel, even if we make sure not to modify the base task until all copies are done.
                    @inbounds threads[tid] = Threads.@spawn newton_polytope_do_taskfun_mpi($V, $(Mosek.Task(task)), $ranges,
                        $nv, $mindeg, $maxdeg, $bk, $cond, $allprogress, $allacceptance, $allcandidates, $notifier, $init_time,
                        $num, $rank)
                end
                @inbounds threads[2] = Threads.@spawn newton_polytope_do_taskfun_mpi($V, $secondtask, $ranges, $nv, $mindeg,
                    $maxdeg, $bk, $cond, $allprogress, $allacceptance, $allcandidates, $notifier, $init_time, $num, $rank)
                # All threads are running and waiting for stuff to do. So let's now feed them with their jobs.
                cutlen = nv - cutat_thread
                cutat += 1 # cutat is now the first entry to be fixed for the worker
                cutat_thread += 1 # cutat_thread is now the first entry to be fixed for a thread in the worker
                # [entries iterated on thread][entries fixed on thread, iterated on task][entries fixed on task]
                # 1                           cutat_thread                               cutat
                whichworker = 0
                minmultideg_thread = minmultideg[cutat_thread:cutat-1]
                maxmultideg_thread = maxmultideg[cutat_thread:cutat-1]
                @inbounds for _ in MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end],
                                                                      maxmultideg[cutat:end], @view(minmultideg[cutat:end]))
                    if rank == whichworker
                        for _ in MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg_thread, maxmultideg_thread,
                                                                    @view(minmultideg[cutat_thread:cutat-1]))
                            copyto!(maxmultideg, cutat_thread, minmultideg, cutat_thread, cutlen)
                            put!(ranges, (copy(minmultideg), copy(maxmultideg)))
                        end
                    end
                    if (whichworker += 1) == nworkers
                        whichworker = 0
                    end
                end
                close(ranges)
                isroot(rank) && @verbose_info("All tasks set up")
                @inbounds threads[1] = Threads.@spawn newton_polytope_do_taskfun_mpi($V, $task, $ranges, $nv, $mindeg, $maxdeg,
                    $bk, $cond, $allprogress, $allacceptance, $allcandidates, $notifier, $init_time, $num, $rank)
                for thread in threads
                    wait(thread)
                end
            finally
                ccall(:jl_exit_threaded_region, Cvoid, ())
            end
        end

        MPI.Barrier(comm)
    end

    if isroot(rank)
        @verbose_info("\33[2KFinished construction of all Newton polytope element on the workers in ", newton_time,
            " seconds. Fetching and sorting data.")
        fetch_time = @elapsed begin
            sizes = Vector{Int}(undef, nworkers -1)
            for _ in 2:nworkers
                size, status = MPI.Recv(Int, comm, MPI.Status, tag=2)
                @inbounds sizes[status.source] = size
            end
            prepare_push!(allcandidates, sum(sizes, init=0))
            buffer = Matrix{Int}(undef, nv, maximum(sizes, init=0))
            @inbounds for (rankᵢ, sizeᵢ) in enumerate(sizes)
                MPI.Recv!(buffer, comm, source=rankᵢ, tag=3)
                for j in 1:sizeᵢ
                    unsafe_push!(allcandidates, buffer[:, j])
                end
            end
            allresult = sort!(finish!(allcandidates), lt=(a, b) -> compare(a, b, Graded{LexOrder}) < 0)
        end

        @verbose_info("Fetched ", length(allresult), " elements in the Newton polytope in ", fetch_time, " seconds")
        return PolynomialOptimization.makemonovec(variables(objective), allresult)
    else
        localresult = reshape(finish!(allcandidates), nv, length(allcandidates) ÷ nv)
        MPI.Send(size(localresult, 2), comm, dest=root, tag=2)
        MPI.Send(localresult, comm, dest=root, tag=3)
        return
    end
end

function PolynomialOptimization.newton_halfpolytope(V::Val{:Mosek}, objective::P, ::Val{true}; kwargs...) where {P<:AbstractPolynomialLike}
    if isone(Threads.nthreads())
        MPI.Init(threadlevel=MPI.THREAD_SINGLE)
    elseif MPI.Init(threadlevel=MPI.THREAD_FUNNELED) < MPI.THREAD_FUNNELED
        @error "Invalid combination of MPI and Julia threads. Exiting."
        return
    elseif !MPI.Is_thread_main()
        @error "This function must be called from the thread that is identified as the MPI main thread."
        return
    end
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    return newton_halfpolytope(V, objective, comm, MPIRank(rank); kwargs...)
end

end