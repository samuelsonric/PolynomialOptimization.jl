module PolynomialOptimizationMPI

using PolynomialOptimization, MultivariatePolynomials, Printf
import MPI, Random, Mosek
import PolynomialOptimization: @verbose_info, @capture, FastVec, prepare_push!, unsafe_push!, finish!,
    haveMPI, newton_polytope_preproc, newton_polytope_do_worker, newton_polytope_do_taskfun, monomial_cut,
    newton_halfpolytope_analyze, newton_halfpolytope_tighten, newton_halfpolytope_do_prepare, newton_halfpolytope_do_execute,
    newton_halfpolytope, makemonovec

__init__() = haveMPI[] = true

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

function newton_polytope_do_taskfun(V::Val{:Mosek}, task, ranges, nv, mindeg, maxdeg, bk, cond, allprogress, allacceptance,
    allcandidates, verbose, rank::MPIRank)
    lastappend = time_ns()
    lastinfo = Ref(lastappend)
    powers = Vector{typeof(maxdeg)}(undef, nv)
    tmp = Vector{Float64}(undef, nv)
    candidates = FastVec{typeof(maxdeg)}()
    Δprogress = Ref(0)
    Δacceptance = Ref(0)
    try
        while true
            local curminrange, curmaxrange
            try
                curminrange, curmaxrange = take!(ranges)
            catch e
                e isa InvalidStateException && break
                rethrow(e)
            end
            newton_polytope_do_worker(V, task, bk,
                MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers), tmp, Δprogress,
                Δacceptance, @capture(p -> append!($candidates, p)),
                !verbose ? nothing : @capture(() -> let
                    nextinfo = time_ns()
                    if nextinfo - $lastinfo[] > 1_000_000_000 && trylock($cond)
                        # no reason to block, we can also just retry in the next iteration
                        $allprogress[] += $Δprogress[]
                        $allacceptance[] += $Δacceptance[]
                        unlock(cond)
                        lastinfo[] = nextinfo
                        Δprogress[] = 0
                        Δacceptance[] = 0
                        yield()
                    end
                end
            ))
            nextinfo = time_ns()
            if verbose && nextinfo - lastinfo[] > 1_000_000_000 && trylock(cond)
                # make sure to update regularly, even if the workers are just minimal
                allprogress[] += Δprogress[]
                allacceptance[] += Δacceptance[]
                unlock(cond)
                lastinfo[] = nextinfo
                Δprogress[] = 0
                Δacceptance[] = 0
                yield()
            end
            # make sure that we update the main list regularly, but not ridiculously often
            nextappend = time_ns()
            if nextappend - lastappend > 10_000_000_000
                lock(cond)
                try
                    if isroot(rank)
                        prepare_push!(allcandidates, length(candidates) ÷ nv)
                        for i in 1:nv:length(candidates)
                            @inbounds unsafe_push!(allcandidates, convert(Vector{Int}, @view(candidates[i:i+nv-1])))
                        end
                    else
                        append!(allcandidates, candidates)
                    end
                finally
                    unlock(cond)
                end
                empty!(candidates)
                lastappend = nextappend
            end
        end
        # at the end, exit gracefully
        lock(cond)
        try
            if isroot(rank)
                prepare_push!(allcandidates, length(candidates) ÷ nv)
                for i in 1:nv:length(candidates)
                    @inbounds unsafe_push!(allcandidates, convert(Vector{Int}, @view(candidates[i:i+nv-1])))
                end
            else
                append!(allcandidates, candidates)
            end
            allprogress[] += Δprogress[]
            allacceptance[] += Δacceptance[]
        finally
            unlock(cond)
        end
    finally
        Mosek.deletetask(task)
    end
end

function newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, rank::MPIRank)
    Δprogress_acceptance = Vector{Int}(undef, 2)
    Δbuffer = MPI.Buffer(Δprogress_acceptance)
    self_running = Ref(true)
    # We take care of the gathering of information in a separate task that must run on the main thread (due to MPI funneling).
    # This requires the other tasks (and the main one) to yield regularly.
    if isroot(rank)
        running = Ref(MPI.Comm_size(comm))
        req = Ref(MPI.Irecv!(Δbuffer, comm, tag=1))
        check = @capture(() -> let
            @inbounds for _ in 1:($self_running[] ? $running[] -1 : running[])
                while !MPI.Test($req[])
                    yield() # this is MPI.Wait with schedular-aware waiting
                end
                if $Δprogress_acceptance[1] < 0
                    running[] -= 1
                else
                    $progress[] += Δprogress_acceptance[1]
                    $acceptance[] += Δprogress_acceptance[2]
                end
                req[] = MPI.Irecv!($Δbuffer, $comm, tag=1)
            end
            prog = max(1, progress[]) # just to be sure that we don't divide by zero, though we really expect progress[] > 0
            rem_sec = round(Int, ((time_ns() - $init_time) / 1_000_000_000prog) * ($num - prog))
            @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
               100prog / num, 100acceptance[] / prog, rem_sec ÷ 60, rem_sec % 60)
            flush(stdout)
        end)
        notifier = Task(@capture(() -> begin
            while !iszero($running[])
                $check()
                if $self_running[] && running[] == 1
                    sleep(1) # we are not waiting for any external data, so we must sleep manually
                else
                    yield() # the sleep will come due to the MPI waiting
                end
            end
            check() # make sure to have a final report
        end))
    else
        running = Ref(1)
        check = @capture(() -> begin
            $Δprogress_acceptance[1] = $progress[]
            Δprogress_acceptance[2] = $acceptance[]
            progress[] = 0
            acceptance[] = 0
            MPI.Send($Δbuffer, $comm, dest=root, tag=1)
        end)
        notifier = Task(@capture(() -> begin
            while $self_running[]
                $check()
                sleep(1)
            end
            $check() # make sure to have a final report
            # and then signal that we are done
            Δprogress_acceptance[1] = -1
            MPI.Send(Δbuffer, comm, dest=root, tag=1)
        end))
    end
    notifier.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), notifier, 0)
    return schedule(notifier), running, self_running
end

function newton_halfpolytope_notifier(num, progress, acceptance, init_time, cond, comm, rank::MPIRank)
    Δprogress_acceptance = Vector{Int}(undef, 2)
    Δbuffer = MPI.Buffer(Δprogress_acceptance)
    self_running = Ref(true)
    # We take care of the gathering of information in a separate task that must run on the main thread (due to MPI
    # funneling). This requires the other tasks (and the main one) to yield regularly.
    if isroot(rank)
        running = Ref(MPI.Comm_size(comm))
        req = Ref(MPI.Irecv!(Δbuffer, comm, tag=1))
        check = @capture(() -> let
            δprogress = 0
            δacceptance = 0
            @inbounds for _ in 1:($self_running[] ? $running[] -1 : running[])
                while !MPI.Test($req[])
                    yield()
                end
                if $Δprogress_acceptance[1] < 0
                    running[] -= 1
                else
                    δprogress += Δprogress_acceptance[1]
                    δacceptance += Δprogress_acceptance[2]
                end
                req[] = MPI.Irecv!($Δbuffer, $comm, tag=1)
            end
            lock($cond)
            prog = max(1, ($progress[] += δprogress)) # we divide, so if this task runs too early, we might be in trouble
            acc = ($acceptance[] += δacceptance)
            unlock(cond)
            rem_sec = round(Int, ((time_ns() - $init_time) / 1_000_000_000prog) * ($num - prog))
            @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                100prog / num, 100acc / prog, rem_sec ÷ 60, rem_sec % 60)
            flush(stdout)
        end)
        notifier = Task(@capture(() -> begin
            while !iszero($running[])
                $check()
                if $self_running[] && running[] == 1
                    sleep(1) # we are not waiting for any external data, so we must sleep manually
                else
                    yield() # the sleep will come due to the MPI waiting
                end
            end
            check() # make sure to have a final report
        end))
    else
        running = Ref(1)
        check = @capture(() -> begin
            lock($cond)
            @inbounds $Δprogress_acceptance[1] = $progress[]
            @inbounds Δprogress_acceptance[2] = $acceptance[]
            progress[] = 0
            acceptance[] = 0
            unlock(cond)
            MPI.Send($Δbuffer, $comm, dest=root, tag=1)
        end)
        notifier = Task(@capture(() -> begin
            while $self_running[]
                $check()
                sleep(1)
            end
            $check() # make sure to have a final report
            # and then signal that we are done
            Δprogress_acceptance[1] = -1
            MPI.Send(Δbuffer, comm, dest=root, tag=1)
        end))
    end
    notifier.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), notifier, 0)
    return schedule(notifier), running, self_running
end

function newton_halfpolytope_do_execute(V::Val{:Mosek}, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, nthreads,
    task, secondtask, comm, rank::MPIRank)
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)

    bk = fill(Mosek.MSK_BK_FX, nv)
    candidates = isroot(rank) ? FastVec{Vector{Int}}() : FastVec{typeof(maxdeg)}()
    progress = Ref(0)
    acceptance = Ref(0)
    cutat_worker = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
    cutlen_worker = nv - cutat_worker
    if !isone(nthreads)
        threadsize = div(workersize, nthreads, RoundUp)
        cutat_thread = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
        if cutat_thread ≥ cutat_worker # why would > even happen?
            nthreads = 1
        else
            cutlen_thread = cutat_worker - cutat_thread
            cutat_thread += 1 # cutat_thread is now the first entry to be fixed for a thread in the worker
            minmultideg_thread = minmultideg[cutat_thread:cutat_worker]
            maxmultideg_thread = maxmultideg[cutat_thread:cutat_worker]
        end
    end

    if verbose && isroot(rank)
        if isone(nthreads)
            println("Preparing to determine Newton polytope using ", nworkers,
                " single-threaded workers, each checking about ", workersize, " candidates")
        else
            println("Preparing to determine Newton polytope using ", nworkers, " workers, each with ", nthreads,
                " threads checking about ", threadsize, " candidates")
        end
        flush(stdout)
    end
    if isone(nthreads)
        powers = Vector{isroot(rank) ? Int : typeof(maxdeg)}(undef, nv)
        tmp = Vector{Float64}(undef, nv)
    else
        ranges = Base.Channel{NTuple{2,Vector{typeof(maxdeg)}}}(typemax(Int))
        cond = Threads.SpinLock()
        ccall(:jl_enter_threaded_region, Cvoid, ())
        # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
        threads = Vector{Task}(undef, nthreads)
        # We can already start all the tasks; this main task that must still feed the data will continue running until we
        # yield to the scheduler.
        for tid in nthreads:-1:3
            # secondtask has a solution, so we just use task (better than deletesolution).
            # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
            # parallel, even if we make sure not to modify the base task until all copies are done.
            @inbounds threads[tid] = Threads.@spawn newton_polytope_do_taskfun($V, $(Mosek.Task(task)), $ranges, $nv, $mindeg,
                $maxdeg, $bk, $cond, $progress, $acceptance, $candidates, $verbose, $rank)
        end
        @inbounds threads[2] = Threads.@spawn newton_polytope_do_taskfun($V, $secondtask, $ranges, $nv, $mindeg, $maxdeg, $bk,
            $cond, $progress, $acceptance, $candidates, $verbose, $rank)
        @inbounds threads[1] = Threads.@spawn newton_polytope_do_taskfun($V, $task, $ranges, $nv, $mindeg, $maxdeg, $bk, $cond,
            $progress, $acceptance, $candidates, $verbose, $rank)
    end
    init_time = time_ns()
    if verbose
        notifier, running, self_running = isone(nthreads) ?
            newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, rank) :
            newton_halfpolytope_notifier(num, progress, acceptance, init_time, cond, comm, rank)
    end
    # We do not want to do a lot of communication between the tasks as is done in the multithreaded approach. There, we just
    # feed every item into the channel and the first one that is available pops it. Here, we check beforehand how many items
    # we can expect in an individual batch and then decide how many batches will be done by the rank.
    occurrences_before = length(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[1:cutat_worker],
        maxmultideg[1:cutat_worker]), Val(:detailed))
    cutat_worker += 1 # cutat is now the first entry to be fixed for the worker
    workload = zeros(Int, nworkers)
    @inbounds for ranktask in MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat_worker:end],
        maxmultideg[cutat_worker:end], @view(minmultideg[cutat_worker:end]))
        fixdeg = sum(ranktask, init=zero(maxdeg))
        # The index i in occurrences_before that would have corresponded to the degree i-1 now corresponds to i-1+fixdeg.
        # Consequently, we only take those degrees into account that are at least mindeg, which now has start index
        # mindeg+1-fixdeg, and we need to introduce a new maxdegree cutoff.
        tasklen = sum(@view(occurrences_before[max(1, Base.bitcast(Int, mindeg + 1 - fixdeg)):maxdeg + 1 - fixdeg]), init=0)
        iszero(tasklen) && continue
        copyto!(maxmultideg, cutat_worker, ranktask, 1, cutlen_worker)
        # The lowest-rank worker with the least workload will get the job.
        minworkloadᵢ = 1
        minworkload = workload[1]
        for i in 2:nworkers
            if workload[i] < minworkload
                minworkloadᵢ = i
                minworkload = workload[i]
            end
        end
        workload[minworkloadᵢ] += tasklen
        if rank == minworkloadᵢ -1
            # it's our job!
            if isone(nthreads)
                lastinfo = Ref(init_time)
                # Here, we cannot just use moniter, although it points to the current minmultideg, maxmultideg - but moniter
                # in its initialization caches some data.
                newton_polytope_do_worker(V, task, bk, MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg,
                    maxmultideg, powers), tmp, progress, acceptance,
                    isroot(rank) ? @capture(p -> push!($candidates, copy(p))) :
                                   @capture(p -> append!($candidates, p)),
                    !verbose ? nothing : @capture(() -> let
                        nextinfo = time_ns()
                        if nextinfo - $lastinfo[] > 1_000_000_000
                            yield()
                            lastinfo[] = nextinfo
                        end
                    end)
                )
                yield()
            else
                # All threads are running and waiting for stuff to do. So let's now feed them with their jobs.
                # [entries iterated on thread][entries fixed on thread, iterated on task][entries fixed on task]
                # 1                           cutat_thread                               cutat
                for _ in MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg_thread, maxmultideg_thread,
                                                            @view(minmultideg[cutat_thread:cutat_worker-1]))
                    copyto!(maxmultideg, cutat_thread, minmultideg, cutat_thread, cutlen_thread)
                    put!(ranges, (copy(minmultideg), copy(maxmultideg)))
                end
                # We won't get a progress report while we're pushing stuff into the channel, but if we yield here, we
                # potentially delay the filling of the channel by allowing the worker task in the main thread to take over.
            end
        end
    end
    if verbose && isroot(rank)
        # there may already be a progress in print, so we'll want to clear the line
        if nworkers ≤ 20
            println("\33[2KExact workload distribution: $workload")
        else
            let minv=typemax(Int), maxv=0, meanv=0., stdv=0.
                @simd for workloadᵢ in workload
                    if workloadᵢ < minv
                        minv = workloadᵢ
                    end
                    if workloadᵢ > maxv
                        maxv = workloadᵢ
                    end
                    meanv += workloadᵢ
                end
                meanv /= nworkers
                @simd for workloadᵢ in workload
                    stdv += (workloadᵢ - meanv)^2
                end
                stdv = sqrt(stdv / (nworkers -1))
                println("\33[2KExact workload distribution: range [$minv, $maxv], mean $meanv, standard deviation $stdv")
            end
        end
        flush(stdout)
    end
    if isone(nthreads)
        Mosek.deletetask(task)
    else
        close(ranges)
        for thread in threads
            wait(thread)
        end
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
    if verbose
        running[] -= 1
        self_running[] = false
        wait(notifier)
    end
    return candidates
end

function newton_halfpolytope(V::Val{:Mosek}, objective::P, comm, rank::MPIRank; verbose::Bool=false, kwargs...) where {P<:AbstractPolynomialLike}
    nworkers = MPI.Comm_size(comm) # we don't need a master, everyone can do the same work
    isone(nworkers) && return newton_halfpolytope(V, objective, Val(false); verbose, kwargs...)

    MPI.Barrier(comm)

    if haskey(kwargs, :preprocess_randomized)
        # We want to get the same results of the randomized preprocessing on all workers
        seed = isroot(rank) ? Ref(rand(Int)) : Ref{Int}()
        MPI.Bcast!(seed, root, comm)

        Random.seed!(seed[])
    end
    parameters, coeffs = newton_polytope_preproc(V, objective; verbose=verbose && isroot(rank),
        warn_disable_randomization=isroot(rank), kwargs...)
    nv = size(coeffs, 1)

    if isroot(rank)
        newton_time = @elapsed allresult = let
            analysis = newton_halfpolytope_analyze(coeffs)
            num, nthreads, task, secondtask = newton_halfpolytope_do_prepare(V, coeffs, analysis..., verbose;
                parameters...)
            tightened = newton_halfpolytope_tighten(analysis...)
            candidates = newton_halfpolytope_do_execute(V, size(coeffs, 1), (isone(nthreads) ? analysis : tightened)...,
                verbose, num, nthreads, task, secondtask, comm, rank)
            T = typeof(tightened[1])
            sizes = Vector{Int}(undef, nworkers -1)
            for _ in 2:nworkers
                size, status = MPI.Recv(Int, comm, MPI.Status, tag=2)
                @inbounds sizes[status.source] = size
            end
            prepare_push!(candidates, sum(sizes, init=0))
            buffer = Matrix{T}(undef, nv, maximum(sizes, init=0))
            @inbounds for (rankᵢ, sizeᵢ) in enumerate(sizes)
                MPI.Recv!(buffer, comm, source=rankᵢ, tag=3)
                for c in eachcol(@view(buffer[:, 1:sizeᵢ]))
                    unsafe_push!(candidates, convert(Vector{Int}, c))
                end
            end
            sort!(finish!(candidates), lt=(a, b) -> compare(a, b, Graded{LexOrder}) < 0)
        end

        @verbose_info("\33[2KFinished construction of all Newton polytope element on the workers in ", newton_time,
            " seconds.")
        return makemonovec(variables(objective), allresult)
    else
        let
            analysis = newton_halfpolytope_analyze(coeffs)
            num, nthreads, task, secondtask = newton_halfpolytope_do_prepare(V, coeffs, analysis..., false;
                parameters...)
            candidates = newton_halfpolytope_do_execute(V, size(coeffs, 1), newton_halfpolytope_tighten(analysis...)...,
                verbose, num, nthreads, task, secondtask, comm, rank)
            localresult = reshape(finish!(candidates), nv, length(candidates) ÷ nv)
            MPI.Send(size(localresult, 2), comm, dest=root, tag=2)
            MPI.Send(localresult, comm, dest=root, tag=3)
            return
        end
    end
end

function newton_halfpolytope(V::Val{:Mosek}, objective::P, ::Val{true}; kwargs...) where {P<:AbstractPolynomialLike}
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