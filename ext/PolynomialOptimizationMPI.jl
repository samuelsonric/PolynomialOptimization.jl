module PolynomialOptimizationMPI

using PolynomialOptimization, MultivariatePolynomials, Printf
import MPI, Random, Mosek
import PolynomialOptimization: @verbose_info, @capture, FastVec, prepare_push!, unsafe_push!, finish!,
    haveMPI, newton_polytope_preproc, newton_polytope_do_worker, newton_polytope_do_taskfun, monomial_cut,
    newton_halfpolytope_analyze, newton_halfpolytope_tighten, newton_halfpolytope_do_prepare, InitialStateIterator,
    moniter_state, newton_halfpolytope_restore_status!, newton_halfpolytope_do_execute, newton_halfpolytope, makemonovec

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

function newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, ::RootRank)
    init_progress = progress[]
    self_running = Ref(true)
    # We take care of the gathering of information in a separate task that must run on the main thread (due to MPI funneling).
    # This requires the other tasks (and the main one) to yield regularly.
    workers = Ref(MPI.Comm_size(comm) -1)
    Δprogress_acceptances = [Vector{Int}(undef, 2) for _ in 1:workers[]]
    Δbuffers = MPI.Buffer.(Δprogress_acceptances)
    reqs = MPI.MultiRequest(workers[])
    for (worker, (req, Δbuffer)) in enumerate(zip(reqs, Δbuffers))
        MPI.Irecv!(Δbuffer, comm, req, tag=1, source=worker)
    end
        check = @capture(() -> let
        # We need to give MPI some time to collect all the data; but we don't want to completely block using Waitall for
        # performance reasons. So we wait at most 10ms (= 1% of the calucation time) before we interrupt and resume the
        # calculations.
        # Note that the strange scheme where we receive on all nodes seems to be the only way not to completely loose some
        # packages. If we just receive without any binding to a source, this surprisingly is very unreliable.
        init_wait = time_ns()
        @inbounds while time_ns() - init_wait < 10_000_000
            _, idx = MPI.Testany($reqs)
            if !isnothing(idx)
                Δprogress_acceptance = $Δprogress_acceptances[idx]
                if Δprogress_acceptance[1] < 0
                    $workers[] -= 1
                else
                    $progress[] += Δprogress_acceptance[1]
                    $acceptance[] += Δprogress_acceptance[2]
                    MPI.Irecv!($Δbuffers[idx], $comm, reqs[idx], tag=1, source=idx)
                end
            end
            end
            prog = max(1, progress[]) # just to be sure that we don't divide by zero, though we really expect progress[] > 0
        rem_sec = round(Int, ((time_ns() - $init_time) / (1_000_000_000 * (prog - $init_progress))) * ($num - prog))
            @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
               100prog / num, 100acceptance[] / prog, rem_sec ÷ 60, rem_sec % 60)
            flush(stdout)
        end)
        notifier = Task(@capture(() -> begin
        while !iszero($workers[]) || $self_running[]
            $check()
            sleep(1)
            end
            check() # make sure to have a final report
        end))
    notifier.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), notifier, 0)
    return schedule(notifier), self_running
end

function newton_halfpolytope_notifier(_, progress, acceptance, _, comm, rank::OtherRank)
    self_running = Ref(true)
    Δprogress_acceptance = Vector{Int}(undef, 2)
    Δbuffer = MPI.Buffer(Δprogress_acceptance)
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
        check() # make sure to have a final report
            # and then signal that we are done
        $Δprogress_acceptance[1] = -1
        MPI.Send($Δbuffer, $comm, dest=root, tag=1)
        end))
    notifier.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), notifier, 0)
    return schedule(notifier), self_running
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

function newton_halfpolytope_print_workload(workload)
    nworkers = length(workload)
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

function newton_halfpolytope_restore_status!(fileprogress, workload::Vector{Int}, powers::Vector{T}) where {T<:Integer}
    lastprogress = UInt8[]
    seekstart(fileprogress)
    s = 2sizeof(Int) + sizeof(workload) + sizeof(powers)
    nb = readbytes!(fileprogress, lastprogress, s)
    GC.@preserve lastprogress begin
        if iszero(nb)
            return nothing
        elseif nb == s
            lpp = Ptr{Int}(pointer(lastprogress))
            unsafe_copyto!(pointer(powers), Ptr{T}(lpp + 2sizeof(Int)), length(powers))
            unsafe_copyto!(pointer(workload), lpp + 2sizeof(Int) + sizeof(powers), length(workload))
            return unsafe_load(lpp), unsafe_load(lpp, 2)
        else
            error("Unknown progress file format - please delete existing files.")
        end
    end
end

function newton_halfpolytope_do_execute(V::Val{:Mosek}, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task,
    filepath, comm, rank::MPIRank)
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)
    isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers,
        " single-threaded workers, each checking about ", workersize, " candidates")
    cutat_worker = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
    cutlen_worker = nv - cutat_worker
    occurrences_before = length(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[1:cutat_worker],
        maxmultideg[1:cutat_worker]), Val(:detailed))
    cutat_worker += 1 # cutat is now the first entry to be fixed for the worker

    bk = fill(Mosek.MSK_BK_FX.value, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    if isroot(rank) && isnothing(filepath)
        candidates = FastVec{Vector{Int}}() # don't try to save on the data type, DynamicPolynomials requires Vector{Int}
    else
        candidates = FastVec{typeof(maxdeg)}()
    end

    progress = Ref(0)
    acceptance = Ref(0)
    workload = zeros(Int, nworkers)
    powers = Vector{isroot(rank) && isnothing(filepath) ? Int : typeof(maxdeg)}(undef, nv)
    initialize = false
    # We don't use the initial state iterator here, as we have to jump into the loop
    rankiter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat_worker:end], maxmultideg[cutat_worker:end],
        @view(maxmultideg[cutat_worker:end])) # we iterate on maxmultideg, so that this one is always current
    if isnothing(filepath)
        fileprogress = nothing
        fileout = nothing
    else
        prefix = "$filepath-n$(convert(Int, rank))"
        fileprogress = open("$prefix.prog", read=true, write=true, create=true, lock=false)
        success = fill(false, nworkers)
        local restore
        try
            restore = newton_halfpolytope_restore_status!(fileprogress, workload, powers)
        catch
            close(fileprogress)
            MPI.Allgather!(MPI.UBuffer(success, 1), comm)
            isroot(rank) && error("Unknown progress file format - please delete existing files.")
            return
        end
        @inbounds success[convert(Int, rank)+1] = true
        MPI.Allgather!(MPI.UBuffer(success, 1), comm)
        if !all(success)
            close(fileprogress)
            isroot(rank) && error("Unknown progress file format - please delete existing files.")
            return
        end
        if !isnothing(restore)
            progress[] = restore[1]
            acceptance[] = restore[2]
            copyto!(rankiter.powers, 1, powers, cutat_worker, cutlen_worker)
            initialize = true
        end
        fileout = open("$prefix.out", append=true, lock=false)
    end
    init_time = time_ns()
    if verbose
        notifier, self_running = newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, rank)
    end
    try
        # We do not want to do a lot of communication between the tasks as is done in the multithreaded approach. There, we just
        # feed every item into the channel and the first one that is available pops it. Here, we check beforehand how many items
        # we can expect in an individual batch and then decide how many batches will be done by the rank.
        tmp = Vector{Float64}(undef, nv)
        lastinfo = Ref(init_time)
        # unroll "for ranktask in rankiter" so that we can jump into the loop starting with the previous iteration
        if initialize
            ranktask = rankiter.powers
            rankiter_state = moniter_state(ranktask)
            @goto start
        end
        rankiter_next = iterate(rankiter)
        @inbounds while !isnothing(rankiter_next)
            ranktask, rankiter_state = rankiter_next
            @label start

            fixdeg = typeof(maxdeg)(sum(ranktask, init=zero(maxdeg)))
            # The index i in occurrences_before that would have corresponded to the degree i-1 now corresponds to i-1+fixdeg.
            # Consequently, we only take those degrees into account that are at least mindeg, which now has start index
            # mindeg+1-fixdeg, and we need to introduce a new maxdegree cutoff.
            tasklen = sum(@view(occurrences_before[max(1, mindeg + 1 - fixdeg):maxdeg + 1 - fixdeg]), init=0)
            # if we don't put this assignment of minworkloadᵢ here (instead of the if), we will get internal runtime errors
            # upon compilation if we later on add access workload[minworkloadᵢ], even if we protect it in the same if.
            # Unfortunately, the case is so complex that it's hard to come up with a minimal example. Can you?
            minworkloadᵢ = 1
            if !iszero(tasklen)
                # The lowest-rank worker with the least workload will get the job.
                minworkload = workload[1]
                for i in 2:nworkers
                    if workload[i] < minworkload
                        minworkloadᵢ = i
                        minworkload = workload[i]
                    end
                end
                if rank == minworkloadᵢ -1
                    # it's our job!
                    lastworkload = Ref{typeof(workload)}()
                    copyto!(minmultideg, cutat_worker, ranktask, 1, cutlen_worker)
                    if initialize
                        iter = InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg,
                            maxmultideg, powers), moniter_state(powers))
                    else
                        iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, powers)
                    end
                    newton_polytope_do_worker(V, task, bk, iter, tmp, progress, acceptance,
                        isroot(rank) && isnothing(filepath) ? @capture(p -> push!($candidates, copy(p))) :
                                                              @capture(p -> append!($candidates, p)),
                        !verbose && isnothing(filepath) ? nothing : @capture(p -> let nextinfo=time_ns()
                            if nextinfo - $lastinfo[] > 1_000_000_000
                                if !isnothing($filepath)
                                    write($fileout, $candidates)
                                    flush(fileout)
                                    seekstart($fileprogress)
                                    write(fileprogress, $progress[], $acceptance[], p)
                                    if !isassigned($lastworkload) || lastworkload[] !== $workload
                                        write(fileprogress, workload)
                                        lastworkload[] = workload
                                    end
                                    flush(fileprogress)
                                    empty!(candidates)
                                end
                                $verbose && yield()
                                lastinfo[] = nextinfo
                            end
                        end)
                    )
                end
            end
            if verbose || !isnothing(filepath)
                nextinfo = time_ns()
                if nextinfo - lastinfo[] > 1_000_000_000
                    if !isnothing(filepath)
                        write(fileout, candidates)
                        flush(fileout)
                        seekstart(fileprogress)
                        # we always write the last completed iteration, which is maxmultideg
                        write(fileprogress, progress[], acceptance[], maxmultideg, workload)
                        flush(fileprogress)
                        empty!(candidates)
                    end
                    verbose && yield()
                    lastinfo[] = nextinfo
                end
            end
            workload[minworkloadᵢ] += tasklen
            initialize = false
            rankiter_next = iterate(rankiter, rankiter_state)
        end
        if !isnothing(filepath)
            write(fileout, candidates)
            seekstart(fileprogress)
            # Even here, we don't signal completion - all the notifications must still run between the workers, so let's just
            # store the last element.
            write(fileprogress, progress[], acceptance[], workload, maxmultideg)
        end
        verbose && isroot(rank) && newton_halfpolytope_print_workload(workload)
    finally
        Mosek.deletetask(task)
        if !isnothing(fileout)
            close(fileout)
            close(fileprogress)
        end
        if verbose
            self_running[] = false
            wait(notifier)
        end
        return isnothing(fileout) ? candidates : nothing
    end
end

function newton_halfpolytope_do_execute(V::Val{:Mosek}, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num,
    nthreads::Integer, task, secondtask, filepath, comm, rank::MPIRank)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath,
            comm, rank)
    end
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

function newton_halfpolytope(V::Val{:Mosek}, objective::P, comm, rank::MPIRank; verbose::Bool=false,
    filepath::Union{<:AbstractString,Nothing}=nothing, kwargs...) where {P<:AbstractPolynomialLike}
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
            candidates = newton_halfpolytope_do_execute(V, size(coeffs, 1),
                (isone(nthreads) && isnothing(filepath) ? analysis : tightened)..., verbose, num, nthreads, task, secondtask,
                filepath, comm, rank)
            if isnothing(filepath)
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
        end

        @verbose_info("\33[2KFinished construction of all Newton polytope element on the workers in ", newton_time,
            " seconds.")
        return isnothing(filepath) ? makemonovec(variables(objective), allresult) : true
    else
        let
            analysis = newton_halfpolytope_analyze(coeffs)
            num, nthreads, task, secondtask = newton_halfpolytope_do_prepare(V, coeffs, analysis..., false;
                parameters...)
            candidates = newton_halfpolytope_do_execute(V, size(coeffs, 1), newton_halfpolytope_tighten(analysis...)...,
                verbose, num, nthreads, task, secondtask, filepath, comm, rank)
            if isnothing(filepath)
            localresult = reshape(finish!(candidates), nv, length(candidates) ÷ nv)
            MPI.Send(size(localresult, 2), comm, dest=root, tag=2)
            MPI.Send(localresult, comm, dest=root, tag=3)
            end
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