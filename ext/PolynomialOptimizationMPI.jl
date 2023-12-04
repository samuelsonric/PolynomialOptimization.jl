module PolynomialOptimizationMPI

using PolynomialOptimization, MultivariatePolynomials, Printf
import MPI, Random
import PolynomialOptimization: @verbose_info, @capture, FastVec, prepare_push!, unsafe_push!, finish!, haveMPI,
    newton_polytope_preproc, newton_polytope_do_worker, newton_polytope_do_taskfun, monomial_cut, newton_halfpolytope_analyze,
    newton_halfpolytope_tighten, newton_halfpolytope_do_prepare, InitialStateIterator, moniter_state,
    newton_halfpolytope_restore_status!, toSigned, newton_halfpolytope_do_execute, newton_halfpolytope_alloc,
    newton_halfpolytope, newton_halfpolytope_clonetask, makemonovec

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

function newton_polytope_do_taskfun(V, tid, task, ranges, nv, mindeg, maxdeg, bk, cond, progresses,
    acceptances, allcandidates, verbose, filestuff, rank::MPIRank)
    lastappend = time_ns()
    lastinfo = Ref(lastappend)
    powers = Vector{typeof(maxdeg)}(undef, nv)
    tmp = Vector{Float64}(undef, nv)
    candidates = FastVec{typeof(maxdeg)}()
    progress = Ref(progresses, tid)
    acceptance = Ref(acceptances, tid)
    workload = nothing # we need it for capturing, but also to be preseved between loops
    if isnothing(filestuff)
        fileprogress = nothing
        fileout = nothing
        cut = 0
        writefile = (p, workload) -> nothing
        appendorwrite = @capture (p, workload) -> begin
            lock($cond)
            try
                if isroot($rank)
                    prepare_push!($allcandidates, length($candidates) ÷ $nv)
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
        end
    else
        fileprogress, fileout = filestuff
        lastworkload = Ref{Vector{Int}}()
        writefile = @capture (p, workload) -> begin
            write($fileout, $candidates)
            flush(fileout)
            seekstart($fileprogress)
            # we always write the last completed iteration, which is maxmultideg
            write(fileprogress, $progress[], $acceptance[], p)
            if !isassigned($lastworkload) || lastworkload[] !== workload
                write(fileprogress, workload)
                lastworkload[] = workload
            end
            flush(fileprogress)
            empty!(candidates)
        end
        appendorwrite = writefile
    end
    try
        while true
            local curminrange, curmaxrange, restore
            restore = nothing
            try
                if isnothing(filestuff)
                    curminrange, curmaxrange = take!(ranges)
                else
                    workload, curminrange, curmaxrange, restore = take!(ranges)
                end
            catch e
                e isa InvalidStateException && break
                rethrow(e)
            end
            iter_ = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers)
            if isnothing(restore)
                iter = iter_
            else
                copyto!(powers, restore)
                iter = InitialStateIterator(iter_, moniter_state(powers))
            end
            newton_polytope_do_worker(V, task, bk, iter, tmp, progress, acceptance, @capture(p -> append!($candidates, p)),
                !verbose && isnothing(filestuff) ? nothing : @capture(p -> let
                    nextinfo = time_ns()
                    if nextinfo - $lastinfo[] > 1_000_000_000
                        @inline $writefile(p, $workload)
                        $verbose && yield()
                        lastinfo[] = nextinfo
                    end
                end)
            )
            if verbose || !isnothing(filepath)
                nextinfo = time_ns()
                if nextinfo - lastinfo[] > 1_000_000_000
                    @inline writefile(powers, workload)
                    verbose && yield()
                    lastinfo[] = nextinfo
                end
            end
            # make sure that we update the main list regularly, but not ridiculously often
            nextappend = time_ns()
            if nextappend - lastappend > 10_000_000_000
                @inline appendorwrite(powers, workload)
                lastappend = nextappend
            end
        end
        # at the end, exit gracefully. It may happen that this thread did not even participate once, so workload could still
        # be unset
        isnothing(workload) || @inline appendorwrite(powers, workload)
    finally
        finalize(task)
        if !isnothing(filestuff)
            close(fileout)
            close(fileprogress)
        end
    end
end

function newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, ::RootRank, threadprogress=missing,
    threadacceptance=missing)
    init_progress = progress[]
    ismissing(threadprogress) || (init_progress += sum(threadprogress, init=0))
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
        δprogress = 0
        δacceptance = 0
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
                    δprogress += Δprogress_acceptance[1]
                    δacceptance += Δprogress_acceptance[2]
                    MPI.Irecv!($Δbuffers[idx], $comm, reqs[idx], tag=1, source=idx)
                end
            end
        end
        # no need to lock here, in the multithreaded case, progress = workerprogress, which is only changed here
        prog = ($progress[] += δprogress)
        acc = ($acceptance[] += δacceptance)
        ismissing($threadacceptance) || (acc += sum(threadacceptance, init=0))
        ismissing($threadprogress) || (prog += sum(threadprogress, init=0))
        iszero(prog) && (prog = 1) # we divide, so if this task runs too early, we might be in trouble
        Δt = prog == $init_progress ? 1 : prog - init_progress # if a finished job is started, this might happen
        rem_sec = round(Int, ((time_ns() - $init_time) / 1_000_000_000Δt) * ($num - prog))
        @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
            100prog / num, 100acc / prog, rem_sec ÷ 60, rem_sec % 60)
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

function newton_halfpolytope_notifier(num, progress, acceptance, init_time, comm, ::OtherRank, threadprogress=missing,
    threadacceptance=missing)
    self_running = Ref(true)
    Δprogress_acceptance = Vector{Int}(undef, 2)
    Δbuffer = MPI.Buffer(Δprogress_acceptance)
    # We take care of the gathering of information in a separate task that must run on the main thread (due to MPI funneling).
    # This requires the other tasks (and the main one) to yield regularly.
    check = @capture(() -> begin
        if ismissing($threadprogress)
            @inbounds $Δprogress_acceptance[1] = $progress[]
            @inbounds Δprogress_acceptance[2] = $acceptance[]
            progress[] = 0
            acceptance[] = 0
        else
            # in the multithreaded case, progress contains the total progress that we sent so far, which allows us to skip any
            # locking here
            Σacceptance = sum(threadacceptance, init=0)
            Σprogress = sum(threadprogress, init=0)
            @inbounds Δprogress_acceptance[1] = Σprogress - progress[]
            @inbounds Δprogress_acceptance[2] = Σacceptance - acceptance[]
            progress[] = Σprogress
            acceptance[] = Σacceptance
        end
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
        MPI.Send($Δbuffer, $comm, dest=root, tag=1)
    end))
    notifier.sticky = true
    ccall(:jl_set_task_tid, Cint, (Any, Cint), notifier, 0)
    return schedule(notifier), self_running
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

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath, comm,
    rank::MPIRank)
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)
    isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers,
        " single-threaded workers, each checking about ", workersize, " candidates")
    cutat_worker = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
    cutlen_worker = nv - cutat_worker
    occurrences_before = length(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[1:cutat_worker],
        maxmultideg[1:cutat_worker]), Val(:detailed))
    cutat_worker += 1 # cutat is now the first entry to be fixed for the worker

    bk = newton_halfpolytope_alloc(V, nv)
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
        finalize(task)
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

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, nthreads::Integer, task,
    secondtask, filepath, comm, rank::MPIRank)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath,
            comm, rank)
    end
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)
    threadsize = div(workersize, nthreads, RoundUp)
    isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers, " workers, each with ", nthreads,
        " threads checking about ", threadsize, " candidates")
    cutat_worker = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, workersize)
    cutlen_worker = nv - cutat_worker
    cutat_thread = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
    if cutat_thread ≥ cutat_worker # why would > even happen?
        finalize(secondtask)
        return newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath,
            comm, rank)
    else
        cutlen_thread = cutat_worker - cutat_thread
        cutat_thread += 1 # cutat_thread is now the first entry to be fixed for a thread in the worker
        minmultideg_thread = minmultideg[cutat_thread:cutat_worker]
        maxmultideg_thread = maxmultideg[cutat_thread:cutat_worker]
    end
    occurrences_before = length(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[1:cutat_worker],
        maxmultideg[1:cutat_worker]), Val(:detailed))
    cutat_worker += 1 # cutat is now the first entry to be fixed for the worker

    bk = newton_halfpolytope_alloc(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    if isroot(rank) && isnothing(filepath)
        candidates = FastVec{Vector{Int}}() # don't try to save on the data type, DynamicPolynomials requires Vector{Int}
    else
        candidates = FastVec{typeof(maxdeg)}()
    end

    threadprogress = zeros(Int, nthreads)
    threadacceptance = zeros(Int, nthreads)
    workerprogress = Ref(0)
    workeracceptance = Ref(0)
    workload = zeros(Int, nworkers)
    workloadcopy = nothing # we need this to avoid a compile error when we later access workloadcopy, although it is only in
                           # cases in which it is defined
    initialize = false
    # We don't use the initial state iterator here, as we have to jump into the loop
    rankiter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat_worker:end], maxmultideg[cutat_worker:end],
        @view(maxmultideg[cutat_worker:end])) # we iterate on maxmultideg, so that this one is always current
    if isnothing(filepath)
        ranges = Base.Channel{NTuple{2,typeof(maxmultideg)}}(typemax(Int))
    else
        # ranges must now cover much more than in the single-worker case:
        # - workload, we have to restore the workload vector as it was before the worker decided to take the task upon itself
        # - minmultideg and maxmultideg for this thread
        # - potentially a position within minmultideg/maxmultideg that was already processed
        ranges = Base.Channel{Tuple{Vector{Int},typeof(maxmultideg),typeof(maxmultideg),
                                    Union{Nothing,typeof(maxmultideg)}}}(typemax(Int))
        prefix = "$filepath-n$(convert(Int, rank))"
        fileprogresses = Vector{IOStream}(undef, nthreads)
        fileouts = Vector{IOStream}(undef, nthreads)
        maxrankitr = missing
        maxitr = missing
        success = fill(false, nworkers)
        @inbounds try
            curworkload = similar(workload)
            for i in 1:nthreads
                fileprogresses[i] = fileprogress = open("$prefix-$i.prog", read=true, write=true, create=true, lock=false)
                fileouts[i] = open("$prefix-$i.out", append=true, lock=false)
                curpower = Vector{typeof(maxdeg)}(undef, nv)
                currestore = newton_halfpolytope_restore_status!(fileprogress, curworkload, curpower)
                if !isnothing(currestore)
                    threadprogress[i], threadacceptance[i] = currestore
                    restartmin = copy(minmultideg)
                    restartmax = copy(maxmultideg)
                    copyto!(restartmin, cutat_thread, curpower, cutat_thread, cutlen_thread + cutlen_worker)
                    copyto!(restartmax, cutat_thread, curpower, cutat_thread, cutlen_thread + cutlen_worker)
                    put!(ranges, (curworkload, restartmin, restartmax, curpower))
                    currankitr = @view(curpower[cutat_worker:end])
                    if ismissing(maxrankitr) || curworkload > workload
                        @assert(
                            ismissing(maxrankitr) ||
                            toSigned(compare(maxrankitr, currankitr, Graded{LexOrder})) < 0
                        )
                        workload, curworkload = curworkload, workload
                        maxrankitr = currankitr
                        maxitr = @view(curpower[cutat_thread:end])
                    elseif currankitr == maxrankitr
                        curitr = @view(curpower[cutat_thread:end])
                        if toSigned(compare(@view(maxitr[1:cutlen_thread]), @view(curitr[1:cutlen_thread]),
                                Graded{LexOrder})) < 0
                            maxitr = curitr
                        end
                    end
                end
            end
        catch
            for i in 1:nthreads
                isassigned(fileprogresses, i) || break
                close(fileprogresses[i])
                isassigned(fileouts, i) && close(fileouts[i])
            end
            MPI.Allgather!(MPI.UBuffer(success, 1), comm)
            isroot(rank) && error("Unknown progress file format - please delete existing files.")
            return
        end
        rankidx = convert(Int, rank) +1
        @inbounds success[rankidx] = true
        MPI.Allgather!(MPI.UBuffer(success, 1), comm)
        if !all(success)
            for (fp, fo) in zip(fileprogresses, fileouts)
                close(fp)
                close(fo)
            end
            isroot(rank) && error("Unknown progress file format - please delete existing files.")
            return
        end
        if isroot(rank)
            worker_pas = Vector{Int}(undef, 2nworkers)
            MPI.Gather!(MPI.IN_PLACE, MPI.UBuffer(worker_pas, 2), comm; root)
            workerprogress[] = sum(@view(worker_pas[3:2:end]), init=0)
            workeracceptance[] = sum(@view(worker_pas[4:2:end]), init=0)
        else
            workerprogress[] = sum(threadprogress, init=0)
            workeracceptance[] = sum(threadacceptance, init=0)
            MPI.Gather!([workerprogress[], workeracceptance[]], nothing, comm; root)
        end

        if !ismissing(maxitr)
            copyto!(maxmultideg, cutat_thread, maxitr, 1, cutlen_thread + cutlen_worker)
            initialize = true
        end
    end
    cond = Threads.SpinLock()

    init_time = time_ns()
    if verbose
        notifier, self_running = newton_halfpolytope_notifier(num, workerprogress, workeracceptance, init_time, comm, rank,
            threadprogress, threadacceptance)
    end
    local notifier, self_running
    ccall(:jl_enter_threaded_region, Cvoid, ())
    try
        # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
        threads = Vector{Task}(undef, nthreads)
        # We can already start all the tasks; this main task that must still feed the data will continue running until we
        # yield to the scheduler.
        @inbounds for (tid, taskₜ) in Iterators.flatten((zip(nthreads:-1:3, Iterators.map(newton_halfpolytope_clonetask,
                                                                                         Iterators.repeated(task))),
                                                        ((2, secondtask), (1, task))))
            if isnothing(filepath)
                filestuff = nothing
            else
                filestuff = (fileprogresses[tid], fileouts[tid])
            end
            # secondtask has a solution, so we just use task (better than deletesolution).
            # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
            # parallel, even if we make sure not to modify the base task until all copies are done.
            threads[tid] = Threads.@spawn newton_polytope_do_taskfun($V, $tid, $taskₜ, $ranges, $nv, $mindeg,
                $maxdeg, $bk, $cond, $threadprogress, $threadacceptance, $candidates, $verbose, $filestuff, $rank)
        end
        # unroll "for ranktask in rankiter" so that we can jump into the loop starting with the previous iteration
        if initialize
            ranktask = rankiter.powers
            rankiter_state = moniter_state(ranktask)
            @goto start_rank
        end
        rankiter_next = iterate(rankiter)
        @inbounds while !isnothing(rankiter_next)
            ranktask, rankiter_state = rankiter_next
            @label start_rank

            fixdeg = typeof(maxdeg)(sum(ranktask, init=zero(maxdeg)))
            # The index i in occurrences_before that would have corresponded to the degree i-1 now corresponds to i-1+fixdeg.
            # Consequently, we only take those degrees into account that are at least mindeg, which now has start index
            # mindeg+1-fixdeg, and we need to introduce a new maxdegree cutoff.
            tasklen = sum(@view(occurrences_before[max(1, mindeg + 1 - fixdeg):maxdeg + 1 - fixdeg]), init=0)
            if !iszero(tasklen)
                # The lowest-rank worker with the least workload will get the job.
                minworkloadᵢ = 1
                minworkload = workload[1]
                for i in 2:nworkers
                    if workload[i] < minworkload
                        minworkloadᵢ = i
                        minworkload = workload[i]
                    end
                end
                if rank == minworkloadᵢ -1
                    # it's our job!
                    copyto!(minmultideg, cutat_worker, ranktask, 1, cutlen_worker)
                    if !isnothing(filepath)
                        workloadcopy = copy(workload) # the threads write down their workload
                    end
                    # All threads are running and waiting for stuff to do. So let's now feed them with their jobs.
                    # [entries iterated on thread][entries fixed on thread, iterated on task][entries fixed on task]
                    # 1                           cutat_thread                               cutat
                    iter_ = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg_thread, maxmultideg_thread,
                                                               @view(maxmultideg[cutat_thread:cutat_worker-1]))
                    if initialize
                        iter = InitialStateIterator(iter_, moniter_state(iter_.powers))
                        initialize = false
                        # we can indeed start with the item succeeding iter_.powers. While this particular item might not be
                        # finished yet, we already added it to ranges when reading the files.
                    else
                        iter = iter_
                    end
                    for threadtask in iter
                        copyto!(minmultideg, cutat_thread, threadtask, 1, cutlen_thread)
                        if isnothing(filepath)
                            put!(ranges, (copy(minmultideg), copy(maxmultideg)))
                        else
                            put!(ranges, (workloadcopy, copy(minmultideg), copy(maxmultideg), nothing))
                        end
                    end
                    # We won't get a progress report while we're pushing stuff into the channel, but if we yield here, we
                    # potentially delay the filling of the channel by allowing the worker task in the main thread to take over.
                end
                workload[minworkloadᵢ] += tasklen
            end
            rankiter_next = iterate(rankiter, rankiter_state)
        end
        verbose && isroot(rank) && newton_halfpolytope_print_workload(workload)
        close(ranges)
        for thread in threads
            wait(thread)
        end
    finally
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
    if verbose
        self_running[] = false
        wait(notifier)
    end
    return candidates
end

function newton_halfpolytope(V, objective::P, comm, rank::MPIRank; verbose::Bool=false,
    filepath::Union{<:AbstractString,Nothing}=nothing, kwargs...) where {P<:AbstractPolynomialLike}
    nworkers = MPI.Comm_size(comm) # we don't need a master, everyone can do the same work
    isone(nworkers) && return newton_halfpolytope(V, objective, Val(false); verbose, filepath, kwargs...)

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

function newton_halfpolytope(V, objective::P, ::Val{true}; kwargs...) where {P<:AbstractPolynomialLike}
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