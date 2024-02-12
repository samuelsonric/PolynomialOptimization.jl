function verbose_worker(num, progress, acceptance, init_time, comm, ::RootRank, threadprogress=missing,
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
    Threads.@spawn(:interactive, begin
        while !iszero($workers[]) || $self_running[]
            $check()
            sleep(1)
        end
        $check() # make sure to have a final report
    end)
    return self_running
end

function verbose_worker(num, progress, acceptance, init_time, comm, ::OtherRank, threadprogress=missing,
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
    Threads.@spawn(:interactive, begin
        while $self_running[]
            $check()
            sleep(1)
        end
        $check() # make sure to have a final report
        # and then signal that we are done
        $Δprogress_acceptance[1] = -1
        MPI.Send($Δbuffer, $comm, dest=root, tag=1)
    end)
    return self_running
end

function execute(V, nv, verbose, iter, num, task, filepath, comm, rank::MPIRank)
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)
    isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers,
        " single-threaded workers, each checking about ", workersize, " candidates")

    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{eltype(eltype(iter))}()
    iter = RangedMonomialIterator(iter, convert(Int, rank) * workersize +1, workersize, copy=false)

    progress = Ref(0)
    acceptance = Ref(0)
    if isnothing(filepath)
        filestuff = nothing
    else
        prefix = "$filepath-n$(convert(Int, rank))"
        fileprogress = open("$prefix.prog", read=true, write=true, create=true, lock=false)
        success = fill(false, nworkers)
        try
            fs = filesize(fileprogress)
            if fs == 2sizeof(Int)
                progress[] = read(fileprogress, Int)
                acceptance[] = read(fileprogress, Int)
            elseif !iszero(fs)
                error()
            end
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
        iter = Iterators.drop(iter, progress[], copy=false)
        fileout = open("$prefix.out", append=true, lock=false)
        filestuff = (fileprogress, fileout)
    end
    if verbose
        self_running = verbose_worker(num, progress, acceptance, time_ns(), comm, rank)
    end
    lastinfo = Ref(time_ns())
    cb = step_callback(verbose, filestuff, lastinfo, candidates, progress, acceptance)
    try
        work(V, task, alloc_global(V, nv), alloc_local(V, nv), iter, progress, acceptance,
            @capture(p -> append!($candidates, p)), cb)
        if !isnothing(filestuff)
            write(fileout, candidates)
            seekstart(fileprogress)
            write(fileprogress, progress[], acceptance[])
        end
    finally
        if verbose
            self_running[] = false
        end
        finalize(task)
        if !isnothing(filestuff)
            close(fileout)
            close(fileprogress)
        end
    end
    return isnothing(filestuff) ? candidates : nothing
end

function execute(V, nv, verbose, iter, num, nthreads::Integer, task, secondtask, filepath, comm, rank::MPIRank)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return execute(V, nv, verbose, iter, num, task, filepath, comm, rank)
    end
    nworkers = MPI.Comm_size(comm)
    workersize = div(num, nworkers, RoundUp)
    threadsize = div(workersize, nthreads, RoundUp)
    if threadsize == workersize
        finalize(secondtask)
        return execute(V, nv, verbose, iter, num, task, filepath, comm, rank)
    end
    isroot(rank) && @verbose_info("Preparing to determine Newton polytope using ", nworkers, " workers, each with ", nthreads,
        " threads checking about ", threadsize, " candidates")

    data_global = alloc_global(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{eltype(eltype(iter))}()
    iter = RangedMonomialIterator(iter, convert(Int, rank) * workersize +1, workersize, copy=false)
    iterators = Vector{typeof(iter)}(undef, nthreads)
    let start=1
        for i in 1:nthreads
            @inbounds iterators[i] = RangedMonomialIterator(iter, start, threadsize, copy=true)
            start += threadsize
        end
    end

    threadprogress = zeros(Int, nthreads)
    threadacceptance = zeros(Int, nthreads)
    workerprogress = Ref(0)
    workeracceptance = Ref(0)
    if !isnothing(filepath)
        prefix = "$filepath-n$(convert(Int, rank))"
        fileprogresses = Vector{IOStream}(undef, nthreads)
        fileouts = Vector{IOStream}(undef, nthreads)
        success = fill(false, nworkers)
        @inbounds try
            for i in 1:nthreads
                fileprogresses[i] = fileprogress = open("$prefix-$i.prog", read=true, write=true, create=true, lock=false)
                fileouts[i] = open("$prefix-$i.out", append=true, lock=false)
                fs = filesize(fileprogress)
                if fs == 2sizeof(Int)
                    threadprogress[i] = read(fileprogress, Int)
                    threadacceptance[i] = read(fileprogress, Int)
                elseif !iszero(fs)
                    error()
                end
                iterators[i] = Iterators.drop(iterators[i], threadprogress[i], copy=false)
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
        @inbounds success[convert(Int, rank)+1] = true
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
    end
    cond = Threads.SpinLock()

    if verbose
        self_running = verbose_worker(num, workerprogress, workeracceptance, time_ns(), comm, rank, threadprogress,
            threadacceptance)
    end
    ccall(:jl_enter_threaded_region, Cvoid, ())
    try
        # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
        threads = Vector{Union{Task,Nothing}}(undef, nthreads)
        # We can already start all the tasks; this main task that must still feed the data will continue running until we
        # yield to the scheduler.
        @inbounds for (tid, taskₜ) in Iterators.flatten((zip(nthreads:-1:3, Iterators.map(clonetask, Iterators.repeated(task))),
                                                        ((2, secondtask), (1, task))))
            iter = iterators[tid]
            if isnothing(filepath)
                filestuff = nothing
            elseif isempty(iter)
                threads[tid] = nothing
                close(fileouts[tid])
                close(fileprogresses[tid])
                continue
            else
                filestuff = (fileprogresses[tid], fileouts[tid])
            end
            # secondtask has a solution, so we just use task (better than deletesolution).
            # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
            # parallel, even if we make sure not to modify the base task until all copies are done.
            threads[tid] = Threads.@spawn execute_taskfun($V, $tid, $taskₜ, $iter, $nv, $data_global, $cond, $threadprogress,
                $threadacceptance, $candidates, $verbose, $filestuff)
        end
        for thread in threads
            isnothing(thread) || wait(thread)
        end
    finally
        if verbose
            self_running[] = false
        end
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
    return isnothing(filestuff) ? candidates : nothing
end