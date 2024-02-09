function execute_taskfun(V, tid, task, ranges, nv, mindeg, maxdeg, data_global, cond, progresses, acceptances, allcandidates,
    notifier, init_time, init_progress, num, filestuff)
    # notifier: 0 - no notification; 1 - the next to get becomes the notifier; 2 - notifier is taken
    verbose = notifier[] != 0
    lastappend = time_ns()
    isnotifier = Ref(false) # necessary due to the capturing/boxing bug
    lastinfo = Ref{Int}(lastappend)
    data_local = alloc_local(V, nv)
    candidates = FastVec{typeof(maxdeg)}()
    @inbounds progress = Ref(progresses, tid)
    @inbounds acceptance = Ref(acceptances, tid)
    local curminrange, curmaxrange, iter
    if isnothing(filestuff)
        powers = Vector{typeof(maxdeg)}(undef, nv)
        fileprogress = nothing # required for capture below
        fileout = nothing
        cut = 0
    else
        fileprogress = filestuff[1]
        fileout = filestuff[2]
        cut = filestuff[3]
        if ismissing(filestuff[4])
            powers = Vector{typeof(maxdeg)}(undef, nv)
        else
            curminrange, curmaxrange, powers = filestuff[4]
            iter = InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers),
                moniter_state(powers))
            # @goto start - must be deferred as it would jump into a try block
        end
    end
    try
        !isnothing(filestuff) && !ismissing(filestuff[4]) && @goto start
        while true
            try
                curminrange, curmaxrange = take!(ranges)
            catch e
                e isa InvalidStateException && break
                rethrow(e)
            end
            iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers)
            @label start
            work(V, task, data_global, data_local, iter, progress, acceptance,
                @capture(p -> append!($candidates, p)),
                !verbose && isnothing(filestuff) ? nothing : @capture(p -> let
                    nextinfo = time_ns()
                    if nextinfo - $lastinfo[] > 1_000_000_000
                        if !isnothing($filestuff)
                            write($fileout, $candidates)
                            flush(fileout)
                            seekstart($fileprogress)
                            write(fileprogress, $progress[], $acceptance[], p)
                            flush(fileprogress)
                            empty!(candidates)
                        end
                        if $verbose
                            if !$isnotifier[] && notifier[] == 1
                                isnotifier[] = true
                                notifier[] = 2
                            end
                            if isnotifier[]
                                allprogress = sum(progresses, init=0)
                                allacceptance = sum(acceptances, init=0)
                                Δt = allprogress == $init_progress ? 1 : allprogress - init_progress
                                # ^ if a finished job is started, this might happen
                                rem_sec = round(Int, ((nextinfo - init_time) / 1_000_000_000Δt) * (num - allprogress))
                                @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                                    100allprogress / num, 100allacceptance / allprogress, rem_sec ÷ 60, rem_sec % 60)
                                flush(stdout)
                            end
                        end
                        lastinfo[] = nextinfo
                    end
                end)
            )
            # make sure that we update the main list regularly, but not ridiculously often
            nextappend = time_ns()
            if nextappend - lastappend > 10_000_000_000
                if isnothing(filestuff)
                    lock(cond)
                    try
                        append!(allcandidates, candidates)
                    finally
                        unlock(cond)
                    end
                else
                    write(fileout, candidates)
                    flush(fileout)
                    seekstart(fileprogress)
                    write(fileprogress, progress[], acceptance[], @view(curminrange[cut:end]))
                    truncate(fileprogress, position(fileprogress))
                end
                empty!(candidates)
                lastappend = nextappend
            end
        end
        if isnotifier[]
            notifier[] = 1
        end
        if isnothing(filestuff)
            lock(cond)
            try
                append!(allcandidates, candidates)
            finally
                unlock(cond)
            end
        else
            write(fileout, candidates)
            seekstart(fileprogress)
            write(fileprogress, progress[], acceptance[])
            truncate(fileprogress, position(fileprogress))
        end
    finally
        finalize(task)
        if !isnothing(filestuff)
            close(fileout)
            close(fileprogress)
        end
    end
end

function execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath)
    @verbose_info("Preparing to determine Newton polytope (single-threaded)")

    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{typeof(maxdeg)}()

    progress = Ref(0)
    acceptance = Ref(0)
    if isnothing(filepath)
        fileprogress = nothing
        fileout = nothing
        iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, true)
    else
        fileprogress = open("$filepath.prog", read=true, write=true, create=true, lock=false)
        local iter
        try
            progress[], acceptance[], iter = restore_status!(fileprogress, mindeg, maxdeg, minmultideg,
                maxmultideg)
        catch
            close(fileprogress)
            rethrow()
        end
        if isnothing(iter)
            close(fileprogress)
            return progress[], acceptance[]
        end
        fileout = open("$filepath.out", append=true, lock=false)
    end
    try
        init_time = time_ns()
        init_progress = progress[]
        lastinfo = Ref(init_time)
        work(V, task, alloc_global(V, nv), alloc_local(V, nv),
            iter, progress, acceptance, @capture(p -> append!($candidates, p)),
            !verbose && isnothing(filepath) ? nothing : @capture(p -> let
                nextinfo = time_ns()
                if nextinfo - $lastinfo[] > 1_000_000_000
                    if !isnothing($filepath)
                        write($fileout, $candidates)
                        flush(fileout)
                        seekstart($fileprogress)
                        write(fileprogress, $progress[], $acceptance[], p)
                        flush(fileprogress)
                        empty!(candidates)
                    end
                    if $verbose
                        Δt = progress[] == $init_progress ? 1 : progress[] - init_progress
                        # ^ if a finished job is started, this might happen
                        rem_sec = round(Int, ((nextinfo - $init_time) / 1_000_000_000Δt) * (num - progress[]))
                        @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                            100progress[] / num, 100 * acceptance[] / progress[], rem_sec ÷ 60, rem_sec % 60)
                        flush(stdout)
                    end
                    lastinfo[] = nextinfo
                end
            end)
        )
        verbose && print("\33[2K")
        # How about the order of monomials? Currently, the monomial order can be defined to be arbitrary, but monomials
        # in DynamicPolynomials always outputs Graded{LexOrder}. Therefore, it is currently not possible to use
        # different ordering in DP, and this translates to PolynomialOptimization, unless the user chooses to generate
        # the monomials manually.
        # Here, we can relatively easily make the basis compliant with the specified monomial ordering just by doing a
        # sorting as a postprocessing. Of course, this is not efficient and it would be much better to create functions
        # that directly generate the appropriate order, but let's defer this at least until #138 in DP is solved, for
        # other monomial orderings won't be in widespread use before this anyway.

        # sort!(candidates, lt=(a, b) -> compare(a, b, monomial_ordering(P)) < 0)
        # TODO: There is no monomial_ordering, so we cannot even do this
        if !isnothing(fileout)
            write(fileout, candidates)
            seekstart(fileprogress)
            write(fileprogress, progress[], acceptance[])
            truncate(fileprogress, position(fileprogress))
        end
    finally
        finalize(task)
        if !isnothing(fileout)
            close(fileout)
            close(fileprogress)
        end
    end
    if isnothing(fileout)
        return SimpleMonomialVector{nv,0}(reshape(finish!(candidates), nv, length(candidates)÷nv))
    else
        return progress[], acceptance[]
    end
end

function execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, nthreads::Integer, task, secondtask,
    filepath)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath)
    end

    threadsize = div(num, nthreads, RoundUp)
    @verbose_info("Preparing to determine Newton polytope using ", nthreads, " threads, each checking about ", threadsize,
        " candidates")
    cutat = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
    cutlen = nv - cutat
    cutat += 1 # cutat is now the first entry to be fixed
    data_global = alloc_global(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{typeof(maxdeg)}()

    ranges = Base.Channel{NTuple{2,Vector{typeof(maxdeg)}}}(typemax(Int))
    threadprogress = zeros(Int, nthreads)
    threadacceptance = zeros(Int, nthreads)
    if isnothing(filepath)
        iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end], maxmultideg[cutat:end],
            @view(minmultideg[cutat:end]))
    else
        fileprogresses = Vector{IOStream}(undef, nthreads)
        fileouts = Vector{IOStream}(undef, nthreads)
        restores = Vector{Union{NTuple{3,Vector{typeof(maxdeg)}},Missing,Nothing}}(undef, nthreads)
        maxitr = missing
        @inbounds try
            for i in 1:nthreads
                fileprogresses[i] = fileprogress = open("$filepath-$i.prog", read=true, write=true, create=true, lock=false)
                fileouts[i] = open("$filepath-$i.out", append=true, lock=false)
                curpower = Vector{typeof(maxdeg)}(undef, nv)
                currestore = restore_status!(fileprogress, curpower, nv - cutat +1)
                if isnothing(currestore)
                    restores[i] = missing
                else
                    threadprogress[i], threadacceptance[i], curitr, curpower_ = currestore
                    if isnothing(curitr)
                        restores[i] = nothing
                    else
                        if ismissing(maxitr) || toSigned(compare(maxitr, curitr, Graded{LexOrder})) < 0
                            maxitr = curitr
                        end
                        if isnothing(curpower_)
                            restores[i] = missing
                        else
                            curminrange, curmaxrange = copy(minmultideg), copy(maxmultideg)
                            copyto!(curminrange, cutat, curitr, 1, cutlen)
                            copyto!(curmaxrange, cutat, curitr, 1, cutlen)
                            restores[i] = (curminrange, curmaxrange, curpower)
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
            rethrow()
        end
        if ismissing(maxitr)
            iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end], maxmultideg[cutat:end],
                @view(minmultideg[cutat:end]))
        else
            iter = InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end],
                maxmultideg[cutat:end], @view(minmultideg[cutat:end])),
                moniter_state(copyto!(@view(minmultideg[cutat:end]), maxitr)))
        end
    end
    cond = Threads.SpinLock()

    ccall(:jl_enter_threaded_region, Cvoid, ())
    try
        # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
        threads = Vector{Union{Task,Nothing}}(undef, nthreads)
        # We can already start all the tasks; this main task that must still feed the data will continue running until we
        # yield to the scheduler.
        init_time = time_ns()
        init_progress = sum(threadprogress, init=0)
        notifier = Ref(verbose ? 1 : 0)
        @inbounds for (tid, taskₜ) in Iterators.flatten((zip(nthreads:-1:3, Iterators.map(clonetask,
                                                                                         Iterators.repeated(task))),
                                                        ((2, secondtask), (1, task))))
            if isnothing(filepath)
                filestuff = nothing
            elseif isnothing(restores[tid])
                threads[tid] = nothing
                close(fileouts[tid])
                close(fileprogresses[tid])
                continue
            else
                filestuff = (fileprogresses[tid], fileouts[tid], cutat, restores[tid])
            end
            # secondtask has a solution, so we just use task (better than deletesolution).
            # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
            # parallel, even if we make sure not to modify the base task until all copies are done.
            threads[tid] = Threads.@spawn execute_taskfun($V, $tid, $taskₜ, $ranges, $nv, $mindeg, $maxdeg, $data_global, $cond,
                $threadprogress, $threadacceptance, $candidates, $notifier, $init_time, $init_progress, $num, $filestuff)
        end
        # All tasks are created and waiting for stuff to do. So let's now feed them with their jobs.

        for subtask in iter
            copyto!(maxmultideg, cutat, subtask, 1, cutlen) # minmultideg is already set appropriately due to the
                                                            # @view trickery
            put!(ranges, (copy(minmultideg), copy(maxmultideg)))
        end
        close(ranges)
        for thread in threads
            isnothing(thread) || wait(thread)
        end
    finally
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
    @verbose_info("\33[2KAll tasks have finished, sorting the output")
    # We need to return the deglex monomial order, but due to the partitioning and multithreading, our output
    # is unordered (not completely, we have ordered of varying length, but does this help?).
    if isnothing(filepath)
        return SimpleMonomialVector{nv,0}(sortslices(reshape(finish!(candidates), nv, length(candidates)÷nv), dims=2,
            lt=isless_degree))
    else
        return sum(threadprogress, init=0), sum(threadacceptance, init=0)
    end
end