module PolynomialOptimizationMPI

using PolynomialOptimization, MultivariatePolynomials, Printf
import MPI, Random, Mosek
import PolynomialOptimization: @verbose_info, FastVec, prepare_push!, unsafe_push!, finish!

__init__() = PolynomialOptimization.haveMPI[] = true

function PolynomialOptimization.newton_halfpolytope(V::Val{:Mosek}, objective::P, ::Val{true}; verbose::Bool=false,
    kwargs...) where {P<:AbstractPolynomialLike}
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
    root = 0 # do not change this. We use root instead of the literal 0 to be more expressive, but the receive code rests
    # on the fact that root is 0.
    nworkers = MPI.Comm_size(comm) # we don't need a master, everyone can do the same work

    MPI.Barrier(comm)

    if haskey(kwargs, :preprocess_randomized)
        # We want to get the same results of the randomized preprocessing on all workers
        if rank == root
            seed = Ref(rand(Int))
        else
            seed = Ref{Int}()
        end
        MPI.Bcast!(seed, root, comm)

        Random.seed!(seed[])
    end
    parameters, coeffs = PolynomialOptimization.newton_polytope_preproc(V, objective; verbose=verbose && rank == root,
        kwargs...)
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

        moniter, num, nthreads, task, secondtask = PolynomialOptimization.newton_halfpolytope_do_prepare(V, coeffs, mindeg,
            maxdeg, minmultideg, maxmultideg, verbose && rank == root; parameters...)
        workersize = div(num, nworkers, RoundUp)
        bk = fill(Mosek.MSK_BK_FX, nv)
        if rank == root
            allcandidates = FastVec{Vector{Int}}()
        else
            localcandidates = FastVec{Int}()
        end
        if isone(nthreads)
            rank == root && @verbose_info("Preparing to determine Newton polytope using ", nworkers,
                " single-threaded workers, each checking about ", workersize, " candidates")
            let
                tmp = Vector{Float64}(undef, nv)
                init_time = time_ns()
                lastinfo = init_time
                progress_acceptance = zeros(Int, 2)
                if verbose && rank == root
                    Δprogress_acceptance = similar(progress_acceptance)
                    Δpabuffer = MPI.Buffer(Δprogress_acceptance)
                    req = MPI.Irecv!(Δpabuffer, comm, tag=1)
                end
                whichworker = 0
                # every worker iterates through all the monomials, but they feel responsible ring-a-ring
                for powers in moniter
                    if rank == whichworker
                        copyto!(tmp, powers)
                        Mosek.putconboundslice(task, 1, nv +1, bk, tmp, tmp)
                        Mosek.optimize(task)
                        if Mosek.getsolsta(task, Mosek.MSK_SOL_BAS) == Mosek.MSK_SOL_STA_OPTIMAL
                            if rank == root
                                push!(allcandidates, copy(powers))
                            else
                                append!(localcandidates, powers)
                            end
                            @inbounds progress_acceptance[2] += 1
                        end
                        if verbose
                            @inbounds progress_acceptance[1] += 1
                            nextinfo = time_ns()
                            if nextinfo - lastinfo > 10_000_000_000
                                if rank == root
                                    while MPI.Test(req)
                                        progress_acceptance .+= Δprogress_acceptance
                                        req = MPI.Irecv!(Δpabuffer, comm, tag=1)
                                    end
                                    @inbounds progress = progress_acceptance[1]
                                    @inbounds acceptance = progress_acceptance[2]
                                    rem_sec = round(Int, ((time_ns() - init_time) / 1_000_000_000progress) * (num - progress))
                                    @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                                        100progress / num, 100acceptance / progress, rem_sec ÷ 60, rem_sec % 60)
                                    flush(stdout)
                                    lastinfo = nextinfo
                                else
                                    MPI.Isend(progress_acceptance, comm, dest=root, tag=1)
                                    @inbounds progress_acceptance[1] = 0
                                    @inbounds progress_acceptance[2] = 0
                                end
                            end
                        end
                    end
                    if (whichworker += 1) == nworkers
                        whichworker = 0
                    end
                end
            end
            Mosek.deletetask(task)
        else
            error("Not implemented")
        end

        MPI.Barrier(comm)
    end

    if rank == root
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
            @assert(rank == 0)
            @inbounds for (rankᵢ, sizeᵢ) in enumerate(Iterators.drop(sizes, 1))
                MPI.Recv!(buffer, comm, source=rankᵢ, tag=4)
                for j in 1:sizeᵢ
                    unsafe_push!(allcandidates, buffer[:, j])
                end
            end
            allresult = sort!(finish!(allcandidates), lt=(a, b) -> compare(a, b, Graded{LexOrder}) < 0)
        end

        @verbose_info("Fetched ", length(allresult), " elements in the Newton polytope in ", fetch_time, " seconds")
        return PolynomialOptimization.makemonovec(variables(objective), allresult)
    else
        localresult = reshape(finish!(localcandidates), nv, length(localcandidates) ÷ nv)
        MPI.Send(size(localresult, 2), comm, dest=root, tag=2)
        MPI.Send(localresult, comm, dest=root, tag=4)
        return
    end
end

end