function halfpolytope(V, objective::P, comm, rank::MPIRank; verbose::Bool=false,
    filepath::Union{<:AbstractString,Nothing}=nothing, zero::AbstractVector{P}=P[], nonneg::AbstractVector{P}=P[],
    psd::AbstractVector{<:AbstractMatrix{P}}=Matrix{P}[],
    degree::Int=maxhalfdegree(objective),kwargs...) where {P<:AbstractPolynomialLike}
    nworkers = MPI.Comm_size(comm) # we don't need a master, everyone can do the same work
    #isone(nworkers) && return halfpolytope(V, objective, Val(false); verbose, filepath, kwargs...)

    MPI.Barrier(comm)

    if haskey(kwargs, :preprocess_randomized)
        # We want to get the same results of the randomized preprocessing on all workers
        seed = isroot(rank) ? Ref(rand(Int)) : Ref{Int}()
        MPI.Bcast!(seed, root, comm)

        Random.seed!(seed[])
    end
    parameters, coeffs = preproc(V, objective; verbose=verbose && isroot(rank), warn_disable_randomization=isroot(rank),
        zero, nonneg, psd, degree, kwargs...)
    nv = size(coeffs, 1)
    analysis = analyze(coeffs)
    iter = MonomialIterator(analysis..., true)
    num = length(iter)

    if isroot(rank)
        @verbose_info("Starting point selection among ", num, " possible monomials")
        newton_time = @elapsed allresult = let
            nthreads, task, secondtask = prepare(V, coeffs, num, verbose; parameters...)
            candidates = execute(V, size(coeffs, 1), verbose, iter, num, nthreads, task, secondtask, filepath, comm, rank)
            if isnothing(filepath)
                sizes = Vector{Int}(undef, nworkers -1)
                for _ in 2:nworkers
                    size, status = MPI.Recv(Int, comm, MPI.Status, tag=2)
                    @inbounds sizes[status.source] = size
                end
                i = length(candidates)
                candidates = resize!(candidates.data, i + sum(sizes, init=0))
                @inbounds for (rankᵢ, sizeᵢ) in enumerate(sizes)
                    MPI.Recv!(@view(candidates[i+1:i+sizeᵢ]), comm, source=rankᵢ, tag=3)
                    i += sizeᵢ
                end
                SimpleMonomialVector{nv,0}(sortslices(reshape(candidates, nv, length(candidates) ÷ nv), dims=2, lt=isless_degree))
            end
        end

        @verbose_info("\33[2KFinished construction of all Newton polytope element on the workers in ", newton_time,
            " seconds.")
        return isnothing(filepath) ? allresult : true
    else
        let
            nthreads, task, secondtask = prepare(V, coeffs, num, false; parameters...)
            candidates = execute(V, size(coeffs, 1), verbose, iter, num, nthreads, task, secondtask, filepath, comm, rank)
            if isnothing(filepath)
                MPI.Send(length(candidates), comm, dest=root, tag=2)
                MPI.Send(finish!(candidates), comm, dest=root, tag=3)
            end
            return
        end
    end
end

function halfpolytope(V, objective::P, ::Val{true}; kwargs...) where {P<:AbstractPolynomialLike}
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
    return halfpolytope(V, objective, comm, MPIRank(rank); kwargs...)
end