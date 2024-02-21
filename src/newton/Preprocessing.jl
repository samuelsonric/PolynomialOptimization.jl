function preproc_prequick(V, coeffs, verbose; parameters...)
    nv = size(coeffs, 1)
    vertexindices = fill(1, 2nv)
    @inbounds lowestidx = @view(vertexindices[1:nv])
    @inbounds highestidx = @view(vertexindices[nv+1:2nv])
    # we might also add the points with the smallest/largest sum of all coordinates, or differences (but there are 2^nv ways to
    # combine, so let's skip it)
    @inbounds for (j, coeff) in enumerate(eachcol(coeffs))
        for (i, coeffᵢ) in enumerate(coeff)
            if coeffs[i, lowestidx[i]] > coeffᵢ
                lowestidx[i] = j
            end
            if coeffs[i, highestidx[i]] < coeffᵢ
                highestidx[i] = j
            end
        end
    end
    Base._groupedunique!(sort!(vertexindices))
    return preproc_quick(V, coeffs, vertexindices, verbose; parameters...)
end

function preproc_randomized_taskfun(V, coeffs, nv, subset_size, required_coeffs, subset, done, event, stop; parameters...)
    @inbounds while !stop[]
        dropped = 0
        for (cfi, rem) in zip(subset, preproc_remove(V, nv, subset_size, i -> @view(coeffs[:, subset[i]]), false, true;
                                                     parameters...))
            required_coeffs[cfi] = rem
            if !rem
                dropped += 1
            end
        end
        put!(done, dropped)
        stop[] && return
        wait(event)
    end
end

function preproc_randomized(V, coeffs, verbose; parameters...)
    nv, nc = size(coeffs)
    nthreads = Threads.nthreads()
    subset_size = min(1000, nc ÷ 20) # samples too small -> no chance of success; samples too large -> takes too long
    @assert(subset_size ≥ 1)
    required_coeffs = fill(true, nc)
    if nthreads * subset_size > nc
        nthreads = nc ÷ subset_size # we could squeeze in another incomplete thread, but this would make the code more
                                    # complicated than necessary without a real benefit
    end
    if isone(nthreads)
        let
            _subset = sample(required_coeffs, subset_size, nc)
            @inbounds while true
                totaldrop = 0
                for (cfi, rem) in zip(_subset, preproc_remove(V, nv, subset_size, i -> @view(coeffs[:, _subset[i]]), false,
                                                              false); parameters...)
                    required_coeffs[cfi] = rem
                    if !rem
                        totaldrop += 1
                    end
                end
                nc -= totaldrop
                if verbose
                    print("\33[2KStatus update: ", nc, " remaining extremal points, last drop was ", totaldrop, "\r")
                    flush(stdout)
                end
                (totaldrop < 20 || 10totaldrop < subset_size) && break
                subset_size > nc && break
                sample!(required_coeffs, _subset, nc)
            end
        end
    else
        ccall(:jl_enter_threaded_region, Cvoid, ())
        try
            # We divide our variables in randomized, disjoint (in the case of multi-threading) subsets and try to find
            # linear dependencies in them. Depending on our success rate, we'll try to increase the size of the subsets.
            allsubs = sample(required_coeffs, nthreads * subset_size, nc)
            done = Base.Channel{Int}(1)
            events = [Base.Event(true) for _ in 1:nthreads]
            stop = Ref(false)
            # Initialize all threads; as everything is already set up, they will directly start
            threads = Vector{Task}(undef, nthreads)
            for (tid, start) in enumerate(1:subset_size:subset_size*nthreads)
                @inbounds threads[tid] = Threads.@spawn preproc_randomized_taskfun($V, $coeffs, $nv, $subset_size,
                    $required_coeffs, $(@view(allsubs[start:start+subset_size-1])), $done, $(events[tid]), $stop;
                    $parameters...)
            end
            while true
                # We wait for all threads to finish - if the have a sufficiently good coverage of the whole set with
                # all available threads, it would make no sense to directly re-start a thread, it would just get the
                # same set again. Instead, when all threads are finished (which happens approximately at the same
                # time), we can rehash the full set.
                totaldrop = 0
                for _ in 1:nthreads
                    totaldrop += take!(done)
                end
                nc -= totaldrop
                if verbose
                    print("\33[2KStatus update: ", nc, " remaining extremal points, last drop was ", totaldrop, "\r")
                    flush(stdout)
                end
                totalsize = nthreads * subset_size
                (totaldrop < 20 || 10totaldrop < totalsize) && break
                if totalsize > nc
                    nthreads = nc ÷ subset_size
                    iszero(nthreads) && break
                    stop[] = true
                    for event in events
                        notify(event)
                    end
                    totalsize = nthreads * subset_size
                end
                subs = @view(allsubs[1:totalsize])
                sample!(required_coeffs, subs, nc)
                for event in @view(events[1:nthreads])
                    notify(event) # don't start those beyond nthreads
                end
            end
            stop[] = true
            for event in events
                notify(event) # to properly finish all tasks
            end
        finally
            ccall(:jl_exit_threaded_region, Cvoid, ())
        end
    end
    return required_coeffs
end

function preproc(V, objective::P; verbose::Bool=false, zero::AbstractVector{P}, nonneg::AbstractVector{P},
    psd::AbstractVector{<:AbstractMatrix{P}}, groupings::RelaxationGroupings, preprocess::Union{Nothing,Bool}=nothing,
    preprocess_quick::Bool=true, preprocess_randomized::Bool=false, preprocess_fine::Bool=false,
    warn_disable_randomization::Bool=true, parameters...) where {P<:SimpleRealPolynomial}
    if !isnothing(preprocess)
        preprocess_quick = preprocess_randomized = preprocess_fine = preprocess
    end
    @verbose_info("Determining Newton polytope (quick preprocessing: ", preprocess_quick, ", randomized preprocessing: ",
        preprocess_randomized, ", fine preprocessing: ", preprocess_fine, ")")
    nv = nvariables(objective)
    nc = length(objective)
    if !isempty(zero) || !isempty(nonneg) || !isempty(psd)
        coeffs = merge_constraints(
            objective, zero, nonneg, psd, groupings, Val(monomials(objective).exponents_real isa DenseMatrix), verbose
        )
    else
        # shortcut, no need to temporarily go to the index-based version. Just copy the coefficient matrix (as preprocessing
        # will make changes to it)
        let objexps=monomials(objective).exponents_real
            if VERSION < v"1.11"
                # due to preprocessing, we might want to shrink this array. In Julia, shrinking is almost unsupported - a new
                # buffer of the smaller size is allocated and things are copied over. However, in case we actually wrap a
                # malloc-allocated buffer into an array, realloc is used properly.
                if objexps isa DenseMatrix
                    coeffs = resizable_copy(objexps)
                else
                    @assert(objexps isa SparseArrays.AbstractSparseMatrixCSC)
                    colptr = SparseArrays.getcolptr(objexps)
                    rowval = rowvals(objexps)
                    nzval = nonzeros(objexps)
                    @inbounds coeffs = typeof(objexps)(size(objexps)..., resizable_copy(colptr), resizable_copy(rowval),
                        resizable_copy(nzval))
                end
            else
                coeffs = copy(objexps)
            end
        end
    end
    if preprocess_quick
        @verbose_info("Removing redundancies from the convex hull - quick heuristic, ", nc, " initial candidates")
        preproc_time = @elapsed begin
            coeffs = keepcol!(coeffs, preproc_prequick(V, coeffs, verbose; parameters...))
            nc = size(coeffs, 2)
        end
        @verbose_info("Found ", nc, " potential extremal points of the convex hull in ", preproc_time, " seconds")
    end
    if preprocess_randomized
        if nc ≥ 100
            @verbose_info("Removing redundancies from the convex hull - randomized, ", nc, " initial candidates")
            preproc_time = @elapsed begin
                coeffs = keepcol!(coeffs, preproc_randomized(V, coeffs, verbose; parameters...))
                nc = size(coeffs, 2)
            end
            @verbose_info("Found ", nc, " extremal points of the convex hull via randomization in ", preproc_time, " seconds")
        else
            warn_disable_randomization &&
                @info("Removing redundancies from the convex hull via randomization was requested, but skipped due to the small size of the problem")
        end
    end
    if preprocess_fine
        # eliminate all the coefficients that are redundant themselves to make the linear system smaller
        @verbose_info("Removing redundancies from the convex hull - fine, ", nc, " initial candidates")
        preproc_time = @elapsed begin
            coeffs = keepcol!(coeffs, preproc_remove(V, nv, nc, @capture(i -> @inbounds(@view $coeffs[:, i])), verbose, false;
                parameters...))
            nc = size(coeffs, 2)
        end
        @verbose_info("Found ", nc, " extremal points of the convex hull in ", preproc_time, " seconds")
    end
    return parameters, coeffs
end

function analyze(coeffs)
    # This is some quick preprocessing to further restrict the potential degrees (actually, this is what SumOfSquares.jl calls
    # Newton polytope)
    nv = size(coeffs, 1)

    maxdeg, mindeg = zero(UInt), typemax(UInt)
    maxmultideg, minmultideg = fill(zero(UInt), nv), fill(typemax(UInt), nv)
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

    # Now that we know the ranges of all the types, we can make them as tight as possible to reduce the memory footprint
    maxval = max(maxdeg, maximum(maxmultideg, init=0))
    local T
    for outer T in (UInt8, UInt16, UInt32, UInt64)
        typemax(T) ≥ maxval && break
    end
    return convert(T, mindeg), convert(T, maxdeg), convert(Vector{T}, minmultideg), convert(Vector{T}, maxmultideg)
end