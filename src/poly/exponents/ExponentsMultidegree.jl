export ExponentsMultideg

mutable struct ExponentsMultideg{N,I<:Integer,V} <: AbstractExponentsDegreeBounded{N,I}
    const mindeg::Int
    const maxdeg::Int
    const minmultideg::V
    const maxmultideg::V
    const Σminmultideg::Int
    const Σmaxmultideg::Int
    counts::Matrix{I}

    @doc """
        ExponentMultideg{N,I}(mindeg, maxdeg, minmultideg::AbstractVector, maxmultideg::AbstractVector)

    Represents an exponent range that is restricted both by a global bound on the degree and by individual bounds on the
    variable degrees. Note that the vectors must not be used afterwards, and the constructor may clip maxmultideg to be no
    larger than maxdeg in each entry.
    """
    function ExponentsMultideg{N,I}(mindeg::Integer, maxdeg::Integer, minmultideg::V, maxmultideg::V) where
        {N,I<:Integer,V<:AbstractVector{<:Integer}}
        0 ≤ N || throw(MethodError(ExponentDegree{N}, (mindeg, maxdeg)))
        0 ≤ mindeg ≤ maxdeg || throw(ArgumentError("Invalid degree specification"))
        length(minmultideg) == length(maxmultideg) == N || throw(ArgumentError("Unsuitable multidegree lengths"))
        Σminmultideg = 0
        Σmaxmultideg = 0
        @inbounds for i in eachindex(minmultideg, maxmultideg)
            maxmultideg[i] > maxdeg && (maxmultideg[i] = maxdeg)
            mindeg < minmultideg[i] && (mindeg = minmultideg[i])
            0 ≤ minmultideg[i] ≤ maxmultideg[i] || throw(ArgumentError("Invalid multidegree specification"))
            Σmaxmultideg = Base.Checked.checked_add(Σmaxmultideg, Int(maxmultideg[i]))
            Σminmultideg += Int(minmultideg[i])
        end
        new{N,I,V}(max(mindeg, Σminmultideg), min(maxdeg, Σmaxmultideg), minmultideg, maxmultideg, Σminmultideg, Σmaxmultideg)
    end
end

ExponentsMultideg{N,I}(range::AbstractUnitRange, minmultideg::V, maxmultideg::V) where {N,I<:Integer,V<:AbstractVector{<:Integer}} =
    ExponentsMultideg{N,I}(first(range), last(range), minmultideg, maxmultideg)

function _calc_index_counts!(e::ExponentsMultideg{N,I}) where {N,I<:Integer}
    maxdeg = e.maxdeg
    if e.mindeg > e.maxdeg
        e.counts = Matrix{I}(undef, 0, N +1)
        return
    end
    minmultideg = e.minmultideg
    maxmultideg = e.maxmultideg
    counts = zeros(I, maxdeg +1, N +1)
    @inbounds @views begin
        if iszero(e.mindeg)
            fill!(counts[:, N+1], 1)
        end
        nextround = counts[:, N]
        for (occₖ, indₖ) in enumerate(minmultideg[N]+1:maxmultideg[N]+1)
           nextround[indₖ] = occₖ
        end
        fill!(nextround[maxmultideg[N]+2:maxdeg+1], maxmultideg[N]-minmultideg[N]+1)
        for j in N-1:-1:1
            lastround = nextround
            nextround = counts[:, j]
            for degᵢ in minmultideg[j]:maxmultideg[j]
                for (degₖ, occₖ) in zip(Iterators.countfrom(0), lastround)
                    newdeg = degᵢ + degₖ
                    newdeg > maxdeg && break
                    nextround[newdeg+1] += occₖ
                end
            end
        end
    end
    e.counts = counts
    return
end

function exponents_to_index(e::ExponentsMultideg{N,I}, exponents) where {N,I<:Integer}
    e.mindeg > e.maxdeg && return zero(I)
    mondeg::Int = sum(exponents, init=0)
    e.mindeg ≤ mondeg ≤ e.maxdeg || return zero(I)
    iszero(mondeg) && return one(I)
    counts, success = @inbounds index_counts(e, mondeg)
    @assert(success)
    # Our index starts with the last exponent that has a degree ≤ the required
    mindex::I = @inbounds counts[mondeg+1, 1]
    iszero(e.mindeg) || (mindex -= @inbounds counts[e.mindeg, 1])
    @inbounds for (i, vardeg, minmultideg, maxmultideg) in zip(2:N, exponents, e.minmultideg, e.maxmultideg)
        minmultideg ≤ vardeg ≤ maxmultideg || return zero(I)
        # We still need to get mondeg for the total degree, but the current variable only has vardeg. Skip over all the
        # exponents where the current variable had a higher degree - these are given by the total number of exponents where the
        # variables to the right of the current one have degree exactly mondeg-(vardeg+1), mondeg-(vardeg+2), ...,
        # mondeg-maxmultideg.
        mondeg > vardeg && (mindex -= counts[mondeg-vardeg, i])
        mondeg > maxmultideg && (mindex += counts[mondeg-maxmultideg, i])
        mondeg -= vardeg
    end
    @inbounds last(e.minmultideg) ≤ mondeg ≤ last(e.maxmultideg) || return zero(I)
    return mindex
end

@inline function degree_from_index(::Unsafe, e::ExponentsMultideg{<:Any,I}, index::I) where {I<:Integer}
    @inbounds counts = @view(index_counts(unsafe, e)[:, 1])
    iszero(e.mindeg) || (index += @inbounds counts[e.mindeg, 1])
    return searchsortedfirst(counts, index) -1
end

Base.@propagate_inbounds function degree_from_index(e::ExponentsMultideg{<:Any,I}, index::I) where {I<:Integer}
    counts = let ic=index_counts(e, e.maxdeg)
        @assert(ic[2])
        @inbounds @view(ic[1][:, 1])
    end
    iszero(e.mindeg) || (index += @inbounds counts[e.mindeg, 1])
    return searchsortedfirst(counts, index) -1
end

function exponents_from_index(e::ExponentsMultideg{<:Any,I}, index::I, degree::Int) where {I<:Integer}
    index > zero(I) || throw(BoundsError(e, index))
    e.mindeg ≤ degree ≤ e.maxdeg || throw(BoundsError(e, index))
    counts, success = index_counts(e, degree) # initialize the cache
    @assert(success)
    allindex = index
    iszero(e.mindeg) || (allindex += @inbounds counts[e.mindeg, 1])
    @inbounds if (iszero(degree) && allindex > 1) ||
        (!iszero(degree) && (counts[degree, 1] ≥ allindex || counts[degree+1, 1] < allindex))
        throw(ArgumentError("Index $index does not have degree $degree"))
    end
    return ExponentIndices(e, index, degree)
end

function Base.iterate(efi::ExponentIndices{I,<:ExponentsMultideg{<:Any,I}}) where {I<:Integer}
    parent = efi.parent
    counts = index_counts(unsafe, parent)
    degree = efi.degree
    @inbounds return iterate(efi, (degree, 2, (iszero(degree) ? efi.index : efi.index - counts[degree, 1])))
end

function Base.iterate(efi::ExponentIndices{I,<:ExponentsMultideg{N,I}}, (degree, i, index)::Tuple{Int,Int,I}) where {N,I<:Integer}
    counts = index_counts(unsafe, efi.parent) # must not be passed in the state - then Julia would have to do an allocation for
                                              # the return type
    minmultideg, maxmultideg = efi.parent.minmultideg, efi.parent.maxmultideg
    @inbounds if i ≤ N
        iszero(degree) && return 0, (degree, i +1, index)
        # Obtain the degree for index i-1 by looking at how large the subspaces to the right of i-1 are when we fix i-1 to
        # a certain value - we are in the last subspace that still fits. The sizes of the exact subspaces are given by the
        # differences of adjacent rows in counts (as counts is accumulating). We must start from the smallest degree for
        # i-1, which means the largest degree for the subspace to the right. So we must accumulate in the opposite order
        # than what is stored in counts. But using the telescopic sum, these accumulations simplify.
        lowerbound = max(1, degree - maxmultideg[i-1])
        tmp = counts[degree-minmultideg[i-1]+1, i]
        remainingdeg = searchsortedlast(@view(counts[lowerbound:degree-minmultideg[i-1], i]), tmp - index) + lowerbound -1
        degᵢ₋₁ = degree - remainingdeg
        index -= tmp - counts[remainingdeg+1, i]
        return degᵢ₋₁, (remainingdeg, i +1, index)
    elseif i == N +1
        return degree, (degree, i +1, zero(index))
    else
        return nothing
    end
end