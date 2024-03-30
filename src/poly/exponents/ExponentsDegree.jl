export ExponentsDegree

mutable struct ExponentsDegree{N,I<:Integer} <: AbstractExponentsDegreeBounded{N,I}
    const mindeg::Int
    const maxdeg::Int
    counts::Matrix{I}

    @doc """
        ExponentsDegree{N,I}(mindeg::Integer, maxdeg::Integer)

    Represents an exponent range that is restricted by a bound on the total degree.
    """
    function ExponentsDegree{N,I}(mindeg::Integer, maxdeg::Integer) where {N,I<:Integer}
        0 ≤ N || throw(MethodError(ExponentDegree{N}, (mindeg, maxdeg)))
        0 ≤ mindeg ≤ maxdeg || throw(ArgumentError("Invalid degree specification"))
        new{N,I}(mindeg, maxdeg)
    end
end

ExponentsDegree{N,I}(range::AbstractUnitRange) where {N,I<:Integer} = ExponentsDegree{N,I}(first(range), last(range))

function _calc_index_counts!(E::ExponentsDegree{N,I}) where {N,I<:Integer}
    # This is not exactly correct if we think about the very last column (corresponding to zero variables), which should be
    # zero if a nonzero mindeg is set. But this last column is not used anyway (apart from the construction of the matrix), so
    # instead of needlessly duplicating data, we just use a reference to the current unbounded one. This is very efficient -
    # unless the unbounded one then grows, because we'll keep the reference alive unneccessarily.
    E.counts, success = index_counts(ExponentsAll{N,I}(), E.maxdeg)
    @assert(success)
    return
end

function _exponents_to_index(e::ExponentsDegree{N,I}, exponents, degree::Int) where {N,I<:Integer}
    e.mindeg ≤ degree ≤ e.maxdeg || return zero(I)
    iszero(degree) && return one(I)
    counts, success = @inbounds index_counts(e, degree)
    @assert(success)
    # Our index starts with the last exponent that has a degree ≤ the required
    index::I = @inbounds counts[degree+1, 1]
    iszero(e.mindeg) || (index -= @inbounds counts[e.mindeg, 1])
    @inbounds for (i, vardeg) in zip(2:N, exponents)
        # We still need to get mondeg for the total degree, but the current variable only has vardeg. Skip over all the
        # exponents where the current variable had a higher degree - these are given by the total number of exponents where the
        # variables to the right of the current one have degree exactly mondeg-(vardeg+1), mondeg-(vardeg+2), ....
        degree > vardeg && (index -= counts[degree-vardeg, i])
        iszero(degree -= vardeg) && break
    end
    return index
end

@inline function degree_from_index(::Unsafe, e::ExponentsDegree{<:Any,I}, index::I) where {I<:Integer}
    @inbounds counts = @view(index_counts(unsafe, e)[:, 1])
    iszero(e.mindeg) || (index += @inbounds counts[e.mindeg, 1])
    return searchsortedfirst(counts, index) -1
end

Base.@propagate_inbounds function degree_from_index(e::ExponentsDegree{<:Any,I}, index::I) where {I<:Integer}
    counts = let ic=index_counts(e, e.maxdeg)
        @assert(ic[2])
        @inbounds @view(ic[1][:, 1])
    end
    iszero(e.mindeg) || (index += @inbounds counts[e.mindeg, 1])
    return searchsortedfirst(counts, index) -1
end

function exponents_from_index(e::ExponentsDegree{<:Any,I}, index::I, degree::Int) where {I<:Integer}
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

function Base.iterate(efi::ExponentIndices{I,<:ExponentsDegree{<:Any,I}}) where {I<:Integer}
    parent = efi.parent
    counts = index_counts(unsafe, parent)
    degree = efi.degree
    @inbounds return iterate(efi, (degree, 2, (iszero(degree) ? efi.index : efi.index - counts[degree, 1]) +
                                              (iszero(parent.mindeg) ? zero(I) : counts[parent.mindeg, 1])))
end

function Base.iterate(efi::ExponentIndices{I,ExponentsDegree{N,I}}, (degree, i, index)::Tuple{Int,Int,I}) where {N,I<:Integer}
    counts = index_counts(unsafe, efi.parent) # must not be passed in the state - then Julia would have to do an allocation for
                                              # the return type
    @inbounds if i ≤ N
        iszero(degree) && return 0, (degree, i +1, index)
        # Obtain the degree for index i-1 by looking at how large the subspaces to the right of i-1 are when we fix i-1 to
        # a certain value - we are in the last subspace that still fits. The sizes of the exact subspaces are given by the
        # differences of adjacent rows in counts (as counts is accumulating). We must start from the smallest degree for
        # i-1, which means the largest degree for the subspace to the right. So we must accumulate in the opposite order
        # than what is stored in counts. But using the telescopic sum, these accumulations simplify.
        tmp = counts[degree+1, i]
        remainingdeg = searchsortedlast(@view(counts[:, i]), tmp - index)
        degᵢ₋₁ = degree - remainingdeg
        index -= tmp - counts[remainingdeg+1, i]
        return degᵢ₋₁, (remainingdeg, i +1, index)
    elseif i == N +1
        return degree, (degree, i +1, zero(index))
    else
        return nothing
    end
end

function iterate!(::Unsafe, v::AbstractVector{Int}, e::ExponentsDegree)
    # we must assume that a valid vector is provided, i.e. sum(v) ≥ e.mindeg
    @inbounds begin
        while true
            # This is not a loop at all, we only go through it once, but we need to be able to leave the block at multiple
            # positions. If we do it with @goto, as would be proper, Julia begins to box all our arrays.

            # find the next exponent that can be decreased
            i = findlast(>(0), v)
            isnothing(i) && break

            # we must increment the exponents to the left by 1 in total
            isone(i) && break
            v[i-1] += 1
            # this implies that we reset everything to the right of the increment to its minimum and then compensate for all
            # the reductions by increasing the exponents again
            δ = v[i] -1
            v[i] = 0
            v[end] += δ
            return true
        end
        # there's still hope: we can perhaps go to the next degree
        deg = sum(v, init=0) +1
        deg > e.maxdeg && return false
        fill!(@view(v[1:end-1]), 0)
        @inbounds v[end] = deg
        return true
    end
end