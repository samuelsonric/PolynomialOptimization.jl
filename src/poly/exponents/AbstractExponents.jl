export index_counts, index_counts, exponents_to_index, degree_from_index

abstract type AbstractExponents{N,I<:Integer} end
# all AbstractExponents descendants must implement index_counts(::Unsafe, ::AbstractExponents{N,I}) -> Matrix{I} that returns a
# unique (potentially uninitialized) cache matrix for the given object (should be fast)

Base.IteratorEltype(::Type{<:AbstractExponents}) = Base.HasEltype()
Base.eltype(::Type{E}) where {I<:Integer,E<:AbstractExponents{<:Any,I}} = ExponentIndices{I,E}

abstract type AbstractExponentsUnbounded{N,I<:Integer} <: AbstractExponents{N,I} end
# AbstractExponentsUnbounded descendants must implement
# - _has_index_counts(::AbstractExponentsUnbounded, degree::Integer) that checks whether the cache matrix is valid for the
#   given degree (should be fast)
# - _calc_index_counts!(::AbstractExponentsUnbounded, degree::Integer) that makes sure that the cache matrix contains valid
#   entries for at least degree+1 rows
Base.IteratorSize(::Type{<:AbstractExponentsUnbounded}) = Base.IsInfinite()

abstract type AbstractExponentsDegreeBounded{N,I<:Integer} <: AbstractExponents{N,I} end
# AbstractExponentsDegreeBounded descendants must implement
# - _has_index_counts(::AbstractExponents) that checks whether the cache matrix is fully initialized (should be fast)
# - _calc_index_counts!(::AbstractExponents) that initializes the full cache matrix.

Base.IteratorSize(::Type{<:AbstractExponentsDegreeBounded}) = Base.HasLength()
"""
    length(unsafe, e::AbstractExponentsDegreeBounded)

Returns the total number of exponents present in `e`. Requires the cache to be set up correctly, else the behavior is
undefined.
"""
function Base.length(::Unsafe, e::AbstractExponentsDegreeBounded)
    counts = index_counts(unsafe, e)
    @inbounds return counts[e.maxdeg+1, 1] - (iszero(e.mindeg) ? zero(eltype(counts)) : counts[e.mindeg, 1])
end
"""
    length(e::AbstractExponentsDegreeBounded)

Returns the total number of exponents present in `e`.
"""
function Base.length(e::AbstractExponentsDegreeBounded)
    counts, success = index_counts(e, e.maxdeg)
    @assert(success)
    @inbounds return counts[e.maxdeg+1, 1] - (iszero(e.mindeg) ? zero(eltype(counts)) : counts[e.mindeg, 1])
end

Base.firstindex(::AbstractExponents{<:Any,I}) where {I<:Integer} = one(I)
Base.lastindex(e::AbstractExponentsDegreeBounded) = length(e)
Base.getindex(e::AbstractExponents{<:Any,I}, index::I) where {I<:Integer} = exponents_from_index(e, index)
# ^ should we maybe dispatch the @inbounds version to unsafe and the normal to the one with known cache? Dangerous, this makes
# assumptions not only about whether the index is valid, but also whether the cache is populated to know about the index...

Base.iterate(e::AbstractExponentsUnbounded{<:Any,I}) where {I<:Integer} =
    ExponentIndices(e, one(I), 0), (one(I), 0, one(I))
function Base.iterate(e::AbstractExponentsDegreeBounded{<:Any,I}) where {I<:Integer}
    iszero(e.mindeg) && return ExponentIndices(e, one(I), e.min), (one(I), e.mindeg, one(I))
    counts, success = index_counts(e, e.mindeg)
    @assert(success)
    @inbounds return ExponentIndices(e, one(I), e.mindeg), (one(I), e.mindeg, counts[e.mindeg+1, 1] - counts[e.mindeg, 1])
end
function Base.iterate(e::AbstractExponentsUnbounded{<:Any,I}, (index, degree, remainingdeg)::Tuple{I,Int,I}) where {I<:Integer}
    if isone(remainingdeg)
        counts, success = index_counts(e, degree +1)
        @assert(success)
        @inbounds return ExponentIndices(e, index + one(I), degree +1),
            (index + one(I), degree +1, counts[degree+2, 1] - counts[degree+1, 1])
    else
        return ExponentIndices(e, index + one(I), degree), (index + one(I), degree, remainingdeg - one(I))
    end
end
function Base.iterate(e::AbstractExponentsDegreeBounded{<:Any, I}, (index, degree, remainingdeg)::Tuple{I,Int,I}) where {I<:Integer}
    if isone(remainingdeg)
        degree == e.maxdeg && return nothing
        counts, success = index_counts(e, degree +1)
        @assert(success)
        @inbounds return ExponentIndices(e, index + one(I), degree +1),
            (index + one(I), degree +1, counts[degree+2, 1] - counts[degree+1, 1])
    else
        return ExponentIndices(e, index + one(I), degree), (index + one(I), degree, remainingdeg - one(I))
    end
end

# Memory layout: We need to have the counts available for all exponents of nvars variables where nvars ∈ {1, ..., N} and
# of a degree that we don't want to fix yet (for the unbounded case, in the degree bounded case, everything is just calculated
# once). We need to quickly access the degrees of a fixed variable number.
# When we iterate through multiple variable numbers, the higher ones always come first.
# These demands are orthogonal: for quick access, we need (degree, N-nvars) as indices. In order to be able to add new
# degrees on demand, we want them to be at the end. So what we do instead is to wrap the matrix in a Ref and only keep this
# Ref alive. When we extend, we generate a new matrix, copy the data all over, then swap the reference. This can be made in
# a thread-safe manner without compromising performance.

"""
    index_counts(unsafe, ::AbstractExponents{N,I})

Returns the current `Matrix{I}` that holds the exponents counts for up to `N` variables in its `N+1` columns (in descending
order, ending with zero variables), and for the maximal degrees in the rows (in ascending order, starting with zero).
The result is neither guaranteed to be defined at all nor have the required form if the cache was not calculated before.
"""
function index_counts end

Base.@assume_effects :total !:inaccessiblememonly index_counts(::Unsafe, e::AbstractExponentsDegreeBounded) =
    (@inline; e.counts)

"""
    index_counts(::AbstractExponents{N,I}, degree::Integer) -> Tuple{Matrix{I},Bool}

Safe version of the above. If the boolean in the result is `true`, the matrix will have at least `degree+1` rows, i.e., all
entries up to `degree` are present. If it is false, the requested degree is not present in the exponents and the matrix will
have fewer rows. Note that `true` does not mean that the degree is actually present in the exponents, only that its information
has been calculated.
"""
@inline function index_counts(e::AbstractExponentsUnbounded, degree::Integer)
    # index_counts will be interpreted in a two-dimensional way. The maximal number of variables is fixed, so this will
    # be the fast dimension; the degrees may increase, which leads to appending elements. This will be the slow dimension.
    @boundscheck 0 ≤ degree || throw(ArgumentError("The degree must be nonnegative"))
    @inline(_has_index_counts(e, degree)) || @noinline(_calc_index_counts!(e, degree)) # zero degree = row 1
    # ^ would be good if we could insert an unlikely hint here
    counts = index_counts(unsafe, e)
    return counts, true
end

Base.@assume_effects :nothrow function index_counts(e::AbstractExponentsDegreeBounded, degree::Integer)
    @inline
    @inline(_has_index_counts(e)) || @noinline(_calc_index_counts!(e))
    return e.counts, size(e.counts, 1) > degree
end

function _has_index_counts end
function _calc_index_counts! end
@inline _has_index_counts(e::AbstractExponentsDegreeBounded) = isdefined(e, :counts)

"""
    exponents_to_index(::AbstractExponents{N,I}, exponents, degree::Int=sum(exponents, init=0))

Calculates the index of a monomial in `N` variables in an exponent set with exponents given by the iterable `exponents` (whose
length should match `N`, else the behavior is undefined). The data type of the output is `I`.
If `exponents` is not present in the exponent set, the result is zero.
`degree` must always match the sum of all elements in the exponent set, but if it is already known, it can be passed to the
function. No validity check is performed.
"""
function exponents_to_index end

"""
    degree_from_index(unsafe, ::AbstractExponents{N,I}, index::I)

Returns the degree that is associated with a given monomial index `index` in `N` variables. This function is unsafe; it assumes
that the required cache for the output degree has already been initialized. If the index was larger than the number of items in
the cache, a degree larger than the maximal degree allowed in the iterator will be contained.


    degree_from_index(::AbstractExponents{N,I}, index::I)

Same as above, but makes sure that the cache is initialized appropriately.
"""
function degree_from_index end