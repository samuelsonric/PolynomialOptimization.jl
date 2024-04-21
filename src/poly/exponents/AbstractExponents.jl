export AbstractExponents, AbstractExponentsUnbounded, AbstractExponentsDegreeBounded,
    index_counts, exponents_to_index, degree_from_index, iterate!, veciter, convert_index, compare_indices, exponents_sum

"""
    AbstractExponents{N,I}

Abstract supertype for all collections of multivariate exponents in `N` variables (`N > 0`). Every collection is iterable (both
using a default lazy iteration and a mutable iteration into a vector by means of [`iterate!`](@ref) or [`veciter`](@ref)) and
indexable (return a lazy collection of exponents) with index type `I`. The collection has a length if it is finite; it is never
empty.
"""
abstract type AbstractExponents{N,I<:Integer} end
# all AbstractExponents descendants must implement index_counts(::Unsafe, ::AbstractExponents{N,I}) -> Matrix{I} that returns a
# unique (potentially uninitialized) cache matrix for the given object (should be fast)

Base.IteratorEltype(::Type{<:AbstractExponents}) = Base.HasEltype()
Base.eltype(::Type{E}) where {I<:Integer,E<:AbstractExponents{<:Any,I}} = ExponentIndices{I,E}

"""
    indextype(e)

Returns the index type of an instance or subtype of [`AbstractExponents`](@ref). This function is not exported.
"""
indextype(::Union{<:AbstractExponents{<:Any,I},<:Type{<:AbstractExponents{<:Any,I}}}) where {I<:Integer} = I

"""
    AbstractExponentsUnbounded{N,I} <: AbstractExponents{N,I}

Abstract supertype for unbounded collections of multivariate exponents. These collections do not have a length; they are
infinite. Their cache is always initialized incrementally as required.
"""
abstract type AbstractExponentsUnbounded{N,I<:Integer} <: AbstractExponents{N,I} end
# AbstractExponentsUnbounded descendants must implement
# - _has_index_counts(::AbstractExponentsUnbounded, degree::Integer) that checks whether the cache matrix is valid for the
#   given degree (should be fast)
# - _calc_index_counts!(::AbstractExponentsUnbounded, degree::Integer) that makes sure that the cache matrix contains valid
#   entries for at least degree+1 rows
Base.IteratorSize(::Type{<:AbstractExponentsUnbounded}) = Base.IsInfinite()

"""
    AbstractExponentsDegreeBounded{N,I} <: AbstractExponents{N,I}

Abstract supertype for finite collections of multivariate exponents, bounded by their degrees. These collections have a length;
they also provide at least the fields `mindeg` and `maxdeg` that describe (a superset of) the described degree range. Their
cache is always initialized completely when required.
"""
abstract type AbstractExponentsDegreeBounded{N,I<:Integer} <: AbstractExponents{N,I} end
# AbstractExponentsDegreeBounded descendants must implement
# - _has_index_counts(::AbstractExponents) that checks whether the cache matrix is fully initialized (should be fast)
# - _calc_index_counts!(::AbstractExponents) that initializes the full cache matrix.

Base.IteratorSize(::Type{<:AbstractExponentsDegreeBounded}) = Base.HasLength()
"""
    length(unsafe, e::AbstractExponentsDegreeBounded)

Unsafe variant of [`length`](@ref length(::AbstractExponentsDegreeBounded)): requires the cache to be set up
correctly, else the behavior is undefined.
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
Base.lastindex(e::AbstractExponentsDegreeBounded{<:Any,I}) where {I<:Integer} = I(length(e))
Base.getindex(e::AbstractExponents{<:Any,I}, index::I) where {I<:Integer} = exponents_from_index(e, index)
# ^ should we maybe dispatch the @inbounds version to unsafe and the normal to the one with known cache? Dangerous, this makes
# assumptions not only about whether the index is valid, but also whether the cache is populated to know about the index...

Base.iterate(e::AbstractExponentsUnbounded{<:Any,I}) where {I<:Integer} =
    ExponentIndices(e, one(I), 0), (one(I), 0, one(I))
function Base.iterate(e::AbstractExponentsDegreeBounded{<:Any,I}) where {I<:Integer}
    counts, success = index_counts(e, e.mindeg) # ExponentIndices requires the cache to be set up
    iszero(e.mindeg) && return ExponentIndices(e, one(I), e.mindeg), (one(I), e.mindeg, one(I))
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

"""
    iterate!(v::AbstractVector{Int}, e::AbstractExponents)

Iterates through a set of exponents by maintaining an explicit representation of all exponents. This is slightly more efficient
that the lazy iteration version if every exponent has to be accessed explicitly. Note that `v` must be initialized with a valid
exponent combination in `e` (this may be done via `copyto!(v, first(e))`).
The function returns `true` if successful and `false` if the end was reached.
"""
@inline function iterate!(v::AbstractVector{Int}, e::AbstractExponents{N}) where {N}
    @boundscheck length(v) == N || throw(DimensionMismatch("Vector length does not match variable count"))
    iterate!(unsafe, v, e)
end

struct ExponentsVectorIterator{V,E<:AbstractExponents}
    v::V
    e::E

    function ExponentsVectorIterator(v::V, e::E) where {V<:AbstractVector{Int},N,E<:AbstractExponents{N}}
        length(v) == N || throw(DimensionMismatch("Vector length does not match variable count"))
        index_counts(e, 0) # make sure that the cache is populated
        new{V,E}(v, e)
    end

    function ExponentsVectorIterator(e::E) where {E<:AbstractExponents}
        index_counts(e, 0)
        new{Nothing,E}(nothing, e)
    end
end

Base.IteratorSize(::Type{<:ExponentsVectorIterator{<:Any,<:AbstractExponentsUnbounded}}) = Base.IsInfinite()
Base.IteratorSize(::Type{<:ExponentsVectorIterator{<:Any,<:AbstractExponentsDegreeBounded}}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ExponentsVectorIterator}) = Base.HasEltype()
Base.length(ei::ExponentsVectorIterator{<:Any,<:AbstractExponentsDegreeBounded}) = length(unsafe, ei.e)
Base.eltype(::Type{<:ExponentsVectorIterator{Nothing}}) = Vector{Int}
Base.eltype(::Type{<:ExponentsVectorIterator{V}}) where {V} = V
# AbstractExponents are never empty
Base.iterate(ei::ExponentsVectorIterator{<:AbstractVector{Int}}) = @inbounds copyto!(ei.v, first(ei.e)), ei.v
function Base.iterate(ei::ExponentsVectorIterator{Nothing})
    @inbounds v = collect(first(ei.e))
    return v, v
end
Base.iterate(ei::ExponentsVectorIterator, state::AbstractVector{Int}) =
    @inbounds iterate!(unsafe, state, ei.e) ? (state, state) : nothing
Base.parent(v::ExponentsVectorIterator) = v.e

"""
    veciter(e::AbstractExponents[, v::AbstractVector{Int}])

Creates an iterator over exponents that stores its result in a vector. This results in zero-allocations per iteration (as the
iteration over `e` also does), but is more efficient if every element in `e` must be accessed.
If the vector `v` is given as an argument, the data will be stored in this vector.
If the vector is omitted, it will be created once at the beginning of the iteration process.
The vector must never be altered, as it also serves as the state for the iterator; therefore, the same iterator may also not be
nested.
"""
veciter(e::AbstractExponents, v::AbstractVector{Int}) = ExponentsVectorIterator(v, e)
veciter(e::AbstractExponents) = ExponentsVectorIterator(e)

# Memory layout: We need to have the counts available for all exponents of nvars variables where nvars ∈ {1, ..., N} and
# of a degree that we don't want to fix yet (for the unbounded case, in the degree bounded case, everything is just calculated
# once). We need to quickly access the degrees of a fixed variable number.
# When we iterate through multiple variable numbers, the higher ones always come first.
# These demands are orthogonal: for quick access, we need (degree, N-nvars) as indices. In order to be able to add new
# degrees on demand, we want them to be at the end. So what we do instead is to wrap the matrix in a Ref and only keep this
# Ref alive. When we extend, we generate a new matrix, copy the data all over, then swap the reference. This can be made in
# a thread-safe manner without compromising performance.

function index_counts end

"""
    index_counts(unsafe, ::AbstractExponents{N,I})

Returns the current `Matrix{I}` that holds the exponents counts for up to `N` variables in its `N+1` columns (in descending
order, ending with zero variables), and for the maximal degrees in the rows (in ascending order, starting with zero).
The result is neither guaranteed to be defined at all nor have the required form if the cache was not calculated before.
"""
index_counts(::Unsafe, ::AbstractExponents{N,I}) where {N,I<:Integer}

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
    exponents_to_index(::AbstractExponents{N,I}, exponents,
        degree::Int=sum(exponents, init=0)[, report_lastexp::Int])

Calculates the index of a monomial in `N` variables in an exponent set with exponents given by the iterable `exponents` (whose
length should match `N`, else the behavior is undefined). The data type of the output is `I`.
If `exponents` is not present in the exponent set, the result is `zero(I)`.
`degree` must always match the sum of all elements in the exponent set, but if it is already known, it can be passed to the
function. No validity check is performed on `degree`.

!!! info "Truncated lengths"
    If the last argument `report_lastexp` is set to a value between `1` and `N`, the function will only consider the first
    `report_lastexp` exponents and return the largest index whose left exponents are compatible with those in `exponents`
    (whose length should still match `N`, unless `degree` is correctly specified manually).
    Again, if no such match can be found, the index is zero.
    If `report_lastexp` is set, the result will be a 2-tuple whose first entry is the index and whose second entry is the value
    of the last considered exponent, i.e. `exponents[report_lastexp]` if `exponents` is indexable.
"""
exponents_to_index(e::AbstractExponents, exponents, degree::Int=sum(exponents, init=0), report_lastexp=nothing) =
    _exponents_to_index(e, exponents, degree, report_lastexp)
function _exponents_to_index end
# implement this worker for all concrete AbstractExponents types with four mandatory args

function degree_from_index end
"""
    degree_from_index(unsafe, ::AbstractExponents{N,I}, index::I)

Unsafe variant of [`degree_from_index`](@ref degree_from_index(::AbstractExponents{N,I}, ::I) where {N,I<:Integer}): assumes
that the required cache for the degree that is associated with `index` has already been initialized, else the behavior is
undefined (and in fact, non-deterministic, as it depends on the current size of the cache).
"""
degree_from_index(::Unsafe, ::AbstractExponents{N,I}, ::I) where {N,I<:Integer}

"""
    degree_from_index(::AbstractExponents{N,I}, index::I)

Returns the degree that is associated with a given monomial index `index` in `N` variables. If the index was larger than the
maximally allowed index in the exponent set, a degree larger than the maximal degree allowed in the iterator will be returned
(not necessarily `lastindex +1`).
"""
degree_from_index(::AbstractExponents{N,I}, ::I) where {N,I<:Integer}

"""
    convert_index(unsafe, target::AbstractExponents{N}, source::AbstractExponents{N,I},
        index::I[, degree::Int])

Unsafe variant of [`convert_index`](@ref convert_index(::AbstractExponents{N,I}, ::AbstractExponents{N,IS}, ::IS, ::Int) where {N,I<:Integer,IS<:Integer}):
assumes that caches for both the source and the target are set up as required for `degree`. If `degree` is omitted, it is
calculated using the unsafe variant of
[`degree_from_index`](@ref degree_from_index(::Unsafe, ::AbstractExponents{N,I}, ::I) where {N,I<:Integer}).`
"""
@inline function convert_index(::Unsafe, target::AbstractExponents{N,I}, source::AbstractExponents{N,IS}, index::IS,
    degree::Int) where {N,I<:Integer,IS<:Integer}
    # there are several paths (and compile time reduces these to at most one if) that provide shorted conditions because we
    # know that the indices must coincide.
    source === target && return index
    if target isa ExponentsAll
        if source isa ExponentsAll
            return I(index)
        elseif source isa ExponentsDegree
            return iszero(source.mindeg) ? I(index) : I(@inbounds(index_counts(unsafe, source)[source.mindeg, 1]) + index)
        elseif source isa ExponentsMultideg && all(≥(degree), source.maxmultideg)
            iszero(source.mindeg) && return I(index) # implies minmultideg = 0
            all(iszero, source.minmultideg) && return I(@inbounds(index_counts(unsafe, source)[source.mindeg, 1]) + index)
        end
    elseif target isa ExponentsDegree
        target.mindeg ≤ degree ≤ target.maxdeg || return zero(I)
        if source isa ExponentsAll
            return iszero(target.mindeg) ? I(index) : I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
        elseif source isa ExponentsDegree
            iszero(source.mindeg) || (index += @inbounds(index_counts(unsafe, source)[source.mindeg, 1]))
            return iszero(target.mindeg) ? I(index) : I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
        elseif source isa ExponentsMultideg && all(≥(degree), source.maxmultideg)
            iszero(source.mindeg) &&
                return iszero(target.mindeg) ? I(index) : I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
            if all(iszero, source.minmultideg)
                index += @inbounds(index_counts(unsafe, source)[source.mindeg, 1])
                return iszero(target.mindeg) ? I(index) : I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
            end
        end
    elseif target isa ExponentsMultideg
        target.mindeg ≤ degree ≤ target.maxdeg || return zero(I)
        if source isa ExponentsAll && all(≥(degree), target.maxmultideg)
            iszero(target.mindeg) && return I(index)
            all(iszero, target.minmultideg) && return I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
        elseif source isa ExponentsDegree && all(≥(degree), target.maxmultideg)
            index0 = index
            iszero(source.mindeg) || (index0 += @inbounds(index_counts(unsafe, source)[source.mindeg, 1]))
            iszero(target.mindeg) && return I(index0)
            all(iszero, target.minmultideg) && return I(index0) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
        elseif source isa ExponentsMultideg && source.minmultideg == target.minmultideg &&
            all(let degree=degree; x -> min(x[1], degree) == min(x[2], degree) end,
                zip(target.maxmultideg, source.maxmultideg))
            iszero(source.mindeg) || (index += @inbounds(index_counts(unsafe, source)[source.mindeg, 1]))
            return iszero(target.mindeg) ? I(index) : I(index) - @inbounds(index_counts(unsafe, target)[target.mindeg, 1])
        end
    end
    # no shortcut available, must do it the hard way
    return _exponents_to_index(target, exponents_from_index(unsafe, source, index, degree), degree, nothing)
end

@inline function convert_index(::Unsafe, target::AbstractExponents{N}, source::AbstractExponents{N,IS}, index::IS) where {N,IS<:Integer}
    source === target && return index
    return convert_index(unsafe, target, source, index, degree_from_index(unsafe, source, index))
end

"""
    convert_index(target::AbstractExponents{N}, source::AbstractExponents{N,I},
        index::I[, degree::Int])

Converts an index from an exponent set `source` to an exponent set `target`.
`index` is always assumed to be a valid index for `source`, else the behavior is undefined.
"""
@inline function convert_index(target::AbstractExponents{N,I}, source::AbstractExponents{N,IS}, index::IS, degree::Int) where {N,I<:Integer,IS<:Integer}
    source === target && return index
    index_counts(source, degree) # populate caches
    index_counts(target, degree)[2] || return zero(I)
    return convert_index(unsafe, target, source, index, degree)
end

@inline function convert_index(target::AbstractExponents{N,I}, source::AbstractExponents{N,IS}, index::IS) where {N,I<:Integer,IS<:Integer}
    source === target && return index
    degree = degree_from_index(source, index)
    index_counts(target, degree)[2] || return zero(I)
    return convert_index(unsafe, target, source, index, degree)
end

_CompareOp = Union{typeof(==),typeof(!=),typeof(<),typeof(≤),typeof(>),typeof(≥)}
"""
    compare_indices(unsafe, e₁::AbstractExponents, index₁, op, e₂::AbstractExponents,
        index₂[, degree::Int])

Unsafe variant of [`compare_indices`](@ref compare_indices(::AbstractExponents{N,I1}, ::I1, ::_CompareOp, ::AbstractExponents{N,I2}, ::I2, ::Int) where {N,I1<:Integer,I2<:Integer}):
assumes that both caches are set up as required for `degree` (which, if given, must be the common degree of both exponents).
If `degree` is omitted, it is calculated using the unsafe variant of
[`degree_from_index`](@ref degree_from_index(::Unsafe, ::AbstractExponents{N,I}, ::I) where {N,I<:Integer}).
"""
@inline function compare_indices(::Unsafe, e₁::AbstractExponents{N,I1}, index₁::I1,
    op::_CompareOp, e₂::AbstractExponents{N,I2}, index₂::I2,
    degree::Int) where {N,I1<:Integer,I2<:Integer}
    # there are several paths (and compile time reduces these to at most one if) that provide shorted conditions because we
    # know that the indices must coincide.
    e₂ === e₁ && return op(index₁, index₂)
    if e₁ isa ExponentsAll
        if e₂ isa ExponentsAll
            return op(index₁, index₂)
        elseif e₂ isa ExponentsDegree
            return op(index₁, iszero(e₂.mindeg) ? index₂ : index₂ + @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
        elseif e₂ isa ExponentsMultideg && all(≥(degree), e₂.maxmultideg)
            iszero(e₂.mindeg) && return op(index₁, index₂) # implies minmultideg = 0
            all(iszero, e₂.minmultideg) && return op(index₁, index₂ + @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
        end
    elseif e₁ isa ExponentsDegree
        if e₂ isa ExponentsAll
            return op(iszero(e₁.mindeg) ? index₁ : index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]), index₂)
        elseif e₂ isa ExponentsDegree
            return op(iszero(e₁.mindeg) ? index₁ : index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]),
                      iszero(e₂.mindeg) ? index₂ : index₂ + @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
        elseif e₂ isa ExponentsMultideg && all(≥(degree), e₂.maxmultideg)
            iszero(e₂.mindeg) &&
                return op(iszero(e₁.mindeg) ? index₁ : index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]), index₂)
            if all(iszero, e₂.minmultideg)
                return op(iszero(e₁.mindeg) ? index₁ : index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]),
                          index₂ + @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
            end
        end
    elseif e₁ isa ExponentsMultideg
        if e₂ isa ExponentsAll && all(≥(degree), e₁.maxmultideg)
            iszero(e₁.mindeg) && return op(index₁, index₂)
            all(iszero, e₁.minmultideg) && return op(index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]), index₂)
        elseif e₂ isa ExponentsDegree && all(≥(degree), e₁.maxmultideg)
            index₂0 = index₂
            iszero(e₂.mindeg) || (index₂0 += @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
            iszero(e₁.mindeg) && return op(index₁, index₂0)
            all(iszero, e₁.minmultideg) && return op(index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]), index₂0)
        elseif e₂ isa ExponentsMultideg && e₂.minmultideg == e₁.minmultideg &&
            all(let degree=degree; x -> min(x[1], degree) == min(x[2], degree) end,
                zip(e₁.maxmultideg, e₂.maxmultideg))
            return op(iszero(e₁.mindeg) ? index₁ : index₁ + @inbounds(index_counts(unsafe, e₁)[e₁.mindeg, 1]),
                      iszero(e₂.mindeg) ? index₂ : index₂ + @inbounds(index_counts(unsafe, e₂)[e₂.mindeg, 1]))
        end
    end
    # no shortcut available, must do it the hard way, but we can perhaps short-circuit along the way
    for (exp₁, exp₂) in zip(exponents_from_index(unsafe, e₁, index₁, degree), exponents_from_index(unsafe, e₂, index₂, degree))
        if op === ==
            exp₁ == exp₂ || return false
        elseif op === !=
            exp₁ == exp₂ || return true
        elseif (op === <) || (op === ≤)
            exp₁ < exp₂ && return true
            exp₁ > exp₂ && return false
        elseif (op === >) || (op === ≥)
            exp₁ > exp₂ && return true
            exp₁ < exp₂ && return false
        end
    end
    return op ∈ (==, ≤, ≥)
end

@inline function compare_indices(::Unsafe, e₁::AbstractExponents{N,I1}, index₁::I1, op::_CompareOp,
    e₂::AbstractExponents{N,I2}, index₂::I2) where {N,I1<:Integer,I2<:Integer}
    e₁ === e₂ && return op(index₁, index₂)
    deg₁ = degree_from_index(unsafe, e₁, index₁)
    deg₂ = degree_from_index(unsafe, e₂, index₂)
    return deg₁ == deg₂ ? compare_indices(unsafe, e₁, index₁, op, e₂, index₂, deg₁) : op(deg₁, deg₂)
end

"""
    compare_indices(e₁::AbstractExponents, index₁, op, e₂::AbstractExponents,
        index₂[, degree::Int])

Compares two indices from two possibly different exponent sets. `degree`, if given, must be the common degree of both
exponents); when `degree` is omitted, also different degrees are possible.
Both indices are assumed to be valid for their respective exponent sets, else the behavior is undefined. However, it is not
necessary that an index is also valid in the other's exponent set.
If ẽ₁ and ẽ₂ denote the conversion of the indices to a common exponent set (say, [`ExponentsAll`](@ref)), then the result is
`op(ẽ₁, ẽ₂)`. The only allowed values for `op` are `==`, `!=`, `<`, `<=`, `>`, and `>=`.
"""
@inline function compare_indices(e₁::AbstractExponents{N,I1}, index₁::I1, op::_CompareOp, e₂::AbstractExponents{N,I2},
    index₂::I2, degree::Int) where {N,I1<:Integer,I2<:Integer}
    e₁ === e₂ && return op(index₁, index₂)
    index_counts(e₁, degree) # populate caches
    index_counts(e₂, degree)
    return compare_indices(unsafe, e₁, index₁, op, e₂, index₂, degree)
end

@inline function compare_indices(e₁::AbstractExponents{N,I1}, index₁::I1, op::_CompareOp, e₂::AbstractExponents{N,I2},
    index₂::I2) where {N,I1<:Integer,I2<:Integer}
    e₁ === e₂ && return op(index₁, index₂)
    deg₁ = degree_from_index(e₁, index₁)
    deg₂ = degree_from_index(e₂, index₂)
    return deg₁ == deg₂ ? compare_indices(unsafe, e₁, index₁, op, e₂, index₂, deg₁) : op(deg₁, deg₂)
end

struct ExponentsSum{N,T<:Tuple}
    e::T

    ExponentsSum{N}(e...) where {N} = new{N,typeof(e)}(e)
end

Base.IteratorSize(::Type{<:ExponentsSum}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ExponentsSum}) = Base.HasEltype()
Base.length(s::ExponentsSum{N}) where {N} = N
Base.eltype(::Type{<:ExponentsSum}) = Int
Base.iterate(s::ExponentsSum{<:Any,T}) where {C,T<:NTuple{C,Any}} = iterate(s, ntuple(i -> iterate(s.e[i])::Tuple, Val(C)))
@generated function Base.iterate(s::ExponentsSum, state)
    items = fieldcount(state)
    quote
        total = +($((:(state[$i][1]) for i in 1:items)...))
        $((:(isnothing($(Expr(:(=), Symbol(:nextstate, i), :(iterate(s.e[$i], state[$i][2]))))) &&
            return nothing) for i in 1:items)...)
        return total, ($(((Symbol(:nextstate, i)) for i in 1:items)...),)
    end
end

"""
    exponents_sum(e::AbstractExponents{N,I}, exponents...) -> Tuple{I,Int}

Calculates the index of the sum of all `exponents` within `e`. If the result cannot be found in `e`, the function will return
zero. Returns the total degree as second entry in the tuple
"""
Base.@assume_effects :consistent @generated function exponents_sum(e::AbstractExponents{N}, exponents...) where {N}
    items = length(exponents)
    quote
        d = +($((:(sum(exponents[$i])) for i in 1:items)...))
        return exponents_to_index(e, ExponentsSum{N}($((:(exponents[$i]) for i in 1:items)...)), d), d
    end
end