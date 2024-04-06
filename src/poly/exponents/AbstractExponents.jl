export AbstractExponents, AbstractExponentsUnbounded, AbstractExponentsDegreeBounded,
    index_counts, exponents_to_index, degree_from_index, iterate!, veciter, convert_index, compare_indices

"""
    AbstractExponents{N,I}

Supertype for all collections of multivariate exponents. Every collection is iterable (both using a default lazy iteration and
a mutable iteration into a vector using [`veciter`](@ref)) and indexable (return a lazy collection of exponents). The
collection has a length if it is finite.
"""
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
Base.length(ei::ExponentsVectorIterator{<:Any,<:AbstractExponentsDegreeBounded}) = length(unsafe, ei.e)
# AbstractExponents are never empty
Base.iterate(ei::ExponentsVectorIterator{<:AbstractVector{Int}}) = copyto!(ei.v, first(ei.e)), ei.v
function Base.iterate(ei::ExponentsVectorIterator{Nothing})
    v = collect(first(ei.e))
    return v, v
end
Base.iterate(ei::ExponentsVectorIterator, state::AbstractVector{Int}) =
    iterate!(unsafe, state, ei.e) ? (state, state) : nothing

"""
    veciter(e::AbstractExponents[, v::AbstractVector{Int}])

Creates an iterator over exponents that stores its result in a vector. This results in zero-allocations (as the iteration over
`e` also does), but is more efficient if every element in `e` must be accessed.
If the vector `v` is given as an argument, the data will be stored in this vector; it is then not allowed to nest the iterator.
If the vector is omitted, it will be created once at the beginning of the iteration process.
The vector must never be altered, as it also serves as the state for the iterator.
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
    exponents_to_index(::AbstractExponents{N,I}, exponents, degree::Int=sum(exponents, init=0)[, report_lastexp::Int])

Calculates the index of a monomial in `N` variables in an exponent set with exponents given by the iterable `exponents` (whose
length should match `N`, else the behavior is undefined). The data type of the output is `I`.
If `exponents` is not present in the exponent set, the result is zero.
`degree` must always match the sum of all elements in the exponent set, but if it is already known, it can be passed to the
function. No validity check is performed.

!!! info "Truncated lengths"
    If the last argument `report_lastexp` is set to a value between `1` and `N`, the function will only consider the first
    `report_lastexp` exponents and return the largest index whose left exponents are compatible with those in `exponents`
    (whose length should still match `N`, unless `degree` is correctly specified manually).
    Again, if no such match can be found, the index is zero.
    If `report_lastexp` is set, the result will be a 2-tuple whose first entry is the index and whose second entry is the value
    of the exponents, i.e. `exponents[report_lastexp]` if `exponents` is indexable.
"""
exponents_to_index(e::AbstractExponents, exponents, degree::Int=sum(exponents, init=0), report_lastexp=nothing) =
    _exponents_to_index(e, exponents, degree, report_lastexp)
function _exponents_to_index end
# implement this worker for all concrete AbstractExponents types with four mandatory args

"""
    degree_from_index(unsafe, ::AbstractExponents{N,I}, index::I)

Returns the degree that is associated with a given monomial index `index` in `N` variables. This function is unsafe; it assumes
that the required cache for the output degree has already been initialized. If the index was larger than the number of items in
the cache, a degree larger than the maximal degree allowed in the iterator will be contained.


    degree_from_index(::AbstractExponents{N,I}, index::I)

Same as above, but makes sure that the cache is initialized appropriately.
"""
function degree_from_index end

"""
    convert_index([unsafe,] target::AbstractExponents, source::AbstractExponents, index[, degree::Int])

Converts an index from an exponent set `source` to an exponent set `target`. The unsafe variant assumes that caches for both
the source and the target are set up as required for `degree`.
`index` is always assumed to be a valid index for `source`, else the behavior is undefined.
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
    compare_indices([unsafe,] e₁::AbstractExponents, index₁, op, e₂::AbstractExponents, index₂[, degree::Int])

Compares two indices from two possibly different exponent sets. The unsafe variant assumes that both caches are set up as
required for `degree` (which, if given, must be the common degree of both exponents).
Both indices are assumed to be valid for their respective exponent sets, else the behavior is undefined. However, it is not
necessary that both indices are also valid in the others exponent sets.
If ẽ₁ and ẽ₂ denote the conversion of the indices to a common exponent set (say, [`ExponentsAll`](@ref)), then the result is
`op(ẽ₁, ẽ₂)`. The only allowed values for `op` are `==`, `<`, `≤`, `>`, and `≥`.
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