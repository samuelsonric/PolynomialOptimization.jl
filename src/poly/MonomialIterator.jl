export AbstractMonomialIterator, MonomialIterator, RangedMonomialIterator, ownexponents, exponents_from_index_prepare

abstract type AbstractMonomialIterator{E,P<:Integer} end

Base.IteratorSize(::Type{<:AbstractMonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:AbstractMonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{<:AbstractMonomialIterator{<:Any,P}}) where {P<:Integer} = Vector{P}
Base.getindex(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=true)
Base.view(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=false)
Base.firstindex(::AbstractMonomialIterator) = 1
Base.lastindex(iter::AbstractMonomialIterator) = length(iter)

struct OwnExponents end
"""
    const ownexponents

This value instructs `MonomialIterator` to own its iterate. It will modify the same vector at every iteration.
"""
const ownexponents = OwnExponents()

"""
    MonomialIterator(mindeg, maxdeg, minmultideg, maxmultideg[, exponents])

This is an advanced iterator that is able to iterate through all monomials with constraints specified not only by a minimum and
maximum total degree, but also by individual variable degrees.
If `exponents` is set to [`ownexponents`](@ref) (or passed a `Vector{<:Integer}` of appropriate length), the iterator use the
same vector of exponents whenever it is used, so it must not be used multiple times simultaneously. Additionally, during
iteration, no copy is created, so the vector must not be modified and accumulation e.g. by `collect` won't work.
Note that the exponents that this iterator returns will be of the common integer type of `mindeg`, `maxdeg`, and the element
types of `minmultideg`, `maxmultideg` (and potentially `ownexponents`).

The monomials will be returned in an order that is compatible with `Graded{LexOrder}`.
"""
struct MonomialIterator{E,P<:Integer} <: AbstractMonomialIterator{E,P}
    n::Int
    mindeg::P
    maxdeg::P
    minmultideg::Vector{P}
    maxmultideg::Vector{P}
    exponents::E
    Σminmultideg::UInt
    Σmaxmultideg::UInt

    function MonomialIterator(mindeg::P, maxdeg::P, minmultideg::Vector{P}, maxmultideg::Vector{P},
        exponents::Union{Nothing,OwnExponents,<:AbstractVector{P}}=nothing) where {P<:Integer}
        (mindeg < 0 || mindeg > maxdeg) && throw(ArgumentError("Invalid degree specification"))
        n = length(minmultideg)
        (n != length(maxmultideg) ||
            any(minmax -> minmax[1] < 0 || minmax[1] > minmax[2], zip(minmultideg, maxmultideg))) &&
            throw(ArgumentError("Invalid multidegree specification"))
        Σminmultideg = sum(minmultideg, init=zero(P))
        Σmaxmultideg = sum(maxmultideg, init=zero(P))
        if exponents isa OwnExponents
            return new{Vector{P},P}(n, mindeg, maxdeg, minmultideg, maxmultideg, Vector{P}(undef, n), Σminmultideg,
                Σmaxmultideg)
        elseif isnothing(exponents)
            return new{Nothing,P}(n, mindeg, maxdeg, minmultideg, maxmultideg, nothing, Σminmultideg, Σmaxmultideg)
        elseif length(exponents) != n
            throw(ArgumentError("Invalid length of ownexponents"))
        else
            return new{typeof(exponents),P}(n, mindeg, maxdeg, minmultideg, maxmultideg, exponents, Σminmultideg, Σmaxmultideg)
        end
    end

    function MonomialIterator(iter::MonomialIterator{E,P}) where {E<:AbstractVector,P<:Integer}
        simp = similar(iter.exponents)
        new{typeof(simp),P}(iter.n, iter.mindeg, iter.maxdeg, iter.minmultideg, iter.maxmultideg, simp, iter.Σminmultideg,
            iter.Σmaxmultideg)
    end

    MonomialIterator(iter::MonomialIterator{Nothing,P}) where {P<:Integer} = iter

    # internal use only
    MonomialIterator{E,P}(mindeg::P, maxdeg::P, minmultideg::Vector{P}, maxmultideg::Vector{P},
        exponents::E, Σminmultideg::UInt, Σmaxmultideg::UInt) where {E,P<:Integer} =
        new{E,P}(length(minmultideg), mindeg, maxdeg, minmultideg, maxmultideg, exponents, Σminmultideg, Σmaxmultideg)
end

function Base.iterate(iter::MonomialIterator{E}) where {E}
    minmultideg, Σminmultideg, Σmaxmultideg = iter.minmultideg, iter.Σminmultideg, iter.Σmaxmultideg
    Σminmultideg > iter.maxdeg && return nothing
    Σmaxmultideg < iter.mindeg && return nothing
    exponents = E === Nothing ? copy(minmultideg) : copyto!(iter.exponents, minmultideg)
    if iter.mindeg > Σminmultideg
        exponents_increment_right(iter, exponents, iter.mindeg - Σminmultideg, 1) || @assert(false)
        deg = iter.mindeg
    else
        deg = typeof(iter.mindeg)(Σminmultideg)
    end
    return E === Nothing ? copy(exponents) : exponents, (deg, exponents)
end

function Base.iterate(iter::MonomialIterator{E}, state::Tuple{P,<:AbstractVector{P}}) where {E,P}
    deg, exponents = state
    deg ≤ iter.maxdeg || return nothing
    minmultideg, maxmultideg = iter.minmultideg, iter.maxmultideg
    @inbounds while true
        # This is not a loop at all, we only go through it once, but we need to be able to leave the block at multiple
        # positions. If we do it with @goto, as would be proper, Julia begins to box all our arrays.

        # find the next exponent that can be decreased
        found = false
        local i
        for outer i in iter.n:-1:1
            if exponents[i] > minmultideg[i]
                found = true
                break
            end
        end
        found || break
        # we must increment the exponents to the left by 1 in total
        found = false
        local j
        for outer j in i-1:-1:1
            if exponents[j] < maxmultideg[j]
                found = true
                break
            end
        end
        found || break

        exponents[j] += 1
        # this implies that we reset everything to the right of the increment to its minimum and then compensate for all
        # the reductions by increasing the exponents again where possible
        δ = sum(k -> exponents[k] - minmultideg[k], j+1:i, init=0) -1
        copyto!(exponents, j +1, minmultideg, j +1, i - j)
        if exponents_increment_right(iter, exponents, δ, j +1)
            return E === Nothing ? copy(exponents) : exponents, (deg, exponents)
        end
        break
    end
    # there's still hope: we can perhaps go to the next degree
    deg += one(P)
    deg > iter.maxdeg && return nothing
    copyto!(exponents, minmultideg)
    if exponents_increment_right(iter, exponents, deg - iter.Σminmultideg, 1)
        return E === Nothing ? copy(exponents) : exponents, (deg, exponents)
    else
        return nothing
    end
end

function _moniter_length(maxdeg, minmultideg, maxmultideg, cache::AbstractMatrix{Int}=Matrix{Int}(undef, maxdeg +1, 2))
    # internal function without checks or quick path
    # ~ O(n*d^2)
    @inbounds occurrences = fill!(@view(cache[:, 1]), 0)
    @inbounds fill!(@view(occurrences[minmultideg[1]+1:min(maxmultideg[1], maxdeg)+1]), 1)
    @inbounds nextround = @view(cache[:, 2])
    for (minᵢ, maxᵢ) in Iterators.drop(zip(minmultideg, maxmultideg), 1)
        fill!(nextround, 0)
        for degᵢ in minᵢ:min(maxᵢ, maxdeg)
            for (degⱼ, occⱼ) in zip(Iterators.countfrom(0), occurrences)
                newdeg = degᵢ + degⱼ
                newdeg > maxdeg && break
                @inbounds nextround[newdeg+1] += occⱼ
            end
        end
        occurrences, nextround = nextround, occurrences
    end
    return occurrences
end
Base.@assume_effects :foldable :nothrow :notaskstate function Base.length(iter::MonomialIterator)
    maxdeg = iter.maxdeg
    iter.Σminmultideg > maxdeg && return 0
    iter.Σmaxmultideg < iter.mindeg && return 0
    @inbounds isone(iter.n) && return min(maxdeg, iter.maxmultideg[1]) - max(iter.mindeg, iter.minmultideg[1]) +1
    @inbounds return sum(@view(@inline(_moniter_length(maxdeg, iter.minmultideg, iter.maxmultideg))[iter.mindeg+1:end]), init=0)
end
Base.@assume_effects :consistent :terminates_globally :nothrow :notaskstate function Base.length(iter::MonomialIterator,
    cache::AbstractMatrix{Int})
    maxdeg = iter.maxdeg
    iter.Σminmultideg > maxdeg && return 0
    iter.Σmaxmultideg < iter.mindeg && return 0
    @inbounds isone(iter.n) && return min(maxdeg, iter.maxmultideg[1]) - max(iter.mindeg, iter.minmultideg[1]) +1
    @inbounds return sum(
        @view(@inline(_moniter_length(maxdeg, iter.minmultideg, iter.maxmultideg, cache))[iter.mindeg+1:end]),
        init=0
    )
end

function exponents_increment_right(iter::MonomialIterator, exponents, δ, from)
    @assert(δ ≥ 0 && from ≥ 0)
    maxmultideg = iter.maxmultideg
    i = iter.n
    @inbounds while δ > 0 && i ≥ from
        δᵢ = maxmultideg[i] - exponents[i]
        if δᵢ ≥ δ
            exponents[i] += δ
            return true
        else
            exponents[i] = maxmultideg[i]
            δ -= δᵢ
        end
        i -= 1
    end
    return iszero(δ)
end

@inline function Base.getindex(iter::MonomialIterator{<:Vector}, i::Integer)
    ret = exponents_from_index!(iter.exponents, iter, i)
    @boundscheck ret || throw(BoundsError(iter, i))
    return iter.exponents
end
@inline function Base.getindex(iter::MonomialIterator{Nothing}, i::Integer)
    result = similar(iter.minmultideg)
    ret = exponents_from_index!(result, iter, i)
    @boundscheck ret || throw(BoundsError(iter, i))
    return result
end
Base.parent(iter::MonomialIterator) = iter

MultivariatePolynomials.mindegree(iter::MonomialIterator) = isempty(iter) ? 0 : Int(max(iter.mindeg, iter.Σminmultideg))
MultivariatePolynomials.maxdegree(iter::MonomialIterator) = isempty(iter) ? 0 : Int(min(iter.maxdeg, iter.Σmaxmultideg))
MultivariatePolynomials.extdegree(iter::MonomialIterator) =
    isempty(iter) ? (0, 0) : (Int(max(iter.mindeg, iter.Σminmultideg)), Int(min(iter.maxdeg, iter.Σmaxmultideg)))

"""
    RangedMonomialIterator(iter, start, length; copy)

Represents a subrange of a full [`MonomialIterator`](@ref), starting from index `start` and with maximal length `length`. The
iterator can alternatively be constructed by taking an index range from a monomial iterator - this will set `copy` to `true` -
or a view of an index range - this will set `copy` to `false`.
`copy` determines whether the underlying MonomialIterator is copied or not (which only plays a role if the iterator is set to
use the same vector of exponents for each iteration).
"""
struct RangedMonomialIterator{E,P<:Integer,M<:MonomialIterator{E,P}} <: AbstractMonomialIterator{E,P}
    iter::M
    start::Int
    length::Int

    function RangedMonomialIterator(iter::M, start::Integer, length::Integer; copy::Bool) where
        {E,P<:Integer,M<:MonomialIterator{E,P}}
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        new{E,P,M}(copy ? MonomialIterator(iter) : iter, start, min(length, max(0, Base.length(iter) - start +1)))
    end

    function RangedMonomialIterator(iter::RangedMonomialIterator{E,P,M}, start::Integer, length::Integer; copy::Bool) where
        {E,P<:Integer,M<:MonomialIterator{E,P}}
        # also an inner constructor, we don't need to calculate length(iter) again
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        len = min(length, max(0, iter.length - start +1))
        start = iter.start + start -1
        new{E,P,M}(copy ? MonomialIterator(iter.iter) : iter.iter, start, len)
    end

    RangedMonomialIterator(iter::RangedMonomialIterator{E,P,M}) where {E,P<:Integer,M<:MonomialIterator{E,P}} =
        new{E,P,M}(MonomialIterator(iter.iter), iter.start, iter.length)
end

function Base.iterate(iter::RangedMonomialIterator)
    iszero(iter.length) && return nothing
    if isnothing(iter.iter.exponents)
        exponents = similar(iter.iter.minmultideg)
    else
        exponents = iter.iter.exponents
    end
    exponents_from_index!(exponents, iter.iter, iter.start) || @assert(false)
    return isnothing(iter.iter.exponents) ? copy(exponents) : exponents,
        (typeof(iter.iter.mindeg)(sum(exponents)), exponents, 1)
end

function Base.iterate(iter::RangedMonomialIterator{<:Any,P}, state::Tuple{P,<:AbstractVector{P},Int}) where {P}
    state[3] ≥ iter.length && return nothing
    result = iterate(iter.iter, state[1:2])
    @assert(!isnothing(result)) # we calculated length appropriately
    return result[1], (result[2]..., state[3] + 1)
end

Base.length(iter::RangedMonomialIterator) = iter.length
function Iterators.drop(iter::RangedMonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, max(0, iter.length - n); copy)
end
function Iterators.drop(iter::MonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, typemax(Int); copy)
end

@inline function Base.getindex(iter::RangedMonomialIterator, i::Integer)
    @boundscheck 0 < i ≤ iter.length || throw(BoundsError(iter, i))
    @inbounds return iter.iter[i+iter.start-1]
end
Base.parent(iter::RangedMonomialIterator) = iter.iter

MultivariatePolynomials.mindegree(iter::RangedMonomialIterator) = iszero(iter.length) ? 0 : Int(sum(first(iter)))
MultivariatePolynomials.maxdegree(iter::RangedMonomialIterator) = iszero(iter.length) ? 0 : Int(sum(iter[length(iter)]))
MultivariatePolynomials.extdegree(iter::RangedMonomialIterator) =
    iszero(iter.length) ? (0, 0) : (Int(sum(first(iter))), Int(sum(iter[length(iter)])))