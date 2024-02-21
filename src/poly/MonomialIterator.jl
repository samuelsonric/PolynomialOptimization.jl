export AbstractMonomialIterator, MonomialIterator, RangedMonomialIterator, ownpowers, exponents_from_index_prepare

abstract type AbstractMonomialIterator{P,DI<:Integer} end

struct OwnPowers end
"""
    const ownpowers

This value instructs `MonomialIterator` to own its iterate. It will modify the same vector at every iteration.
"""
const ownpowers = OwnPowers()

"""
    MonomialIterator(mindeg, maxdeg, minmultideg, maxmultideg[, powers])

This is an advanced iterator that is able to iterate through all monomials with constraints specified not only by a minimum and
maximum total degree, but also by individual variable degrees.
If `powers` is set to [`ownpowers`](@ref) (or passed a `Vector{<:Integer}` of appropriate length), the iterator use the same
vector of powers whenever it is used, so it must not be used multiple times simultaneously. Additionally, during iteration, no
copy is created, so the vector must not be modified and accumulation e.g. by `collect` won't work.
Note that the powers that this iterator returns will be of the common integer type of `mindeg`, `maxdeg`, and the element types
of `minmultideg`, `maxmultideg` (and potentially `ownpowers`).

The monomials will be returned in an order that is compatible with `Graded{LexOrder}`.
"""
struct MonomialIterator{P,DI<:Integer} <: AbstractMonomialIterator{P,DI}
    n::Int
    mindeg::DI
    maxdeg::DI
    minmultideg::Vector{DI}
    maxmultideg::Vector{DI}
    powers::P
    Σminmultideg::UInt
    Σmaxmultideg::UInt

    function MonomialIterator(mindeg::DI, maxdeg::DI, minmultideg::Vector{DI}, maxmultideg::Vector{DI},
        powers::Union{Nothing,OwnPowers,<:AbstractVector{DI}}=nothing) where {DI<:Integer}
        (mindeg < 0 || mindeg > maxdeg) && throw(ArgumentError("Invalid degree specification"))
        n = length(minmultideg)
        (n != length(maxmultideg) ||
            any(minmax -> minmax[1] < 0 || minmax[1] > minmax[2], zip(minmultideg, maxmultideg))) &&
            throw(ArgumentError("Invalid multidegree specification"))
        Σminmultideg = sum(minmultideg, init=zero(DI))
        Σmaxmultideg = sum(maxmultideg, init=zero(DI))
        if powers isa OwnPowers
            return new{Vector{DI},DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, Vector{DI}(undef, n), Σminmultideg,
                Σmaxmultideg)
        elseif isnothing(powers)
            return new{Nothing,DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, nothing, Σminmultideg, Σmaxmultideg)
        elseif length(powers) != n
            throw(ArgumentError("Invalid length of ownpowers"))
        else
            return new{typeof(powers),DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, powers, Σminmultideg, Σmaxmultideg)
        end
    end

    function MonomialIterator(iter::MonomialIterator{P,DI}) where {P<:AbstractVector,DI<:Integer}
        simp = similar(iter.powers)
        new{typeof(simp),DI}(iter.n, iter.mindeg, iter.maxdeg, iter.minmultideg, iter.maxmultideg, simp, iter.Σminmultideg,
            iter.Σmaxmultideg)
    end

    MonomialIterator(iter::MonomialIterator{Nothing,DI}) where {DI<:Integer} = iter
end

function Base.iterate(iter::MonomialIterator{P}) where {P}
    minmultideg, Σminmultideg, Σmaxmultideg = iter.minmultideg, iter.Σminmultideg, iter.Σmaxmultideg
    Σminmultideg > iter.maxdeg && return nothing
    Σmaxmultideg < iter.mindeg && return nothing
    powers = P === Nothing ? copy(minmultideg) : copyto!(iter.powers, minmultideg)
    if iter.mindeg > Σminmultideg
        powers_increment_right(iter, powers, iter.mindeg - Σminmultideg, 1) || @assert(false)
        deg = iter.mindeg
    else
        deg = typeof(iter.mindeg)(Σminmultideg)
    end
    return P === Nothing ? copy(powers) : powers, (deg, powers)
end

function Base.iterate(iter::MonomialIterator{P}, state::Tuple{DI,<:AbstractVector{DI}}) where {P,DI}
    deg, powers = state
    deg ≤ iter.maxdeg || return nothing
    minmultideg, maxmultideg = iter.minmultideg, iter.maxmultideg
    @inbounds while true
        # This is not a loop at all, we only go through it once, but we need to be able to leave the block at multiple
        # positions. If we do it with @goto, as would be proper, Julia begins to box all our arrays.

        # find the next power that can be decreased
        found = false
        local i
        for outer i in iter.n:-1:1
            if powers[i] > minmultideg[i]
                found = true
                break
            end
        end
        found || break
        # we must increment the powers to the left by 1 in total
        found = false
        local j
        for outer j in i-1:-1:1
            if powers[j] < maxmultideg[j]
                found = true
                break
            end
        end
        found || break

        powers[j] += 1
        # this implies that we reset everything to the right of the increment to its minimum and then compensate for all
        # the reductions by increasing the powers again where possible
        δ = sum(k -> powers[k] - minmultideg[k], j+1:i, init=0) -1
        copyto!(powers, j +1, minmultideg, j +1, i - j)
        if powers_increment_right(iter, powers, δ, j +1)
            return P === Nothing ? copy(powers) : powers, (deg, powers)
        end
        break
    end
    # there's still hope: we can perhaps go to the next degree
    deg += one(DI)
    deg > iter.maxdeg && return nothing
    copyto!(powers, minmultideg)
    if powers_increment_right(iter, powers, deg - iter.Σminmultideg, 1)
        return P === Nothing ? copy(powers) : powers, (deg, powers)
    else
        return nothing
    end
end

Base.IteratorSize(::Type{<:MonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:MonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{<:MonomialIterator{<:Any,DI}}) where {DI<:Integer} = Vector{DI}
function _moniter_length(maxdeg, minmultideg, maxmultideg)
    # internal function without checks or quick path
    # ~ O(n*d^2)
    occurrences = zeros(Int, maxdeg +1)
    @inbounds fill!(@view(occurrences[minmultideg[1]+1:min(maxmultideg[1], maxdeg)+1]), 1)
    nextround = similar(occurrences)
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

function powers_increment_right(iter::MonomialIterator, powers, δ, from)
    @assert(δ ≥ 0 && from ≥ 0)
    maxmultideg = iter.maxmultideg
    i = iter.n
    @inbounds while δ > 0 && i ≥ from
        δᵢ = maxmultideg[i] - powers[i]
        if δᵢ ≥ δ
            powers[i] += δ
            return true
        else
            powers[i] = maxmultideg[i]
            δ -= δᵢ
        end
        i -= 1
    end
    return iszero(δ)
end

@inline function Base.getindex(iter::MonomialIterator{<:Vector}, i::Integer)
    ret = exponents_from_index!(iter.powers, iter, i)
    @boundscheck ret || throw(BoundsError(iter, i))
    return iter.powers
end
@inline function Base.getindex(iter::MonomialIterator{Nothing}, i::Integer)
    result = similar(iter.minmultideg)
    ret = exponents_from_index!(result, iter, i)
    @boundscheck ret || throw(BoundsError(iter, i))
    return result
end

MultivariatePolynomials.mindegree(iter::MonomialIterator) = isempty(iter) ? 0 : Int(max(iter.mindeg, iter.Σminmultideg))
MultivariatePolynomials.maxdegree(iter::MonomialIterator) = isempty(iter) ? 0 : Int(min(iter.maxdeg, iter.Σmaxmultideg))
MultivariatePolynomials.extdegree(iter::MonomialIterator) =
    isempty(iter) ? (0, 0) : (Int(max(iter.mindeg, iter.Σminmultideg)), Int(min(iter.maxdeg, iter.Σmaxmultideg)))

"""
    exponents_from_index!(powers::AbstractVector{<:Integer}, iter::AbstractMonomialIterator, index::Integer)

Constructs the vector of powers that is associated with the monomial index `index` in the given iterator `iter` and stores it
in `powers`. The method will return `false` if the index was out of bounds (with undefined state of `powers`), else it will
return `true`.


    exponents_from_index!(powers::AbstractVector{<:Integer}, iter::AbstractMonomialIterator, prepared, index::Integer)

A faster version of [`exponents_from_index!`](@ref) that is suitable if it has to be called often with the same iterator.
The `prepared` data must be constructed with [`exponents_from_index_prepare`](@ref)
"""
function exponents_from_index!(powers::AbstractVector{DI}, iter::MonomialIterator{<:Any,DI},
    index::Integer) where {DI}
    length(powers) == iter.n || throw(ArgumentError("powers and iter have different number of variables"))
    @inbounds return exponents_from_index!(powers, iter, exponents_from_index_prepare(iter), index)
end

@inline function exponents_from_index!(powers::AbstractVector{DI}, iter::MonomialIterator{<:Any,DI},
    ::Nothing, ::Integer) where {DI}
    @boundscheck length(powers) == iter.n || throw(ArgumentError("powers and iter have different number of variables"))
    return false
end

@inline function exponents_from_index!(powers::AbstractVector{DI}, iter::MonomialIterator{<:Any,DI}, ::Val{1},
    index::Integer) where {DI}
    @boundscheck length(powers) == iter.n || throw(ArgumentError("powers and iter have different number of variables"))
    powers[1] = max(iter.mindeg, iter.minmultideg[1]) + index -1
    powers[1] > min(iter.maxdeg, iter.maxmultideg[1]) && return false
    return powers
end

@inline function exponents_from_index!(powers::AbstractVector{DI}, iter::MonomialIterator{<:Any,DI},
    occurrences::Matrix{Int}, index::Integer) where {DI}
    # While this is a rather long function, given that this form is used for speed, inlining should probably be done
    @boundscheck length(powers) == iter.n || throw(ArgumentError("powers and iter have different number of variables"))
    # Now our occurrences matrix will fill the role of the binomial coefficient: it contains in each column j the number of
    # monomials if there were only the j right variables, while the row specifies the total degree.
    degree, maxdeg, minmultideg, maxmultideg = iter.mindeg, iter.maxdeg, iter.minmultideg, iter.maxmultideg
    while true
        @inbounds next = occurrences[degree+1, 1]
        if next ≥ index
            break
        else
            index -= next
            degree += 1
            degree > maxdeg && return false
        end
    end
    for i in 1:iter.n-1
        total = 0
        for degᵢ in minmultideg[i]:min(maxmultideg[i], maxdeg)
            @inbounds next = occurrences[degree-degᵢ+1, i+1]
            if total + next ≥ index
                degree -= degᵢ
                index -= total
                @inbounds powers[i] = degᵢ
                break
            else
                total += next
            end
        end
    end
    if 1 ≥ index && minmultideg[iter.n] ≤ degree ≤ maxmultideg[iter.n]
        @inbounds powers[iter.n] = degree
        return true
    else
        return false
    end
end

"""
    exponents_from_index_prepare(iter::AbstractMonomialIterator)

Prepares all the data necessary to quickly perform multiple calls to [`exponents_from_index!`](@ref) in a row.
"""
function exponents_from_index_prepare(iter::MonomialIterator{<:Any,DI}) where {DI}
    maxdeg = iter.maxdeg
    iter.Σminmultideg > maxdeg && return nothing
    iter.Σmaxmultideg < iter.mindeg && return nothing
    isone(iter.n) && return Val(1)
    minmultideg = iter.minmultideg
    maxmultideg = iter.maxmultideg
    j = iter.n
    occurrences = zeros(Int, maxdeg +1, j)
    # This is similar to _moniter_length, however, we will need to store all intermediate results and we'll need to iterate
    # backwards through the variables.
    @inbounds fill!(@view(occurrences[minmultideg[j]+1:min(maxmultideg[j], maxdeg)+1, j]), 1)
    for j in iter.n-1:-1:1
        lastround = @view(occurrences[:, j+1])
        nextround = @view(occurrences[:, j])
        for degᵢ in minmultideg[j]:min(maxmultideg[j], maxdeg)
            for (degₖ, occₖ) in zip(Iterators.countfrom(0), lastround)
                newdeg = degᵢ + degₖ
                newdeg > maxdeg && break
                @inbounds nextround[newdeg+1] += occₖ
            end
        end
    end
    return occurrences
end

"""
    RangedMonomialIterator(iter, start, length; copy)

Represents a subrange of a full [`MonomialIterator`](@ref), starting from index `start` and with maximal length `length`. The
iterator alternatively constructed by taking an index range from a monomial iterator - this will set `copy` to `true` - or a
view of an index range - this will set `copy` to `false`.
`copy` determines whether the underlying MonomialIterator is copied or not (which only plays a role if the iterator is set to
use the same vector of powers for each iteration).
"""
struct RangedMonomialIterator{P,DI<:Integer,M<:MonomialIterator{P,DI}} <: AbstractMonomialIterator{P,DI}
    iter::M
    start::Int
    length::Int

    function RangedMonomialIterator(iter::M, start::Integer, length::Integer; copy::Bool) where {P,DI<:Integer,M<:MonomialIterator{P,DI}}
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        new{P,DI,M}(copy ? MonomialIterator(iter) : iter, start, min(length, max(0, Base.length(iter) - start +1)))
    end

    function RangedMonomialIterator(iter::RangedMonomialIterator{P,DI,M}, start::Integer, length::Integer; copy::Bool) where
        {P,DI<:Integer,M<:MonomialIterator{P,DI}}
        # also an inner constructor, we don't need to calculate length(iter) again
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        len = min(length, max(0, iter.length - start +1))
        start = iter.start + start -1
        new{P,DI,M}(copy ? MonomialIterator(iter.iter) : iter.iter, start, len)
    end

    RangedMonomialIterator(iter::RangedMonomialIterator{P,DI,M}) where {P,DI<:Integer,M<:MonomialIterator{P,DI}} =
        new{P,DI,M}(MonomialIterator(iter.iter), iter.start, iter.length)
end

function Base.iterate(iter::RangedMonomialIterator)
    iszero(iter.length) && return nothing
    if isnothing(iter.iter.powers)
        powers = similar(iter.iter.minmultideg)
    else
        powers = iter.iter.powers
    end
    exponents_from_index!(powers, iter.iter, iter.start) || @assert(false)
    return isnothing(iter.iter.powers) ? copy(powers) : powers, (typeof(iter.iter.mindeg)(sum(powers)), powers, 1)
end

function Base.iterate(iter::RangedMonomialIterator{<:Any,DI},
    state::Tuple{DI,<:AbstractVector{DI},Int}) where {DI}
    state[3] ≥ iter.length && return nothing
    result = iterate(iter.iter, state[1:2])
    @assert(!isnothing(result)) # we calculated length appropriately
    return result[1], (result[2]..., state[3] + 1)
end

Base.IteratorSize(::Type{<:RangedMonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:RangedMonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{RangedMonomialIterator{<:Any,<:Integer,M}}) where {M<:MonomialIterator} = eltype(M)
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
Base.getindex(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=true)
Base.view(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=false)

MultivariatePolynomials.mindegree(iter::RangedMonomialIterator) = iszero(iter.length) ? 0 : Int(sum(first(iter)))
MultivariatePolynomials.maxdegree(iter::RangedMonomialIterator) = iszero(iter.length) ? 0 : Int(sum(iter[length(iter)]))
MultivariatePolynomials.extdegree(iter::RangedMonomialIterator) =
    iszero(iter.length) ? (0, 0) : (Int(sum(first(iter))), Int(sum(iter[length(iter)])))

Base.@propagate_inbounds exponents_from_index!(powers::AbstractVector{DI},
    iter::RangedMonomialIterator{<:Any,DI}, index::Integer) where {DI} =
    1 ≤ index ≤ iter.length && exponents_from_index!(powers, iter.iter, index + iter.start -1)
Base.@propagate_inbounds exponents_from_index!(powers::AbstractVector{DI},
    iter::RangedMonomialIterator{<:Any,DI}, prepared, index::Integer) where {DI} =
    1 ≤ index ≤ iter.length && exponents_from_index!(powers, iter.iter, prepared, index + iter.start -1)
exponents_from_index_prepare(iter::RangedMonomialIterator) = exponents_from_index_prepare(iter.iter)