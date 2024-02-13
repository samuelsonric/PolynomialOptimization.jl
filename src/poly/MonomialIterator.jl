export AbstractMonomialIterator, MonomialIterator, RangedMonomialIterator

abstract type AbstractMonomialIterator{O<:AbstractMonomialOrdering,P,DI<:Integer} end

"""
    MonomialIterator{O}(mindeg, maxdeg, minmultideg, maxmultideg, ownpowers=false)

This is an advanced iterator that is able to iterate through all monomials with constraints specified not only by a minimum and
maximum total degree, but also by individual variable degrees. `ownpowers` can be set to `true` (or be passed a
`Vector{<:Integer}` of appropriate length), which will make the iterator use the same vector of powers whenever it is used, so
it must not be used multiple times simultaneously. Additionally, during iteration, no copy is created, so the vector must not
be modified and accumulation e.g. by `collect` won't work.
Note that the powers that this iterator returns will be of the common integer type of `mindeg`, `maxdeg`, and the element types
of `minmultideg`, `maxmultideg` (and potentially `ownpowers`).

Currently, the only supported monomial ordering `O` is `Graded{LexOrder}`.
"""
struct MonomialIterator{O<:AbstractMonomialOrdering,P,DI<:Integer} <: AbstractMonomialIterator{O,P,DI}
    n::Int
    mindeg::DI
    maxdeg::DI
    minmultideg::Vector{DI}
    maxmultideg::Vector{DI}
    powers::P
    Σminmultideg::UInt
    Σmaxmultideg::UInt

    function MonomialIterator{O}(mindeg::DI, maxdeg::DI, minmultideg::Vector{DI}, maxmultideg::Vector{DI},
        ownpowers::Union{Bool,<:AbstractVector{DI}}=false) where {O,DI<:Integer}
        (mindeg < 0 || mindeg > maxdeg) && throw(ArgumentError("Invalid degree specification"))
        n = length(minmultideg)
        (n != length(maxmultideg) ||
            any(minmax -> minmax[1] < 0 || minmax[1] > minmax[2], zip(minmultideg, maxmultideg))) &&
            throw(ArgumentError("Invalid multidegree specification"))
        Σminmultideg = sum(minmultideg, init=zero(DI))
        Σmaxmultideg = sum(maxmultideg, init=zero(DI))
        if ownpowers === true
            return new{O,Vector{DI},DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, Vector{DI}(undef, n), Σminmultideg,
                Σmaxmultideg)
        elseif ownpowers === false
            return new{O,Nothing,DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, nothing, Σminmultideg, Σmaxmultideg)
        elseif length(ownpowers) != n
            throw(ArgumentError("Invalid length of ownpowers"))
        else
            return new{O,typeof(ownpowers),DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, ownpowers, Σminmultideg,
                Σmaxmultideg)
        end
    end

    function MonomialIterator(iter::MonomialIterator{O,P,DI}) where {O<:AbstractMonomialOrdering,P<:AbstractVector,DI<:Integer}
        simp = similar(iter.powers)
        new{O,typeof(simp),DI}(iter.n, iter.mindeg, iter.maxdeg, iter.minmultideg, iter.maxmultideg, simp, iter.Σminmultideg,
            iter.Σmaxmultideg)
    end

    MonomialIterator(iter::MonomialIterator{O,Nothing,DI}) where {O<:AbstractMonomialOrdering,DI<:Integer} = iter
end

function Base.iterate(iter::MonomialIterator{Graded{LexOrder},P}) where {P}
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

function Base.iterate(iter::MonomialIterator{Graded{LexOrder},P}, state::Tuple{DI,<:AbstractVector{DI}}) where {P,DI}
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
        δ = sum(k -> powers[k] - minmultideg[k], j+1:i, init=zero(DI)) -1
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
Base.eltype(::Type{<:MonomialIterator{<:AbstractMonomialOrdering,<:Any,DI}}) where {DI<:Integer} = Vector{DI}
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
moniter_state(powers::AbstractVector{DI}) where {DI<:Integer} = (DI(sum(powers, init=zero(DI))), powers)

function powers_increment_right(iter::MonomialIterator{Graded{LexOrder}}, powers, δ, from)
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

"""
    exponents_from_index!(powers::AbstractVector{<:Integer}, iter::AbstractMonomialIterator, index::Integer)

Constructs the vector of powers that is associated with the monomial index `index` in the given iterator `iter` and stores it
in `powers`. The method will return `false` if the index was out of bounds (with undefined state of `powers`), else it will
return `true`.
"""
function exponents_from_index!(powers::AbstractVector{DI}, iter::MonomialIterator{Graded{LexOrder},<:Any,DI}, index::Integer) where {DI}
    length(powers) == iter.n || throw(ArgumentError("powers and iter have different number of variables"))
    index < 1 && return false
    maxdeg = iter.maxdeg
    iter.Σminmultideg > maxdeg && return false
    iter.Σmaxmultideg < iter.mindeg && return false
    minmultideg = iter.minmultideg
    maxmultideg = iter.maxmultideg
    @inbounds if isone(iter.n)
        powers[1] = max(iter.mindeg, iter.minmultideg[1]) + index -1
        powres[1] > min(maxdeg, iter.maxmultideg[1]) && return false
        return powers
    end
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
    # Now our occurrences matrix will fill the role of the binomial coefficient: it contains in each column j the number of
    # monomials if there were only the j right variables, while the row specifies the total degree.
    degree = iter.mindeg
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
    RangedMonomialIterator(iter, start, length; copy)

Represents a subrange of a full [`MonomialIterator`](@ref), starting from index `start` and with maximal length `length`. The
iterator alternatively constructed by taking an index range from a monomial iterator - this will set `copy` to `true` - or a
view of an index range - this will set `copy` to `false`.
`copy` determines whether the underlying MonomialIterator is copied or not (which only plays a role if the iterator is set to
use the same vector of powers for each iteration).
"""
struct RangedMonomialIterator{O<:AbstractMonomialOrdering,P,DI<:Integer,M<:MonomialIterator{O,P,DI}} <: AbstractMonomialIterator{O,P,DI}
    iter::M
    start::Int
    length::Int

    function RangedMonomialIterator(iter::M, start::Integer, length::Integer; copy::Bool) where
        {O<:AbstractMonomialOrdering,P,DI<:Integer,M<:MonomialIterator{O,P,DI}}
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        new{O,P,DI,M}(copy ? MonomialIterator(iter) : iter, start, min(length, max(0, Base.length(iter) - start +1)))
    end

    function RangedMonomialIterator(iter::R, start::Integer, length::Integer; copy::Bool) where
        {O<:AbstractMonomialOrdering,P,DI<:Integer,R<:RangedMonomialIterator{O,P,DI}}
        # also an inner constructor, we don't need to calculate length(iter) again
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        len = min(length, max(0, iter.length - start +1))
        start = iter.start + start -1
        new{O,P,DI,R}(copy ? MonomialIterator(iter.iter) : iter.iter, start, len)
    end
end

function Base.iterate(iter::RangedMonomialIterator{<:Graded})
    iszero(iter.length) && return nothing
    if isnothing(iter.iter.powers)
        powers = similar(iter.iter.minmultideg)
    else
        powers = iter.iter.powers
    end
    exponents_from_index!(powers, iter.iter, iter.start) || @assert(false)
    return isnothing(iter.iter.powers) ? copy(powers) : powers, (typeof(iter.iter.mindeg)(sum(powers)), powers, 1)
end

function Base.iterate(iter::RangedMonomialIterator{<:AbstractMonomialOrdering,<:Any,DI},
    state::Tuple{DI,<:AbstractVector{DI},Int}) where {DI}
    state[3] ≥ iter.length && return nothing
    result = iterate(iter.iter, state[1:2])
    @assert(!isnothing(result)) # we calculated length appropriately
    return result[1], (result[2]..., state[3] + 1)
end

Base.IteratorSize(::Type{<:RangedMonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:RangedMonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{RangedMonomialIterator{<:AbstractMonomialOrdering,<:Any,<:Integer,M}}) where {M<:MonomialIterator} =
    eltype(M)
Base.length(iter::RangedMonomialIterator) = iter.length
function Iterators.drop(iter::RangedMonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, max(0, iter.length - n); copy)
end
function Iterators.drop(iter::MonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, typemax(Int); copy)
end

Base.getindex(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=true)
Base.view(iter::AbstractMonomialIterator, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=false)

exponents_from_index!(powers::AbstractVector{DI}, iter::RangedMonomialIterator{Graded{LexOrder},<:Any,DI}, index::Integer) where {DI} =
    1 ≤ index ≤ iter.length && exponents_from_index!(powers, iter.iter, index + iter.start -1)