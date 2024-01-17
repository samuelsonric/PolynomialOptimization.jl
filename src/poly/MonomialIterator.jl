export MonomialIterator

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
struct MonomialIterator{O<:AbstractMonomialOrdering,P,DI<:Integer}
    n::Int
    mindeg::DI
    maxdeg::DI
    minmultideg::Vector{DI}
    maxmultideg::Vector{DI}
    powers::P
    Σminmultideg::DI
    Σmaxmultideg::DI

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
        deg = Σminmultideg
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
Base.eltype(::Type{MonomialIterator{<:AbstractMonomialOrdering,<:Any,DI}}) where {DI<:Integer} = Vector{DI}
function Base.length(iter::MonomialIterator, ::Val{:detailed})
    # internal function without checks or quick path
    # ~ O(n*d^2)
    maxdeg = iter.maxdeg
    occurrences = zeros(Int, maxdeg +1)
    @inbounds for deg₁ in iter.minmultideg[1]:min(iter.maxmultideg[1], maxdeg)
        occurrences[deg₁+1] = 1
    end
    nextround = similar(occurrences)
    for (minᵢ, maxᵢ) in Iterators.drop(zip(iter.minmultideg, iter.maxmultideg), 1)
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
    @inbounds return sum(@view(@inline(length(iter, Val(:detailed)))[iter.mindeg+1:end]), init=0)
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