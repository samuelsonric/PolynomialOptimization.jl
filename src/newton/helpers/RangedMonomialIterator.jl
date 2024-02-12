struct RangedMonomialIterator{M<:MonomialIterator}
    iter::M
    start::Int
    length::Int

    function RangedMonomialIterator(iter::M, start::Integer, length::Integer; copy::Bool) where {M<:MonomialIterator}
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        new{M}(copy ? MonomialIterator(iter) : iter, start, min(length, max(0, Base.length(iter) - start +1)))
    end

    function RangedMonomialIterator(iter::RangedMonomialIterator, start::Integer, length::Integer; copy::Bool)
        # also an inner constructor, we don't need to calculate length(iter) again
        start ≤ 0 && throw(ArgumentError("Nonnegative start positions are not allowed"))
        length < 0 && throw(ArgumentError("Negative lengths are not allowed"))
        len = min(length, max(0, iter.length - start +1))
        start = iter.start + start -1
        new{typeof(iter.iter)}(copy ? MonomialIterator(iter.iter) : iter.iter, start, len)
    end
end

function Base.iterate(iter::RangedMonomialIterator{<:MonomialIterator{<:Graded}})
    iszero(iter.length) && return nothing
    if isnothing(iter.iter.powers)
        powers = similar(iter.iter.minmultideg)
    else
        powers = iter.iter.powers
    end
    exponents_from_index!(powers, iter.iter, iter.start) || @assert(false)
    return isnothing(iter.iter.powers) ? copy(powers) : powers, (typeof(iter.iter.mindeg)(sum(powers)), powers, 1)
end

function Base.iterate(iter::RangedMonomialIterator, state::Tuple{DI,<:AbstractVector{DI},Int}) where {DI}
    state[3] ≥ iter.length && return nothing
    result = iterate(iter.iter, state[1:2])
    @assert(!isnothing(result)) # we calculated length appropriately
    return result[1], (result[2]..., state[3] + 1)
end

Base.IteratorSize(::Type{<:RangedMonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:RangedMonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{RangedMonomialIterator{M}}) where {M<:MonomialIterator} = eltype(M)
Base.length(iter::RangedMonomialIterator) = iter.length
function Iterators.drop(iter::RangedMonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, max(0, iter.length - n); copy)
end
function Iterators.drop(iter::MonomialIterator, n::Integer; copy::Bool)
    n < 0 && throw(ArgumentError("Drop length must be nonnegative"))
    return RangedMonomialIterator(iter, n +1, typemax(Int); copy)
end

Base.getindex(iter::Union{<:MonomialIterator,<:RangedMonomialIterator}, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=true)
Base.view(iter::Union{<:MonomialIterator,<:RangedMonomialIterator}, range::AbstractUnitRange) =
    RangedMonomialIterator(iter, first(range), length(range), copy=false)