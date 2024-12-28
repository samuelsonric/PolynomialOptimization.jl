export Counters, addtocounter!, @counter_is_scalar, @counter_alias

"""
    Counters

Mutable struct that keeps track of the number of conic constraints/variables currently in use. Every possible symbol listed in
the documentation for [`extract_sos`](@ref) has an `Int` field of the same name.
"""
mutable struct Counters
    fix::Int
    nonnegative::Int
    quadratic::Int
    rotated_quadratic::Int
    psd::Int
    psd_complex::Int
    dd::Int
    dd_complex::Int
    lnorm::Int
    lnorm_complex::Int
    sdd::Int
    sdd_complex::Int

    Counters() = new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
end

"""
    addtocounter!(state, counters::Counters, ::Val{type}, dim::Integer) where {type} -> Int
    addtocounter!(state, counters::Counters, ::Val{type}, num::Integer, dim::Integer) where {type} -> UnitRange{Int}

Increments the internal information about how many constraints/variables of a certain type were already added. Usually, a
solver implementation does not have to overwrite the default implementation; but it might be useful to do so if some types are
for example counted in a single counter.
`dim` is the length of the conic constraint or variable, while `num` indicates that multiple such constraints or variables of
the same type are added.
For a list of possible symbols `type`, see the documentation for [`extract_sos`](@ref).

See also [`Counters`](@ref), [`@counter_is_scalar`](@ref).
"""
function addtocounter! end

@inline addtocounter!(state::AbstractSolver, counters::Counters, ::Val{type}, ::Integer) where {type} =
    setproperty!(counters, type, getproperty(counters, type) +1)

@inline function addtocounter!(state::AbstractSolver, counters::Counters, ::Val{type}, num::Integer, ::Integer) where {type}
    v = getproperty(counters, type)
    return v+1:setproperty!(counters, type, v + num)
end

"""
    @counter_is_scalar(::Type{<:AbstractSolver}, counter[, alias])

Defines the [`addtocounter!`](@ref) function in such a way that contraints of type `counter` are always counted as if they were
all one-dimensional, regardless of how they were added.
This function may be called at most once for each counter when defining a solver.
As a consequence, [`extract_sos`](@ref) will always have a `UnitRange{Int}` as `index` parameter for the `counter` type.
If the `alias` parameter is present, whenever the type `counter` is encountered, the counter of type `alias` is incremented
instead. Do not use [`@counter_alias`](@ref) together with this macro on the same `counter`.
"""
macro counter_is_scalar(S, counter::QuoteNode, alias::QuoteNode=counter)
    m = Base.parentmodule(addtocounter!)
    f = alias.value
    esc(quote
        @inline $m.addtocounter!(::$S, counters::$Counters, ::Val{$counter}, dim::Integer) =
            counters.$f+1:(counters.$f += dim)
        @inline $m.addtocounter!(::$S, counters::$Counters, ::Val{$counter}, num::Integer, dim::Integer) =
            counters.$f+1:(counters.$f += num * dim)
    end)
end

"""
    @counter_alias(::Type{<:AbstractSolver}, counter, alias)

Defines the [`addtocounter!`](@ref) function in such a way that contraints of type `counter` instead only affect the counter
`alias`. Do not use [`@counter_is_scalar`](@ref) together with this macro on the same `counter`.
"""
macro counter_alias(S, counter::QuoteNode, alias::QuoteNode)
    m = Base.parentmodule(addtocounter!)
    f = alias.value
    esc(quote
        @inline $m.addtocounter!(::$S, counters::$Counters, ::Val{$counter}, dim::Integer) =
            counters.$f += 1
        @inline $m.addtocounter!(::$S, counters::$Counters, ::Val{$counter}, num::Integer, dim::Integer) =
            counters.$f+1:(counters.$f += num)
    end)
end