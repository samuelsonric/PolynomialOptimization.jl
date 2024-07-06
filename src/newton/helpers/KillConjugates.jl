struct KillConjugates{P}
    parent::P
end

Base.IteratorSize(::Type{KillConjugates{P}}) where {P} = Base.IteratorSize(P)
Base.IteratorEltype(::Type{KillConjugates{P}}) where {P} = Base.HasEltype()
Base.length(k::KillConjugates) = length(k.parent)
Base.eltype(::Type{KillConjugates{P}}) where {P} = eltype(P)
function Base.iterate(k::KillConjugates)
    result = iterate(k.parent)
    isnothing(result) && return nothing
    return result[1], (false, result[2])
end
function Base.iterate(k::KillConjugates, (use, state))
    result = iterate(k.parent, state)
    isnothing(result) && return nothing
    return (use ? result[1] : zero(result[1])), (!use, state)
end

Base.sum(k::KillConjugates) = sum(k.parent)