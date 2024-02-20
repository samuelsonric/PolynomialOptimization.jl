# Define a Missing replacement, as we want to provide some conveniences that probably would not be welcome in Base...
struct Absent end
const absent = Absent()
Base.IteratorSize(::Absent) = Base.HasLength()
Base.IteratorEltype(::Absent) = Base.HasEltype()
Base.eltype(::Absent) = Nothing
Base.length(::Absent) = 0
Base.iterate(::Absent) = nothing
Base.similar(::Absent) = absent
function Base.similar(::Absent, dims::Integer...)
    any(iszero, dims) || error("Cannot create absent with length")
    return absent
end
isabsent(x) = x === absent
Base.getindex(::Absent, ::Vararg{Any}) = absent
Base.view(::Absent, ::Vararg{Any}) = absent
SparseArrays.rowvals(::Absent) = absent
SparseArrays.nonzeros(::Absent) = absent
matrix_delete_end!(::Absent, ::Integer) = absent

const XorTX{X} = Union{<:X,<:Type{<:X}}
const XorA{X} = Union{<:X,Absent}

# Find the smallest possible type that can hold values between 0 and maxval
Base.@assume_effects :total function smallest_unsigned(maxval::Integer)
    maxval ≤ typemax(UInt8) && return UInt8
    maxval ≤ typemax(UInt16) && return UInt16
    maxval ≤ typemax(UInt32) && return UInt32
    return UInt64
end
smallest_unsigned(::Val{T}) where {T} = Val(typeintersect(T, Unsigned))
smallest_unsigned(::T) where {T} = typeintersect(T, Unsigned)

Base.Unsigned(U::Type{<:Unsigned}) = U
Base.Unsigned(S::Type{<:Signed}) = smallest_unsigned(typemax(S))

const isnonzero = ∘(!, iszero)