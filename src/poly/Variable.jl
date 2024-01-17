export SimpleVariable, SimpleRealVariable, SimpleComplexVariable, variable_index

"""
    SimpleVariable{I} <: AbstractVariable

`SimpleVariable` is the supertype for real- and complex-valued simple variables, which are identified by their index of type
`I` alone.

This is an abstract type with exactly two children: [`SimpleRealVariable`](@ref) and [`SimpleComplexVariable`](@ref).
"""
abstract type SimpleVariable{Nr,Nc} <: AbstractVariable end

_get_nr(::XorTX{SimpleVariable{Nr}}) where {Nr} = Nr
_get_nr(::XorTX{SimpleVariable}) = Val{Any}
_get_nc(::XorTX{SimpleVariable{<:Any,Nc}}) where {Nc} = Nc
_get_nc(::XorTX{SimpleVariable}) = Val{Any}

struct SimpleRealVariable{Nr,Nc,I<:Unsigned} <: SimpleVariable{Nr,Nc}
    index::I

    @doc """
        SimpleRealVariable{Nr,Nc}(index::Integer)

    Creates a new real-valued simple variable whose identity is determined by `index`. Real-valued variables with the same
    index are considered identical; however, they are different from complex-valued variables of the same index.
    A real variable will print as `xᵢ`, where the subscript is given by `index`.
    The variable is part of the polynomial ring with `Nr` real and `Nc` complex variables.

    See also [`SimpleVariable`](@ref), [`SimpleComplexVariable`](@ref).
    """
    function SimpleRealVariable{Nr,Nc}(index::Integer) where {Nr,Nc}
        0 ≤ Nr || throw(DomainError(Nr, "Invalid ring: Nr must be a nonnegative integer"))
        0 ≤ Nc || throw(DomainError(Nc, "Invalid ring: Nc must be nonnegative integer"))
        0 < index ≤ Nr || throw(DomainError(index, "Invalid index: must be between 1 and $Nr"))
        I = smallest_unsigned(Nr)
        new{Nr,Nc,I}(I(index))
    end
end

struct SimpleComplexVariable{Nr,Nc,I<:Unsigned} <: SimpleVariable{Nr,Nc}
    index::I
    isconj::Bool

    @doc """
        SimpleComplexVariable{Nr,Nc}(index::Integer, isconj::Bool=false)

    Creates a new complex-valued simple variable whose identity is determined by `index`, and which is a conjugate variable if
    `isconj` is set appropriately. Complex-valued variables with the same index and `isconj` state are considered identical;
    however, they are different from real-valued variables of the same index, even if they are not conjugate.
    A complex variable will print as `zᵢ` (if `isconj=false`) or `z̄ᵢ` (if `isconj=true`), where the subscript is given by
    `index`.
    The variable is part of the polynomial ring with `Nr` real and `Nc` complex variables.

    See also [`SimpleVariable`](@ref), [`SimpleRealVariable`](@ref).
    """
    function SimpleComplexVariable{Nr,Nc}(index::Integer, isconj::Bool=false) where {Nr,Nc}
        0 ≤ Nr || throw(DomainError(Nr, "Invalid ring: Nr must be a nonnegative integer"))
        0 ≤ Nc || throw(DomainError(Nc, "Invalid ring: Nc must be nonnegative integer"))
        0 < index ≤ Nc || throw(DomainError(index, "Invalid index: must be between 1 and $Nc"))
        I = smallest_unsigned(Nc)
        new{Nr,Nc,I}(I(index), isconj)
    end
end

MultivariatePolynomials.name(v::SimpleRealVariable) = "x" * MultivariatePolynomials.unicode_subscript(v.index)
MultivariatePolynomials.name(v::SimpleComplexVariable) = "z" * MultivariatePolynomials.unicode_subscript(v.index)

MultivariatePolynomials.name_base_indices(v::SimpleRealVariable) = (:x, v.index)
MultivariatePolynomials.name_base_indices(v::SimpleComplexVariable) = (:z, v.index)

Base.:(==)(x::SimpleRealVariable{Nr,Nc,I}, y::SimpleRealVariable{Nr,Nc,I}) where {Nr,Nc,I} = x.index == y.index
Base.:(==)(x::SimpleComplexVariable{Nr,Nc,I}, y::SimpleComplexVariable{Nr,Nc,I}) where {Nr,Nc,I} =
    x.index == y.index && x.isconj == y.isconj
Base.:(==)(::SimpleRealVariable{Nr,Nc}, ::SimpleComplexVariable{Nr,Nc}) where {Nr,Nc} = false
Base.:(==)(::SimpleComplexVariable{Nr,Nc}, ::SimpleRealVariable{Nr,Nc}) where {Nr,Nc} = false

# variables is defined later together with SimpleMonomialVector

MultivariatePolynomials.nvariables(::XorTX{SimpleVariable{Nr,Nc}}) where {Nr,Nc} = Nr + 2Nc

#region exponents iterator
struct SimpleVariableExponents{Nr,Nc,V<:SimpleVariable{Nr,Nc}} <: AbstractVector{UInt8}
    v::V
end

Base.IteratorSize(::Type{<:SimpleVariableExponents{Nr,Nc}}) where {Nr,Nc} = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleVariableExponents}) = Base.HasEltype()
Base.eltype(::Type{SimpleVariableExponents}) = UInt8
Base.length(::SimpleVariableExponents{Nr,Nc}) where {Nr,Nc} = Nr + 2Nc
Base.size(::SimpleVariableExponents{Nr,Nc}) where {Nr,Nc} = (Nr + 2Nc,)

@inline function Base.getindex(sve::SimpleVariableExponents{Nr,Nc,V}, idx::Integer) where {Nr,Nc,V<:SimpleVariable{Nr,Nc}}
    @boundscheck checkbounds(sve, idx)
    if V <: SimpleRealVariable
        return UInt8(idx == sve.v.index)
    elseif sve.v.isconj
        return UInt8(idx == sve.v.index + Nr + Nc)
    else
        return UInt8(idx == sve.v.index + Nr)
    end
end

function Base.collect(sve::SimpleVariableExponents{Nr,Nc,V}) where {Nr,Nc,V<:SimpleVariable{Nr,Nc}}
    result = zeros(UInt8, Nr + 2Nc)
    if V <: SimpleRealVariable
        @inbounds result[sve.v.index] = 0x1
    else
        @inbounds result[sve.v.index + Nr + (sve.v.isconj ? Nc : 0)] = 0x1
    end
    return result
end

Base.iterate(sve::SimpleVariableExponents, state::Int=1) =
    state ≤ length(sve) ? (@inbounds(sve[state]), state +1) : nothing

MultivariatePolynomials._zip(t::Tuple, e::SimpleVariableExponents) = zip(t, e)
#endregion
MultivariatePolynomials.exponents(v::SimpleVariable) = SimpleVariableExponents(v)

Base.conj(v::V) where {Nr,Nc,V<:SimpleComplexVariable{Nr,Nc}} = SimpleComplexVariable{Nr,Nc}(v.index, !v.isconj)

Base.real(::SimpleComplexVariable) = error("Not implemented")
Base.imag(::SimpleComplexVariable) = error("Not implemented")

MultivariatePolynomials.isreal(::SimpleComplexVariable) = false

MultivariatePolynomials.isconj(v::SimpleComplexVariable) = v.isconj

MultivariatePolynomials.ordinary_variable(v::V) where {Nr,Nc,V<:SimpleComplexVariable{Nr,Nc}} =
    SimpleComplexVariable{Nr,Nc}(v.index)