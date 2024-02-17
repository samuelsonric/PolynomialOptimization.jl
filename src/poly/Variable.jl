export SimpleVariable, SimpleRealVariable, SimpleComplexVariable, variable_index

"""
    SimpleVariable{I} <: AbstractVariable

`SimpleVariable` is the basic type for any real- or complex-valued simple variables, which is identified by its index of type
`I` alone.

To construct a real-valued or complex-valued variable, see [`SimpleRealVariable`](@ref) and [`SimpleComplexVariable`](@ref).
"""
struct SimpleVariable{Nr,Nc,I<:Unsigned} <: AbstractVariable
    index::I

    function SimpleVariable{Nr,Nc}(index::Integer) where {Nr,Nc}
        0 ≤ Nr || throw(DomainError(Nr, "Invalid ring: Nr must be a nonnegative integer"))
        0 ≤ Nc || throw(DomainError(Nc, "Invalid ring: Nc must be nonnegative integer"))
        0 < index ≤ Nr + 2Nc || throw(DomainError(index, "Invalid index: must be between 1 and $(Nr + 2Nc)"))
        I = smallest_unsigned(Nr + 2Nc)
        new{Nr,Nc,I}(I(index))
    end
end

_get_nr(::XorTX{SimpleVariable{Nr}}) where {Nr} = Nr
_get_nr(::XorTX{SimpleVariable}) = Val{Any}
_get_nc(::XorTX{SimpleVariable{<:Any,Nc}}) where {Nc} = Nc
_get_nc(::XorTX{SimpleVariable}) = Val{Any}

# we want curly-initialization style, so use this singleton that can never be constructed
struct SimpleRealVariable{Nr,Nc}
    @doc """
        SimpleRealVariable{Nr,Nc}(index::Integer)

    Creates a new real-valued simple variable whose identity is determined by `index`. Real-valued variables with the same
    index are considered identical; however, they are different from complex-valued variables of the same index.
    A real variable will print as `xᵢ`, where the subscript is given by `index`.
    The variable is part of the polynomial ring with `Nr` real and `Nc` complex variables.

    See also [`SimpleVariable`](@ref), [`SimpleComplexVariable`](@ref).

    """
    function SimpleRealVariable{Nr,Nc}(index::Integer) where {Nr,Nc}
        0 < index ≤ Nr || throw(DomainError(index, "Invalid index: must be between 1 and $Nr"))
        return SimpleVariable{Nr,Nc}(index)
    end
end

struct SimpleComplexVariable{Nr,Nc}
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
        0 < index ≤ Nc || throw(DomainError(index, "Invalid index: must be between 1 and $Nc"))
        return SimpleVariable{Nr,Nc}(index + Nr + (isconj ? Nc : 0))
    end
end

function MultivariatePolynomials.name(v::SimpleVariable{Nr,Nc}) where {Nr,Nc}
    if v.index ≤ Nr
        return "x" * MultivariatePolynomials.unicode_subscript(v.index)
    else
        return "z" * MultivariatePolynomials.unicode_subscript(v.index ≤ Nr + Nc ? v.index - Nr : v.index - Nr - Nc)
    end
end

function MultivariatePolynomials.name_base_indices(v::SimpleVariable{Nr,Nc}) where {Nr,Nc}
    if v.index ≤ Nr
        return (:x, v.index)
    else
        return (:z, v.index ≤ Nr + Nc ? v.index - Nr : v.index - Nr - Nc)
    end
end

Base.:(==)(x::V, y::V) where {V<:SimpleVariable} = x.index == y.index
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

@inline function Base.getindex(sve::SimpleVariableExponents, idx::Integer)
    @boundscheck checkbounds(sve, idx)
    return UInt8(idx == sve.v.index)
end

function Base.collect(sve::SimpleVariableExponents{Nr,Nc}) where {Nr,Nc}
    result = zeros(UInt8, Nr + 2Nc)
    @inbounds result[sve.v.index] = 0x1
    return result
end

Base.iterate(sve::SimpleVariableExponents, state::Int=1) =
    state ≤ length(sve) ? (@inbounds(sve[state]), state +1) : nothing

MultivariatePolynomials._zip(t::Tuple, e::SimpleVariableExponents) = zip(t, e)
#endregion
MultivariatePolynomials.exponents(v::SimpleVariable) = SimpleVariableExponents(v)

Base.conj(v::SimpleVariable{<:Any,0}) = v
Base.conj(v::SimpleVariable{Nr,Nc}) where {Nr,Nc} =
    SimpleVariable{Nr,Nc}(v.index ≤ Nr ? v.index : (v.index ≤ Nr + Nc ? v.index + Nc : v.index - Nc))

Base.real(::SimpleVariable) = error("Not implemented")
Base.imag(::SimpleVariable) = error("Not implemented")

MultivariatePolynomials.isreal(::SimpleVariable{<:Any,0}) = true
MultivariatePolynomials.isreal(v::SimpleVariable{Nr}) where {Nr} = v.index ≤ Nr
MultivariatePolynomials.isconj(::SimpleVariable{<:Any,0}) = false
MultivariatePolynomials.isconj(v::SimpleVariable{Nr,Nc}) where {Nr,Nc} = v.index > Nr + Nc

MultivariatePolynomials.ordinary_variable(v::SimpleVariable{<:Any,0}) = v
MultivariatePolynomials.ordinary_variable(v::SimpleVariable{Nr,Nc}) where {Nr,Nc} =
    SimpleVariable{Nr,Nc}(v.index ≤ Nr + Nc ? v.index : v.index - Nc)