export SimpleMonomial

"""
    SimpleMonomial{Nr,Nc,P<:Unsigned,V<:AbstractVector{P}} <: AbstractMonomial

`SimpleMonomial` represents a monomial. In order to be used together in operations, the number of real-valued variables `Nr`
and the number of complex-valued variables `Nc` are fixed in the type. The possible exponents in a SimpleMonomial are of type
`P`, which should usually be chosen in a space-optimal way, i.e., typically, it will be `UInt8`.
Note that the total degree of a `SimpleMonomial` will always be an `Int` (as required by the `compare` functions of
`MultivariatePolynomials`), but variable-specific degrees are of type `P`.
"""
struct SimpleMonomial{Nr,Nc,P<:Unsigned,V<:AbstractVector{P},Vr<:XorA{V},Vc<:XorA{V}} <: AbstractMonomial
    exponents_real::Vr
    exponents_complex::Vc
    exponents_conj::Vc

    # internal functions, don't use
    SimpleMonomial{Nr,0,P,V}(exponents_real::V, ::Absent, ::Absent) where {Nr,P<:Unsigned,V<:AbstractVector{P}} =
        new{Nr,0,P,V,V,Absent}(exponents_real, absent, absent)

    SimpleMonomial{0,Nc,P,V}(::Absent, exponents_complex::V, exponents_conj::V) where {Nc,P<:Unsigned,V<:AbstractVector{P}} =
        new{0,Nc,P,V,Absent,V}(absent, exponents_complex, exponents_conj)

    SimpleMonomial{Nr,Nc,P,V}(exponents_real::V, exponents_complex::V, exponents_conj::V) where
        {Nr,Nc,P<:Unsigned,V<:AbstractVector{P}} =
        new{Nr,Nc,P,V,V,V}(exponents_real, exponents_complex, exponents_conj)
end

"""
    SimpleMonomial{Nr,0}(exponents_real::AbstractVector{<:Integer})
    SimpleMonomial{0,Nc}(exponents_complex::AbstractVector{<:Integer}, exponents_conj::AbstractVector{<:Integer})
    SimpleMonomial{Nr,Nc}(exponents_real::AbstractVector{<:Integer}, exponents_complex::AbstractVector{<:Integer},
        exponents_conj::AbstractVector{<:Integer})

Creates a monomial. The vectors will be owned by the monomial afterwards.
"""
function SimpleMonomial{Nr,0}(exponents_real::AbstractVector{<:Integer}) where {Nr}
    length(exponents_real) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(length(exponents_real))"))
    P = Unsigned(eltype(exponents_real))
    exps = convert(AbstractVector{P}, exponents_real)
    return SimpleMonomial{Nr,0,P,typeof(exps)}(exps, absent, absent)
end

function SimpleMonomial{0,Nc}(exponents_complex::AbstractVector{<:Integer}, exponents_conj::AbstractVector{<:Integer}) where {Nc}
    length(exponents_complex) == length(exponents_conj) ||
        throw(ArgumentError("Complex and conjugate exponent lengths are different"))
    length(exponents_complex) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(length(exponents_complex))"))
    P = promote_type(Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    V1 = Base.promote_op(convert, Type{AbstractVector{P}}, typeof(exponents_complex))
    V2 = Base.promote_op(convert, Type{AbstractVector{P}}, typeof(exponents_conj))
    V = promote_type(V1, V2)
    return SimpleMonomial{0,Nc,P,V}(absent, convert(V, exponents_complex), convert(V, exponents_conj))
end

function SimpleMonomial{Nr,Nc}(exponents_real::AbstractVector{<:Integer}, exponents_complex::AbstractVector{<:Integer},
    exponents_conj::AbstractVector{<:Integer}) where {Nr,Nc}
    length(exponents_real) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(length(exponents_real))"))
    length(exponents_complex) == length(exponents_conj) ||
        throw(ArgumentError("Complex and conjugate exponent lengths are different"))
    length(exponents_complex) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(length(exponents_complex))"))
    P = promote_type(Unsigned(eltype(exponents_real)), Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    V1 = Base.promote_op(convert, Type{AbstractVector{P}}, typeof(exponents_real))
    V2 = Base.promote_op(convert, Type{AbstractVector{P}}, typeof(exponents_complex))
    V3 = Base.promote_op(convert, Type{AbstractVector{P}}, typeof(exponents_conj))
    V = promote_type(V1, V2, V3)
    return SimpleMonomial{Nr,Nc,P,V}(convert(V, exponents_real), convert(V, exponents_complex), convert(V, exponents_conj))
end

const SimpleRealMonomial{Nr,P<:Unsigned,V<:AbstractVector{P}} = SimpleMonomial{Nr,0,P,V,V,Absent}
const SimpleComplexMonomial{Nc,P<:Unsigned,V<:AbstractVector{P}} = SimpleMonomial{0,Nc,P,V,Absent,V}
const SimpleMixedMonomial{Nr,Nc,P<:Unsigned,V<:AbstractVector{P}} = SimpleMonomial{Nr,Nc,P,V,V,V}
const SimpleDenseMonomial{Nr,Nc,P<:Unsigned} = SimpleMonomial{Nr,Nc,P,<:DenseVector}
const SimpleSparseMonomial{Nr,Nc,P<:Unsigned} = SimpleMonomial{Nr,Nc,P,<:AbstractSparseVector}
const SimpleRealDenseMonomial{Nr,P<:Unsigned} = SimpleRealMonomial{Nr,P,<:DenseVector}
const SimpleComplexDenseMonomial{Nc,P<:Unsigned} = SimpleComplexMonomial{Nc,P,<:DenseVector}
const SimpleRealSparseMonomial{Nr,P<:Unsigned} = SimpleRealMonomial{Nr,P,<:AbstractSparseVector}
const SimpleComplexSparseMonomial{Nc,P<:Unsigned} = SimpleComplexMonomial{Nc,P,<:AbstractSparseVector}
const SimpleDenseMonomialOrView{Nr,Nc,P<:Unsigned} = SimpleMonomial{Nr,Nc,P,<:Union{<:DenseVector{P},<:(SubArray{P,1,<:DenseArray{P}})}}
const SimpleSparseMonomialOrView{Nr,Nc,P<:Unsigned} = SimpleMonomial{Nr,Nc,P,<:Union{<:AbstractSparseVector{P},<:(SubArray{P,1,<:AbstractSparseArray{P}})}}
const SimpleRealDenseMonomialOrView{Nr,P<:Unsigned} = SimpleRealMonomial{Nr,P,<:Union{<:DenseVector{P},<:(SubArray{P,1,<:DenseArray{P}})}}
const SimpleComplexDenseMonomialOrView{Nc,P<:Unsigned} = SimpleComplexMonomial{Nc,P,<:Union{<:DenseVector{P},<:(SubArray{P,1,<:DenseArray{P}})}}
const SimpleRealSparseMonomialOrView{Nr,P<:Unsigned} = SimpleRealMonomial{Nr,P,<:Union{<:AbstractSparseVector{P},<:(SubArray{P,1,<:AbstractSparseArray{P}})}}
const SimpleComplexSparseMonomialOrView{Nc,P<:Unsigned} = SimpleComplexMonomial{Nc,P,<:Union{<:AbstractSparseVector{P},<:(SubArray{P,1,<:AbstractSparseArray{P}})}}

_get_nr(::XorTX{SimpleMonomial{Nr}}) where {Nr} = Nr
_get_nr(::XorTX{SimpleMonomial}) = Val(Any)
_get_nc(::XorTX{SimpleMonomial{<:Any,Nc}}) where {Nc} = Nc
_get_nc(::XorTX{SimpleMonomial}) = Val(Any)
_get_p(::XorTX{SimpleMonomial{<:Any,<:Any,P}}) where {P<:Unsigned} = P
_get_p(::XorTX{SimpleMonomial}) = Val(Unsigned)
_get_v(::XorTX{SimpleMonomial{<:Any,<:Any,P,V}}) where {P<:Unsigned,V<:AbstractVector{P}} = V
_get_v(::XorTX{SimpleMonomial{<:Any,<:Any,P}}) where {P<:Unsigned} = Val(AbstractVector{P})
_get_v(::XorTX{SimpleMonomial}) = Val(AbstractVector)

MultivariatePolynomials.monomial(m::SimpleMonomial) = m

# variables is defined later together with SimpleMonomialVector

MultivariatePolynomials.nvariables(::XorTX{SimpleMonomial{Nr,Nc}}) where {Nr,Nc} = Nr + 2Nc

function MultivariatePolynomials.convert_constant(T::Type{<:SimpleMonomial}, α)
    isone(α) || error("Cannot convert `$α` to a `SimpleMonomial` as it is not one")
    return constant_monomial(T)
end

# We must convert degree to Ints. MP will do subtractions in the comparisons, so an Unsigned will fail.
MultivariatePolynomials.degree(m::SimpleMonomial) =
    Int(sum(m.exponents_real, init=0) + sum(m.exponents_complex, init=0) + sum(m.exponents_conj, init=0))
MultivariatePolynomials.degree(m::SimpleMonomial{Nr,Nc}, v::SimpleVariable{Nr,Nc}) where {Nr,Nc} =
    @inbounds v.index ≤ Nr ? m.exponents_real[v.index] : (v.index ≤ Nr + Nc ? m.exponents_complex[v.index-Nr] :
                                                          m.exponents_conj[v.index-Nr-Nc])

# These do not correspond to how they are defined in MP. But this definition makes more sense (and is what is needed).
MultivariatePolynomials.degree_complex(m::SimpleDenseMonomial) =
    Int(sum(m.exponents_real, init=0) + max(sum(m.exponents_complex, init=0), sum(m.exponents_conj, init=0)))
MultivariatePolynomials.degree_complex(m::SimpleDenseMonomial{Nr,Nc}, v::SimpleVariable{Nr,Nc}) where {Nr,Nc} =
    @inbounds v.index ≤ Nr ? m.exponents_real[v.index] :
        (v.index ≤ Nr + Nc ? max(m.exponents_complex[v.index-Nr], m.exponents_conj[v.index-Nr]) :
                             max(m.exponents_complex[v.index-Nr-Nc], m.exponents_conj[v.index-Nr-Nc]))

MultivariatePolynomials.halfdegree(m::SimpleDenseMonomial) = Int(div(sum(m.exponents_real, init=0), 2, RoundUp) +
    max(sum(m.exponents_complex, init=0), sum(m.exponents_conj, init=0)))

#region exponents iterator
struct SimpleMonomialExponents{Nr,Nc,P<:Unsigned,M<:SimpleMonomial{Nr,Nc,P}} <: AbstractVector{P}
    m::M
end

Base.IteratorSize(::Type{<:SimpleMonomialExponents{Nr,Nc}}) where {Nr,Nc} = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialExponents{<:Any,<:Any,P}}) where {P<:Unsigned} = Base.HasEltype()
Base.eltype(::Type{SimpleMonomialExponents{<:Any,<:Any,P}}) where {P<:Unsigned} = P
Base.length(::SimpleMonomialExponents{Nr,Nc}) where {Nr,Nc} = Nr + 2Nc
Base.size(::SimpleMonomialExponents{Nr,Nc}) where {Nr,Nc} = (Nr + 2Nc,)

@inline function Base.getindex(sme::SimpleMonomialExponents{Nr,Nc}, idx::Integer) where {Nr,Nc}
    @boundscheck checkbounds(sme, idx)
    idx ≤ Nr && @inbounds return sme.m.exponents_real[idx]
    idx ≤ Nr + Nc && @inbounds return sme.m.exponents_complex[idx-Nr]
    @inbounds return sme.m.exponents_conj[idx-Nr-Nc]
end

Base.collect(sme::SimpleMonomialExponents) =
    [sme.m.exponents_real; sme.m.exponents_complex; sme.m.exponents_conj]

Base.iterate(sme::SimpleMonomialExponents, state::Int=1) =
    state ≤ length(sme) ? (@inbounds(sme[state]), state +1) : nothing

function Base.iterate(sme::SimpleMonomialExponents{Nr,Nc,P,<:SimpleSparseMonomial}, state::NTuple{2,Int}=(1,1)) where {Nr,Nc,P<:Unsigned}
    # we provide an extra sparse implementation, as there's no need to search for the next index again and again
    outeridx, inneridx = state
    if outeridx ≤ Nr
        let idxs=rowvals(sme.m.exponents_real), vals=nonzeros(sme.m.exponents_real)
            @inbounds inneridx ≤ lastindex(idxs) && outeridx == idxs[inneridx] &&
                return vals[inneridx], (outeridx +1, outeridx == Nr ? 1 : inneridx +1)
            return zero(P), (outeridx +1, outeridx == Nr ? 1 : inneridx)
        end
    end
    if outeridx ≤ Nr + Nc
        let idxs=rowvals(sme.m.exponents_complex), vals=nonzeros(sme.m.exponents_complex)
            @inbounds inneridx ≤ lastindex(idxs) && outeridx - Nr == idxs[inneridx] &&
                return vals[inneridx], (outeridx +1, outeridx == Nr + Nc ? 1 : inneridx +1)
            return zero(P), (outeridx +1, outeridx == Nr + Nc ? 1 : inneridx)
        end
    end
    if outeridx ≤ Nr + 2Nc
        let idxs=rowvals(sme.m.exponents_conj), vals=nonzeros(sme.m.exponents_conj)
            @inbounds inneridx ≤ lastindex(idxs) && outeridx - Nr - Nc == idxs[inneridx] &&
                return vals[inneridx], (outeridx +1, outeridx == Nr + 2Nc ? 1 : inneridx +1)
        end
        return zero(P), (outeridx +1, outeridx == Nr + 2Nc ? 1 : inneridx)
    end
    return nothing
end

MultivariatePolynomials._zip(t::Tuple, e::SimpleMonomialExponents) = zip(t, e)
#endregion
MultivariatePolynomials.exponents(m::SimpleMonomial) = SimpleMonomialExponents(m)

# implement an iteration method although there is the exponents function - this one gives a (SimpleVariable, power) tuple and
# skips over zero powers
Base.IteratorSize(::Type{<:SimpleMonomial}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomial}) = Base.HasEltype()
Base.eltype(::Type{<:SimpleMonomial{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned} =
    Tuple{SimpleVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)},P}
function Base.iterate(m::SimpleMonomial{Nr,Nc}, state::Int=0) where {Nr,Nc}
    @inbounds for i in state+1:Nr
        iszero(m.exponents_real[i]) || return (SimpleVariable{Nr,Nc}(i), m.exponents_real[i]), i
    end
    @inbounds for i in max(1, state - Nr +1):Nc
        iszero(m.exponents_complex[i]) || return (SimpleVariable{Nr,Nc}(i + Nr), m.exponents_complex[i]), i + Nr
    end
    @inbounds for i in max(1, state - Nr - Nc +1):Nc
        iszero(m.exponents_conj[i]) || return (SimpleVariable{Nr,Nc}(i + Nr + Nc), m.exponents_conj[i]), i + Nr + Nc
    end
    return nothing
end
function Base.iterate(m::SimpleSparseMonomial{Nr,Nc}, state::Int=0) where {Nr,Nc}
    local δ, δ₂
    let idxs=rowvals(m.exponents_real), vals=nonzeros(m.exponents_real)
        δ = length(idxs)
        @inbounds for i in state+1:δ
            iszero(vals[i]) || return (SimpleVariable{Nr,Nc}(idxs[i]), vals[i]), i
        end
    end
    let idxs=rowvals(m.exponents_complex), vals=nonzeros(m.exponents_complex)
        δ₂ = δ + length(idxs)
        @inbounds for i in state-δ+1:length(idxs)
            iszero(vals[i]) || return (SimpleVariable{Nr,Nc}(idxs[i] + Nr), vals[i]), i + δ
        end
    end
    let idxs=rowvals(m.exponents_conj), vals=nonzeros(m.exponents_conj)
        @inbounds for i in state-δ₂+1:length(idxs)
            iszero(idxs[i]) || return (SimpleVariable{Nr,Nc}(idxs[i] + Nr + Nc), vals[i]), i + δ₂
        end
    end
    return nothing
end
Base.length(m::SimpleMonomial) = count(isnonzero, m.exponents_real, init=0) +
    count(isnonzero, m.exponents_complex, init=0) + count(m.exponents_conj, init=0)

MultivariatePolynomials.isconstant(m::SimpleMonomial) =
    all(iszero, m.exponents_real) && all(iszero, m.exponents_complex) && all(iszero, m.exponents_conj)

SparseArrays.indtype(::Type{<:AbstractSparseArray{<:Any,Ti}}) where {Ti} = Ti # not defined for types
function MultivariatePolynomials.constant_monomial(::XorTX{SimpleMonomial{Nr,Nc,P,V}}) where {Nr,Nc,P<:Unsigned,V<:AbstractVector{P}}
    # we cannot use similar (even for the concrete case) - V might be a view-backed vector, which cannot be recreated.
    if V <: AbstractSparseVector
        vals = P[]
    end
    if iszero(Nr)
        re_v = absent
    elseif V <: AbstractSparseVector
        re_v = FixedSparseVector(Nr, SparseArrays.indtype(V)[], vals)
    else
        re_v = zeros(P, Nr)
    end
    if iszero(Nc)
        im_v = absent
        T = typeof(re_v)
    elseif V <: AbstractSparseVector
        im_v = FixedSparseVector(Nc, SparseArrays.indtype(V)[], vals)
        T = typeof(im_v)
    else
        im_v = zeros(P, Nc)
        T = typeof(im_v)
    end
    return SimpleMonomial{Nr,Nc,P,T}(re_v, im_v, im_v)
end

# also provide a version for the unknown type which defaults to the sparse case. Note that here, we don't really bother with
# incomplete specifications of SimpleMonomial, where maybe one of the vector types is missing.
function MultivariatePolynomials.constant_monomial(::Type{SimpleMonomial{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned}
    vals = P[]
    su = promote_type(smallest_unsigned(Nr), smallest_unsigned(Nc))
    idxs = su[]
    if iszero(Nr)
        re_v = absent
    else
        re_v = FixedSparseVector(Nr, idxs, vals)
    end
    if iszero(Nc)
        im_v = absent
    else
        im_v = FixedSparseVector(Nc, idxs, vals)
    end
    return SimpleMonomial{Nr,Nc,P,FixedSparseVector{P,su}}(re_v, im_v, im_v)
end
MultivariatePolynomials.constant_monomial(::Type{SimpleMonomial{Nr,Nc}}) where {Nr,Nc} =
    constant_monomial(SimpleMonomial{Nr,Nc,UInt8})

Base.conj(m::SimpleMonomial{Nr,Nc,P,V}) where {Nr,Nc,P<:Unsigned,V<:AbstractVector{P}} =
    isreal(m) ? m : SimpleMonomial{Nr,Nc,P,V}(m.exponents_real, m.exponents_conj, m.exponents_complex)
# ^ we do this additional check so that for monomials found to be real, Julia knows that conj is an identity and can propagate
# the effects.
Base.@assume_effects :consistent Base.isreal(m::SimpleMonomial) = m.exponents_complex == m.exponents_conj
# ^ while these are in principle mutable vectors, we explictly demand Simplexxx to be immutable, so let's put this knowledge
# to use.

# we hash the monomial to its index, assuming that usually we are interested in a list of monomials of the same type, so no
# collisions should occur in this way.
Base.hash(m::SimpleMonomial, h::UInt) = hash(monomial_index(m), h)

effective_variables_in(m::SimpleMonomial, in) = all(vp -> ordinary_variable(vp[1]) ∈ in, m)