export SimplePolynomial

struct SimplePolynomial{C,Nr,Nc,P<:Unsigned,M<:SimpleMonomialVector{Nr,Nc,P}} <: AbstractPolynomial{C}
    coeffs::Vector{C}
    monomials::M

    """
        SimplePolynomial(coeffs::Vector{C}, monomials::SimpleMonomialVector)

    Creates a `SimplePolynomial` with well-defined coefficients and monomials, which must have the same length.
    """
    function SimplePolynomial(coeffs::Vector{C}, monomials::M) where {C,Nr,Nc,P<:Unsigned,M<:SimpleMonomialVector{Nr,Nc,P}}
        length(coeffs) == length(monomials) || throw(ArgumentError("Lengths are different"))
        return new{C,Nr,Nc,P,M}(coeffs, monomials)
    end
end

const SimpleRealPolynomial{C,Nr,P<:Unsigned,M<:SimpleRealMonomialVector{Nr,P}} = SimplePolynomial{C,Nr,0,P,M}
const SimpleComplexPolynomial{C,Nc,P<:Unsigned,M<:SimpleComplexMonomialVector{Nc,P}} = SimplePolynomial{C,0,Nc,P,M}
const SimpleDensePolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleDenseMonomialVectorOrView{Nr,Nc,P}}
const SimpleRealDensePolynomial{C,Nr,P<:Unsigned} = SimpleRealPolynomial{C,Nr,P,<:SimpleRealDenseMonomialVectorOrView{Nr,P}}
const SimpleComplexDensePolynomial{C,Nc,P<:Unsigned} = SimpleComplexPolynomial{C,Nc,P,<:SimpleComplexDenseMonomialVectorOrView{Nc,P}}
const SimpleSparsePolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleSparseMonomialVectorOrView{Nr,Nc,P}}
const SimpleRealSparsePolynomial{C,Nr,P<:Unsigned} = SimpleRealPolynomial{C,Nr,P,<:SimpleRealSparseMonomialVectorOrView{Nr,P}}
const SimpleComplexSparsePolynomial{C,Nc,P<:Unsigned} = SimpleComplexPolynomial{C,Nc,P,<:SimpleComplexSparseMonomialVectorOrView{Nc,P}}

_get_c(::XorTX{SimplePolynomial{C}}) where {C} = C
_get_c(::XorTX{SimplePolynomial}) = Val(Any)
_get_nr(::XorTX{SimplePolynomial{<:Any,Nr}}) where {Nr} = Nr
_get_nr(::XorTX{SimplePolynomial}) = Val(Any)
_get_nc(::XorTX{SimplePolynomial{<:Any,<:Any,Nc}}) where {Nc} = Nc
_get_nc(::XorTX{SimplePolynomial}) = Val(Any)
_get_p(::XorTX{SimplePolynomial{<:Any,<:Any,<:Any,P}}) where {P<:Unsigned} = P
_get_p(::XorTX{SimplePolynomial}) = Val(Unsigned)
_get_m(::XorTX{SimplePolynomial{<:Any,Nr,Nc,P,M}}) where {Nr,Nc,P<:Unsigned,M<:SimpleMonomialVector{Nr,Nc,P}} = M
_get_m(::Type{SimplePolynomial{<:Any,Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned} = Val(SimpleMonomialVector{Nr,Nc,P})
_get_m(::Type{SimplePolynomial{<:Any,Nr,Nc}}) where {Nr,Nc} = Val(SimpleMonomialVector{Nr,Nc})
_get_m(::Type{SimplePolynomial{<:Any,Nr,<:Any,P}}) where {Nr,P<:Unsigned} = Val(SimpleMonomialVector{Nr,<:Any,P})
_get_m(::Type{SimplePolynomial{<:Any,<:Any,Nc,P}}) where {Nc,P<:Unsigned} = Val(SimpleMonomialVector{<:Any,Nc,P})
_get_m(::Type{SimplePolynomial{<:Any,Nr}}) where {Nr} = Val(SimpleMonomialVector{Nr})
_get_m(::Type{SimplePolynomial{<:Any,<:Any,Nc}}) where {Nc} = Val(SimpleMonomialVector{<:Any,Nc})
_get_m(::Type{SimplePolynomial{<:Any,<:Any,<:Any,P}}) where {P<:Unsigned} = Val(SimpleMonomialVector{<:Any,<:Any,P})
_get_m(::Type{SimplePolynomial}) = Val(SimpleMonomialVector)
_monvectype(::XorTX{SimplePolynomial{<:Any,Nr,Nc,P,M}}) where {Nr,Nc,P<:Unsigned,M<:SimpleMonomialVector{Nr,Nc,P}} =
    _monvectype(M)
_monvectype(::XorTX{SimplePolynomial{<:Any,<:Any,<:Any,P}}) where {P<:Unsigned} = Val(AbstractVector{P})
_monvectype(::XorTX{SimplePolynomial}) = Val(AbstractVector)

MultivariatePolynomials.variables(p::XorTX{SimplePolynomial}) = variables(monomial_type(p))

MultivariatePolynomials.coefficients(p::SimplePolynomial) = p.coeffs

MultivariatePolynomials.monomials(p::SimplePolynomial) = p.monomials

MultivariatePolynomials.terms(p::SimplePolynomial) = collect(p)

MultivariatePolynomials.nterms(p::SimplePolynomial) = length(p.coeffs)

MultivariatePolynomials.nvariables(::XorTX{SimplePolynomial{C,Nr,Nc} where {C}}) where {Nr,Nc} =
    Nr + 2Nc

Base.iterate(p::SimplePolynomial) =
    isempty(p.coeffs) ? nothing : (@inbounds(term_type(p)(first(p.coeffs), first(p.monomials))), 2)
Base.iterate(p::SimplePolynomial, i::Integer) =
    i > length(p.coeffs) ? nothing : (@inbounds(term_type(p)(p.coeffs[i], p.monomials[i])), i +1)
Base.IteratorSize(::Type{<:SimplePolynomial}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SimplePolynomial}) = Base.HasEltype()
Base.eltype(::Type{P}) where {P<:SimplePolynomial} = term_type(P)
Base.length(p::SimplePolynomial) = length(p.coeffs)

for f in (:mindegree, :maxdegree, :extdegree, :mindegree_complex, :maxdegree_complex, :extdegree_complex, :minhalfdegree,
          :maxhalfdegree, :exthalfdegree, :variables)
    @eval MultivariatePolynomials.$f(p::SimplePolynomial) = $f(p.monomials)
end

MultivariatePolynomials.constant_monomial(m::SimplePolynomial) = constant_monomial(first(m.monomials))

MultivariatePolynomials.constant_monomial(P::Type{<:SimplePolynomial}) = constant_monomial(eltype(P))

MultivariatePolynomials.map_coefficients(f::Function, p::SimplePolynomial) =
    SimplePolynomial(map(f, p.coeffs), copy(p.monomials))

function MultivariatePolynomials.map_coefficients!(f::Function, p::SimplePolynomial)
    @inbounds for i in 1:length(p.coeffs)
        p.coeffs[i] = f(p.coeffs[i])
    end
    return p
end

# these are tricky. They shouldn't even be there since they might (or will) violate the no-allocation policy of
# SimplePolynomials
Base.conj(p::SimplePolynomial) = SimplePolynomial(conj(p.coeffs), conj(p.monomials))
MultivariatePolynomials.LinearAlgebra.adjoint(p::SimplePolynomial) =
    SimplePolynomial(adjoint.(p.coeffs), conj(p.monomials))
Base.isreal(p::SimpleRealPolynomial) = all(∘(iszero, imag), p.coeffs)
function Base.isreal(p::SimplePolynomial)
    # we can rely on the fact that the monomial vector must not contain duplicates; however, we don't know about monomial
    # ordering
    @inbounds for i in 1:length(p)
        mon = p.monomials[i]
        coeff = p.coeffs[i]
        if isreal(mon)
            isreal(coeff) ? continue : return false
        end
        found_conj = false
        mon = conj(mon)
        for j in 1:length(p)
            if mon == p.monomials[j]
                @assert(i != j)
                found_conj = coeff == conj(p.coeffs[j])
                break
            end
        end
        found_conj || return false
    end
    return true
end

MultivariatePolynomials.LinearAlgebra.transpose(p::SimplePolynomial) =
    SimplePolynomial(transpose.(p.coeffs), p.monomials)

SimplePolynomial(p::SimplePolynomial) = p
"""
    SimplePolynomial(p::AbstractPolynomialLike{C}, coefficient_type=C; kwargs...) where {C}

Creates a new `SimplePolynomial` based on any polynomial-like object that satisfied `MultivariatePolynomials`'s
`AbstractPolynomialLike` interface. The coefficients will be of type `coefficient_type`. Keyword arguments are passed on to
[`SimpleMonomialVector`](@ref SimpleMonomialVector(::AbstractVector{<:AbstractMonomialLike})), which allows to influence the
internal representation of the monomial vector.
"""
function SimplePolynomial(p::AbstractPolynomialLike{C1}, ::Type{C}=C1; kwargs...) where {C1,C}
    mv = SimpleMonomialVector(monomials(p); kwargs...)
    coeffs = let c=coefficients(p)
        if c isa Vector{C}
            c
        elseif eltype(c) == C && c isa AbstractVector
            Vector{C}(c)
        elseif eltype(c) == C
            collect(c)
        else
            convert(Vector{C}, collect(c))
        end
    end
    return SimplePolynomial(coeffs, mv)
end

function (p::SimplePolynomial{C,Nr,Nc})(values::AbstractVector{V}) where {C,V,Nr,Nc}
    if length(values) == Nr + 2Nc
        all(v -> v[1] == conj(v[2]), zip(@view(values[Nr+1:Nr+Nc]), @view(values[Nr+Nc+1:end]))) ||
            throw(ArgumentError("Conjugate values did not match conjugate of their actual values"))
    else
        length(values) == Nr + Nc || throw(ArgumentError("Invalid number of values"))
    end
    T = promote_type(C, V)
    result = zero(T)
    real_values = @view(values[1:Nr])
    complex_values = @view(values[Nr+1:Nr+Nc])
    # This is not StaticPolynomials - we are interested in evaluating the polynomials, but only once or few times, so the
    # effort of generating efficient code is not really worth it.
    for t in p
        val = T(coefficient(t))
        for (var, pow) in monomial(t)
            if isreal(var)
                val *= real_values[var.index]^pow
            elseif isconj(var)
                val *= conj(complex_values[var.index])^pow
            else
                val *= complex_values[var.index]^pow
            end
        end
        result += val
    end
    return result
end

function effective_variables_in(p::SimplePolynomial, in)
    for t in p
        iszero(coefficient(t)) || effective_variables_in(monomial(t), in) || return false
    end
    return true
end