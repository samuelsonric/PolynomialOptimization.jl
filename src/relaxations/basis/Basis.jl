abstract type AbstractBasisRelaxation{Prob<:POProblem} <: AbstractPORelaxation{Prob} end

basis(relaxation::AbstractBasisRelaxation) = relaxation.basis

function basis(relaxation::AbstractBasisRelaxation, i::Int)
    i == 1 || throw(ArgumentError("Unknown clique index: $i"))
    return relaxation.basis
end
# sparse: filter(Base.Fix2(effective_variables_in, clique), a1)

function groupings(relaxation::AbstractBasisRelaxation)
    p = relaxation.problem
    b = relaxation.basis
    zb = relaxation.zero_basis
    d = relaxation.degree
    full = @view(b[begin:end])
    T = typeof(full)
    return RelaxationGroupings(
        [full],
        Vector{T}[[truncate_basis(zb, 2d - maxdegree(x))] for x in p.constr_zero],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in p.constr_nonneg],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in p.constr_psd],
        [filter(∘(!, isconj), variables(p.objective))]
    )
end

iterate!(::AbstractBasisRelaxation; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

supports(relaxation::AbstractBasisRelaxation) = relaxation.basis

MultivariatePolynomials.degree(relaxation::AbstractBasisRelaxation) = relaxation.degree

function Base.show(io::IO, m::MIME"text/plain", relaxation::AbstractBasisRelaxation)
    _show(io, m, relaxation)
    print(io, "\nRelaxation degree: ", relaxation.degree)
end

struct _DummyMonomial
    degree::Int
end

MultivariatePolynomials.degree(d::_DummyMonomial) = d.degree

function truncate_basis(v::SimpleMonomialVector, maxdeg::Integer)
    idx = searchsortedlast(v, _DummyMonomial(maxdeg), by=degree)
    if idx < firstindex(v)
        return @view(v[begin:end])
    else
        return @view(v[1:idx])
    end
end

function zero_maxdegs(constr_zero::AbstractVector{P}, degree::Integer) where {Nr,Nc,P<:SimplePolynomial{<:Any,Nr,Nc}}
    maxpower_T = SimplePolynomials.smallest_unsigned(2degree)
    isempty(constr_zero) && return 0, zeros(maxpower_T, Nr + 2Nc)
    # For zero constraints, the prefactor is not given by ∑ᵢⱼ zᵢ PSDᵢⱼ conj(zⱼ), but it is instead an arbitrary polynomial of
    # twice the degree. But since the basis for twice the degree is much smaller that the square of the degree (due to
    # duplicates), we use it directly.
    # We also don't need to construct the whole basis for degree 2d - as we anyway will only take the parts of the basis whose
    # degree plus the degree of the constraint is ≤ 2d, we can look for the minimal constraint degree.
    maxmultideg = fill(typemax(maxpower_T), Nr + 2Nc)
    @inbounds for constr in constr_zero
        mons = monomials(constr)
        if !iszero(Nr)
            for (i, degs_r) in enumerate(eachrow(mons.exponents_real))
                deg_r = maximum(degs_r, init=0)
                if deg_r < maxmultideg[i]
                    maxmultideg[i] = deg_r
                end
            end
        end
        if !iszero(Nc)
            for (i, degs_c) in zip(Iterators.countfrom(Nr +1), eachrow(mons.exponents_complex))
                deg_c = maximum(degs_c, init=0)
                if deg_c < maxmultideg[i]
                    maxmultideg[i] = deg_c
                end
            end
            for (i, degs_c) in zip(Iterators.countfrom(Nr + Nc +1), eachrow(mons.exponents_conj))
                deg_c = maximum(degs_c, init=0)
                if deg_c < maxmultideg[i]
                    maxmultideg[i] = deg_c
                end
            end
        end
    end
    # maxmultideg currently holds smallest maximal degree that can be found per variable. We need to flip this around and
    # turn it into the largest allowed degree for the prefactor
    @inbounds for i in 1:Nr
        maxmultideg[i] = 2degree - maxmultideg[i]
    end
    @inbounds for i in Nr+1:Nr+2Nc
        maxmultideg[i] = degree - maxmultideg[i]
    end
    return 2degree - minimum(maxdegree_complex, constr_zero), maxmultideg
end