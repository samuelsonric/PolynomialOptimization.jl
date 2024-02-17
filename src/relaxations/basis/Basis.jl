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
    d = relaxation.degree
    full = @view(b[begin:end])
    T = typeof(full)
    return RelaxationGroupings(
        [full],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in p.constr_zero],
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