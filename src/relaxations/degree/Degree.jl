"""
    AbstractRelaxationDegree{Prob} <: AbstractPORelaxation{Prob}

An `AbstractRelaxationDegree` is a relaxation of a polynomial optimization problem that is built on a global maximal degree
cutoff. The groupings for the individual elements will come from a degree truncation of the same shared basis for all
constituents of the problem (intersected with a parent grouping).
"""
abstract type AbstractRelaxationDegree{Prob<:POProblem} <: AbstractPORelaxation{Prob} end

basis(relaxation::AbstractRelaxationDegree) = relaxation.basis

function basis(relaxation::AbstractRelaxationDegree, i::Int)
    i == 1 || throw(ArgumentError("Unknown clique index: $i"))
    return relaxation.basis
end
# sparse: filter(Base.Fix2(effective_variables_in, clique), a1)

function groupings(problem::POProblem{Prob}, basis::AbstractVector{M}, degree::Integer, parent) where
    {Nr,Nc,M<:SimpleMonomial{Nr,Nc},Prob<:SimplePolynomial{<:Any,Nr,Nc}}
    return intersect(RelaxationGroupings(
        [basis],
        [[truncate_basis(basis, degree - maxhalfdegree(x))] for x in problem.constr_zero],
        [[truncate_basis(basis, degree - maxhalfdegree(x))] for x in problem.constr_nonneg],
        [[truncate_basis(basis, degree - maxhalfdegree(x))] for x in problem.constr_psd],
        [filter(∘(!, isconj), variables(problem.objective))]
    ), parent)
end

MultivariatePolynomials.degree(relaxation::AbstractRelaxationDegree) = relaxation.degree

function Base.show(io::IO, m::MIME"text/plain", relaxation::AbstractRelaxationDegree)
    _show(io, m, relaxation)
    print(io, "\nRelaxation degree: ", relaxation.degree)
end

struct _DummyMonomial
    degree::Int
end

MultivariatePolynomials.degree(d::_DummyMonomial) = d.degree

function truncate_basis(v::AbstractVector{M} where {M<:SimpleMonomial}, maxdeg::Integer)
    idx = searchsortedlast(v, _DummyMonomial(maxdeg), by=degree)
    if idx < firstindex(v)
        return @view(v[begin:end])
    else
        return @view(v[1:idx])
    end
end

include("./Dense.jl")
include("./Custom.jl")
include("./Newton.jl")