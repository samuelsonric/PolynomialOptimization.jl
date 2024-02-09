export AbstractPORelaxation, basis, groupings, iterate!

"""
    AbstractPORelaxation

This is the general abstract type for any kind of relaxation of a polynomial optimization problem.
Its concrete types can be used for analyzing and optimizing the problem.

See also [`poly_problem`](@ref), [`POProblem`](@ref), [`poly_optimize`](@ref).
"""
abstract type AbstractPORelaxation{Prob<:POProblem} end

"""
    RelaxationGroupings

Contains information about how the elements in a certain (sparse) polynomial optimization problem combine.
Groupings are contained in the fields `obj`, `zero`, `nonneg`, and `psd`.
The field `var_cliques` contains a list of sets of variables, each corresponding to a variable clique in the total problem. In
the complex case, only the declared variables are returned, not their conjugates.
"""
struct RelaxationGroupings{MV,V}
    obj::Vector{MV}
    zero::Vector{Vector{MV}}
    nonneg::Vector{Vector{MV}}
    psd::Vector{Vector{MV}}
    var_cliques::Vector{V}
end

"""
    poly_problem(relaxation::AbstractPORelaxation)

Returns the original problem associated with a relaxation.
"""
poly_problem(relaxation::AbstractPORelaxation) = relaxation.problem

"""
    basis(relaxation::AbstractPORelaxation[, clique::Int]) -> SimpleMonomialVector

Constructs the basis that is associated with a given polynomial relaxation. If `clique` is given, only the monomials that are
relevant for the given clique must be returned.
"""
function basis end

"""
    groupings(state::AbstractPORelaxation) -> RelaxationGroupings

Analyze the current state and return the bases and cliques as indicated by its relaxation in a [`RelaxationGroupings`](@ref)
struct.
"""
function groupings end

"""
    iterate!(state::AbstractPORelaxation; objective=true, zero=true, nonneg=true, psd=true)

Some sparse polynomial optimization relaxations allow to iterate their sparsity, which will lead to a more dense representation
and might give better bounds at the expense of a more costly optimization. Return `nothing` if the iterations converged
(`state` did not change any more), else return the new state. Note that `state` will be modified.

The keyword arguments allow to restrict the iteration to certain elements. This does not necessarily mean that the bases
associated to other elements will not change as well to keep consistency; but their own contribution will not be considered.
The parameters `nonneg` and `psd` may either be `true` (to iterate all those constraints) or a set of integers that refer to
the indices of the constraints, as they were originally given to [`poly_problem`](@ref).
"""
function iterate! end

# internal function
function supports end

function _show(io::IO, m::MIME"text/plain", x::AbstractPORelaxation)
    groups = groupings(x)
    sort!.(groups.var_cliques)
    sort!(groups.var_cliques)
    # we don't want to print the fully parameterized type type
    print(io, typeof(x).name.name, " of a polynomial optimization problem\nVariable cliques:")
    for va in groups.var_cliques
        print(io, "\n  ", join(va, ", "))
    end
    bs = StatsBase.countmap(length.(groups.obj))
    for constrs in (groups.zero, groups.nonneg, groups.psd)
        for constr in constrs
            mergewith!(+, bs, StatsBase.countmap(length.(constr)))
        end
    end
    print(io, "\nBlock sizes:\n  ", sort!(collect(bs), rev=true))
end

Base.show(io::IO, m::MIME"text/plain", x::AbstractPORelaxation) = _show(io, m, x)

# make working with the relaxation as simple as working with the problem itself
Base.getproperty(relaxation::AbstractPORelaxation, f::Symbol) =
    hasfield(typeof(relaxation), f) ? getfield(relaxation, f) : getproperty(getfield(relaxation, :problem), f)
Base.propertynames(relaxation::AbstractPORelaxation{P}) where {P<:POProblem} =
    (fieldnames(typeof(relaxation))..., fieldnames(P)...)
MultivariatePolynomials.variables(relaxation::AbstractPORelaxation) = variables(relaxation.problem)
MultivariatePolynomials.nvariables(relaxation::AbstractPORelaxation) = nvariables(relaxation.problem)
"""
    degree(problem::AbstractPOProblem)

Returns the degree associated with a polynomial optimization problem.

See also [`poly_problem`](@ref).
"""
MultivariatePolynomials.degree(relaxation::AbstractPORelaxation) = maxdegree(basis(relaxation))
Base.isreal(relaxation::AbstractPORelaxation) = isreal(relaxation.problem)

include("./basis/Basis.jl")
include("./basis/Dense.jl")
include("./basis/Custom.jl")
