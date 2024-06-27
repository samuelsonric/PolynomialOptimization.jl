module Relaxation

using ..SimplePolynomials, .SimplePolynomials.MultivariateExponents, ..PolynomialOptimization, MultivariatePolynomials,
    ..PolynomialOptimization.FastVector
import StatsBase, Graphs
using ..PolynomialOptimization: Problem, @verbose_info
import ..PolynomialOptimization: poly_problem, iterate!

export AbstractRelaxation, basis, groupings, iterate!

"""
    AbstractRelaxation

This is the general abstract type for any kind of relaxation of a polynomial optimization problem.
Its concrete types can be used for analyzing and optimizing the problem.

See also [`poly_problem`](@ref), [`Problem`](@ref), [`poly_optimize`](@ref).
"""
abstract type AbstractRelaxation{Prob<:Problem} end

@doc raw"""
    RelaxationGroupings

Contains information about how the elements in a certain (sparse) polynomial optimization problem combine.
Groupings are contained in the fields `obj`, `zero`, `nonneg`, and `psd`:
- ``\sum_i \mathit{obj}_i^\top \sigma_i \overline{\mathit{obj}_i}`` is the SOS representation of the objective with
  ``\sigma_i \succeq 0``
- ``\sum_i \mathit{zero}_{k, i}^\top f_k`` is the prefactor for the kᵗʰ equality
  constraint with ``f_k`` a free vector
- ``\sum_i \mathit{nonneg}_{k, i}^\top \sigma_{k, i} \overline{\mathit{nonneg}_{k, i}}`` is the SOS representation of
  the prefactor of the kᵗʰ nonnegative constraint with ``\sigma_{k, i} \succeq 0``
- ``\sum_i (\mathit{psd}_{k, i}^\top \otimes \mathbb1) Z_{k, i} (\overline{\mathit{psd}_{k, i}} \otimes \mathbb1)``
  is the SOS matrix representation of the prefactor of the kᵗʰ PSD constraint with ``Z_{k, i} \succeq 0``
The field `var_cliques` contains a list of sets of variables, each corresponding to a variable clique in the total problem. In
the complex case, only the declared variables are returned, not their conjugates.
"""
struct RelaxationGroupings{Nr,Nc,I<:Integer,V<:SimpleVariable{Nr,Nc}}
    # These can all be different types of monomial vectors, but we don't want to have a type explosion with nested relaxation
    # groupings. Keep it dynamic.
    obj::Vector{SimpleMonomialVector{Nr,Nc,I}}
    zeros::Vector{Vector{SimpleMonomialVector{Nr,Nc,I}}}
    nonnegs::Vector{Vector{SimpleMonomialVector{Nr,Nc,I}}}
    psds::Vector{Vector{SimpleMonomialVector{Nr,Nc,I}}}
    var_cliques::Vector{Vector{V}}
end

_lensort(x) = (-length(x), x) # use as "by" argument for sort to sort with descending length, standard tie-breaker

    lg = length(grouping)
    print(io, lg, " block", isone(lg) ? "" : "s")
    lensorted = sort(grouping, by=_lensort)
    len = floor(Int, log10(length(first(lensorted)))) +1
    limit = get(io, :limit, false)::Bool
    for block in Iterators.take(lensorted, limit ? 5 : length(lensorted) -1)
        # we must do the printing manually to avoid all the type cluttering. We can assume that a grouping is never empty.
        print(io, "\n  ", lpad(length(block), len, " "), " [")
        a, rest = Iterators.peel(block)
        show(io, "text/plain", a)
        for x in Iterators.take(rest, limit ? 10 : length(block) -1)
            print(io, ", ")
            show(io, "text/plain", x)
        end
        limit && length(block) > 10 && print(io, ", ...")
        print(io, "]")
    end
    limit && length(lensorted) > 5 && print(io, "\n  ", lpad("⋮", len, " "))
end

function Base.show(io::IO, m::MIME"text/plain", groupings::RelaxationGroupings{Nr,Nc}) where {Nr,Nc}
    println(io, "Groupings for the relaxation of a polynomial optimization problem\nVariable cliques\n================")
    for clique in groupings.var_cliques
        print(io, "[")
        a, rest = Iterators.peel(clique)
        show(io, "text/plain", a)
        for x in rest
            print(io, ", ")
            show(io, "text/plain", x)
        end
        println(io, "]")
    end
    print(io, "\nBlock groupings\n===============\nObjective: ")
    _show_groupings(io, groupings.obj)
    for (name, f) in (("Equality", :zeros), ("Nonnegative", :nonnegs), ("Semidefinite", :psds))
        block = getproperty(groupings, f)::Vector{<:Vector{SimpleMonomialVector{Nr,Nc,I}}}
        if !isempty(block)
            for (i, constr) in enumerate(block)
                print(io, "\n", name, " constraint #", i, ": ")
                _show_groupings(io, constr)
            end
        end
    end
end

function embed!(to::AbstractVector{X}, new::X, olds::AbstractVector{X}) where {X}
    for oldᵢ in olds
        if new ⊆ oldᵢ
            push!(to, new)
            return
        end
    end
    temp = sort!(intersect.((new,), olds), by=_lensort)
    @inbounds for i in length(temp):-1:2
        tempᵢ = temp[i]
        for tempⱼ in @view(temp[1:i-1])
            tempᵢ ⊆ tempⱼ && deleteat!(temp, i)
        end
    end
    append!(to, temp)
    return
end

function embed(news::AbstractVector{X}, olds::AbstractVector{X}) where {X}
    to = FastVec{X}(buffer=length(news))
    for new in news
        embed!(to, new, olds)
    end
    if X <: AbstractVector
        for toᵢ in to
            sort!(toᵢ)
        end
    end
    return Base._groupedunique!(sort!(finish!(to), by=_lensort))
end

function embed(new::RG, old::RG) where {Nr,Nc,I<:Integer,RG<:RelaxationGroupings{Nr,Nc,I}}
    (length(new.zeros) == length(old.zeros) && length(new.nonnegs) == length(old.nonnegs) &&
        length(new.psds) == length(old.psds)) ||
        throw(ArgumentError("Cannot embed two relaxation groupings for different optimization problems"))
    newobj = embed(new.obj, old.obj)
    newzeros = embed.(new.zeros, old.zeros)
    newnonnegs = embed.(new.nonnegs, old.nonnegs)
    newpsds = embed.(new.psds, old.psds)
    newcliques = embed(new.var_cliques, old.var_cliques)
    return RG(newobj, newzeros, newnonnegs, newpsds, newcliques)
end
embed(new::RelaxationGroupings, ::Nothing) = new

Base.:(==)(g₁::G, g₂::G) where {G<:RelaxationGroupings} = g₁.obj == g₂.obj && g₁.zeros == g₂.zeros &&
    g₁.nonnegs == g₂.nonnegs && g₁.psds == g₂.psds && g₁.var_cliques == g₂.var_cliques

"""
    poly_problem(relaxation::AbstractRelaxation)

Returns the original problem associated with a relaxation.
"""
poly_problem(relaxation::AbstractRelaxation) = relaxation.problem

"""
    basis(relaxation::AbstractRelaxation[, clique::Int]) -> SimpleMonomialVector

Constructs the basis that is associated with a given polynomial relaxation. If `clique` is given, only the monomials that are
relevant for the given clique must be returned.
"""
function basis end

"""
    groupings(relaxation::AbstractRelaxation) -> RelaxationGroupings

Analyze the current state and return the bases and cliques as indicated by its relaxation in a [`RelaxationGroupings`](@ref)
struct.
"""
groupings(relaxation::AbstractRelaxation) = relaxation.groupings
groupings(::Nothing) = nothing

"""
    iterate!(state::AbstractRelaxation; objective=true, zero=true, nonneg=true, psd=true)

Some sparse polynomial optimization relaxations allow to iterate their sparsity, which will lead to a more dense representation
and might give better bounds at the expense of a more costly optimization. Return `nothing` if the iterations converged
(`state` did not change any more), else return the new state. Note that `state` will be modified.

The keyword arguments allow to restrict the iteration to certain elements. This does not necessarily mean that the bases
associated to other elements will not change as well to keep consistency; but their own contribution will not be considered.
The parameters `nonneg` and `psd` may either be `true` (to iterate all those constraints) or a set of integers that refer to
the indices of the constraints, as they were originally given to [`poly_problem`](@ref).
"""
iterate!(::AbstractRelaxation; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

"""
    Relaxation.XXX(problem::Problem[, degree]; kwargs...)

This is a convenience wrapper for `Relaxation.XXX(Relaxation.Dense(problem, degree))` that works for any
[`AbstractRelaxation`](@ref) `XXX`.
`degree` is the degree of the Lasserre relaxation, which must be larger or equal to the halfdegree of all polynomials that are
involved. If `degree` is omitted, the minimum required degree will be used.
Specifying a degree larger than the minimal only makes sense if there are inequality or PSD constraints present, else it
needlessly complicates calculations without any benefit.

The keyword arguments will be passed on to the constructor of `XXX`.
"""
function (r::Type{<:AbstractRelaxation})(problem::Problem, args...; kwargs...)
    r <: Dense && throw(MethodError(r, (problem, args...)))
    # to avoid infinite recursion if the arguments did not match
    return r(Dense(problem, args...); kwargs...)
end

function _show(io::IO, m::MIME"text/plain", x::AbstractRelaxation)
    groups = groupings(x)
    # we don't want to print the fully parameterized type type
    print(io, "Relaxation.", typeof(x).name.name, " of a polynomial optimization problem\nVariable cliques:")
    for va in groups.var_cliques
        print(io, "\n  ", join(va, ", "))
    end
    bs = StatsBase.countmap(length.(groups.obj))
    for constrs in (groups.nonnegs, groups.psds)
        for constr in constrs
            mergewith!(+, bs, StatsBase.countmap(length.(constr)))
        end
    end
    print(io, "\nPSD block sizes:\n  ", sort!(collect(bs), rev=true))
    if !isempty(groups.zeros)
        empty!(bs)
        for constr in groups.zeros
            mergewith!(+, bs, StatsBase.countmap(length.(constr)))
        end
        print(io, "\nFree block sizes:\n  ", sort!(collect(bs), rev=true))
    end
end

Base.show(io::IO, m::MIME"text/plain", x::AbstractRelaxation) = _show(io, m, x)

# make working with the relaxation as simple as working with the problem itself
Base.getproperty(relaxation::AbstractRelaxation, f::Symbol) =
    hasfield(typeof(relaxation), f) ? getfield(relaxation, f) : getproperty(getfield(relaxation, :problem), f)
Base.propertynames(relaxation::AbstractRelaxation{P}) where {P<:Problem} =
    (fieldnames(typeof(relaxation))..., fieldnames(P)...)
MultivariatePolynomials.variables(relaxation::AbstractRelaxation) = variables(relaxation.problem)
MultivariatePolynomials.nvariables(relaxation::AbstractRelaxation) = nvariables(relaxation.problem)
"""
    degree(problem::AbstractRelaxation)

Returns the degree associated with the relaxation of a polynomial optimization problem.

See also [`poly_problem`](@ref).
"""
function MultivariatePolynomials.degree(relaxation::AbstractRelaxation)
    gr = groupings(relaxation)
    subdegree = v -> maximum(maxdegree_complex, v)
    return max(
        maximum(maxdegree_complex, gr.obj),
        maximum(subdegree, gr.zeros, init=0),
        maximum(subdegree, gr.nonnegs, init=0),
        maximum(subdegree, gr.psds, init=0)
    )
end
Base.isreal(relaxation::AbstractRelaxation) = isreal(relaxation.problem)

include("./basis/Basis.jl")
include("./sparse/Sparse.jl")

end