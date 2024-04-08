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
Groupings are contained in the fields `obj`, `zero`, `nonneg`, and `psd`:
- `∑ᵢ transpose(objᵢ) * σᵢ * conj(objᵢ)` is the SOS representation of the objective with `σᵢ` PSD
- `∑ᵢ transpose(zeroₖᵢ) * fₖ * conj(zeroₖᵢ)` is the prefactor for the kᵗʰ equality constraint with `fₖ` a free matrix
- `∑ᵢ transpose(nonnegₖᵢ) * σₖᵢ * conj(nonnegₖᵢ)` is the SOS representation of the prefactor of the kᵗʰ nonnegative constraint
  with `σₖᵢ` PSD
- `∑ᵢ (transpose(psdₖᵢ) ⊗ 𝟙) * Zₖᵢ * (conj(psdₖᵢ) ⊗ 𝟙)` is the SOS matrix representation of the prefactor of the kᵗʰ PSD
  constraint with Zₖᵢ PSD
The field `var_cliques` contains a list of sets of variables, each corresponding to a variable clique in the total problem. In
the complex case, only the declared variables are returned, not their conjugates.
"""
struct RelaxationGroupings{Nr,Nc,V<:SimpleVariable{Nr,Nc}}
    # These can all be different types of monomial vectors, but we don't want to have a type explosion with nested relaxation
    # groupings. Keep it dynamic.
    obj::Vector{<:SimpleMonomialVector{Nr,Nc}}
    zeros::Vector{<:Vector{<:SimpleMonomialVector{Nr,Nc}}}
    nonnegs::Vector{<:Vector{<:SimpleMonomialVector{Nr,Nc}}}
    psds::Vector{<:Vector{<:SimpleMonomialVector{Nr,Nc}}}
    var_cliques::Vector{Vector{V}}
end

@eval function Base.intersect(a::RG, b::RG) where {Nr,Nc,RG<:RelaxationGroupings{Nr,Nc}}
    (length(a.zeros) == length(b.zeros) && length(a.nonnegs) == length(b.nonnegs) && length(a.psds) == length(b.psds)) ||
        error("Cannot intersect two relaxation groupings for different optimization problems")
    # TODO: move from Iterators.product to something that is indexable, so that every loop can be parallelized
    newobj = Vector{SimpleMonomialVector{Nr,Nc}}(undef, length(a.obj) * length(b.obj))
    for (i, (obj_a, obj_b)) in enumerate(Iterators.product(a.obj, b.obj))
        @inbounds newobj[i] = intersect(obj_a, obj_b)
    end
    newobj = Base._groupedunique!(sort!(newobj))

    $((quote
        $(Symbol(:new, name)) = Vector{Vector{<:SimpleMonomialVector{Nr,Nc}}}(undef, length(a.$name))
        for k in 1:length(a.$name)
            @inbounds as, bs = a.$name[k], b.$name[k]
            newₖ = Vector{SimpleMonomialVector{Nr,Nc}}(undef, length(as) * length(bs))
            for (i, (asᵢ, bsᵢ)) in enumerate(Iterators.product(as, bs))
                @inbounds newₖ[i] = intersect(asᵢ, bsᵢ)
            end
            @inbounds $(Symbol(:new, name))[k] = Base._groupedunique!(sort!(newₖ))
        end
    end for name in (:zeros, :nonnegs, :psds))...)

    newcliques = Vector{Vector{SimpleVariable{Nr,Nc,SimplePolynomials.smallest_unsigned(Nr + 2Nc)}}}(
        undef, length(a.var_cliques) * length(b.var_cliques)
    )
    for (i, (clique_a, clique_b)) in enumerate(Iterators.product(a.var_cliques, b.var_cliques))
        @inbounds newcliques[i] = sort!(intersect(clique_a, clique_b))
    end
    newcliques = Base._groupedunique!(sort!(newcliques, by=x -> (-length(x), x)))

    return RG(newobj, newzeros, newnonnegs, newpsds, newcliques)
end
Base.intersect(a::RelaxationGroupings, ::Nothing) = a
Base.intersect(::Nothing, b::RelaxationGroupings) = b

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
    groupings(relaxation::AbstractPORelaxation) -> RelaxationGroupings

Analyze the current state and return the bases and cliques as indicated by its relaxation in a [`RelaxationGroupings`](@ref)
struct.
"""
groupings(relaxation::AbstractPORelaxation) = relaxation.groupings
groupings(::Nothing) = nothing

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
iterate!(::AbstractPORelaxation; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

"""
    RelaxationXXX(problem::POProblem[, degree]; kwargs...)

This is a convenience wrapper for `RelaxationXXX(RelaxationDense(problem, degree))` that works for any
[`AbstractPORelaxation`](@ref) `RelaxationXXX`.
`degree` is the degree of the Lasserre relaxation, which must be larger or equal to the halfdegree of all polynomials that are
involved. If `degree` is omitted, the minimum required degree will be used.
Specifying a degree larger than the minimal only makes sense if there are inequality or PSD constraints present, else it
needlessly complicates calculations without any benefit.

The keyword arguments will be passed on to the constructor of `RelaxationXXX`.
"""
(r::Type{<:AbstractPORelaxation})(problem::POProblem, args...; kwargs...) = r(RelaxationDense(problem, args...); kwargs...)

function _show(io::IO, m::MIME"text/plain", x::AbstractPORelaxation)
    groups = groupings(x)
    # we don't want to print the fully parameterized type type
    print(io, typeof(x).name.name, " of a polynomial optimization problem\nVariable cliques:")
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
function MultivariatePolynomials.degree(relaxation::AbstractPORelaxation)
    gr = groupings(relaxation)
    subdegree = v -> maximum(maxdegree_complex, v)
    return max(
        maximum(maxdegree_complex, gr.obj),
        maximum(subdegree, gr.zeros, init=0),
        maximum(subdegree, gr.nonnegs, init=0),
        maximum(subdegree, gr.psds, init=0)
    )
end
Base.isreal(relaxation::AbstractPORelaxation) = isreal(relaxation.problem)

include("./degree/Degree.jl")
include("./sparse/Sparse.jl")