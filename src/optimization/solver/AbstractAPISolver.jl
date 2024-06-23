export AbstractAPISolver

"""
    AbstractAPISolver{K<:Integer}

Superclass for a solver that requires new variables/constraints to be added via API calls. Solvers that are of this type must
implement [`append!`](@ref append!(::AbstractAPISolver{K}, ::K) where {K<:Integer}) in such a way that they directly add a
variable (moment-case) to or constraint (SOS-case) to the solver.
Concrete types that inherit from `AbstractAPISolver` must have a property `mon_to_solver::Dict{FastKey{K},solver indextype}`.
"""
abstract type AbstractAPISolver{K<:Integer} end

"""
    append!(solver::AbstractAPISolver{K}, key::K)

Appends at least one new variable (moment-case) or constraint (SOS-case) to the solver `state` that represents the monomial
given by `key`.
"""
append!(::AbstractAPISolver{K}, ::K) where {K<:Integer}

@inline function mindex(solver::AbstractAPISolver, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc}
    idx = monomial_index(monomials...)
    dictidx = Base.ht_keyindex(solver.mon_to_solver, FastKey(idx))
    @inbounds return (dictidx < 0 ?
        append!(solver, idx) :
        solver.mon_to_solver.vals[dictidx]
    )::valtype(solver.mon_to_solver)
end

"""
    MomentVector(relaxation::AbstractRelaxation, moments::Vector{<:Real},
        solver::AbstractAPISolver)

Given the moments vector as obtained from an [`AbstractAPISolver`](@ref), convert it to a [`MomentVector`](@ref). Note that this
function is not fully type-stable, as the result may be based either on a dense or sparse vector depending on the relaxation.
"""
function MomentVector(relaxation::AbstractRelaxation, moments::Vector{V}, solver::AbstractAPISolver{K}) where {K<:Integer,V<:Real}
    @assert(length(moments) ≥ length(solver.mon_to_solver))
    max_mons = relaxation_bound(relaxation)
    @assert(length(solver.mon_to_solver) ≤ max_mons)
    # In any case, our variables will not be in the proper order. First figure this out.
    # Dict does not preserve the insertion order. While we could use OrderedDict instead, we need to do lots of insertions and
    # lookups and only once access the elements in the insertion order; so it is probably better to do the sorting once.
    mon_pos = convert.(K, keys(solver.mon_to_solver))
    var_vals = collect(values(solver.mon_to_solver))
    sort_along!(var_vals, mon_pos)
    # Now mon_pos is in insertion order. There might still not be a 1:1 correspondence between x and mon_pos, as slack
    # variables could be present if the solver requires those.
    if length(var_vals) < length(moments)
        var_vals .+= one(Cint)
        @inbounds moments = moments[var_vals] # remove the slack
    end
    # Now we have the 1:1 correspondence, but we want the actual monomial order.
    sort_along!(mon_pos, moments)
    # Finally, x is ordered!
    if length(moments) == max_mons # dense case
        solution = moments
    elseif 3length(mon_pos) < max_mons
        solution = SparseVector(max_mons, mon_pos, moments)
    else
        solution = fill(NaN, max_mons)
        copy!(@view(solution[mon_pos]), moments)
    end
    return MomentVector(relaxation, solution)
end