export SparsityNone

"""
    SparsityNone(problem::PolyOptProblem)

This is a wrapper that does not perform any kind of sparsity analysis.

See also [`poly_problem`](@ref), [`sparse_optimize`](@ref).
"""
struct SparsityNone <: AbstractSparsity
    problem::PolyOptProblem
end

sparse_problem(ns::SparsityNone) = ns.problem

sparse_iterate!(::SparsityNone; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

sparse_groupings(ns::SparsityNone) =
    [[ns.problem.basis], ([x.basis] for x in ns.problem.constraints)...], [ns.problem.variables]

sparse_supports(ns::SparsityNone) = ns.problem.basis

default_solution_method(::SparsityNone, ::Any) = :mvhankel