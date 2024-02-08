# we don't construct a new type for this, we just provide a sparse interface for POProblem

sparse_iterate!(::POProblem; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

function sparse_groupings(problem::POProblem)
    b = problem.basis
    d = problem.degree
    full = @view(b[begin:end])
    T = typeof(full)
    return SparseGroupings(
        [full],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in problem.constr_zero],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in problem.constr_nonneg],
        Vector{T}[[truncate_basis(b, d - maxhalfdegree(x))] for x in problem.constr_psd],
        [filter(∘(!, isconj), variables(problem.objective))]
    )
end

sparse_supports(problem::POProblem) = problem.basis

default_solution_method(::POProblem) = :mvhankel