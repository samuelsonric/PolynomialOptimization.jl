export optimality_certificate

"""
    optimality_certificate(result::POResult, ϵ=1e-6)

This function applies the flat extension/truncation criterion to determine whether the optimality of the given problem can be
certified, in which case it returns `:Optimal`. If no such certificate is found, the function returns `:Unknown`. The criterion
is meaningless for sparse problems or if a full basis is not available.
The parameter `ϵ` controls the bound below which singular values are considered to be zero, and its negative below which
eigenvalues are considered to be negative.

See also [`poly_optimize`](@ref).
"""
function optimality_certificate(result::POResult{<:RealPOProblem}, ϵ::R=1e-6) where {R<:Real}
    problem = poly_problem(result)
    problem.degree < 1 && return :CertificateUnavailable
    d₀ = maxhalfdegree(problem.objective)
    d₁ = max(1, maximum(maxhalfdegree, problem.constr_zero, init=0), maximum(maxhalfdegree, problem.constr_nonneg, init=0),
        maximum(maxhalfdegree, problem.constr_psd, init=0))
    for t in problem.degree:-1:max(d₀, d₁)
        rkMₜ = rank(moment_matrix(result, max_deg=t), atol=ϵ)
        if rkMₜ == 1 || rkMₜ == rank(moment_matrix(result, max_deg=t-d₁), atol=ϵ)
            return :Optimal
        end
    end
    return :Unknown
end
