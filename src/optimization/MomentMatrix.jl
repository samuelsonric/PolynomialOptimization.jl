export moment_matrix

moment_matrix(moments::FullMonomialVector{R,Nr,Nc}, ::Val{:symmetric}, rows, cols, prefix::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc,R} =
    R[i ≤ j ? get(moments, let n=R(NaN); () -> n end, rows[i], cols[j], prefix...) : R(NaN)
      for i in 1:length(rows), j in 1:length(cols)]

moment_matrix(moments::FullMonomialVector{R,Nr,Nc}, ::Val{:throw}, rows, cols, prefix::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc,R} =
    R[get(moments, () -> throw(MonomialMissing()), rows[i], cols[j], prefix...)
      for i in 1:length(rows), j in 1:length(cols)]

"""
    moment_matrix(problem::POResult; max_deg=Inf, prefix=1)

After a problem has been optimized, this function assembles the associated moment matrix (possibly by imposing a degree bound
`max_deg`, and possibly multiplying each monomial by the monomial or variable `prefix`, which does not add to `max_deg`).
Note that `prefix` has to be a valid `SimpleMonomial` or `SimpleVariable` of appropriate type.

See also [`poly_optimize`](@ref), [`poly_optimize`](@ref).
"""
function moment_matrix(result::POResult{Prob}; max_deg=Inf,
    prefix::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc},Nothing}=nothing) where {Nr,Nc,P<:SimplePolynomial{<:Any,Nr,Nc},Prob<:AbstractPOProblem{P}}
    problem = dense_problem(result)
    if max_deg < Inf
        upto = monomial_count(max_deg, Nr + Nc)
        if upto < lastindex(problem.basis)
            if degree(problem.basis[upto]) == max_deg && degree(problem.basis[upto+1]) == max_deg +1
                b = @view(problem.basis[1:upto])
            else
                # this can happen for a custom basis or the Newton polytope
                b = @view(problem.basis[1:searchsortedlast(problem.basis, max_deg, by=degree)])
            end
        else
            @assert(degree(problem.basis[end]) ≤ max_deg)
            b = problem.basis
        end
    else
        b = problem.basis
    end
    return (isreal(problem) ? Symmetric : Hermitian)(
        isnothing(prefix) ? moment_matrix(result.moments, Val(:symmetric), b, conj(b)) :
                            moment_matrix(result.moments, Val(:symmetric), b, conj(b), prefix))
end