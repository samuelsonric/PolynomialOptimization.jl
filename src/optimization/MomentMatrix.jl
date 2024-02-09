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
function moment_matrix(result::POResult{R}; max_deg=Inf,
    prefix::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc},Nothing}=nothing) where {Nr,Nc,R<:AbstractPORelaxation{<:POProblem{<:SimplePolynomial{<:Any,Nr,Nc}}}}
    relaxation = result.relaxation
    b = basis(relaxation)
    if max_deg < Inf
        upto = monomial_count(max_deg, Nr + Nc)
        if upto < lastindex(b)
            if degree(b[upto]) == max_deg && degree(b[upto+1]) == max_deg +1
                b = @view(b[1:upto])
            else
                # this can happen for a custom basis or the Newton polytope
                b = @view(b[1:searchsortedlast(b, max_deg, by=degree)])
            end
        else
            @assert(degree(b[end]) ≤ max_deg)
        end
    end
    return (isreal(relaxation) ? Symmetric : Hermitian)(
        isnothing(prefix) ? moment_matrix(result.moments, Val(:symmetric), b, conj(b)) :
                            moment_matrix(result.moments, Val(:symmetric), b, conj(b), prefix))
end