export POResult

struct FullMonomialVector{V,Nr,Nc,D<:AbstractVector{V}} <: AbstractVector{V}
    values::D

    FullMonomialVector{M}(values::D) where {Nr,Nc,M<:SimpleMonomial{Nr,Nc},V,D<:AbstractVector{V}} = new{V,Nr,Nc,D}(values)
end

Base.length(d::FullMonomialVector) = length(d.values)
Base.size(d::FullMonomialVector) = (length(d.values),)
Base.haskey(d::FullMonomialVector{<:Any,Nr,Nc}, keys::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc} =
    monomial_index(keys...) ≤ length(d.values)
function Base.getindex(d::FullMonomialVector{<:Any,Nr,Nc}, keys::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc}
    idx = monomial_index(keys...)
    idx ≤ length(d.values) && return @inbounds(d.values[idx])
    return V(NaN)
end
Base.@propagate_inbounds Base.getindex(d::FullMonomialVector, i::Integer) = d.values[i]
Base.@propagate_inbounds Base.setindex!(d::FullMonomialVector, value, i::Integer) = setindex!(d.values, value, i)
function Base.get(d::FullMonomialVector{<:Any,Nr,Nc}, not_found::Function,
    keys::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc}
    idx = monomial_index(keys...)
    if idx ≤ length(d.values)
        val = @inbounds(d.values[idx])
        isnan(val) || return val
    end
    return not_found()
end
Base.iterate(d::FullMonomialVector, args...) = iterate(d.values, args...)

"""
    POResult

Result of a polynomial optimization, returned by calling [`poly_optimize`](@ref) on an [`AbstractPOProblem`](@ref).
A `POResult` struct `r` contains information about
- the optimized problem (available via [`poly_problem`](@ref))
- the used method (`r.method`)
- the time required for the optimization in seconds (`r.time`)
- the status of the solver (`r.status`), which also depends on the solver type
- the returned primal value of the solver (`r.objective`), which, if the status was successful, is a lower bound to the true
  minimum
- the moment information which allows to construct a [moment matrix](@ref moment_matrix), extract solutions
  ([`poly_all_solutions`](@ref) or [`poly_solutions`](@ref)), and an [optimality certificate](@ref optimality_certificate).
"""
struct POResult{P<:AbstractPOProblem,V,M<:FullMonomialVector{V}}
    problem::P
    method::Symbol
    time::Float64
    status
    objective::Float64
    moments::M
end

function Base.show(io::IO, ::MIME"text/plain", x::POResult)
    println(io, "Polynomial optimization result")
    println(io, "Used optimization method: ", x.method)
    println(io, "Status of the solver: ", x.status)
    println(io, "Lower bound to optimum (in case of good status): ", x.objective)
    println(io, "Time required for optimization: ",
        iszero(x.time) ? "less than one second" :
        (isone(x.time) ? "one second" : "$(x.time) seconds")
    )
end

Base.eltype(::Type{<:(POResult{<:AbstractPOProblem,V})}) where {V} = V

"""
    poly_problem(r::POResult)

Returns the problem that was associated with the optimization result.
"""
poly_problem(r::POResult) = r.problem
dense_problem(r::POResult) = dense_problem(r.problem)