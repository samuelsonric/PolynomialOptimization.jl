"""
    MomentVector(relaxation::AbstractRelaxation, values::AbstractVector{R} where {R<:Real})

`MomentVector` is a representation of the result of a polynomial optimization. It contains all the values of the moments
that were present in the optimization problem. This vector can be indexed in two ways:
- linearly, which will just transparently yield what linearly indexing `values` would yield
- with a monomial (or multiple monomials, which means that the product of all the monomials is to be considered), which will
  yield the value that is associated with this monomial; if the problem was complex-valued, this will be a `Complex{R}`.

This type is not exported.
"""
struct MomentVector{R,V<:Union{R,Complex{R}},Nr,Nc,D<:AbstractVector{R}} <: AbstractVector{V}
    values::D

    MomentVector(::AbstractRelaxation{<:Problem{<:SimplePolynomial{<:Any,Nr,Nc}}}, values::D) where {Nr,Nc,R,D<:AbstractVector{R}} =
        new{R,iszero(Nc) ? R : Complex{R},Nr,Nc,D}(values)
end

Base.length(d::MomentVector) = length(d.values)
Base.size(d::MomentVector) = (length(d.values),)
Base.haskey(d::MomentVector{<:Any,<:Any,Nr,Nc}, keys::Union{<:SimpleMonomialOrConj{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc} =
    monomial_index(keys...) ≤ length(d.values)
Base.haskey(d::MomentVector{<:Any,<:Any,Nr,Nc,<:AbstractSparseVector},
    keys::Union{<:SimpleMonomialOrConj{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc} =
    insorted(monomial_index(keys...), rowvals(d.values))
Base.getindex(d::MomentVector{<:Any,V,Nr,Nc}, keys::Union{<:SimpleMonomialOrConj{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc,V} =
    @inline get(d, () -> V(NaN), keys...)
Base.@propagate_inbounds Base.getindex(d::MomentVector, i::Integer) = d.values[i]
Base.@propagate_inbounds Base.setindex!(d::MomentVector, value, i::Integer) = setindex!(d.values, value, i)
function Base.get(d::MomentVector{R,R,Nr,0} where {R}, not_found,
    keys::Union{<:SimpleMonomialOrConj{Nr,0},<:SimpleVariable{Nr,0}}...) where {Nr}
    idx = monomial_index(keys...)
    if idx ≤ length(d.values)
        val = @inbounds d.values[idx]
        isnan(val) || return val
    end
    return not_found()
end
function Base.get(d::MomentVector{R,R,Nr,0,<:AbstractSparseVector} where {R}, not_found,
    keys::Union{<:SimpleMonomialOrConj{Nr,0},<:SimpleVariable{Nr,0}}...) where {Nr}
    idx = monomial_index(keys...)
    ridx = searchsorted(rowvals(d.values), idx)
    isempty(ridx) && return not_found()
    @inbounds return nonzeros(d.values)[first(ridx)]
end
function Base.get(d::MomentVector{R,Complex{R},Nr,Nc}, not_found,
    keys::Union{<:SimpleMonomialOrConj{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {R,Nr,Nc}
    idx = monomial_index(keys...)
    idx > length(d.values) && return not_found()
    idx_c = monomial_index((m isa AbstractMonomial ? SimpleConjMonomial(m) : conj(m) for m in keys)...)
    if idx_c == idx
        val = @inbounds d.values[idx]
        return isnan(val) ? not_found() : Complex{R}(val)
    else
        @assert(idx_c ≤ length(d.values))
        idx_re, idx_im = minmax(idx, idx_c)
        @inbounds val_re, val_im = d.values[idx_re], d.values[idx_im]
        return isnan(val_re) || isnan(val_im) ? not_found() : Complex{R}(val_re, idx < idx_c ? val_re : -val_re)
    end
end
function Base.get(d::MomentVector{R,Complex{R},Nr,Nc,<:AbstractSparseVector}, not_found,
    keys::Union{<:SimpleMonomialOrConj{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {R,Nr,Nc}
    rv, nz = rowvals(d.values), nonzeros(d.values)
    idx = monomial_index(keys...)
    ridx = searchsorted(rv, idx)
    isempty(ridx) && return not_found()
    idx_c = monomial_index((m isa AbstractMonomial ? SimpleConjMonomial(m) : conj(m) for m in keys)...)
    if idx_c == idx
        val = @inbounds nz[first(ridx)]
        return isnan(val) ? not_found() : Complex{R}(val)
    else
        ridx_c = searchsorted(rv, idx_c)
        @assert(!isempty(ridx_c))
        idx_re, idx_im = minmax(idx, idx_c)
        @inbounds val_re, val_im = nz[idx_re], nz[idx_im]
        return isnan(val_re) || isnan(val_im) ? not_found() : Complex{R}(val_re, idx < idx_c ? val_re : -val_re)
    end
end
Base.iterate(d::MomentVector, args...) = iterate(d.values, args...)

"""
    Result

Result of a polynomial optimization, returned by calling [`poly_optimize`](@ref) on an [`AbstractRelaxation`](@ref).
A `Result` struct `r` contains information about
- the relaxation employed for the optimization (`r.relaxation`)
- the optimized problem (available via [`poly_problem`](@ref))
- the used method (`r.method`)
- the time required for the optimization in seconds (`r.time`)
- the status of the solver (`r.status`), which also depends on the solver type
- the returned primal value of the solver (`r.objective`), which, if the status was successful, is a lower bound to the true
  minimum
- the moment information in vector form (`r.moments`), which allows to construct a [moment matrix](@ref moment_matrix),
  extract solutions ([`poly_all_solutions`](@ref) or [`poly_solutions`](@ref)), and an
  [optimality certificate](@ref optimality_certificate).

This type is not exported.
"""
struct Result{R<:AbstractRelaxation,V,M<:MomentVector{<:Any,V}}
    relaxation::R
    method::Symbol
    time::Float64
    status
    objective::Float64
    moments::M
end

function Base.show(io::IO, ::MIME"text/plain", x::Result)
    println(io, "Polynomial optimization result")
    println(io, "Relaxation method: ", typeof(x.relaxation).name.name)
    println(io, "Used optimization method: ", x.method)
    println(io, "Status of the solver: ", x.status)
    println(io, "Lower bound to optimum (in case of good status): ", x.objective)
    println(io, "Time required for optimization: ",
        iszero(x.time) ? "less than one second" :
        (isone(x.time) ? "one second" : "$(x.time) seconds")
    )
end

Base.eltype(::Type{<:(Result{<:AbstractRelaxation,V})}) where {V} = V

"""
    poly_problem(r::Result)

Returns the problem that was associated with the optimization result.
"""
poly_problem(r::Result) = poly_problem(r.relaxation)