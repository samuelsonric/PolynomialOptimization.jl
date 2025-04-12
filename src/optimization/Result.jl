export issuccess

import LinearAlgebra: issuccess

"""
    MomentVector(relaxation::AbstractRelaxation, values::AbstractVector{R} where {R<:Real})

`MomentVector` is a representation of the result of a polynomial optimization. It contains all the values of the moments
that were present in the optimization problem. This vector can be indexed in two ways:
- linearly, which will just transparently yield what linearly indexing `values` would yield
- with a monomial (or multiple monomials, which means that the product of all the monomials is to be considered), which will
  yield the value that is associated with this monomial; if the problem was complex-valued, this will be a `Complex{R}`.
In order to get an association-like iterator, use [`MomentAssociation`](@ref).

This type is not exported.
"""
struct MomentVector{R,V<:Union{R,Complex{R}},Nr,Nc,D<:AbstractVector{R},E<:ExponentsAll} <: AbstractVector{V}
    e::E
    values::D

    function MomentVector(r::AbstractRelaxation{<:Problem{<:IntPolynomial{<:Any,Nr,Nc}}}, exponents::E, values::D) where {Nr,Nc,R,D<:AbstractVector{R},N,E<:ExponentsAll{N}}
        N == Nr + 2Nc || throw(MethodError(MomentVector, (r, exponents, values)))
        new{R,iszero(Nc) ? R : Complex{R},Nr,Nc,D,E}(exponents, values)
    end
end

MomentVector(r::AbstractRelaxation{<:Problem{<:IntPolynomial{<:Any,Nr,Nc,<:IntMonomialVector{Nr,Nc,I}}}}, ::Missing) where {Nr,Nc,I<:Integer} =
    MomentVector(r, ExponentsAll{Nr+2Nc,I}(), Float64[]) # this is the fallback if no data was provided by the solver

Base.length(d::MomentVector) = length(d.values)
Base.size(d::MomentVector) = (length(d.values),)
Base.haskey(d::MomentVector{<:Any,<:Any,Nr,Nc}, keys::Union{<:IntMonomialOrConj{Nr,Nc},<:IntVariable{Nr,Nc}}...) where {Nr,Nc} =
    monomial_index(d.e, keys...) ≤ length(d.values)
Base.haskey(d::MomentVector{<:Any,<:Any,Nr,Nc,<:AbstractSparseVector},
    keys::Union{<:IntMonomialOrConj{Nr,Nc},<:IntVariable{Nr,Nc}}...) where {Nr,Nc} =
    insorted(monomial_index(d.e, keys...), rowvals(d.values))
Base.getindex(d::MomentVector{<:Any,V,Nr,Nc}, keys::Union{<:IntMonomialOrConj{Nr,Nc},<:IntVariable{Nr,Nc}}...) where {Nr,Nc,V} =
    @inline get(d, () -> V(NaN), keys...)
Base.@propagate_inbounds Base.getindex(d::MomentVector, i::Integer) = d.values[i]
Base.@propagate_inbounds Base.setindex!(d::MomentVector, value, i::Integer) = setindex!(d.values, value, i)
function Base.get(d::MomentVector{R,R,Nr,0} where {R}, not_found,
    keys::Union{<:IntMonomialOrConj{Nr,0},<:IntVariable{Nr,0}}...) where {Nr}
    idx = monomial_index(d.e, keys...)
    if idx ≤ length(d.values)
        val = @inbounds d.values[idx]
        isnan(val) || return val
    end
    return not_found()
end
function Base.get(d::MomentVector{R,R,Nr,0,<:AbstractSparseVector{R}} where {R}, not_found,
    keys::Union{<:IntMonomialOrConj{Nr,0},<:IntVariable{Nr,0}}...) where {Nr}
    idx = monomial_index(d.e, keys...)
    ridx = searchsorted(rowvals(d.values), idx)
    isempty(ridx) && return not_found()
    @inbounds return nonzeros(d.values)[ridx[begin]]
end
function Base.get(d::MomentVector{R,Complex{R},Nr,Nc}, not_found,
    keys::Union{<:IntMonomialOrConj{Nr,Nc},<:IntVariable{Nr,Nc}}...) where {R,Nr,Nc}
    idx = monomial_index(d.e, keys...)
    idx > length(d.values) && return not_found()
    idx_c = monomial_index((m isa AbstractMonomial ? IntConjMonomial(m) : conj(m) for m in keys)...)
    if idx_c == idx
        val = @inbounds d.values[idx]
        return isnan(val) ? not_found() : Complex{R}(val)
    else
        @assert(idx_c ≤ length(d.values))
        idx_re, idx_im = minmax(idx, idx_c)
        @inbounds val_re, val_im = d.values[idx_re], d.values[idx_im]
        return isnan(val_re) || isnan(val_im) ? not_found() : Complex{R}(val_re, idx < idx_c ? val_im : -val_im)
    end
end
function Base.get(d::MomentVector{R,Complex{R},Nr,Nc,<:AbstractSparseVector{R}}, not_found,
    keys::Union{<:IntMonomialOrConj{Nr,Nc},<:IntVariable{Nr,Nc}}...) where {R,Nr,Nc}
    rv, nz = rowvals(d.values), nonzeros(d.values)
    idx = monomial_index(d.e, keys...)
    ridx = searchsorted(rv, idx)
    isempty(ridx) && return not_found()
    idx_c = monomial_index((m isa AbstractMonomial ? IntConjMonomial(m) : conj(m) for m in keys)...)
    if idx_c == idx
        val = @inbounds nz[ridx[begin]]
        return isnan(val) ? not_found() : Complex{R}(val)
    else
        ridx_c = searchsorted(rv, idx_c)
        @assert(!isempty(ridx_c))
        @inbounds idx_re, idx_im = minmax(ridx[begin], ridx_c[begin]) # rowvals is ascending, so this is transitive
        @inbounds val_re, val_im = nz[idx_re], nz[idx_im]
        return isnan(val_re) || isnan(val_im) ? not_found() : Complex{R}(val_re, idx < idx_c ? val_im : -val_im)
    end
end

Base.mapreduce(f, op, d::MomentVector; kw...) = mapreduce(f, op, d.values; kw...)
# not implemented in SparseArray for views (but we don't want to inject this, so let's just do it for MomentVector)
if !isdefined(SparseArrays, :SparseVectorPartialView) # requires SparseArrays 1.12
    const SparseVectorPartialView{Tv,Ti} = SubArray{Tv,1,<:AbstractSparseVector{Tv,Ti},<:Tuple{AbstractUnitRange},false}
    function _partialview_end_indices(x::SparseVectorPartialView)
        p = parent(x)
        nzinds = SparseArrays.nonzeroinds(p)
        if isempty(nzinds)
            last_idx = length(nzinds)
            first_idx = last_idx + 1
        else
            first_idx = findfirst(>=(x.indices[1][begin]), nzinds)
            last_idx = findlast(<=(x.indices[1][end]), nzinds)
            # empty view
            if first_idx === nothing || last_idx === nothing
                last_idx = length(nzinds)
                first_idx = last_idx+1
            end
        end
        return (first_idx, last_idx)
    end
    SparseArrays.nnz(x::SparseVectorPartialView) = 1 - (-(_partialview_end_indices(x)...,))
    function SparseArrays.nonzeros(x::SparseVectorPartialView)
        (first_idx, last_idx) = _partialview_end_indices(x)
        nzvals = nonzeros(parent(x))
        return view(nzvals, first_idx:last_idx)
    end
    function SparseArrays.nonzeroinds(x::SparseVectorPartialView)
        isempty(x.indices[1]) && return indtype(parent(x))[]
        (first_idx, last_idx) = _partialview_end_indices(x)
        nzinds = SparseArrays.nonzeroinds(parent(x))
        return @view(nzinds[first_idx:last_idx]) .- (x.indices[1][begin] - 1)
    end
    SparseArrays.indtype(x::SparseVectorPartialView) = indtype(parent(x))
end
function Base._mapreduce(f, op, ::IndexCartesian, A::SubArray{V,1,<:MomentVector{R,V,<:Any,<:Any,<:AbstractSparseVector{R}},
                                                              <:Tuple{AbstractUnitRange{<:Integer}},false}) where {R,V<:Union{R,Complex{R}}}
    isempty(A) && return Base.mapreduce_empty(f, op, V)
    d = parent(A)
    dv = view(d.values, A.indices...)
    z = nnz(dv)
    rest, ini = if z == 0
        length(dv)-z-1, f(zero(V))
    else
        length(dv)-z, Base.mapreduce_impl(f, op, nonzeros(dv), 1, z)
    end
    return SparseArrays._mapreducezeros(f, op, V, rest, ini)
end
# we just define rmul!, as it is used in the solution extraction. Of course, many others could be defined.
function LinearAlgebra.rmul!(A::SubArray{V,1,<:MomentVector{R,V,<:Any,<:Any,<:AbstractSparseVector{R}},
                                         <:Tuple{AbstractUnitRange{<:Integer}},false}, s::Number) where {R,V<:Union{R,Complex{R}}}
    d = parent(A)
    dv = view(d.values, A.indices...)
    rmul!(nonzeros(dv), s)
    return A
end

"""
    MomentAssociation(m::MomentVector)

Creates a associative iterator over the moment vector `m` that, upon iteration, returns `Pair`s assigning values to monomials.

This type is not exported.
"""
struct MomentAssociation{M<:MomentVector}
    d::M
end

Base.IteratorSize(::Type{<:MomentAssociation{<:(MomentVector{R,<:Union{R,Complex{R}},<:Any,0} where {R})}}) = Base.HasLength()
Base.IteratorSize(::Type{<:MomentAssociation}) = Base.SizeUnknown() # during iteration, we combine the real/complex parts
Base.eltype(::Type{MomentAssociation{M}}) where {R,V<:Union{R,Complex{R}},Nr,Nc,D<:AbstractVector{R},I<:Integer,
                                                 E<:AbstractExponents{<:Any,I},M<:MomentVector{R,V,Nr,Nc,D,E}} =
    Pair{IntMonomial{Nr,Nc,I,E},V}
Base.length(m::MomentAssociation{<:(MomentVector{R,<:Union{R,Complex{R}},<:Any,0} where {R})}) = length(m.d)
Base.get(m::MomentAssociation, args...) = get(m.d, args...)
function Base.iterate(m::MomentAssociation{<:MomentVector{R,<:Union{R,Complex{R}},Nr,Nc,<:AbstractVector{R},
                                                          <:ExponentsAll{<:Any,I}}},
    (i, rem, deg, degIncAt)=(1, length(m.d.values), 0, 2)) where {R,Nr,Nc,I<:Integer}
    d = m.d
    while true
        iszero(rem) && return nothing
        rem -= 1
        # Determining the degree for every single iteration anew is quite wasteful. We can make things simpler by reusing the
        # degree and storing the next index at which it is incremented.
        if i == degIncAt
            deg += 1
            degIncAt = length(ExponentsDegree{Nr+2Nc,I}(0, deg)) +1
        end
        mon = IntMonomial{Nr,Nc}(unsafe, d.e, IntPolynomials.unsafe_cast(I, i), deg)
        iszero(Nc) && @inbounds return Pair(mon, d.values[i]), (i +1, rem, deg, degIncAt)
        cmon = conj(mon)
        cmon.index == mon.index && @inbounds return Pair(mon, Complex(d.values[i])), (i +1, rem, deg, degIncAt)
        mon.index > cmon.index &&
            @inbounds return Pair(mon, Complex(d.values[cmon.index], -d.values[i])), (i +1, rem, deg, degIncAt)
        # skip over the conjugates
        i += 1
    end
end
function Base.iterate(m::MomentAssociation{<:MomentVector{R,<:Union{R,Complex{R}},Nr,Nc,<:AbstractSparseVector{R},
                                                          <:ExponentsAll{<:Any,I}}}) where {R,Nr,Nc,I<:Integer}
    d = m.d
    rv = rowvals(d.values)
    isempty(rv) && return nothing
    idx = 1
    rem = length(rv)
    deg = @inbounds degree_from_index(unsafe, d.e, rv[begin])
    degIncAt = length(ExponentsDegree{Nr+2Nc,I}(0, deg)) +1
    return iterate(m, (idx, rem, deg, degIncAt))
end
function Base.iterate(m::MomentAssociation{<:MomentVector{R,<:Union{R,Complex{R}},Nr,Nc,<:AbstractSparseVector{R},
                                                          <:ExponentsAll{<:Any,I}}},
    (idx, rem, deg, degIncAt)) where {R,Nr,Nc,I<:Integer}
    d = m.d
    rv, nz = rowvals(d.values), nonzeros(d.values)
    while true
        iszero(rem) && return nothing
        rem -= 1
        i = @inbounds rv[idx]
        # Determining the degree for every single iteration anew is quite wasteful. We can make things simpler by reusing the
        # degree and storing the next index at which it is incremented. Let's assume that we probably don't skip too many
        # complete degrees here, so that an incremental approach is the best.
        if i ≥ degIncAt
            counts = index_counts(unsafe, d.e)
            while i ≥ degIncAt
                deg += 1
                degIncAt = @inbounds(counts[deg+1, 1]) +1
            end
        end
        mon = IntMonomial{Nr,Nc}(unsafe, d.e, i, deg)
        iszero(Nc) && @inbounds return Pair(mon, nz[idx]), (idx +1, rem, deg, degIncAt)
        cmon = conj(mon)
        cmon.index == mon.index && @inbounds return Pair(mon, Complex(nz[idx])), (idx +1, rem, deg, degIncAt)
        if mon.index > cmon.index
            cidx = searchsorted(rv, cmon.index)
            @assert(!isempty(cidx))
            @inbounds return Pair(mon, Complex(nz[cidx[begin]], -nz[idx])), (idx +1, rem, deg, degIncAt)
        end
        # skip over the conjugates
        idx += 1
    end
end

moment_vector_type(::Type{Rx}, ::Type{V}) where {R<:Real,V<:Union{R,Complex{R}},Nr,Nc,I<:Integer,
                                                 Rx<:AbstractRelaxation{<:Problem{<:IntPolynomial{
                                                     <:Any,Nr,Nc,<:IntMonomialVector{Nr,Nc,I}
                                                 }}}} =
    MomentVector{R,V,Nr,Nc,<:Union{Vector{R},SparseVector{R,I},SubArray{R,1,Vector{R},Tuple{UnitRange{I}},true}},
                 ExponentsAll{Nr+2Nc,I}}

"""
    Result

Result of a polynomial optimization, returned by calling [`poly_optimize`](@ref) on an [`AbstractRelaxation`](@ref).
A `Result` struct `r` contains information about
- the relaxation employed for the optimization (`r.relaxation`)
- the optimized problem (available via [`poly_problem`](@ref poly_problem(::Result)))
- the used method (`r.method`)
- the time required for the optimization in seconds (`r.time`)
- the status of the solver (`r.status`), which also depends on the solver type. Use [`issuccess`](@ref) to check whether this
  is a successful status.
- the returned primal value of the solver (`r.objective`), which, if the status was successful, is a lower bound to the true
  minimum
- the moment information in vector form (`r.moments`), which allows to construct a [moment matrix](@ref moment_matrix),
  extract solutions ([`poly_all_solutions`](@ref) or [`poly_solutions`](@ref)), and an
  [optimality certificate](@ref optimality_certificate).

This type is not exported.
"""
mutable struct Result{Rx<:AbstractRelaxation,R<:Real,V<:Union{R,Complex{R}}}
    const relaxation::Rx
    const method::Symbol
    const time::Float64
    state
    const status
    const objective::R
    moments

    Result(relaxation::AbstractRelaxation{<:Problem{<:IntPolynomial{<:Any,<:Any,Nc}}}, method::Symbol, time::Float64,
        @nospecialize(state), @nospecialize(status), objective::R) where {R<:Real,Nc} =
        new{typeof(relaxation),R,iszero(Nc) ? R : Complex{R}}(relaxation, method, time, state, status, objective, missing)
end

function Base.getproperty(r::Result{Rx,R,V}, f::Symbol) where {Rx<:AbstractRelaxation,R<:Real,V<:Union{R,Complex{R}}}
    if f === :moments
        T = moment_vector_type(Rx, V)
        if ismissing(getfield(r, :moments))
            return setfield!(r, :moments, Solver.extract_moments(r.relaxation, r.state))::T
        else
            return getfield(r, f)::T
        end
    end
    return getfield(r, f)
end

function Base.show(io::IO, ::MIME"text/plain", x::Result)
    println(io, "Polynomial optimization result")
    println(io, "Relaxation method: ", typeof(x.relaxation).name.name)
    println(io, "Used optimization method: ", x.method)
    println(io, "Status of the solver: ", x.status)
    if !iszero(poly_problem(x).prefactor)
        println(io, "Lower bound to optimum (in case of good status): ", x.objective)
    else
        println(io, "Membership in the SOS cone was ", issuccess(x) ? "" : "not ", "certified")
    end
    println(io, "Time required for optimization: ",
        iszero(x.time) ? "less than one second" :
        (isone(x.time) ? "one second" : "$(x.time) seconds")
    )
end

Base.eltype(::Type{<:(Result{<:AbstractRelaxation,<:Real,V})}) where {V} = V

"""
    issuccess(r::Result)

Returns `true` if the solver successfully solved the relaxation and provided a solution, and `false` otherwise.

!!! info
    Solvers often do not have just a single "good" status code, but also "near successes". Whether they will return `true` or
    `false` is dependent on the implementation. The `status` field of the result is always available to get the original return
    value.
"""
issuccess(r::Result) = issuccess(Val(r.method), r.status)

"""
    poly_problem(r::Result)

Returns the problem that was associated with the optimization result.
"""
poly_problem(r::Result) = poly_problem(r.relaxation)