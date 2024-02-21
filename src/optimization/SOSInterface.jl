"""
    sos_solver_mindex(state, monomials::SimpleMonomial...)

Calculates the index that the product of all monomials will have in the SDP represented by `state`.
The default implementation calculates one-based monomial index according to a dense deglex order and returns an Int.
Make sure that the return value of this function can always be inferred using promote_op.
The returned index is arbitrary as long as it satisfies the following condition: If `iscanonical(m)`, then
`sos_solver_mindex(m) ≤ sos_solver_mindex(conj(m))`.
"""
@inline sos_solver_mindex(_, monomials::SimpleMonomial...) = monomial_index(monomials...)

"""
    sos_solver_add_scalar!(state, index, value::Real)

Put a SOS constraint of side dimension 1 (one new nonnegative decision variable) at the monomial position indicated by `index`
(which is of the type returned by [`sos_solver_mindex`](@ref)) and weighted by `value`.
If this method is not implemented, a fallback to the more general vector-based version is used.

    sos_solver_add_scalar!(state, indices::AbstractVector{T},
        values::AbstractVector{V}) where {T,V<:Real}

Put a SOS constraint of side dimension 1 at the monomial positions indicated by `indices` (whose eltype is of the type returned
by [`sos_solver_mindex`](@ref)) and weighted by `values`.
"""
@inline function sos_solver_add_scalar!(state, index, value::Real)
    sos_solver_add_scalar!(state, StackVec(index), StackVec(value))
    return
end

@doc raw"""
    sos_solver_add_quadratic!(state, index₁::T, value₁::V, index₂::T, value₂::V,
        rest::Tuple{T,V}...) where {T,V<:Real}

Put a SOS constraint of side dimension 2 (three decision variables satisfying (``x_1 \cdot x_2 \geq x_3^2``, corresponding to
the semidefinite constraint ``\begin{pmatrix} x_1 & x_3 \\ x_3 x_2 \end{pmatrix} \succeq 0``) at the monomial position
indicated by the indices (which are of the type returned by [`sos_solver_mindex`](@ref)) and weighted by their values (which
are all the same unless the constraint comes from a PSD matrix). If this method is not implemented, a fallback to the more
general vector-based version is used.
!!! note "`rest` vs. ``x_3``"
    In the real-valued case, `rest` is a simple tuple corresponding to `index₃` and `value₃`; however, in the complex-valued
    case, it will be two tuples; the equivalent transformation is
    ``x_1 \cdot x_2 \geq \operatorname{Re}^2(x_3) + \operatorname{Im}^2(x_3)``, where the decomposition into real and imaginary
    parts is already done - they should be created as different real variables.
    `rest` will always have either one or two elements.

    sos_solver_add_quadratic!(state, indices₁::AbstractVector{T}, values₁::AbstractVector{V},
        indices₂::AbstractVector{T}, values₂::AbstractVector{V},
        rest::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Put a SOS constraint of side dimension 2 at the monomial positions indicated by the indices (whose eltypes are of the type
returned by [`sos_solver_mindex`](@ref)) and weighted by their values (which most likely are all the same, but as the indices
are guaranteed to be duplicate-free, the values might need to compensate for this).
If this method is not implemented (if the solver does not support quadratic constraints), a fallback to a 2x2 PSD constraint is
used (but in this case, consider overwriting the single-index version of this function to provide a more efficient fallback).

    sos_solver_add_quadratic!(state, index₊::T, value₊::V,
        rest_free::Tuple{T,V}...) where {T,V<:Real}
    sos_solver_add_quadratic!(state, indices₊::AbstractVector{T}, values₊::AbstractVector{V},
        rest_free::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Special case of a SOS constraint of side dimension 2 which has been simplified to x₊ ≥ 0, rest_free ∈ ℝ. If the scalar method
is not implemented, it falls back to the vector method.

!!! warning
    These functions will only be called if [`sos_solver_supports_quadratic`](@ref) is defined to return `true` for the given
    state.
"""
function sos_solver_add_quadratic!(::Val{:check}, state, index₁::T, value₁::V, index₂::T, value₂::V, rest::Tuple{T,V}...) where {T,V<:Real}
    if all(∘(iszero, last), rest)
        if iszero(value₁)
            iszero(value₂) || sos_solver_add_scalar!(state, index₂, value₂)
        elseif iszero(value₂)
            sos_solver_add_scalar!(state, index₁, value₁)
        else
            sos_solver_add_scalar!(state, index₁, value₁)
            sos_solver_add_scalar!(state, index₂, value₂)
        end
    elseif iszero(value₁)
        sos_solver_add_quadratic!(state, index₂, value₂, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    elseif iszero(value₂)
        sos_solver_add_quadratic!(state, index₁, value₁, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    else
        sos_solver_add_quadratic!(state, index₁, value₁, index₂, value₂, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    end
    return
end

function sos_solver_add_quadratic!(::Val{:check}, state, index₁::AbstractVector{T}, value₁::AbstractVector{V},
    index₂::AbstractVector{T}, value₂::AbstractVector{V}, rest::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}
    # value₁ and value₂ are constructed by pushing nonzero values into it
    # the rest is constructed by also potentially modifying entries
    if all(∘(Base.Fix1(all, iszero), last), rest)
        if isempty(value₁)
            isempty(value₂) || sos_solver_add_scalar!(state, index₂, value₂)
        elseif iszero(value₂)
            sos_solver_add_scalar!(state, index₁, value₁)
        else
            sos_solver_add_scalar!(state, index₁, value₁)
            sos_solver_add_scalar!(state, index₂, value₂)
        end
    elseif isempty(value₁)
        sos_solver_add_quadratic!(state, index₂, value₂, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    elseif isempty(value₂)
        sos_solver_add_quadratic!(state, index₁, value₁, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    else
        sos_solver_add_quadratic!(state, index₁, value₁, index₂, value₂, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    end
    return
end

sos_solver_add_quadratic!(state, index₁::T, value₁::V, index₂::T, value₂::V, rest::Tuple{T,V}...) where {T,V<:Real} =
    sos_solver_add_quadratic!(state, StackVec(index₁), StackVec(value₁), StackVec(index₂), StackVec(value₂),
        ((StackVec(indexᵣ), StackVec(valueᵣ)) for (indexᵣ, valueᵣ) in rest)...)

sos_solver_add_quadratic!(state, index₊::T, value₊::V, rest_free::Tuple{T,V}...) where {T,V<:Real} =
    sos_solver_add_quadratic!(state, StackVec(index₊), StackVec(value₊),
        ((StackVec(indexᵣ), StackVec(valueᵣ)) for (indexᵣ, valueᵣ) in rest_free)...)

sos_solver_add_quadratic!(state, indices₊::AbstractVector{T}, values₊::AbstractVector{V},
    rest_free::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real} =
    sos_solver_add_quadratic!(state, indices₊, values₊, StackVec{T}(), StackVec{V}(), rest_free...)

"""
    sos_solver_supports_quadratic(state)

Indicates whether the solver can deal with quadratic constraints of the form `x₁ x₂ ≥ ∑ᵢ yᵢ²`.
[`sos_solver_add_quadratic!`](@ref) will only be called if this method returns `true`; else, quadratic constraints will also be
modeled using the semidefinite cone.
The default implementation returns `false`.
"""
sos_solver_supports_quadratic(_) = false

"""
    sos_solver_add_psd!(state, dim::Int,
        data::Dict{FastKey{T},<:Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Real}

Put a SOS constraint of side dimension `dim` ≥ 3 (a PSD variable whose requested triangle is indexed according to the return
value of [`sos_solver_psd_indextype`](@ref)) at the monomial positions indicated by the keys of the `data` dictionary (which
are of the type returned by [`sos_solver_mindex`](@ref)) and weighted by their values (maybe scaled).
Note that if [`sos_solver_add_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`sos_solver_psd_indextype`](@ref) returns a 3-tuple. The first index of the tuple determines if there
is one or there are two indices in the value-tuple of the dictionary; the second index determines which triangle is given; and
the third index determines the offset of the index/indices in the value-tuple.

!!! hint "`FastKey`"
    The key of the dictionary actually is a `FastKey{T}`, where `T` is the type returned by [`sos_solver_mindex`](@ref). The
    non-exported `FastKey` can be converted to `T` and circumvents Julia's long-winded hash calculations for integers.
    Note that `FastKey` is not exported.

    sos_solver_add_psd!(state, dim::Int, data::SOSPSDIterable{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of monomial positions (which are of the type returned by
[`sos_solver_mindex`](@ref)) and an `AbstractVector` of their weights.
This method is called if [`sos_solver_psd_indextype`](@ref) returns a single symbol, which determines the travelling order.

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`sos_solver_supports_complex_psd`](@ref)
    returns `false`. The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`sos_solver_add_psd_complex!`](@ref) must be implemented.
"""
function sos_solver_add_psd! end

"""
    sos_solver_add_psd_complex!(state, dim::Int,
        data::Dict{FastKey{T},Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Complex}

Put a complex-valued SOS constraint of side dimension `dim` ≥ 3 (a Hermitian PSD variable whose requested triangle is indexed
according to the return value of [`sos_solver_psd_indextype`](@ref)) at the monomial positions indicated by the keys of the
`data` dictionary (which are of the type returned by [`sos_solver_mindex`](@ref)) and weighted by their values. The real part
of the weight corresponds to the coefficient in front of the real part of the matrix entry, the imaginary part is the
coefficient for the imaginary part of the matrix entry.
Note that if [`sos_solver_add_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`sos_solver_psd_indextype`](@ref) returns a 3-tuple. The first index of the tuple determines if there
is one or there are two indices in the value-tuple of the dictionary; the second index determines which triangle is given; and
the third index determines the offset of the index/indices in the value-tuple.

    sos_solver_add_psd!(state, dim::Int, data::SOSPSDIterable{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of monomial positions (which are of the type returned by
[`sos_solver_mindex`](@ref)) and an `AbstractVector` of their weights.
This method is called if [`sos_solver_psd_indextype`](@ref) returns a single symbol, which determines the travelling order.
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the weights are real-valued.

!!! warning
    This function will only be called if [`sos_solver_supports_complex_psd`](@ref) is defined to return `true` for the given
    state.
"""
function sos_solver_add_psd_complex! end

struct SOSPSDIterable{T,V,L}
    indices::Vector{T}
    values::Vector{V}
    lens::L

    function SOSPSDIterable(indices::Vector{T}, values::Vector{V}, len::I) where {T,V,I<:Integer}
        length(indices) == length(values) || error("Invalid SOSPSDIterable construction")
        new{T,V,I}(indices, values, len)
    end
    function SOSPSDIterable(indices::Vector{T}, values::Vector{V}, lens::L) where {T,V,L<:AbstractVector{<:Integer}}
        length(indices) == length(values) || error("Invalid SOSPSDIterable construction")
        (isempty(lens) || length(indices) != sum(lens, init=0)) && error("Invalid SOSPSDIterable construction")
        new{T,V,L}(indices, values, lens)
    end
end

Base.IteratorSize(::Type{<:SOSPSDIterable}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SOSPSDIterable}) = Base.HasEltype()
Base.eltype(::Type{SOSPSDIterable{T,V,<:Any}}) where {T,V} =
    Tuple{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true},SubArray{V,1,Vector{V},Tuple{UnitRange{Int}},true}}
Base.length(spi::SOSPSDIterable{<:Any,<:Any,<:Integer}) = length(spi.indices) ÷ spi.lens
Base.length(spi::SOSPSDIterable{<:Any,<:Any,<:AbstractVector{<:Integer}}) =
    length(spi.indices) * length(spi.lens) ÷ sum(spi.lens, init=0)
@inline function Base.iterate(spi::SOSPSDIterable{<:Any,<:Any,<:Integer}, state=1)
    endpos = state + spi.lens -1
    if endpos ≤ length(spi.indices)
        @inbounds return (view(spi.indices, state:endpos), view(spi.values, state:endpos)), endpos +1
    else
        return nothing
    end
end
@inline function Base.iterate(spi::SOSPSDIterable{<:Any,<:Any,<:AbstractVector{<:Integer}}, state=(1, 1))
    endpos = @inbounds(state[1] + spi.lens[state[2]] -1)
    if endpos ≤ length(spi.indices)
        @inbounds return (view(spi.indices, state:endpos), view(spi.values, state:endpos)), (endpos +1, state[2] +1)
    else
        return nothing
    end
end
SparseArrays.rowvals(spi::SOSPSDIterable) = spi.indices
SparseArrays.nonzeros(spi::SOSPSDIterable) = spi.values
Base.index_lengths(spi::SOSPSDIterable{<:Any,<:Any,<:Integer}) = Iterators.repeated(spi.lens, length(spi))
Base.index_lengths(spi::SOSPSDIterable{<:Any,<:Any,<:AbstractVector{<:Integer}}) = spi.lens

"""
    sos_solver_psd_indextype(state)

This function must indicate how the lower triangle of a PSD variable is indexed; this is either a 3-tuple
- with the first index corresponding to the type, which must either be a scalar Integer type (linear indexing) or a subtype of
  `Tuple{Integer,Integer}` (Cartesian indexing)
- with the second index being either `:L` or `:U`
- with the third index being an integer of the appropriate type that specifies an offset, typically either 0 or 1
or it is a single symbol: either `:L`, `:U`, or `:F`.

!!! info "Meaning of the second index (first case)"
    The second index specifies whether the upper or lower triangle is indexed. In the case of linear indexing, this is always
    col-major. Note that while only one triangle is indexed, it is assumed that the solver will by default double the values
    associated with the off-diagonal elements, i.e., the value will be half of what it would be if the corresponding index was
    addressed in a full matrix.
    This corresponds to the typical behavior of solvers that expose PSD variables and allow accessing elements in them via
    sparse symmetric matrices where only one triangle is given, but the other half is implicit.

!!! info "Meaning of the symbol (second case)"
    The symbol specifies whether the upper or lower triangle or the full matrix is indexed in a col-major fashion.
    Note that for the triangles, it is assumed that the solver expects the data in scaled form, where off-diagonal variables
    that enter the cone are implicitly multiplied by ``\\frac{1}{\\sqrt2}`` in the matrix; so the weights will be premultiplied
    by ``\\sqrt2``.
"""
function sos_solver_psd_indextype end

"""
    sos_solver_supports_complex_psd(state)

This function indicates whether the solver natively supports a complex-valued PSD cone. If it returns `false` (default), the
complex-valued PSD constraints will be rewritten into real-valued PSD constraints (using the
[double-dualization technique](https://doi.org/10.48550/arXiv.2307.11599) that does not require additional constraints); this
is completely transparent for the solver.
If the function returns `true`, the solver must additionally implement [`sos_solver_add_psd_complex!`](@ref).
"""
sos_solver_supports_complex_psd(_) = false

"""
    sos_solver_add_free_prepare!(state, num::Int)

Prepares to add no more than `num` free variables (corresponding to polynomial equality constraints) to `state`; the actual
data is then put into the problem by subsequent calls by [`sos_solver_add_free!`](@ref) and the whole transaction is completed
by [`sos_solver_add_free_finalize!`](@ref).
The return value of this function is passed on as `eqstate` to both functions, and will also be mutated by
[`sos_solver_add_free!`](@ref).
The default implementation does nothing.
"""
sos_solver_add_free_prepare!(_, _) = nothing

"""
    sos_solver_add_free!(state, eqstate, indices::AbstractVector{T}, values::AbstractVector{V},
        obj::Bool) where {T,V<:Real}

Put a free variable at the monomial positions indicated by `indices` (whose eltype is of the type returned by
[`sos_solver_mindex`](@ref)) and weighted by `values.` If `obj` is `true`, the variable should also be put into the objective
with coefficient `1`.
The parameter `eqstate` is, upon first call, the value returned by [`sos_solver_add_free_prepare!`](@ref); and on all further
calls, it will be the return value of the previous call.
"""
function sos_solver_add_free! end
"""
    sos_solver_add_free_finalize!(state, eqstate)

Finishes the addition of free variables to `state`; the value of `eqstate` is the return value of the last call to
[`sos_solver_add_free!`](@ref). Note that less variables than initially requested might have been added; in this case, the
implementation might decide to delete the unused ones or just leave them as they are.
The default implementation does nothing.
"""
sos_solver_add_free_finalize!(_, _) = nothing

"""
    sos_solver_fix_constraints!(state, indices::Vector{T},
        values::Vector{V}) where {T,V<:Real}

Ensures that all constraints in the optimization problem that appear in `indices` are fixed to their entries in `values`, while
all the rest is fixed to zero. The eltype of `indices` if the one returned by [`sos_solver_mindex`](@ref).
This function will be called exactly once after all constraints have been set up.
"""
function sos_solver_fix_constraints! end