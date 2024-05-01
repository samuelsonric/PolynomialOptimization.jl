export mindex, add_nonnegative!, add_quadratic!, supports_quadratic, add_psd!, add_psd_complex!, supports_complex_psd,
    PSDIndextypeMatrixLinear, PSDIndextypeMatrixCartesian, PSDIndextypeVector, add_free_prepare!, add_free!,
    add_free_finalize!, fix_constraints!, SOSPSDIterable

"""
    mindex(state, monomials::SimpleMonomialOrConj...)

Calculates the index that the product of all monomials will have in the SDP represented by `state`.
The default implementation calculates the one-based monomial index according to a dense deglex order and returns an `UInt`.
Make sure that the return value of this function can always be inferred using `promote_op`.
The returned index is arbitrary as long as it is unique for the total monomial.
"""
@inline mindex(_, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc} =
    monomial_index(ExponentsAll{Nr+2Nc,UInt}(), monomials...)

"""
    add_nonnegative!(state, index, value::Real)

Add a nonnegative decision variable to the solver and put its value into the linear constraint with index (row in the linear
constraint matrix) `index`, which is of the type returned by [`mindex`](@ref), with coefficient `value`.
If this method is not implemented, a fallback to the more general vector-based version is used.
"""
@inline function add_nonnegative!(state, index, value::Real)
    add_nonnegative!(state, StackVec(index), StackVec(value))
    return
end

"""
    add_nonnegative!(state, indices::AbstractVector{T},
        values::AbstractVector{V}) where {T,V<:Real}

Same as above, but put the new decision variable in multiple constraints `indices`, whose eltype is of the type returned by
[`mindex`](@ref), with coefficients `values`.
"""
add_nonnegative!(::Any, ::AbstractVector{T}, values::AbstractVector{V}) where {T,V<:Real}

function add_quadratic!(::Val{:check}, state, index₁::T, value₁::V, index₂::T, value₂::V, rest::Tuple{T,V}...) where {T,V<:Real}
    if all(∘(iszero, last), rest)
        if iszero(value₁)
            iszero(value₂) || add_nonnegative!(state, index₂, value₂)
        elseif iszero(value₂)
            add_nonnegative!(state, index₁, value₁)
        else
            add_nonnegative!(state, index₁, value₁)
            add_nonnegative!(state, index₂, value₂)
        end
    elseif iszero(value₁)
        add_quadratic!(state, index₂, value₂, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    elseif iszero(value₂)
        add_quadratic!(state, index₁, value₁, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    else
        add_quadratic!(state, index₁, value₁, index₂, value₂, (restᵢ for restᵢ in rest if !iszero(restᵢ[2]))...)
    end
    return
end

function add_quadratic!(::Val{:check}, state, index₁::AbstractVector{T}, value₁::AbstractVector{V},
    index₂::AbstractVector{T}, value₂::AbstractVector{V}, rest::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}
    # value₁ and value₂ are constructed by pushing nonzero values into it
    # the rest is constructed by also potentially modifying entries
    if all(∘(Base.Fix1(all, iszero), last), rest)
        if isempty(value₁)
            isempty(value₂) || add_nonnegative!(state, index₂, value₂)
        elseif iszero(value₂)
            add_nonnegative!(state, index₁, value₁)
        else
            add_nonnegative!(state, index₁, value₁)
            add_nonnegative!(state, index₂, value₂)
        end
    elseif isempty(value₁)
        add_quadratic!(state, index₂, value₂, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    elseif isempty(value₂)
        add_quadratic!(state, index₁, value₁, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    else
        add_quadratic!(state, index₁, value₁, index₂, value₂, (restᵢ for restᵢ in rest if !all(iszero, restᵢ[2]))...)
    end
    return
end

@doc raw"""
    add_quadratic!(state, index₁::T, value₁::V, index₂::T, value₂::V,
        rest::Tuple{T,V}...) where {T,V<:Real}

Add three three decision variables satisfying ``x_1, x_2 \geq 0``, ``x_1 \cdot x_2 \geq x_3^2``, corresponding to the
semidefinite constraint ``\left(\begin{smallmatrix} x_1 & x_3 \\ x_3 & x_2 \end{smallmatrix}\right) \succeq 0``) to the solver
and put their values into the linear constraints (rows in the linear constraint matrix) indicated by the indices, which are of
the type returned by [`mindex`](@ref), with coefficients given by the values (which are all the same unless the constraint
comes from a PSD matrix).
If this method is not implemented, a fallback to the more general vector-based version is used.
!!! note "rest vs. x₃"
    In the real-valued case, `rest` is a simple tuple corresponding to `index₃` and `value₃`; however, in the complex-valued
    case, it will be two tuples; the equivalent transformation is
    ``x_1 \cdot x_2 \geq \operatorname{Re}^2(x_3) + \operatorname{Im}^2(x_3)``, where the decomposition into real and imaginary
    parts is already done - they should be created as different real variables.
    `rest` will always have either one or two elements.
"""
add_quadratic!(state, index₁::T, value₁::V, index₂::T, value₂::V, rest::Tuple{T,V}...) where {T,V<:Real} =
    add_quadratic!(state, StackVec(index₁), StackVec(value₁), StackVec(index₂), StackVec(value₂),
        ((StackVec(indexᵣ), StackVec(valueᵣ)) for (indexᵣ, valueᵣ) in rest)...)

"""
    add_quadratic!(state, indices₁::AbstractVector{T}, values₁::AbstractVector{V},
        indices₂::AbstractVector{T}, values₂::AbstractVector{V},
        rest::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Same as above, but put the new decision variables in multiple constraints indicated by the indices, whose eltypes are of the
type returned by [`mindex`](@ref), with coefficients given by the values (which most likely are all the same, but as the
indices are guaranteed to be duplicate-free, the values might need to compensate for this).
If this method is not implemented (if the solver does not support quadratic constraints), a fallback to a 2x2 PSD constraint is
used (but in this case, consider overwriting the single-index version of this function to provide a more efficient fallback).
"""
add_quadratic!(::Any, ::AbstractVector{T}, ::AbstractVector{V}, ::AbstractVector{T}, ::AbstractVector{V},
    ::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

"""
    add_quadratic!(state, index₊::T, value₊::V,
        rest_free::Tuple{T,V}...) where {T,V<:Real}
    add_quadratic!(state, indices₊::AbstractVector{T}, values₊::AbstractVector{V},
        rest_free::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Special case of a quadratic constraint of side dimension 2 which has been simplified to x₊ ≥ 0, rest_free ∈ ℝ. If the scalar
method is not implemented, it falls back to the vector method.

!!! warning
    These functions will only be called if [`supports_quadratic`](@ref) is defined to return `true` for the given state.
"""
add_quadratic!(state, index₊::T, value₊::V, rest_free::Tuple{T,V}...) where {T,V<:Real} =
    add_quadratic!(state, StackVec(index₊), StackVec(value₊),
        ((StackVec(indexᵣ), StackVec(valueᵣ)) for (indexᵣ, valueᵣ) in rest_free)...)

add_quadratic!(state, indices₊::AbstractVector{T}, values₊::AbstractVector{V},
    rest_free::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real} =
    add_quadratic!(state, indices₊, values₊, StackVec{T}(), StackVec{V}(), rest_free...)

"""
    supports_quadratic(state)

Indicates whether the solver can deal with quadratic constraints of the form ``x_1 x_2 \\geq \\sum_i y_i^2``.
[`add_quadratic!`](@ref) will only be called if this method returns `true`; else, quadratic constraints will also be modeled
using the semidefinite cone.
The default implementation returns `false`.
"""
supports_quadratic(_) = false

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

function add_psd! end

"""
    add_psd!(state, dim::Int,
        data::Dict{FastKey{T},<:Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Real}

Add a PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the return value
of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear constraint
matrix) indicated by the keys of the `data` dictionary, which are of the type returned by [`mindex`](@ref), with coefficients
given by their values (maybe scaled).
Note that if [`add_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns an [`AbstractPSDIndextypeMatrix`](@ref).
For [`PSDIndextypeMatrixLinear`](@ref), there is one index in the value-tuple of the dictionary; for
[`PSDIndextypeMatrixCartesian`](@ref), there are two.

!!! hint "FastKey"
    The key of the dictionary actually is a `FastKey{T}`, where `T` is the type returned by [`mindex`](@ref). `FastKey` can be
    converted to `T` and circumvents Julia's long-winded hash calculations for integers.

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_psd_complex!`](@ref) must be implemented.
"""
add_psd!(::Any, ::Int, ::Dict{FastKey{T},<:Tuple{AbstractVector{I},AbstractVector{V}}}) where {T,I,V<:Real}

"""
    add_psd!(state, dim::Int, data::SOSPSDIterable{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of constraint indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their coefficients.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_psd_complex!`](@ref) must be implemented.
"""
add_psd!(::Any, ::Int, ::SOSPSDIterable{T,V}) where {T,V<:Real}

function add_psd_complex! end

"""
    add_psd_complex!(state, dim::Int,
        data::Dict{FastKey{T},Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Complex}

Add a Hermitian PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the
return value of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear
constraint matrix) indicated by the keys of the `data` dictionary, which are of the type returned by [`mindex`](@ref), with
coefficients given by their values. The real part of the coefficient corresponds to the coefficient in front of the real part
of the matrix entry, the imaginary part is the coefficient for the imaginary part of the matrix entry.
Note that if [`add_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns an [`AbstractPSDIndextypeMatrix`](@ref).
For [`PSDIndextypeMatrixLinear`](@ref), there is one index in the value-tuple of the dictionary; for
[`PSDIndextypeMatrixCartesian`](@ref), there are two.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_psd_complex!(::Any, ::Int, ::Dict{FastKey{T},Tuple{AbstractVector{I},AbstractVector{V}}}) where {T,I,V<:Complex}

"""
    add_psd_complex!(state, dim::Int, data::SOSPSDIterable{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of constraint indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their weights.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_psd_complex!(::Any, ::Int, ::SOSPSDIterable{T,V}) where {T,V<:Real}

"""
    AbstractPSDIndextype

Abstract class for all supported types in which a solver can represent a PSD matrix.

See also [`AbstractPSDIndextypeMatrix`](@ref), [`PSDIndextypeMatrixLinear`](@ref), [`PSDIndextypeMatrixCartesian`](@ref),
[`PSDIndextypeVector`](@ref).
"""
abstract type AbstractPSDIndextype end

"""
    AbstractPSDIndextypeMatrix <: AbstractPSDIndextype

Abstract class for a solver that implements PSD matrix constraints by using a monolithic PSD matrix variable from which values
can then be extracted.

See also [`PSDIndextypeMatrixLinear`](@ref), [`PSDIndextypeMatrixCartesian`](@ref).
"""
abstract type AbstractPSDIndextypeMatrix{I<:Integer,Tri,Offset} <: AbstractPSDIndextype end

"""
    PSDIndextypeMatrixLinear(Indextype, triangle, offset) <: AbstractPSDIndextypeMatrix

Entries from a PSD matrix variable are obtained by using a linear index of type `Indextype`. This index represents one triangle
of the matrix (the lower if `triangle === :L`, the upper if `triangle === :U`), taken in a col-major way. The first entry has
the index `offset`, typically either `zero(Indextype)` or `one(Indextype)`.

!!! info
    Note that while only one triangle is indexed, it is assumed that the solver will by default double the values associated
    with the off-diagonal elements, i.e., the value will be half of what it would be if the corresponding index was addressed
    in a full matrix.
    This corresponds to the typical behavior of solvers that expose PSD variables and allow accessing elements in them via
    sparse symmetric matrices where only one triangle is given, but the other half is implicit.

See also [`PSDIndextypeMatrixCartesian`](@ref).
"""
struct PSDIndextypeMatrixLinear{I<:Integer,Tri,Offset} <: AbstractPSDIndextypeMatrix{I,Tri,Offset}
    function PSDIndextypeMatrixLinear(Indextype::Type{I}, triangle::Symbol, offset::I) where {I<:Integer}
        triangle ∈ (:L, :U) || throw(MethodError(PSDIndextypeMatrixLinear, (Indextype, triangle, offset)))
        new{Indextype,triangle,offset}()
    end
end

"""
    PSDIndextypeMatrixCartesian(Indextype, triangle, offset) <: AbstractPSDIndextypeMatrix

Entries from a PSD matrix variable are obtained by using a cartesian index of two integers of type `Indextype`. This index
represents one triangle of the matrix (the lower if `triangle === :L`, the upper if `triangle === :U`). The first entry has the
index `(offset, offset)`, typically either `zero(Indextype)` or `one(Indextype)`.

!!! info
    Note that while only one triangle is indexed, it is assumed that the solver will by default double the values associated
    with the off-diagonal elements, i.e., the value will be half of what it would be if the corresponding index was addressed
    in a full matrix.
    This corresponds to the typical behavior of solvers that expose PSD variables and allow accessing elements in them via
    sparse symmetric matrices where only one triangle is given, but the other half is implicit.

See also [`PSDIndextypeMatrixLinear`](@ref).
"""
struct PSDIndextypeMatrixCartesian{I<:Integer,Tri,Offset} <: AbstractPSDIndextypeMatrix{I,Tri,Offset}
    function PSDIndextypeMatrixCartesian(Indextype::Type{I}, triangle::Symbol, offset::I) where {I<:Integer}
        triangle ∈ (:L, :U) || throw(MethodError(PSDIndextypeMatrixCartesian, (Indextype, triangle, offset)))
        new{Indextype,triangle,offset}()
    end
end

"""
    PSDIndextypeVector(triangle) <: AbstractPSDIndextype

The solver implements PSD matrix constraints by demanding that the matrixization of a vector of decision variables be PSD.
If `triangle === :F`, the vector is formed by stacking all the columns of the matrix.
If `triangle === :L`, the columns of the lower triangle are assumed to be stacked _and scaled_, i.e., off-diagonal variables
that enter the cone are implicitly multiplied by ``1 / \\sqrt2`` in the matrix; so the coefficients will already be
premultiplied by ``\\sqrt2``.
If `triangle === :U`, the columns of the upper triangle are assumed to be stacked and scaled.

See also [`AbstractPSDIndextypeMatrix`](@ref).
"""
struct PSDIndextypeVector{Tri} <: AbstractPSDIndextype
    function PSDIndextypeVector(triangle::Symbol)
        triangle ∈ (:L, :U, :F) || throw(MethodError(PSDIndextypeVector, (triangle,)))
        new{triangle}()
    end
end

"""
    psd_indextype(state)

This function must indicate in which format the solver expects its data for PSD variables. The return type must be an instance
of an [`AbstractPSDIndextype`](@ref) subtype.

See also [`PSDIndextypeMatrixLinear`](@ref), [`PSDIndextypeMatrixCartesian`](@ref), [`PSDIndextypeVector`](@ref).
"""
function psd_indextype end

@doc raw"""
    supports_complex_psd(state)

This function indicates whether the solver natively supports a complex-valued PSD cone. If it returns `false` (default), the
complex-valued PSD constraints will be rewritten into real-valued PSD constraints (using the standard
``\left(\begin{smallmatrix} \operatorname{Re} & -\operatorname{Im} \\ \operatorname{Im} & \operatorname{Re} \end{smallmatrix}\right)``
form for the vector interface and the [double-dualization technique](https://doi.org/10.48550/arXiv.2307.11599) for the matrix
interface, so no additional equality constraints are required in either case); this is completely transparent for the solver.
If the function returns `true`, the solver must additionally implement [`add_psd_complex!`](@ref).
"""
supports_complex_psd(_) = false

"""
    add_free_prepare!(state, num::Int)

Prepares to add no more than `num` free variables to `state`; the actual data is then put into the solver by subsequent calls
of [`add_free!`](@ref) and the whole transaction is completed by [`add_free_finalize!`](@ref).
The return value of this function is passed on as `eqstate` to [`add_free!`](@ref).
The default implementation does nothing.
"""
add_free_prepare!(_, _) = nothing

"""
    add_free!(state, eqstate, indices::AbstractVector{T}, values::AbstractVector{V},
        obj::Bool) where {T,V<:Real}

Add a free variable to the solver and put its value into the linear constraints with indices (rows in the linear constraint
matrix) `indices`, whose eltype is of the type returned by [`mindex`](@ref), and with coefficients `values`.
If `obj` is `true`, the variable should also be put into the objective with coefficient `1`.
The parameter `eqstate` is, upon first call, the value returned by [`add_free_prepare!`](@ref); and on all further calls, it
will be the return value of the previous call.
"""
function add_free! end

"""
    add_free_finalize!(state, eqstate)

Finishes the addition of free variables to `state`; the value of `eqstate` is the return value of the last call to
[`add_free!`](@ref). Note that less variables than initially requested might have been added; in this case, the implementation
might decide to do some bookkeeping.
The default implementation does nothing.
"""
add_free_finalize!(_, _) = nothing

"""
    fix_constraints!(state, indices::Vector{T}, values::Vector{V}) where {T,V<:Real}

Ensures that all constraints in the optimization problem that appear in `indices` are fixed to their entries in `values`, while
all the rest is fixed to zero. The eltype of `indices` if the one returned by [`mindex`](@ref).
This function will be called exactly once by [`sos_setup!`](@ref) after all constraints have been set up.
"""
function fix_constraints! end