export mindex, supports_quadratic, supports_complex_psd, psd_indextype,
    PSDIndextypeMatrixCartesian, PSDMatrixCartesian, PSDIndextypeVector, PSDVector

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
    supports_quadratic(state)

Indicates whether the solver can deal with quadratic constraints of the form ``2x_1 x_2 \\geq \\sum_i y_i^2``.
[`add_var_quadratic!`](@ref) or [`add_constr_quadratic!`](@ref) will only be called if this method returns `true`; else,
quadratic constraints will also be modeled using the semidefinite cone.
The default implementation returns `false`.
"""
supports_quadratic(_) = false

@doc raw"""
    supports_complex_psd(state)

This function indicates whether the solver natively supports a complex-valued PSD cone. If it returns `false` (default), the
complex-valued PSD constraints will be rewritten into real-valued PSD constraints (using the standard
``\left(\begin{smallmatrix} \operatorname{Re} & -\operatorname{Im} \\ \operatorname{Im} & \operatorname{Re} \end{smallmatrix}\right)``
form for the vector interface and the [double-dualization technique](https://doi.org/10.48550/arXiv.2307.11599) for the matrix
interface, so no additional equality constraints are required in either case); this is completely transparent for the solver.
If the function returns `true`, the solver must additionally implement [`add_var_psd_complex!`](@ref) and
[`add_constr_psd_complex!`](@ref).
"""
supports_complex_psd(_) = false

"""
    AbstractPSDIndextype{Tri}

Abstract class for all supported types in which a solver can represent a PSD matrix.

See also [`PSDIndextypeMatrixCartesian`](@ref), [`PSDIndextypeVector`](@ref).
"""
abstract type AbstractPSDIndextype{Tri} end

"""
    PSDIndextypeMatrixCartesian(triangle, offset) <: AbstractPSDIndextype

The solver implements PSD matrix constraints by using a monolithic PSD matrix variable.
Entries from the variable are obtained by using a cartesian index of two integers of the return type of [`mindex`](@ref). This
index represents one triangle of the matrix (the lower if `triangle === :L`, the upper if `triangle === :U`). The first entry
has the index `(offset, offset)`, typically either `0` or `1`.

!!! info
    Note that while only one triangle is indexed, it is assumed that the solver will by default populate the other triangle in
    a completely symmetric way.
    This corresponds to the typical behavior of solvers that expose PSD variables and allow accessing elements in them via
    sparse symmetric matrices where only one triangle is given, but the other half is implicit.

See also [`PSDMatrixCartesian`](@ref).
"""
struct PSDIndextypeMatrixCartesian{Tri,Offset} <: AbstractPSDIndextype{Tri}
    function PSDIndextypeMatrixCartesian(triangle::Symbol, offset::Integer)
        triangle ∈ (:L, :U) || throw(MethodError(PSDIndextypeMatrixCartesian, (triangle, offset)))
        new{triangle,offset}()
    end
end

_get_offset(::PSDIndextypeMatrixCartesian{<:Any,Offset}) where {Offset} = Offset

"""
    PSDMatrixCartesian

An iterable that returns matrix elements of a PSD cone.
Iterating through it will give `Pair`s with keys that contain the [`mindex`](@ref) of the monomial, and a 3-Tuple of
`AbstractVector`s as the values which contain the row and column indices together with their coefficients to describe where
the monomial appears.

See also [`PSDIndextypeMatrixCartesian`](@ref).
"""
struct PSDMatrixCartesian{T,V,Tri,Offset}
    rowind::Vector{T}
    colind::Vector{T}
    nzval::Vector{V}
    trinumbers::Vector{T}
    symrows::FastVec{T}
    symcols::FastVec{T}

    function PSDMatrixCartesian{Offset}(dim::Int, tri::Symbol, rowind::Vector{T}, colind::Vector{T}, nzval::Vector{V}) where {T,V,Offset}
        tri === :L || tri === :U || error("Invalid triangle")
        length(rowind) == length(colind) == length(nzval) || error("Invalid PSDMatrixCartesian construction")
        sort_along!(colind, rowind, nzval)
        new{T,V,tri,
            if isbitstype(T) # be careful with large integers. But assume the offset will never exceed machine numbers
                T(Offset)
            elseif T <: Unsigned
                UInt(Offset)
            else
                Int(Offset)
            end
           }(rowind, colind, nzval, trisize.(zero(T):T(dim)-1), FastVec{T}(), FastVec{T}())
    end
end

Base.IteratorSize(::Type{<:PSDMatrixCartesian}) = Base.HasLength()
Base.IteratorEltype(::Type{<:PSDMatrixCartesian}) = Base.HasEltype()
Base.eltype(::Type{<:PSDMatrixCartesian{T,V}}) where {T,V} =
    Pair{T,Tuple{FastVec{T},FastVec{T},SubArray{V,1,Vector{V},Tuple{UnitRange{Int}},true}}}
function Base.length(psdi::PSDMatrixCartesian)
    isempty(psdi.colind) && return 0
    result = 1
    @inbounds for i in 2:length(psdi.colind)
        if psdi.colind[i] != psdi.colind[i-1]
            result += 1
        end
    end
    return result
end
@inline function Base.iterate(psdi::PSDMatrixCartesian{<:Any,<:Any,Tri,Offset}, (i, remaining)=(1, length(psdi.colind))) where {Tri,Offset}
    iszero(remaining) && return nothing
    rowind, colind, trinumbers, symrows, symcols = psdi.rowind, psdi.colind, psdi.trinumbers, psdi.symrows, psdi.symcols
    empty!(symrows)
    empty!(symcols)
    @inbounds curcol = colind[i]
    @inbounds while true
        idx = rowind[i]
        triindex = searchsortedlast(trinumbers, idx)
        # The input data is always in U format, apart from the fact that the nzval are already conjugated if Tri === :L
        push!(Tri === :L ? symrows : symcols, triindex - one(triindex) + Offset)
        push!(Tri === :L ? symcols : symrows, idx - trinumbers[triindex] + Offset)
        if isone(remaining) || colind[i+1] != curcol
            return Pair(curcol, (symrows, symcols, view(psdi.nzval, i-length(symrows)+1:i))), (i +1, remaining -1)
        end
        i += 1
        remaining -= 1
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

See also [`PSDVector`](@ref).
"""
struct PSDIndextypeVector{Tri} <: AbstractPSDIndextype{Tri}
    function PSDIndextypeVector(triangle::Symbol)
        triangle ∈ (:L, :U, :F) || throw(MethodError(PSDIndextypeVector, (triangle,)))
        new{triangle}()
    end
end

"""
    PSDVector

An iterable that returns consecutive elements in a vectorized PSD cone.
This type stores a vector of indices and values together with information about the length of the individual subsequences.
Iterating through it will give 2-Tuples that contain views into the indices and the values.
The vector of indices is available via `SparseArrays.rowvals`, the vector of values via `SparseArrays.nonzeros`, and the
lengths of the subsequences via `Base.index_lengths`.

See also [`PSDIndextypeVector`](@ref).
"""
struct PSDVector{T,V,L}
    indices::Vector{T}
    values::Vector{V}
    lens::L

    function PSDVector(indices::Vector{T}, values::Vector{V}, len::I) where {T,V,I<:Integer}
        length(indices) == length(values) || error("Invalid PSDVector construction")
        new{T,V,I}(indices, values, len)
    end
    function PSDVector(indices::Vector{T}, values::Vector{V}, lens::L) where {T,V,L<:AbstractVector{<:Integer}}
        length(indices) == length(values) || error("Invalid PSDVector construction")
        (isempty(lens) || length(indices) != sum(lens, init=0)) && error("Invalid PSDVector construction")
        new{T,V,L}(indices, values, lens)
    end
end

Base.IteratorSize(::Type{<:PSDVector}) = Base.HasLength()
Base.IteratorEltype(::Type{<:PSDVector}) = Base.HasEltype()
Base.eltype(::Type{PSDVector{T,V,<:Any}}) where {T,V} =
    Tuple{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true},SubArray{V,1,Vector{V},Tuple{UnitRange{Int}},true}}
Base.length(psdi::PSDVector{<:Any,<:Any,<:Integer}) = length(psdi.indices) ÷ psdi.lens
Base.length(psdi::PSDVector{<:Any,<:Any,<:AbstractVector{<:Integer}}) = length(psdi.lens)
@inline function Base.iterate(psdi::PSDVector{<:Any,<:Any,<:Integer}, state=1)
    endpos = state + psdi.lens -1
    if endpos ≤ length(psdi.indices)
        @inbounds return (view(psdi.indices, state:endpos), view(psdi.values, state:endpos)), endpos +1
    else
        return nothing
    end
end
@inline function Base.iterate(psdi::PSDVector{<:Any,<:Any,<:AbstractVector{<:Integer}}, state=(1, 1))
    state[2] ≤ length(psdi.lens) || return nothing
    startpos = state[1]
    endpos = @inbounds(startpos + psdi.lens[state[2]] -1)
    @inbounds return (view(psdi.indices, startpos:endpos), view(psdi.values, startpos:endpos)), (endpos +1, state[2] +1)
end
SparseArrays.rowvals(psdi::PSDVector) = psdi.indices
SparseArrays.nonzeros(psdi::PSDVector) = psdi.values
Base.index_lengths(psdi::PSDVector{<:Any,<:Any,<:Integer}) = Iterators.repeated(psdi.lens, length(psdi))
Base.index_lengths(psdi::PSDVector{<:Any,<:Any,<:AbstractVector{<:Integer}}) = psdi.lens

"""
    psd_indextype(state)

This function must indicate in which format the solver expects its data for PSD variables. The return type must be an instance
of an [`AbstractPSDIndextype`](@ref) subtype.

See also [`PSDIndextypeMatrixLinear`](@ref), [`PSDIndextypeMatrixCartesian`](@ref), [`PSDIndextypeVector`](@ref).
"""
function psd_indextype end

#include("./SOSInterface.jl")
include("./MomentInterface.jl")