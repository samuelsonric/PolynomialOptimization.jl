export SparseMatrixSolver, SparseMatrixCOO, coo_to_csc!

"""
    SparseMatrixSolver{I<:Integer,K<:Integer,V<:Real}

Superclass for a solver that requires its data in sparse matrix form. The data is aggregated in COO form using
[`append!`](@ref append!(::SparseMatrixCOO{I,K,V,Offset}, ::Tuple{AbstractVector{K,AbstractVector{V}}...) where {I<:Integer,K<:Integer,V<:Real,Offset})
and can be converted to CSC form. The type of the indices in final CSC form is `I`, where the monomials during construction
will be represented by numbers of type `K`.

See also [`SparseMatrixCOO`](@ref).
"""
abstract type SparseMatrixSolver{I<:Integer,K<:Integer,V<:Real} end

Solver.mindex(::SparseMatrixSolver{<:Integer,K,<:Real}, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {K,Nr,Nc} =
    monomial_index(monomials...)::K

"""
    SparseMatrixCOO{I<:Integer,K<:Integer,V<:Real,Offset}

Representation of a sparse matrix in COO form. Fields are `rowinds::FastVec{I}`, `moninds::FastVec{K}` (where `K` is of the
type returned by `monomial_index`), and `nzvals::FastVec{V}`. The first row/column for the solver has index `Offset` (of type
`I`).
"""
struct SparseMatrixCOO{I<:Integer,K<:Integer,V<:Real,Offset}
    rowinds::FastVec{I}
    moninds::FastVec{K}
    nzvals::FastVec{V}

    function SparseMatrixCOO{I,K,V,Offset}() where {I<:Integer,K<:Integer,V<:Real,Offset}
        Offset isa I || throw(MethodError(SparseMatrixCOO{I,K,V,Offset}, ()))
        new{I,K,V,Offset}(FastVec{I}(), FastVec{K}(), FastVec{V}())
    end
end

Base.length(smc::SparseMatrixCOO) = length(smc.rowinds)
@inline function Base.size(smc::SparseMatrixCOO{<:Integer,<:Integer,<:Real,Offset}, dim) where {Offset}
    dim == 1 || error("Not implemented")
    @inbounds return isempty(smc.rowinds) ? 0 : Int(last(smc.rowinds)) + (1 - Int(Offset))
end

function FastVector.prepare_push!(smc::SparseMatrixCOO, new_items::Integer)
    prepare_push!(smc.rowinds, new_items)
    prepare_push!(smc.moninds, new_items)
    prepare_push!(smc.nzvals, new_items)
    return smc
end

"""
    append!(coo::SparseMatrixCOO, indvals::AbstractIndvals...)

Appends the data given in `indvals` into successive rows in `coo` (`indvals[1]` to the first rows, `indvals[2]` to the second,
...). Returns the index of the last row that was added.
"""
@inline function Base.append!(coo::SparseMatrixCOO{I,K,V,Offset}, indvals::AbstractIndvals{K,V}...) where {I<:Integer,K<:Integer,V<:Real,Offset}
    prep = 0
    for indval in indvals
        prep += length(indval)
    end
    prepare_push!(coo, prep)
    v = isempty(coo.rowinds) ? Offset : last(coo.rowinds) + one(I)
    for indval in indvals
        for (monind, nzval) in indval
            unsafe_push!(coo.rowinds, v)
            unsafe_push!(coo.moninds, monind)
            unsafe_push!(coo.nzvals, nzval)
        end
        v += one(I)
    end
    return v - one(I)
end

"""
    append!(coo::SparseMatrixCOO, psd::PSDVector)

Appends the data given in `psd` into successive rows in `coo`. Returns the index of the last row that was added.
"""
@inline function Base.append!(coo::SparseMatrixCOO{I,K,V,Offset}, psd::PSDVector{K,V}) where {I<:Integer,K<:Integer,V<:Real,Offset}
    prepare_push!(coo.rowinds, length(rowvals(psd)))
    v = isempty(coo.rowinds) ? Offset : last(coo.rowinds) + one(I)
    for l in Base.index_lengths(psd)
        for _ in 1:l
            unsafe_push!(coo.rowinds, v)
        end
        v += one(I)
    end
    append!(coo.moninds, rowvals(psd))
    append!(coo.nzvals, nonzeros(psd))
    return v - one(I)
end

struct COO_to_CSC_callback{Offset,CC<:Tuple,CP<:Tuple,CV<:Tuple,DV<:Tuple,I<:Tuple}
    coo_columns::CC
    csc_colptrs::CP
    coo_vecs::CV
    dense_vecs::DV
    ivecs::I

    function COO_to_CSC_callback{Offset}(coo_columns::Tuple, csc_colptrs::Tuple, coo_vecs::Tuple, dense_vecs::Tuple) where {Offset}
        ivecs = ntuple(_ -> Ref(1), length(coo_vecs))
        new{Offset,typeof(coo_columns),typeof(csc_colptrs),typeof(coo_vecs),typeof(dense_vecs),typeof(ivecs)}(
            coo_columns, csc_colptrs, coo_vecs, dense_vecs, ivecs
        )
    end
end

function (ctc::COO_to_CSC_callback{Offset,<:Tuple{Any}})(index, i) where {Offset}
    @inbounds begin
        colidx = ctc.coo_columns[1][i]
        for (ivec, coo_vec, dense_vec) in zip(ctc.ivecs, ctc.coo_vecs, ctc.dense_vecs)
            if ivec[] ≤ length(coo_vec[1]) && coo_vec[1][ivec[]] == colidx
                dense_vec[index] = coo_vec[2][ivec[]]
                ivec[] += 1
            end
        end
        ctc.csc_colptrs[1][index+1] = i + Offset
    end
    return
end

function (ctc::COO_to_CSC_callback{Offset,<:Tuple{Any,Any}})(index, i₁, i₂) where {Offset}
    @inbounds begin
        colidx = ismissing(i₁) ? ctc.coo_columns[2][i₂] : ctc.coo_columns[1][i₁]
        for (ivec, coo_vec, dense_vec) in zip(ctc.ivecs, ctc.coo_vecs, ctc.dense_vecs)
            if ivec[] ≤ length(coo_vec[1]) && coo_vec[1][ivec[]] == colidx
                dense_vec[index] = coo_vec[2][ivec[]]
                ivec[] += 1
            end
        end
        ctc.csc_colptrs[1][index+1] = ismissing(i₁) ? ctc.csc_colptrs[1][index] : i₁ + Offset
        ctc.csc_colptrs[2][index+1] = ismissing(i₂) ? ctc.csc_colptrs[2][index] : i₂ + Offset
    end
    return
end

"""
    coo_to_csc(coo::Union{SparseMatrixCOO{I,K,V}},Tuple{FastVec{K},FastVec{V}}}...)

Converts sparse COO matrix or vector representations, where the monomial indices of the `coo` matrices or the entries of the
vectors can be arbitrarily sparse, to a CSC-based matrix representation with continuous columns, and the vectors are converted
to dense ones. No more than two matrices may be supplied. The input data may be mutated; however, it is still usable to pass on
to [`MomentVector`](@ref MomentVector(::AbstractRelaxation, ::Vector{V}, ::SparseMatrixCOO{<:Integer,K,V,Offset}...) where {K<:Integer,V<:Real,Offset}).
The following values are returned:
- number of distinct columns
- for each input, if it is a matrix, a tuple containing the colptr, rowval, nzval vectors
- for each input, if it is a vector, the corresponding dense vector
"""
@generated function coo_to_csc!(coo::Union{SparseMatrixCOO{I,K,V,Offset},<:Tuple{AbstractVector{K},AbstractVector{V}}}...) where {I<:Integer,K<:Integer,V<:Real,Offset}
    result = Expr(:block)
    mats = Int[]
    vecs = Int[]
    for (i, data) in enumerate(coo)
        if data <: SparseMatrixCOO
            length(mats) == 2 && throw(MethodError(coo_to_csc!, coo)) # no more than two such matrices are allowed
            # sort according to col indices, but keep row indices in order (sort_along is in general not order-preserving, so
            # use row indices as secondary option)
            push!(result.args, :(sort_along!(coo[$i].moninds, coo[$i].rowinds, coo[$i].nzvals, relevant=2)))
            push!(mats, i)
        else
            push!(result.args, :(sort_along!(coo[$i][1], coo[$i][2])))
            push!(vecs, i)
        end
    end
    isempty(mats) && throw(MethodError(coo_to_csc!, coo))
    # how many distinct monomials do we have? - we only count the matrices, which contains the conic data
    push!(result.args, :(moncount = count_uniques($((:(coo[$i].moninds) for i in mats)...),)))
    for i in mats
        push!(result.args,
            :($(Symbol(:colptr, i)) = Vector{I}(undef, moncount +1)),
            :(@inbounds $(Symbol(:colptr, i))[1] = $Offset)
        )
    end
    append!(result.args, (:($(Symbol(:vec, i)) = zeros(V, moncount)) for i in vecs))
    push!(result.args, :(count_uniques(
        $((:(coo[$i].moninds) for i in mats)...),
        COO_to_CSC_callback{Offset}(
            ($((:(coo[$i].moninds) for i in mats)...),),
            ($((Symbol(:colptr, i) for i in mats)...),),
            ($((:(coo[$i]) for i in vecs)...),),
            ($((Symbol(:vec, i) for i in vecs)...),)
        )
    )))
    returning = Expr(:tuple, :(moncount))
    for (i, data) in enumerate(coo)
        if data <: SparseMatrixCOO
            push!(returning.args, :(($(Symbol(:colptr, i)), finish!(coo[$i].rowinds), finish!(coo[$i].nzvals))))
        else
            push!(returning.args, :($(Symbol(:vec, i))))
        end
    end
    push!(result.args, :(return $returning))
    result
end

"""
    MomentVector(relaxation::AbstractRelaxation, moments::Vector{<:Real}, coo::SparseMatrixCOO...)

Given the moments vector as obtained from a [`SparseMatrixSolver`](@ref) solver, convert it to a [`MomentVector`](@ref). Note
that this function is not fully type-stable, as the result may be based either on a dense or sparse vector depending on the
relaxation. To establish the mapping between the solver output and the actual moments, the all COO data used in the problem
construction needs to be passed on.
"""
function MomentVector(relaxation::AbstractRelaxation, moments::Vector{V}, coo₁::SparseMatrixCOO{<:Integer,K,V,Offset},
    cooₙ::SparseMatrixCOO{<:Integer,K,V,Offset}...) where {K<:Integer,V<:Real,Offset}
    # we need at least one coo here for dispatch
    max_mons = relaxation_bound(relaxation)
    @assert(length(moments) ≤ max_mons)
    if length(moments) == max_mons # dense case
        solution = moments
    else
        # We need to build the vector of monomial indices.
        mon_pos = Vector{Int}(undef, length(moments))
        if length(cooₙ) == 0
            count_uniques(coo₁.moninds, @capture((index, i) -> @inbounds begin
                $mon_pos[index] = $coo₁.moninds[i]
            end))
        elseif length(cooₙ) == 1
            count_uniques(coo₁.moninds, cooₙ[1].moninds, @capture((index, i₁, i₂) -> @inbounds begin
                $mon_pos[index] = ismissing(i₁) ? $cooₙ[1].moninds[i₂] : $coo₁.moninds[i₁]
            end))
        else
            throw(MethodError(MomentVector, (relaxation, moments, coo₁, cooₙ...)))
        end
        if 3length(moments) < max_mons
            solution = SparseVector(max_mons, mon_pos, moments)
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), moments)
        end
    end
    return MomentVector(relaxation, solution)
end