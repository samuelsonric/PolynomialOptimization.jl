module SortAlong

using SparseArrays

export sort_along!

# Here, we provide the helper function sort_along!, which allows to sort! a vector (the first argument) while at the same time
# sorting various other vectors using the exact same permutation. Hence,
#    sort_along!(v, a, b, c)
# is equivalent to
#    p = sortperm(v)
#    permute!.((v, a, b, c), (p,))
# but does not require any new intermediate allocations.
# Note that the sorting functions are exactly identical to the ones used in Julia 1.8 (there was a big overhaul in 1.9 which
# made everything extremely complicated; we don't reproduce the new behavior).
Base.@propagate_inbounds function swap_items!(v::AbstractVector, i, j)
    @assert(i < j)
    v[i], v[j] = v[j], v[i]
    return
end
Base.@propagate_inbounds function swap_items!(v::AbstractMatrix, i, j)
    @assert(i < j)
    part₁ = @view(v[:, i])
    part₂ = @view(v[:, j])
    @inbounds @simd for k in eachindex(part₁, part₂)
        part₁[k], part₂[k] = part₂[k], part₁[k]
    end
    return
end
Base.@propagate_inbounds function swap_items!(v::SparseArrays.AbstractSparseMatrixCSC, i, j)
    @assert(i < j)
    colptrs = SparseArrays.getcolptr(v)
    rowvals = SparseArrays.rowvals(v)
    nzvals = SparseArrays.nonzeros(v)
    δ = (colptrs[j+1] - colptrs[j]) - (colptrs[i+1] - colptrs[i])
    if !iszero(δ)
        # We do four successive reversions. Regarding cache lines, this is not too bad.
        for (start, stop) in ((colptrs[i], colptrs[i+1] -1), # first column
                              (colptrs[i+1], colptrs[j] -1), # columns in between
                              (colptrs[j], colptrs[j+1] -1), # last column
                              (colptrs[i], colptrs[j+1] -1)) # everything
            if start < stop
                reverse!(rowvals, start, stop)
                reverse!(nzvals, start, stop)
            end
        end
        @inbounds @simd for k in i+1:j
            colptrs[k] += δ
        end
    else
        l = colptrs[j]
        @inbounds @simd for k in colptrs[i]:colptrs[i+1]-1
            rowvals[k], rowvals[l] = rowvals[l], rowvals[k]
            nzvals[k], nzvals[l] = nzvals[l], nzvals[k]
            l += 1
        end
    end
    return
end
Base.@propagate_inbounds function rotate_items_left!(v::AbstractVector, i, j, k)
    @assert(i < j < k)
    v[i], v[j], v[k] = v[j], v[k], v[i]
    return
end
Base.@propagate_inbounds function rotate_items_left!(v::AbstractMatrix, i, j, k)
    @assert(i < j < k)
    part₁ = @view(v[:, i])
    part₂ = @view(v[:, j])
    part₃ = @view(v[:, k])
    @inbounds @simd for l in eachindex(part₁, part₂, part₃)
        part₁[l], part₂[l], part₃[l] = part₂[l], part₃[l], part₁[l]
    end
    return
end
Base.@propagate_inbounds function rotate_items_left!(v::SparseArrays.AbstractSparseMatrixCSC, i, j, k)
    @assert(i < j < k)
    swap_items!(v, i, j)
    swap_items!(v, j, k)
    return
end

can_extract(_) = true

function sort_along!(v::AbstractVector, along::AbstractVector...; lo::Integer=1, hi::Integer=length(v),
    o::Base.Ordering=Base.Order.Forward)
    lv = length(v)
    all(x -> length(x) == lv, along) || error("sort_along! requires vectors of the same length")
    lo ≥ 1 || error("sort_along! parameter lo must be strictly positive")
    hi ≤ lv || error("sort_along! parameter hi must not exceed length of the vectors")
    return sort_along!(v, lo, hi, o, along...)
end

@generated function sort_along!(v::AbstractVector, lo::Integer, hi::Integer, o::Base.Ordering, along::Vararg{<:AbstractVector,N}) where {N}
    extract_v = can_extract(v)
    extract_along = can_extract.(along)
    quote
        @inbounds while lo < hi
            if hi - lo <= Base.SMALL_THRESHOLD
                # function sort!(v::AbstractVector, lo::Integer, hi::Integer, ::InsertionSortAlg, o::Ordering)
                lo_plus_1 = (lo +1)::Integer
                for i in lo_plus_1:hi
                    j = i
                    $(extract_v ? :(x = v[i]) : :())
                    $((:($(Symbol(:x, i)) = along[$i][i]) for i in 1:N if extract_along[i])...)
                    # If !can_extract, v[i] is a view into data in v, so we cannot simply store its value so easily.
                    while j > lo
                        y = v[j-1]
                        Base.lt(o, $(extract_v ? :(x) : :(v[j])), y)::Bool || break
                        $(extract_v ? :(v[j] = y) : :(swap_items!(v, j -1, j)))
                        $((extract_along[i] ? :(along[$i][j] = along[$i][j-1]) :
                                              :(swap_items!(along[$i], j -1, j)) for i in 1:N)...)
                        j -= 1
                    end
                    $(extract_v ? :(v[j] = x) : :())
                    $((:(along[$i][j] = $(Symbol(:x, i))) for i in 1:N if extract_along[i])...)
                end
                return v, along...
                # end
            end
            # function partition!(v::AbstractVector, lo::Integer, hi::Integer, o::ordering)
                # function selectpivot!(v::AbstractVector, lo::Integer, hi::Integer, o::Ordering)
                mi = Base.midpoint(lo, hi)
                if Base.lt(o, v[lo], v[mi])
                    swap_items!(v, lo, mi)
                    $((:(swap_items!(along[$i], lo, mi)) for i in 1:N)...)
                end
                if Base.lt(o, v[hi], v[lo])
                    if Base.lt(o, v[hi], v[mi])
                        rotate_items_left!(v, lo, mi, hi)
                        $((:(rotate_items_left!(along[$i], lo, mi, hi)) for i in 1:N)...)
                    else
                        swap_items!(v, lo, hi)
                        $((:(swap_items!(along[$i], lo, hi)) for i in 1:N)...)
                    end
                end
                pivot = v[lo]
                # end
                i, j = lo, hi
                while true
                    i += 1; j -= 1
                    while Base.lt(o, v[i], pivot); i += 1; end;
                    while Base.lt(o, pivot, v[j]); j -= 1; end;
                    i >= j && break
                    swap_items!(v, i, j) # j > i > lo, so this swap can never affect pivot even if it is a view into v
                    $((:(swap_items!(along[$i], i, j)) for i in 1:N)...)
                end
                swap_items!(v, lo, j)
                $((:(swap_items!(along[$i], lo, j)) for i in 1:N)...)
            # end
            if j - lo < hi - j
                lo < (j -1) && sort_along!(v, lo, j -1, o, along...)
                lo = j +1
            else
                j +1 < hi && sort_along!(v, j +1, hi, o, along...)
                hi = j -1
            end
        end
        return v, along...
    end
end

end