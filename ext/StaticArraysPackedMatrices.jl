module StaticArraysPackedMatrices

using PolynomialOptimization.PackedMatrices
import StaticArrays

# We need to be extra careful here - StaticArrays's broadcasting has higher precedence, but will try to write n^2 elements to
# our linear index. But we can spell this out and only copy the upper triangle.
@generated function StaticArrays._broadcast!(f, ::StaticArrays.Size{newsize}, dest::PackedMatrix,
    s::Tuple{Vararg{StaticArrays.Size}}, a...) where {newsize}
    sizes = [sz.parameters[1] for sz in s.parameters]
    @assert(length(newsize) == 2 && newsize[1] == newsize[2])

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr, packedsize(newsize[1]))
    j = 1
    for current_ind ∈ indices
        @inbounds if current_ind[1] ≤ current_ind[2]
            exprs_vals = (StaticArrays.broadcast_getindex(sz, i, current_ind) for (i, sz) in enumerate(sizes))
            exprs[j] = :(dest[$j] = f($(exprs_vals...)))
            j += 1
        end
    end

    return quote
        StaticArrays.@_inline_meta
        @inbounds $(Expr(:block, exprs...))
        return dest
    end
end

end