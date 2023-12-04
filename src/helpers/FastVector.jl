# Julia exploits the sizehint!/push! combination very poorly (https://github.com/JuliaLang/julia/issues/24909). Here, we define
# a (limited) fast vector, which rectifies this issue. It is similar to the PushVectors.jl package, but allows for even more
# speed by allowing to drop the bounds check on push!. It also uses Julia's precise algorithm to determine the next size when
# growing.
"""
    FastVec{V}(undef, n::Integer; buffer::Integer=n)

Creates a new FastVec, which is a vector of elements of type `V`. The elements are initially undefined; there are `n` items.
The vector has a capacity of size `buffer`; while it can hold arbitrarily many, pushing to the vector is fast as long as the
capacity is not exceeded.
"""
mutable struct FastVec{V} <: AbstractVector{V}
    # We could in principle make this more compact by redefining Julia's internal jl_array_t structure (which already has the
    # length, but let's keep up compatibility)
    const data::Vector{V}
    len::Int

    function FastVec{V}(::UndefInitializer, n::Integer; buffer::Integer=n) where {V}
        buffer ≥ n || error("The buffer must not be smaller than the number of elements")
        new{V}(Vector{V}(undef, buffer), n)
    end
end

"""
    FastVec{V}(; buffer::Integer=0)

Shorthand to create an empty FastVec with a certain buffer size.
"""
FastVec{V}(; buffer::Integer=0) where {V} = FastVec{V}(undef, 0; buffer)

Base.size(v::FastVec) = (v.len,)

Base.@propagate_inbounds function Base.getindex(v::FastVec, i::Int)
    @boundscheck checkbounds(v, i)
    @inbounds return v.data[i]
end

Base.IndexStyle(::FastVec) = IndexLinear()

Base.@propagate_inbounds function Base.setindex!(v::FastVec{V}, el::V, i::Int) where {V}
    @boundscheck checkbounds(v, i)
    @inbounds return v.data[i] = el
end

Base.length(v::FastVec) = v.len

"""
    sizehint!(v::FastVec, len::Integer)

Changes the size of the internal buffer that is kept available to quickly manage pushing into the vector. If len is smaller
than the actual length of this vector, this is a no-op.
"""
function Base.sizehint!(v::FastVec, len::Integer)
    len ≥ v.len && resize!(v.data, len)
    return v
end

"""
    empty!(v::FastVec)

Clears the vector without freeing the internal buffer.
"""
function Base.empty!(v::FastVec)
    v.len = 0
    return v
end

"""
    prepare_push!(v::FastVec, new_items::Integer)

Prepares pushing (or appending) at last `new_items` in the future, in one or multiple calls. This ensures that the internal
buffer is large enough to hold all the new items that will be pushed without allocations in between.
"""
function prepare_push!(v::FastVec, new_items::Integer)
    new_items ≥ 0 || error("prepare_push! expects a positive number of items")
    v.len - length(v.data) < new_items && resize!(v.data, overallocation(v.len + new_items))
    return v
end

"""
    push!(v::FastVec, el)

Adds `el` to the end of `v`, increasing the length of `v` by one. If there is not enough space, grows `v`.
Note that if you made sure beforehand that the capacity of `v` is sufficient for the addition of this element, consider calling
[`unsafe_push!`](@ref) instead, which avoids the length check.
"""
@inline function Base.push!(v::FastVec{V}, el) where {V}
    elV = convert(V, el)
    v.len += 1
    length(v.data) < v.len && resize!(v.data, overallocation(v.len))
    @inbounds v.data[v.len] = elV
    return v
end

"""
    unsafe_push!(v::FastVec, el)

Adds `el` to the end of `v`, increasing the length of `v` by one.
This function assumes that the internal buffer of `v` holds enough space to add at least one element; if this is not the case,
it will lead to memory corruption. Call [`push!`](@ref) instead if you cannot guarantee the necessary buffer size.
"""
@inline function unsafe_push!(v::FastVec{V}, el) where {V}
    elV = convert(V, el)
    v.len += 1
    @assert(length(v.data) ≥ v.len)
    @inbounds v.data[v.len] = elV
    return v
end

"""
    append!(v::FastVec, els)

Appends all items in `els` to the end of `v`, increasing the length of `v` by `length(els)`. If there is not enough space,
grows `v`.
Note that if you made sure beforehand that the capacity of `v` is sufficient for the addition of these elements, consider
calling [`unsafe_append!`](@ref) instead, which avoids the length check.
"""
@inline function Base.append!(v::FastVec{V}, els) where {V}
    oldlen = v.len
    v.len += length(els)
    length(v.data) < v.len && resize!(v.data, overallocation(v.len))
    @inbounds copyto!(v.data, oldlen +1, els, 1, length(els))
    return v
end

@inline function Base.append!(v::FastVec{V}, els::FastVec{V}) where {V}
    oldlen = v.len
    v.len += els.len
    length(v.data) < v.len && resize!(v.data, overallocation(v.len))
    @inbounds copyto!(v.data, oldlen +1, els.data, 1, els.len)
    return v
end

"""
    unsafe_append!(v::FastVec, els)

Appends all items in `els` to the end of `v`, increasing the length of `v` by `length(els)`.
This function assumes that the internal buffer of `v` holds enough space to add at least all elements in `els`; if this is not
the case, it will lead to memory corruption. Call [`append!`](@ref) instead if you cannot guarantee the necessary buffer size.
"""
@inline function unsafe_append!(v::FastVec{V}, els) where {V}
    oldlen = v.len
    v.len += length(els)
    @assert(length(v.data) ≥ v.len)
    @inbounds copyto!(v.data, oldlen +1, els, 1, length(els))
    return v
end

@inline function unsafe_append!(v::FastVec{V}, els::FastVec{V}) where {V}
    oldlen = v.len
    v.len += els.len
    @assert(length(v.data) ≥ v.len)
    @inbounds copyto!(v.data, oldlen +1, els.data, 1, els.len)
    return v
end

"""
    prepend!(v::FastVec, els)

Prepends all items in `els` to the beginning of `v`, increasing the length `v` by `length(els)`. If there is not enough space,
grows `v`.
Note that if you made sure beforehand that the capacity of `v` is sufficient for the addition of these elements, consider
calling [`unsafe_prepend!`](@ref) instead, which avoids the length check.
"""
@inline function Base.prepend!(v::FastVec{V}, els) where {V}
    oldlen = v.len
    v.len += length(els)
    length(v.data) < v.len && resize!(v.data, overallocation(v.len))
    @inbounds copyto!(v.data, length(els) +1, v.data, 1, oldlen) # must always work correctly with overlapping
    @inbounds copyto!(v.data, els)
    return v
end

@inline function Base.prepend!(v::FastVec{V}, els::FastVec{V}) where {V}
    oldlen = v.len
    v.len += els.len
    length(v.data) < v.len && resize!(v.data, overallocation(v.len))
    @inbounds copyto!(v.data, els.len +1, v.data, 1, oldlen) # must always work correctly with overlapping
    @inbounds copyto!(v.data, 1, els.data, 1, els.len)
    return v
end

"""
    unsafe_prepend!(v::FastVec, els::AbstractVector)

Prepends all items in `els` to the beginning of `v`, increasing the length `v` by `length(els)`.
This function assumes that the internal buffer of `v` holds enough space to add at least all elements in `els`; if this is not
the case, it will lead to memory corruption. Call [`prepend!`](@ref) instead if you cannot guarantee the necessary buffer size.
"""
@inline function unsafe_prepend!(v::FastVec{V}, els) where {V}
    oldlen = v.len
    v.len += length(els)
    @assert(length(v.data) ≥ v.len)
    @inbounds copyto!(v.data, length(els) +1, v.data, 1, oldlen) # must always work correctly with overlapping
    @inbounds copyto!(v.data, els)
    return v
end

@inline function unsafe_prepend!(v::FastVec{V}, els::FastVec{V}) where {V}
    oldlen = v.len
    v.len += els.len
    @assert(length(v.data) ≥ v.len)
    @inbounds copyto!(v.data, els.len +1, v.data, 1, oldlen) # must always work correctly with overlapping
    @inbounds copyto!(v.data, 1, els.data, 1, els.len)
    return v
end

"""
    similar(v::FastVec)

Creates a FastVec of the same type and length as `v`. All the common variants to supply different element types or lengths are
also available; when changing the length, you might additionally specify the keyword argument `buffer` that also allows to
change the internal buffer size.
"""
Base.similar(v::FastVec{V}) where {V} = FastVec{V}(undef, v.len, buffer=length(v.data))
Base.similar(v::FastVec{V}, ::Type{S}) where {V,S} = FastVec{S}(undef, v.len, buffer=length(v.data))
Base.similar(::FastVec{V}, len::Integer; buffer::Integer=overallocation(len)) where {V} = FastVec{V}(undef, len; buffer)
Base.similar(::FastVec{V}, ::Type{S}, len::Integer; buffer::Integer=overallocation(len)) where {V,S} =
    FastVec{S}(undef, len; buffer)

Base.unsafe_convert(::Type{Ptr{V}}, v::FastVec{V}) where {V} = Base.unsafe_convert(Ptr{V}, v.data)

# inspired by KristofferC's PushVector
"""
    finish!(v::FastVec)

Returns the `Vector` representation that internally underlies the FastVec instance. This function should only be called when no
further operations on the FastVec itself are carried out.
"""
finish!(v::FastVec) = resize!(v.data, v.len)

# This is how Julia internally grows vectors:
# size_t overallocation(size_t maxsize)
# {
#     if (maxsize < 8)
#         return 8;
#     // compute maxsize = maxsize + 4*maxsize^(7/8) + maxsize/8
#     // for small n, we grow faster than O(n)
#     // for large n, we grow at O(n/8)
#     // and as we reach O(memory) for memory>>1MB,
#     // this means we end by adding about 10% of memory each time
#     int exp2 = sizeof(maxsize) * 8 -
# #ifdef _P64
#         __builtin_clzll(maxsize);
# #else
#         __builtin_clz(maxsize);
# #endif
#     maxsize += ((size_t)1 << (exp2 * 7 / 8)) * 4 + maxsize / 8;
#     return maxsize;
# }
overallocation(maxsize) = maxsize < 8 ? 8 : let exp2 = 8sizeof(maxsize) - leading_zeros(maxsize)
    return maxsize + 4(1 << (7exp2 ÷ 8)) + maxsize ÷ 8
end