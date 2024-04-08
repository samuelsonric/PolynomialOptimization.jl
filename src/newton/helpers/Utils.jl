function isless_degree(x::AbstractVector, y::AbstractVector)
    dx = sum(x)
    dy = sum(y)
    if dx == dy
        return isless(x, y)
    else
        return isless(dx, dy)
    end
end

# We provide some setter functions for SimpleDenseMonomialVector vector mainly due to the need for sorting when the vector is
# constructed in the parallelized Newton algorithm. This is the only reason - Simplexxx is not supposed to be mutable!
@inline function Base.setindex!(x::SimplePolynomials.SimpleRealDenseMonomialVectorOrView{Nr,P},
    val::SimplePolynomials.SimpleRealDenseMonomial{Nr,P}, i::Integer) where {Nr,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_real[:, i]), val.exponents_real)
    return val
end
@inline function Base.setindex!(x::SimplePolynomials.SimpleComplexDenseMonomialVectorOrView{Nc,P},
    val::SimplePolynomials.SimpleComplexDenseMonomial{Nc,P}, i::Integer) where {Nc,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_complex[:, i]), val.exponents_complex)
    @inbounds copyto!(@view(x.exponents_conj[:, i]), val.exponents_conj)
    return val
end
@inline function Base.setindex!(x::SimplePolynomials.SimpleDenseMonomialVectorOrView{Nc,P},
    val::SimplePolynomials.SimpleDenseMonomial{Nc,P}, i::Integer) where {Nc,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_real[:, i]), val.exponents_real)
    @inbounds copyto!(@view(x.exponents_complex[:, i]), val.exponents_complex)
    @inbounds copyto!(@view(x.exponents_conj[:, i]), val.exponents_conj)
    return val
end

# The same goes for resizing, which is done after the monomial vector was sorted by calling unique!.
function Base.resize!(x::SimpleMonomialVector{Nr,0,P,M}, len) where {Nr,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{Nr,0,P,M}(matrix_delete_end!(x.exponents_real, n))
end
function Base.resize!(x::SimpleMonomialVector{0,Nc,P,M}, len) where {Nc,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{0,Nc,P,M}(matrix_delete_end!(x.exponents_complex, n), matrix_delete_end!(x.exponents_conj, n))
end
function Base.resize!(x::SimpleMonomialVector{Nr,Nc,P,M}, len) where {Nr,Nc,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{Nr,Nc,P,M}(matrix_delete_end!(x.exponents_real, n), matrix_delete_end!(x.exponents_complex, n),
        matrix_delete_end!(x.exponents_conj, n))
end