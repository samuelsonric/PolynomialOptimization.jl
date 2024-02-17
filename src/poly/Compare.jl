Base.isless(x::V, y::V) where {V<:SimpleVariable} = isless(x.index, y.index)

# comparison of monomial
function MultivariatePolynomials.compare(x::SimpleMonomial{Nr,Nc,P}, y::SimpleMonomial{Nr,Nc,P}, ::Type{LexOrder}) where {Nr,Nc,P<:Unsigned}
    # don't use the default implementation, which subtracts exponents: these are unsigned types!
    for (xᵢ, yᵢ) in zip(exponents(x), exponents(y))
        if xᵢ > yᵢ
            return 1
        elseif xᵢ < yᵢ
            return -1
        end
    end
    return 0
end

Base.isless(x::SimpleMonomial{Nr,Nc,P}, y::SimpleMonomial{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} =
    MultivariatePolynomials.compare(x, y, Graded{LexOrder}) < 0

Base.:(==)(x::SimpleMonomial{Nr,Nc,P}, y::SimpleMonomial{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} =
    iszero(MultivariatePolynomials.compare(x, y, Graded{LexOrder}))