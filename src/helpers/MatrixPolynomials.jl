MultivariatePolynomials.variables(m::AbstractMatrix{<:AbstractPolynomialLike}) = union(variables.(m)...)

MultivariatePolynomials.monomials(m::AbstractMatrix{<:AbstractPolynomialLike}) = union(monomials.(m)...)

MultivariatePolynomials.coefficients(m::AbstractMatrix{<:AbstractPolynomialLike}) = coefficients.(m, (monomials(m),))
MultivariatePolynomials.coefficients(m::AbstractMatrix{<:AbstractPolynomialLike}, X::AbstractVector) = coefficients.(m, (X,))

MultivariatePolynomials.mindegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    minimum((mindegree(p, args...) for p in m), init=0)::Int
MultivariatePolynomials.mindegree_complex(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    minimum((mindegree_complex(p, args...) for p in m), init=0)::Int
MultivariatePolynomials.minhalfdegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    minimum((minhalfdegree(p, args...) for p in m), init=0)::Int
MultivariatePolynomials.maxdegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    maximum((maxdegree(p, args...) for p in m), init=0)::Int
MultivariatePolynomials.maxdegree_complex(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    maximum((maxdegree_complex(p, args...) for p in m), init=0)::Int
MultivariatePolynomials.maxhalfdegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...) =
    maximum((maxhalfdegree(p, args...) for p in m), init=0)::Int
function MultivariatePolynomials.extdegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...)
    l = typemax(Int)
    u = 0
    for p in m
        (newl, newu) = extdegree(p, args...)
        newl < l && (l = newl)
        newu > u && (u = newu)
    end
    return l, u
end
function MultivariatePolynomials.extdegree_complex(m::AbstractMatrix{<:AbstractPolynomialLike}, args...)
    l = typemax(Int)
    u = 0
    for p in m
        (newl, newu) = extdegree_complex(p, args...)
        newl < l && (l = newl)
        newu > u && (u = newu)
    end
    return l, u
end
function MultivariatePolynomials.exthalfdegree(m::AbstractMatrix{<:AbstractPolynomialLike}, args...)
    l = typemax(Int)
    u = 0
    for p in m
        (newl, newu) = exthalfdegree(p, args...)
        newl < l && (l = newl)
        newu > u && (u = newu)
    end
    return l, u
end

MultivariatePolynomials.effective_variables(m::AbstractMatrix{<:AbstractPolynomialLike}) = union(effective_variables.(m)...)

SimplePolynomials.effective_variables_in(m::AbstractMatrix{<:AbstractPolynomialLike}, in) =
    all(Base.Fix2(effective_variables_in, in), m)

Base.isreal(m::AbstractMatrix{<:AbstractPolynomialLike}) = all(isreal, m)