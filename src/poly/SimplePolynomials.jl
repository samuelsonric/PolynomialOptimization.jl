module SimplePolynomials

using MultivariatePolynomials, PolynomialOptimization.FastVector, PolynomialOptimization.SortAlong
using PolynomialOptimization: @assert

include("./Utils.jl")
include("./exponents/MultivariateExponents.jl")
using .MultivariateExponents
using .MultivariateExponents: ExponentIndices, Unsafe, index_counts
include("./Variable.jl")
include("./Monomial.jl")
include("./MonomialVector.jl")
include("./Polynomial.jl")
include("./TypeHandling.jl")

end