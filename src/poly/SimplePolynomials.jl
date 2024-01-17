module SimplePolynomials

using MultivariatePolynomials, SparseArrays, PolynomialOptimization.FastVector
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseVector
import StatsBase

include("./Utils.jl")
include("./Variable.jl")
include("./Monomial.jl")
include("./MonomialIndex.jl")
include("./MonomialIterator.jl")
include("./MonomialVector.jl")
include("./Term.jl")
include("./Polynomial.jl")
include("./TypeHandling.jl")
include("./Compare.jl")

end