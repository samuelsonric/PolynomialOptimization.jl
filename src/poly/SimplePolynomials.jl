module SimplePolynomials

using MultivariatePolynomials, SparseArrays, PolynomialOptimization.FastVector
using SparseArrays: AbstractSparseVector, AbstractSparseMatrixCSC, FixedSparseVector
using PolynomialOptimization: @assert, resizable_array
import PolynomialOptimization: matrix_delete_end!
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