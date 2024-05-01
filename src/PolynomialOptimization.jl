__precompile__()
module PolynomialOptimization

using MultivariatePolynomials
using SparseArrays
using LinearAlgebra
using Printf
import Graphs
import Combinatorics
import MutableArithmetics
import StatsBase

export Newton

const sqrt2 = sqrt(2.0)
const haveMPI = Ref{Bool}(false)
const debugging = true

include("./helpers/Helpers.jl")

include("./Problem.jl")
include("./relaxations/Relaxation.jl")
include("./newton/Newton.jl")
import .Newton
include("./optimization/Optimization.jl")
include("./Tightening.jl")

end