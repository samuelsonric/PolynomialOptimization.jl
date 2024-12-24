__precompile__()
module PolynomialOptimization

using MultivariatePolynomials
using SparseArrays
using LinearAlgebra
using Printf
import MutableArithmetics
import StatsBase

export Newton, Relaxation

const sqrt2 = sqrt(2.)
const haveMPI = Ref{Bool}(false)
const debugging = false

include("./helpers/Helpers.jl")

include("./Problem.jl")
include("./relaxations/Relaxation.jl")
using .Relaxation
include("./optimization/Optimization.jl")
include("./newton/Newton.jl")
import .Newton
include("./solutions/SolutionExtraction.jl")
include("./Tightening.jl")

end