module PolynomialOptimizationLancelot

using MultivariatePolynomials, LinearAlgebra, StandardPacked, Printf, PolynomialOptimization
using PolynomialOptimization: @assert, @inbounds, @verbose_info, Solver
using GALAHAD: libgalahad_double
import StaticPolynomials

include("./Bindings.jl")
include("./Lancelot.jl")

__init__() = push!(Solver.solver_methods, :Lancelot)

end