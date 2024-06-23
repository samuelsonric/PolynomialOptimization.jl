module PolynomialOptimizationHypatia

using Hypatia, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.Solver
using PolynomialOptimization: @assert, @inbounds
using Hypatia: Cones, Models, Solvers

include("./HypatiaMoment.jl")

__init__() = push!(solver_methods, :Hypatia, :HypatiaMoment)

@solver_alias Hypatia HypatiaMoment

end