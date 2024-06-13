module PolynomialOptimizationHypatia

using PolynomialOptimization, Hypatia, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.FastVector,
    PolynomialOptimization.Solver
using PolynomialOptimization: @assert, @inbounds, POProblem, RelaxationGroupings, @verbose_info, @capture, MomentVector,
    sort_along!
using PolynomialOptimization.SimplePolynomials: monomial_index, _get_I
using Hypatia: Cones, Models, Solvers

include("./HypatiaMoment.jl")

__init__() = push!(Solver.solver_methods, :HypatiaMoment)

end