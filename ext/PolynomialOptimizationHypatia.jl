module PolynomialOptimizationHypatia

using PolynomialOptimization, Hypatia, MultivariatePolynomials, SparseArrays
using PolynomialOptimization: FastVec, PolyOptProblem, MonomialComplexContainer, pctEqualitySimple, pctEqualityGröbner,
    pctEqualityNonneg, pctNonneg, pctPSD, EmptyGröbnerBasis, solver_methods, @verbose_info, sqrt2
using Hypatia: Cones, Models, Solvers

include("./Hypatia/HypatiaMoment.jl")

__init__() = push!(solver_methods, :HypatiaMoment)

end