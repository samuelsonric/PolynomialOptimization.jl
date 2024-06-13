module PolynomialOptimizationSCS

using PolynomialOptimization, SCS, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.FastVector,
    PolynomialOptimization.Solver
using PolynomialOptimization: @assert, @inbounds, POProblem, RelaxationGroupings, @verbose_info, @capture, MomentVector,
    sort_along!
using PolynomialOptimization.SimplePolynomials: monomial_index, _get_I
using SCS: ScsCone, ScsData, ScsSettings, ScsSolution, ScsInfo, ScsMatrix, scsint_t, scs_init, scs_solve, scs_finish,
    LinearSolver

include("./SCSMoment.jl")

__init__() = push!(Solver.solver_methods, :SCSMoment)

end