module Solver

using ..SimplePolynomials, ..PolynomialOptimization, ..FastVector, MultivariatePolynomials, LinearAlgebra, SparseArrays
using ..SimplePolynomials: SimpleMonomialOrConj, SimpleConjMonomial
using ..PolynomialOptimization: @assert, @capture, FastKey, StackVec, POProblem, RelaxationGroupings
using ..SimplePolynomials.MultivariateExponents: ExponentsAll, ExponentsDegree
export SimpleMonomialOrConj, SimpleConjMonomial, FastKey

function poly_optimize end

const solver_methods = Symbol[]

function default_solver_method()
    isempty(solver_methods) && error("No solver method is available. Load a solver package that provides such a method (e.g., Mosek)")
    return first(solver_methods)
end

include("./Interface.jl")
include("./SOSHelpers.jl")

end