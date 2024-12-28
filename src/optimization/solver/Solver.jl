module Solver

using ..SimplePolynomials, ..PolynomialOptimization, MultivariatePolynomials, LinearAlgebra, SparseArrays, Reexport
using ..SimplePolynomials: SimpleMonomialOrConj, SimpleConjMonomial, _get_I
@reexport using ..FastVector
using ..FastVector: overallocation
using ..PolynomialOptimization: @assert, @inbounds, @verbose_info, @capture, @unroll, FastKey, StackVec, Problem, sort_along!
import ..PolynomialOptimization: MomentVector
using ..SimplePolynomials.MultivariateExponents: ExponentsAll, ExponentsDegree, Unsafe, unsafe
using ..Relaxation: AbstractRelaxation, RelaxationGroupings
import LinearAlgebra: issuccess
# We re-export things that implementations of solvers (which is the only place where this module should be use'd) will most
# likely need
export
    poly_problem, Problem, MomentVector, StackVec, FastKey, sort_along!, @verbose_info, @capture, # from PolynomialOptimization
    AbstractRelaxation, RelaxationGroupings, # from Relaxation
    SimpleMonomialOrConj, SimpleConjMonomial, monomial_index, _get_I, # from SimplePolynomials
    overallocation, # from FastVector (not exported)
    poly_optimize, solver_methods, @solver_alias,
    issuccess # from LinearAlgebra

function poly_optimize end

const solver_methods = Symbol[]

function default_solver_method()
    isempty(solver_methods) &&
        error("No solver method is available. Load a solver package that provides such a method (e.g., Mosek)")
    @inbounds return solver_methods[begin]
end

"""
    @solver_alias(alias, original)

Defines the solver identifier `alias` to map to the same optimization routine as `original`.
"""
macro solver_alias(alias, original)
    quote
        $Solver.poly_optimize(::Val{$(QuoteNode(alias))}, relaxation::$AbstractRelaxation, args...; kwargs...) =
            $poly_optimize(Val($(QuoteNode(original))), relaxation, args...; kwargs...)
    end
end

"""
    issuccess(::Val{method}, status)

A solver must implement this method for all of its possible methods to indicate whether a status `status` signifies success.
"""
issuccess(::Val, ::Any)


include("./Helpers.jl")
include("./Interface.jl")
include("./SOSHelpers.jl")
include("./MomentHelpers.jl")
include("./AbstractAPISolver.jl")
include("./AbstractSparseMatrixSolver.jl")

end