module PolynomialOptimizationCOPT

using PolynomialOptimization, COPT, MultivariatePolynomials, PolynomialOptimization.FastVector,
    PolynomialOptimization.SortAlong, SparseArrays
using PolynomialOptimization: @assert, POProblem, RelaxationGroupings, @verbose_info, monomial_count, MomentVector, StackVec,
    FastKey
using COPT: _check_ret, Env, libcopt

global copt_env::Env

include("./COPTSOS.jl")

function __init__()
    global copt_env = Env()
    push!(PolynomialOptimization.solver_methods, :COPTSOS)
end

end