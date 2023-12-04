module PolynomialOptimizationCOPT

using PolynomialOptimization, COPT, MultivariatePolynomials
using PolynomialOptimization: FastVec, PolyOptProblem, pctEqualitySimple, pctEqualityGröbner, pctEqualityNonneg, pctNonneg,
    pctPSD, EmptyGröbnerBasis, solver_methods, @verbose_info
using COPT: _check_ret, Env, libcopt

global copt_env::Env

include("./COPT/COPTSOS.jl")

function __init__()
    global copt_env = Env()
    push!(solver_methods, :COPTSOS)
end

end