module PolynomialOptimizationMosek

using PolynomialOptimization, Mosek, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.FastVector
using PolynomialOptimization: POProblem, RelaxationGroupings, @verbose_info, @allocdiff, monomial_count, sos_solution,
    StackVec, FastKey
using Mosek: msk_global_env, Env, deletetask

printstream(msg::String) = (print(msg); flush(stdout))

include("./MosekSOS.jl")
include("./Newton.jl")
include("./Tightening.jl")

function __init__()
    pushfirst!(PolynomialOptimization.solver_methods, :MosekSOS)
    pushfirst!(PolynomialOptimization.newton_methods, :Mosek)
    pushfirst!(PolynomialOptimization.tightening_methods, :Mosek)
end

end