module PolynomialOptimizationMosek

using PolynomialOptimization, Mosek, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.FastVector
using PolynomialOptimization: @assert, POProblem, RelaxationGroupings, @verbose_info, @allocdiff, MomentVector,
    StackVec, FastKey, Newton, sort_along!
using PolynomialOptimization.SimplePolynomials: SimpleMonomial, monomial_index, veciter
using Mosek: msk_global_env, Env, deletetask

printstream(msg::String) = (print(msg); flush(stdout))

# Do we have Mosek version at least 10?
#isdefined(Mosek, :appendafes) && include("./MosekMoment.jl")
include("./MosekSOS.jl")
include("./Newton.jl")
include("./Tightening.jl")

function __init__()
    # isdefined(Mosek, :appendafes) && pushfirst!(solver_methods, :MosekMoment)
    pushfirst!(PolynomialOptimization.solver_methods, :MosekSOS)
    pushfirst!(Newton.newton_methods, :Mosek)
    pushfirst!(PolynomialOptimization.tightening_methods, :Mosek)
end

end