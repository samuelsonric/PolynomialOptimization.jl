module PolynomialOptimizationMosek

using PolynomialOptimization, Mosek, MultivariatePolynomials, LinearAlgebra, SparseArrays, PolynomialOptimization.Solver,
    PolynomialOptimization.Newton
using PolynomialOptimization: @assert, @inbounds, @allocdiff
using PolynomialOptimization.SimplePolynomials: veciter
using Mosek: msk_global_env, Env, deletetask

printstream(msg::String) = (print(msg); flush(stdout))

# Do we have Mosek version at least 10?
isdefined(Mosek, :appendafes) && include("./MosekMoment.jl")
# include("./MosekSOS.jl")
include("./Newton.jl")
include("./Tightening.jl")

function __init__()
    isdefined(Mosek, :appendafes) && pushfirst!(solver_methods, :MosekMoment)
    # pushfirst!(solver_methods, :MosekSOS)
    pushfirst!(Newton.newton_methods, :Mosek)
    pushfirst!(PolynomialOptimization.tightening_methods, :Mosek)
end

end