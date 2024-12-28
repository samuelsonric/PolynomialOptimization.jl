module PolynomialOptimizationMosek

using Mosek, MultivariatePolynomials, PolynomialOptimization.Solver, PolynomialOptimization.Newton
using PolynomialOptimization: @assert, @inbounds, @allocdiff
using PolynomialOptimization.SimplePolynomials: veciter
using Mosek: msk_global_env, Env, deletetask
using StandardPacked: SPMatrix
import PolynomialOptimization

printstream(msg::String) = (print(msg); flush(stdout))

include("./MosekSOS.jl")
# Do we have Mosek version at least 10?
isdefined(Mosek, :appendafes) && include("./MosekMoment.jl")
include("./Newton.jl")
include("./Tightening.jl")

function __init__()
    isdefined(Mosek, :appendafes) && push!(solver_methods, :MosekMoment)
    pushfirst!(solver_methods, :Mosek, :MosekSOS)
    pushfirst!(Newton.newton_methods, :Mosek)
    pushfirst!(PolynomialOptimization.tightening_methods, :Mosek)
end

@solver_alias Mosek MosekSOS

end