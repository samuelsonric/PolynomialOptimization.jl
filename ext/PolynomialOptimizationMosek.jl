module PolynomialOptimizationMosek

using PolynomialOptimization, Mosek, MultivariatePolynomials, SparseArrays
using PolynomialOptimization: FastVec, unsafe_push!, finish!, MonomialComplexContainer, PolyOptProblem, pctEqualitySimple,
    pctEqualityGröbner, pctEqualityNonneg, pctNonneg, pctPSD, EmptyGröbnerBasis, solver_methods, newton_methods,
    tightening_methods, @verbose_info, @allocdiff, sqrt2
using Mosek: msk_global_env, Env, deletetask

# Do we have Mosek version at least 10?
isdefined(Mosek, :appendafes) && include("./Mosek/MosekMoment.jl")
include("./Mosek/MosekSOS.jl")
include("./Mosek/Newton.jl")
include("./Mosek/Tightening.jl")

function __init__()
    isdefined(Mosek, :appendafes) && pushfirst!(solver_methods, :MosekMoment)
    pushfirst!(solver_methods, :MosekSOS)
    pushfirst!(newton_methods, :Mosek)
    pushfirst!(tightening_methods, :Mosek)
end

end