module PolynomialOptimizationCOSMO

using PolynomialOptimization, COSMO, MultivariatePolynomials, SparseArrays
using PolynomialOptimization: FastVec, finish!, PolyOptProblem, pctEqualitySimple, pctEqualityGröbner, pctEqualityNonneg,
    pctNonneg, pctPSD, EmptyGröbnerBasis, solver_methods, @verbose_info
using COSMO: AbstractConvexSet, Constraint, Model, Settings, ZeroSet, PsdCone, optimize!

include("./COSMO/COSMOMoment.jl")

__init__() = push!(solver_methods, :COSMOMoment)

end