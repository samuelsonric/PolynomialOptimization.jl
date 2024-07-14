module PolynomialOptimizationLancelot

using MultivariatePolynomials, LinearAlgebra, StandardPacked, Printf, PolynomialOptimization
using PolynomialOptimization: @assert, @inbounds, @verbose_info, Solver
using GALAHAD: libgalahad_double
import StaticPolynomials

include("./Bindings.jl")
include("./Lancelot.jl")

function __init__()
    push!(Solver.solver_methods, :Lancelot)
    # This weakdep introduces a new binding, which should be accessible and documented in the main package.
    PolynomialOptimization.LANCELOT_simple = LANCELOT_simple
    POmeta = Docs.meta(PolynomialOptimization)
    POLmeta = Docs.meta(PolynomialOptimizationLancelot)
    PObinding = Docs.Binding(PolynomialOptimization, :LANCELOT_simple)
    POLbinding = Docs.Binding(PolynomialOptimizationLancelot, :LANCELOT_simple)
    POmeta[PObinding] = POLmeta[POLbinding]
    delete!(POLmeta, POLbinding)
end

end