# while this is not a weak dependecy as the solver interface is shipped with PO, we follow the naming convention
module PolynomialOptimizationLoRADS

using ..Solvers.LoRADS, MultivariatePolynomials, LinearAlgebra, Printf, ..Solver
using ..PolynomialOptimization: @assert, @inbounds
using ..Solvers.LoRADS: LoRADSInt

include("./LoRADSMoment.jl")

__init__() = !isempty(LoRADS.solverlib) && push!(solver_methods, :LoRADS, :LoRADSMoment)

@solver_alias LoRADS LoRADSMoment

end