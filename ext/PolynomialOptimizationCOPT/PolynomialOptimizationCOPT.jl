module PolynomialOptimizationCOPT

using PolynomialOptimization, COPT, MultivariatePolynomials, SparseArrays, PolynomialOptimization.FastVector,
    PolynomialOptimization.Solver
using PolynomialOptimization: @assert, @inbounds, POProblem, RelaxationGroupings, @verbose_info, MomentVector, StackVec,
    FastKey, sort_along!
using PolynomialOptimization.SimplePolynomials: monomial_index, _get_I
using COPT: _check_ret, Env

global copt_env::Env

mutable struct COPTProb
    ptr::Ptr{copt_prob}
    finalize_called::Bool

    function COPTProb(env::Env)
        p_ptr = Ref{Ptr{copt_prob}}(C_NULL)
        _check_ret(env, COPT_CreateProb(env.ptr, p_ptr))
        problem = new(p_ptr[], false)
        finalizer(problem) do p
            p.finalize_called = true
            if p.ptr != C_NULL
                COPT_DeleteProb(Ref(p.ptr))
                p.ptr = C_NULL
            end
        end
        return problem
    end
end

Base.unsafe_convert(::Type{Ptr{copt_prob}}, problem::COPTProb) = problem.ptr

# include("./COPTSOS.jl")
include("./COPTMoment.jl")

function __init__()
    global copt_env = Env()
    push!(Solver.solver_methods, :COPTMoment)
    # push!(Solver.solver_methods, :COPTSOS)
end

end