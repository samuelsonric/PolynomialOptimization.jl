module PolynomialOptimizationCOPT

using COPT, MultivariatePolynomials, SparseArrays, PolynomialOptimization.Solver
using PolynomialOptimization: @assert, @inbounds
using COPT: _check_ret, Env#, libcopt

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

include("./COPTMoment.jl")

function __init__()
    global copt_env = Env()
    pushfirst!(solver_methods, :COPT, :COPTMoment)
end

@solver_alias COPT COPTMoment

end