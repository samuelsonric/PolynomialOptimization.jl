export poly_optimize, optimality_certificate

include("./Result.jl")
include("./MomentMatrix.jl")
include("./OptimalityCertificate.jl")
include("./CliqueMerging.jl")
include("./solver/Solver.jl")
using .Solver: default_solver_method, monomial_count
import .Solver: poly_optimize

"""
    poly_optimize([method, ]relaxation::AbstractRelaxation; verbose=false,
        clique_merging=false, solutions::Bool=false, certificate::Bool=false, kwargs...)

Optimize a relaxed polynomial optimization problem that was construced via [`poly_problem`](@ref) and then wrapped into an
[`AbstractRelaxation`](@ref). Returns a [`Result`](@ref) object.

Clique merging is a way to improve the performance of the solver in case a sparse analysis led to cliques with a lot of
overlap; however, the process itself may be time-consuming and is therefore disabled by default.

`verbose=true` will enable logging; this will print basic information about the relaxation itself as well as instruct the
solver to output a detailed log. The PSD block sizes reported accurately represent the side dimensions of semidefinite
variables and how many of these variables appear. The free block sizes are only very loose upper bounds on the maximal number
of equality constraints that will be constructed by multiplying two elements from a block, as duplicates will be ignored.
Any additional keyword argument is passed on to the solver.

For a list of supported methods, see [the solver reference](@ref solvers_poly_optimize). If `method` is omitted, the default
solver is used. Note that this depends on the loaded solver packages, and possibly also their loading order if no preferred
solver has been loaded.
"""
function poly_optimize(v::Val{S}, relaxation::AbstractRelaxation; verbose::Bool=false, clique_merging::Bool=false, kwargs...) where {S}
    otime = @elapsed begin
        @verbose_info("Beginning optimization...")
        groups = groupings(relaxation) # This is instantaneous, as the groupings were already calculated when the relaxation
                                       # was constructed.
        if clique_merging
            clique_merging && @verbose_info("Merging cliques...")
            t = @elapsed begin
                groups = merge_cliques(groups)
            end
            @verbose_info("Cliques merged in ", t, " seconds.")
        else
            @verbose_info("Clique merging disabled.")
        end
        if verbose
            bs = StatsBase.countmap(length.(groups.obj))
            @unroll for constrs in (groups.nonnegs, groups.psds)
                for constr in constrs
                    mergewith!(+, bs, StatsBase.countmap(length.(constr)))
                end
            end
            print("PSD block sizes:\n  ", sort!(collect(bs), rev=true))
            if !isempty(groups.zeros)
                empty!(bs)
                for constr in groups.zeros
                    mergewith!(+, bs, StatsBase.countmap(length.(constr)))
                end
                print("\nFree block sizes:\n  ", sort!(collect(bs), rev=true))
            end
            println("\nStarting solver...")
        end
        result = poly_optimize(v, relaxation, groups; verbose, kwargs...)
    end
    return Result(relaxation, S, otime, result...)
end

"""
    poly_optimize([method, ]problem::Problem[, degree::Int]; kwargs...)

Construct a [`Relaxation.Dense`](@ref) by default.
"""
poly_optimize(v::Val, problem::Problem, rest...; kwargs...) =
    poly_optimize(v, Relaxation.Dense(problem, rest...); kwargs...)

poly_optimize(s::Symbol, rest...; kwrest...) = poly_optimize(Val(s), rest...; kwrest...)

function poly_optimize(args...; kwargs...)
    if !isempty(args) && args[1] isa Val
        error("Unknown solver method specified. Are the required solver packages loaded?")
    end
    method = default_solver_method()
    @info("No solver method specified: choosing $method")
    poly_optimize(Val(method), args...; kwargs...)
end