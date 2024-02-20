export poly_optimize, optimality_certificate

include("./Result.jl")
include("./MomentMatrix.jl")
include("./OptimalityCertificate.jl")
include("./CliqueMerging.jl")

"""
    poly_optimize(method, relaxation::AbstractPORelaxation; verbose=false, clique_merging=false, solutions::Bool=false,
        certificate::Bool=false, kwargs...)

Optimize a relaxed polynomial optimization problem that was construced via [`poly_problem`](@ref) and then wrapped into a
[`AbstractPORelaxation`](@ref). Returns a [`POResult`](@ref) object.

Clique merging is a way to improve the performance of the solver in case a sparse analysis led to cliques with a lot of
overlap; however, the process itself may be time-consuming and is therefore disabled by default.

Any additional keyword argument is passed on to the solver.

For a list of supported methods, see [the solver reference](@ref solvers_poly_optimize).


    poly_optimize(relaxation::AbstractPORelaxation; kwargs...)

Uses the default solver. Note that this depends on the loaded solver packages, and possibly also their loading order if no
preferred solver has been loaded.


    poly_optimize([method, ]problem::POProblem[, degree::Int]; kwargs...)

Construct a [`RelaxationDense`](@ref) by default.
"""
function poly_optimize(v::Val{S}, relaxation::AbstractPORelaxation; verbose::Bool=false, clique_merging::Bool=false, kwargs...) where {S}
    otime = @elapsed begin
        @verbose_info("Determining groupings...")
        t = @elapsed begin
            groups = groupings(relaxation)
        end
        @verbose_info("Determined grouping in ", t, " seconds")
        if clique_merging
            clique_merging && @verbose_info("Merging cliques...")
            t = @elapsed begin
                groups = merge_cliques(relaxation, groups)
            end
            @verbose_info("Cliques merged in ", t, " seconds. Block sizes:")
        else
            @verbose_info("Clique merging disabled. Block sizes:")
        end
        if verbose
            block_sizes = Dict{Int,Int}()
            for m in groups.obj
                block_sizes[length(m)] = get(block_sizes, length(m), 0) +1
            end
            for constrgroup in (groups.zeros, groups.nonnegs, groups.psds)
                for constr in constrgroup
                    for m in constr
                        block_sizes[length(m)] = get(block_sizes, length(m), 0) +1
                    end
                end
            end
            println(sort!(collect(block_sizes), rev=true), "\nStarting optimization")
        end
        result = poly_optimize(v, relaxation, groups; verbose, kwargs...)
    end
    return POResult(relaxation, S, otime, result...)
end

poly_optimize(v::Val, problem::POProblem, degree=problem.mindegree; kwargs...) =
    poly_optimize(v, RelaxationDense(problem, degree); kwargs...)

poly_optimize(s::Symbol, rest...; kwrest...) = poly_optimize(Val(s), rest...; kwrest...)

function poly_optimize(args...; kwargs...)
    if !isempty(args) && args[1] isa Val
        error("Unknown solver method specified. Are the required solver packages loaded?")
    end
    method = default_solver_method()
    @info("No solver method specified: choosing $method")
    poly_optimize(Val(method), args...; kwargs...)
end

const solver_methods = Symbol[]

function default_solver_method()
    isempty(solver_methods) && error("No solver method is available. Load a solver package that provides such a method (e.g., Mosek)")
    return first(solver_methods)
end

include("./SOSInterface.jl")
include("./SOSHelpers.jl")

include("./SolutionExtraction.jl")