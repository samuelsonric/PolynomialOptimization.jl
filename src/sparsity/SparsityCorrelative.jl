export SparsityCorrelative

mutable struct SparsityCorrelative <: SparseAnalysisState
    problem::PolyOptProblem
    graph::Graphs.SimpleGraph
    chordal_completion::Bool
    cache

    @doc """
        SparsityCorrelative(problem::PolyOptProblem; chordal_completion=true)

    Analyze the correlative sparsity of a problem.
    [Correlative sparsity](https://doi.org/10.1137/050623802) is a rough sparsity analysis that groups the variables into
    cliques based on the constraints in which they appear (or the terms of the objective in which they appear). By default,
    the correlative sparsity graph is completed to a chordal graph before the cliques are determined, which guarantees that the
    maximal cliques can be determined quickly; however, this may degrade the sparsity and it may be favorable not to carry out
    the completion.
    Correlative sparsity patterns cannot be iterated.

    See also [`poly_problem`](@ref), [`sparse_optimize`](@ref), [`chordal_cliques!`](@ref),
    [`SparsityCorrelativeTerm`](@ref).
    """
    function SparsityCorrelative(problem::PolyOptProblem; chordal_completion::Bool=true)
        g = Graphs.SimpleGraph(length(problem.variables))
        # objective: check which pairs of exponents appear together in a term
        for term in problem.objective
            for (var1, var2) in Combinatorics.combinations(problem.complex ? ordinary_variable(effective_variables(term)) :
                                                           effective_variables(term), 2)
                Graphs.add_edge!(g, Graphs.Edge(problem.var_map[var1], problem.var_map[var2]))
            end
        end
        # constraints: check which variables are jointly present in a constraint at all
        for constr in problem.constraints
            for (var1, var2) in Combinatorics.combinations(problem.complex ? ordinary_variable(effective_variables(constr.constraint)) :
                                                        effective_variables(constr.constraint), 2)
                Graphs.add_edge!(g, Graphs.Edge(problem.var_map[var1], problem.var_map[var2]))
            end
        end
        return new(problem, g, chordal_completion, nothing)
    end
end

sparse_problem(cs::SparsityCorrelative) = cs.problem

sparse_iterate!(::SparsityCorrelative; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

function sparse_groupings(cs::SparsityCorrelative; chordal_completion::Bool=cs.chordal_completion, verbose::Bool=false,
    return_assignments::Bool=false)
    if cs.cache === nothing || cs.cache[1] != chordal_completion
        problem = sparse_problem(cs)
        @verbose_info("Determining cliques")
        gentime = @elapsed(cliques = chordal_completion ? chordal_cliques!(cs.graph) : Graphs.maximal_cliques(cs.graph))
        @verbose_info("Obtained cliques in ", gentime, " seconds. Generating objective basis partitions.")
        variable_groupings = getindex.((problem.variables,), cliques)
        sort!(variable_groupings, by=length)
        sort!.(variable_groupings, rev=true)
        if cs.problem.complex
            inequality_groups = Int[findfirst(v -> effective_variables_in_complex(constr.constraint, v), variable_groupings)
                                    for constr in problem.constraints]
        else
            inequality_groups = Int[findfirst(v -> effective_variables_in_real(constr.constraint, v), variable_groupings)
                                    for constr in problem.constraints]
        end
        data = Vector{Vector{typeof(problem.basis)}}(undef, length(problem.constraints) +1)
        @inbounds data[1] = Vector{typeof(problem.basis)}(undef, length(variable_groupings))
        gentime = @elapsed begin
            Threads.@threads for i in 1:length(variable_groupings)
                @inbounds data[1][i] = subbasis(problem.basis, variable_groupings[i])
            end
        end
        @verbose_info("Objective basis partitions generated in ", gentime, " seconds. Generating constraint basis partitions.")
        gentime = @elapsed begin
            Threads.@threads for i in 2:length(problem.constraints)+1
                @inbounds data[i] = [subbasis(problem.constraints[i-1].basis, variable_groupings[inequality_groups[i-1]])]
            end
        end
        @verbose_info("Constraint basis partitions generated in ", gentime, " seconds.")
        cs.cache = (chordal_completion, data, variable_groupings, (variable_groupings, inequality_groups))
    end
    if return_assignments
        return cs.cache[2:end]
    else
        return cs.cache[2], cs.cache[3]
    end
end

default_solution_method(::SparsityCorrelative, ::Any) = :mvhankel