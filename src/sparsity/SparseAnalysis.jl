export sparse_problem, sparse_groupings, sparse_iterate!, sparse_optimize

"""
    SparseAnalysisState

This is the general abstract type for any kind of sparse analysis of a polynomial optimization problem.
Its concrete types can be used for analyzing and optimizing the problem.

See also [`poly_problem`](@ref), [`PolyOptProblem`](@ref), [`sparse_optimize`](@ref).
"""
abstract type SparseAnalysisState end

"""
    sparse_problem(state::SparseAnalysisState)

Return the instance of the [`PolyOptProblem`](@ref) that is associated with the current sparse problem.
"""
function sparse_problem end
"""
    sparse_groupings(state::SparseAnalysisState)

Analyze the current state and return the bases and cliques as indicated by its sparsity.
Return `(groupings, cliques)`, where `groupings` is a vector whose first entry corresponds to the objective and the others to
the respective constraint. The entries themselves contain for every such polynomial a list of all bases used for indexing the
SOS or moment matrices. `cliques` in turn contains a list of sets of variables, each corresponding to a variable clique in the
total problem. In the complex case, only the declared variables are returned.
"""
function sparse_groupings end

"""
    sparse_iterate!(state::SparseAnalysisState; objective=true, zero=true, nonneg=true, psd=true)

Iterate the sparsity, which will lead to a more dense representation and might give better bounds at the expense of a more
costly optimization. Return `nothing` if the iterations converged (`state` did not change any more), else return the new state.

The keyword arguments allow to restrict the iteration to certain elements. This does not necessarily mean that the bases
associated to other elements will not change as well to keep consistency; but their own contribution will not be considered.
The parameters `nonneg` and `psd` may either be `true` (to iterate all those constraints) or a set of integers that refer to
the indices of the constraints, as they were originally given to [`poly_problem`](@ref).

Note that zero constraints are handled by Gröbner basis methods in the real case and hence do not enter into the problem at
all, so they cannot be iterated; the `zero` parameter is only useful for the complex case, in which the constraints are modeled
via inequalities.

    sparse_iterate!(cts::SparsityCorrelativeTerm; term_modes=default, cliques=true, kwargs...)

Allows to overwrite the `term_modes` setting specified while constructing the object for each iteration. This may either be a
`TermMode`, or a vector of `TermMode`s with the same length as there are cliques, or a vector of pairs that maps clique indices
to term modes (unspecified ones will be default, and also a single pair without a vector is valid), or a dictionary.
Note that the choice of the mode will be permanent unless changed again, and this will reflect in the internal storage. In
particular, setting a mode to `tm_none` will remove all term iteration information for this particular clique. If such a mode
is reset to a term sparsity later on, it will again start from scratch.

The keyword arguments `cliques` allow to further restrict the iteration to certain cliques. This does not necessarily mean that
the bases associated to other cliques will not change as well to keep consistency; but their own contribution will not be
considered. If `cliques` is `true`, all cliques are considered; else, it should be a set of the indices of all relevant
cliques (as can be observed by printing the sparsity pattern).
"""
function sparse_iterate! end

# internal function
function sparse_supports end

function Base.show(io::IO, m::MIME"text/plain", x::SparseAnalysisState)
    basis_groupings, variable_groupings = sparse_groupings(x)
    sort!.(variable_groupings, rev=true)
    sort!(variable_groupings, rev=true)
    print(io, typeof(x), " with ", length(sparse_problem(x).constraints), " constraint(s)\nVariable cliques:",
        (s for va in variable_groupings for s in ("\n  ", join(va, ", ")))...,
        "\nBlock sizes:",
        (s for bs in basis_groupings for s in ("\n  ", sort!(collect(StatsBase.countmap(length.(bs))), rev=true)))...)
end

function merge_cliques!(cliques::AbstractVector{<:AbstractSet{T}}) where {T}
    # directly drop cliques of length 1 and 2. They are so efficient (linear vs. quadratic constraints) that we don't even
    # consider them in the merge process
    smallcliques = @view cliques[length.(cliques).≤2]
    cliques = @view cliques[length.(cliques).≥3]
    # first form the clique graph; this time, we work with the adjacency matrix
    n = length(cliques)
    n ≤ 1 && return [smallcliques; cliques]
    @inbounds adjmOwn = [i > j ? length(cliques[i])^3 + length(cliques[j])^3 - length(cliques[i] ∪ cliques[j])^3 : 0
                         for i in 1:n, j in 1:n]
    idxOwn = fill(true, n)
    GC.@preserve adjmOwn idxOwn begin
        adjm = unsafe_wrap(Array, pointer(adjmOwn), (n, n), own=false)
        idx = unsafe_wrap(Array, pointer(idxOwn), n, own=false)
        deleted = 0
        @inbounds while true
            # select two permissible cliques with the highest weight
            w, maxidx = findmax(adjm)
            i = maxidx[1]
            j = maxidx[2]
            # while clique graph contains positive weights
            w ≤ 0 && break
            # merge cliques
            union!(cliques[i], cliques[j])
            idx[j] = false
            # In every iteration, we have to search through n^2 elements, we have to reset n-1 elements and also recalculate
            # n-1 elements. The calculation a^3+b^3-c^3 itself consists of 6 multiplications, one addition and one subtraction
            # (and a couple of loads). Multiplication cost is about 3*addition cost, but with out-of-order execution this may
            # be reduced again, but assuming 20 cycles per calcuation (which is also conditional) is fair. Resetting to zero
            # can be done using ymm registers for four items in one step, but vmovups also has a latency of ~4 (varies greatly
            # between the architectures). But it is unrolled with four instructions following each other, which may partially
            # compensate. And we have the counter add, check and jump, so about 4.25(n-1) clock cycles for the reset would be
            # fair. In comparison, the @inbounds findmax requires about 8 clock cycles per element.
            # So the total cost is 8n^2 [search] + 4.25(n-1) [reset] + 20(n-1) [recalculate] for every iteration, until we are
            # done. If we instead restart the evaluation, we have to recalculate everything: 20n^2, but then the n is smaller.
            # So when is n * (8n^2 + 24.25(n +1)) ≥ deleted * (8n^2 + 24.25(n +1)) + 20(n - deleted)^2 +
            #                                       (n - deleted)*(8(n - deleted)^2 + 24.25(n - deleted +1))
            # As soon as n > deleted ≥ 2, we find this to be fulfilled. However, these theoretical considerations do not seem
            # to be particularly successful.
            deleted += 1
            if n > 100 && deleted == 50
                cliques = @view cliques[idx]
                n -= deleted
                # we already have enough space allocated at adjm; we will now simply overwrite it.
                adjm = unsafe_wrap(Array, pointer(adjmOwn), (n, n), own=false)
                adjm .= [i > j ? length(cliques[i])^3 + length(cliques[j])^3 - length(cliques[i] ∪ cliques[j])^3 : 0
                         for i in 1:n, j in 1:n]
                # same for idx
                idx = unsafe_wrap(Array, pointer(idxOwn), n, own=false)
                idx .= true
                deleted = 0
            else
                # update clique graph
                adjm[j, 1:j-1] .= 0
                adjm[j+1:n, j] .= 0
                # recompute weights for updated clique graph
                newLenCube = length(cliques[i])^3
                for k in 1:i-1
                    idx[k] && (adjm[i, k] = length(cliques[k])^3 + newLenCube - length(cliques[k] ∪ cliques[i])^3)
                end
                for k in i+1:n
                    idx[k] && (adjm[k, i] = newLenCube + length(cliques[k])^3 - length(cliques[k] ∪ cliques[i])^3)
                end
            end
        end
    end
    return [smallcliques; cliques[idx]]
end

merge_cliques(cliques::AbstractVector{<:AbstractVector{T}}) where {T} =
    monomial_vector.(collect.(merge_cliques!(Set.(cliques))))

"""
    sparse_optimize(method, state::SparseAnalysisState; verbose=false, clique_merging=true, solutions=false,
        certificate=false, kwargs...)

Optimize a polynomial optimization problem that was construced via [`poly_problem`](@ref) and wrapped into a
[`SparseAnalysisState`](@ref). Return the optimizer state, the best bound, and a list of potential optimal points. The latter
is a vector of 2-tuples, where the first entry in each tuple corresponds to the optimal point and the second entry yields the
largest violation of a constraint or the difference from the actual value of the objective at this point to the bound,
whichever is larger. Unless this value is numerically zero, the optimal point is bogus; if it is numerically zero, the bound is
the optimal value of the full problem. Multiple optimal points may be returned.
Clique merging is a way to improve the performance of the solver; however, the process itself may be time-consuming and is
therefore disabled by default.
Solution extraction uses [`poly_all_solutions`](@ref) to give a graded list of all possible solution vectors; note that this
may be slow if lots of cliques arise from the sparsity pattern. In this case, it might be better to use the iterator-based
function [`poly_solutions`](@ref) in order to get one solution.
An optimality certificate can be provided if no sparsity pattern was in effect.

Any additional keyword argument is passed on to the solver.

# Methods
The following methods are currently supported:
- `:MosekMoment`: for any kind of problem, requires Mosek 10+ and uses a moment-matrix approach. This is precise and moderately
  fast.
- `:MosekSOS`: for real-valued problems, requires Mosek 9+ and uses a SOS approach. This is precise and typically fastest.
- `:COSMOMoment`: for real-valued problems, requires COSMO and uses a moment-matrix approach. This is imprecise and not too
  fast, but can scale to very large sizes.
- `:HypatiaMoment`: for any kind of problem, requires Hypatia. This is moderately precise and not too fast.

See also [`poly_all_solutions`](@ref), [`poly_solutions`](@ref), [`poly_solution_badness`](@ref),
[`optimality_certificate`](@ref).
"""
function sparse_optimize(v::V, state::SparseAnalysisState; verbose::Bool=false, clique_merging::Bool=false,
    solutions::Bool=false, certificate::Bool=false, kwargs...) where {V<:Val}
    @verbose_info("Determining groupings...")
    t = @elapsed begin
        groupings, _ = sparse_groupings(state)
    end
    @verbose_info("Determined grouping in ", t, " seconds")
    if clique_merging
        clique_merging && @verbose_info("Merging cliques...")
        t = @elapsed begin
            groupings .= merge_cliques.(groupings)
        end
        @verbose_info("Cliques merged in ", t, " seconds. Block sizes:")
    else
        @verbose_info("Clique merging disabled. Block sizes:")
    end
    @verbose_info(sort!(collect(StatsBase.countmap(length.(Iterators.flatten(groupings)))), rev=true), "\nStarting optimization")
    problem = sparse_problem(state)
    result = sparse_optimize(v, problem, groupings; verbose, kwargs...)
    problem.last_objective = result[2]
    if solutions
        result = (result..., poly_all_solutions(state; verbose, method=default_solution_method(state, v)))
    end
    if certificate
        if state isa SparsityNone
            result = (result..., optimality_certificate(problem))
        else
            result = (result..., :CertificateUnavailable)
        end
    end
    return result
end

sparse_optimize(s::Symbol, rest...; kwrest...) = sparse_optimize(Val(s), rest...; kwrest...)

last_moments(state::SparseAnalysisState) = last_moments(sparse_problem(state))
last_objective(state::SparseAnalysisState) = last_objective(sparse_problem(state))

function Base.mergewith(combine, d::AbstractVector{Pair{K,V}}) where {K,V}
    return mergewith(combine, d, K, V)
end

function Base.mergewith(combine, itr, K::Type, V::Type)
    result = Dict{K,V}()
    Base.haslength(itr) && sizehint!(result, length(itr))
    for (k, v) in itr
        result[k] = haskey(result, k) ? combine(result[k], V(v)) : V(v)
    end
    return result
end

for fun in (:variables, :nvariables, :degree)
    @eval MultivariatePolynomials.$fun(state::SparseAnalysisState) = $fun(sparse_problem(state))
end
Base.isreal(state::SparseAnalysisState) = isreal(sparse_problem(state))

printstream(msg::String) = (print(msg); flush(stdout))