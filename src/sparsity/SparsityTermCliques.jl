export SparsityTermCliques

struct SparsityTermCliques <: SparsityTerm
    logic::SparsityTermLogic
    chordal_completion::Bool

    @doc """
        SparsityTermCliques(problem::PolyOptProblem; chordal_completion=true)

    Analyze the term sparsity of the problem using chordal cliques.
    [Chordal term sparsity](https://doi.org/10.1137/20M1323564) is a recent iterative sparsity analysis that groups terms
    with shared supports. Even in its last iteration, it may give strictly smaller values than the dense problem.
    The basis elements are grouped in terms of chordal cliques of the term sparsity graph. The completion of the graph to a
    chordal one may be turned off by using the parameter `chordal_completion`, leading to even smaller problem sizes, which may
    however not have a sound theoretical justification (but can still work).

    See also [`poly_problem`](@ref), [`sparse_optimize`](@ref), [`SparsityTermBlock`](@ref),
    [`SparsityCorrelativeTerm`](@ref).

        SparsityTermCliques(term_sparsity::SparsityTerm; chordal_completion=true)

    Re-interprets a different kind of term sparsity pattern as chordal term sparsity. Note that the data between
    `term_sparsity` and the resulting object will be shared - `sparse_groupings` will always give back the result of the last
    iteration with the type that the pattern had _at the time of iteration_.

    See also [`SparsityTerm`](@ref).
    """
    function SparsityTermCliques(problem::PolyOptProblem, args...; chordal_completion::Bool=true)
        return new(SparsityTermLogic(problem, args..., chordal_completion ? :chordal_cliques : :cliques), chordal_completion)
    end

    function SparsityTermCliques(term_sparsity::SparsityTerm; chordal_completion::Bool=true)
        return new(term_sparsity.logic, chordal_completion)
    end
end

sparse_iterate!(ts::SparsityTermCliques; chordal_completion::Bool=ts.chordal_completion, kwargs...) =
   isnothing(sparse_iterate!(ts.logic, chordal_completion ? :chordal_cliques : :cliques; kwargs...)) ? nothing : ts

function extend_graphs!(g::Graphs.SimpleGraph{I}, ::Val{:cliques}) where {I<:Integer}
    # cliques can be overlapping
    cliques = sort!(sort!.(Graphs.maximal_cliques(g)))
    edge_sets = [Set{I}() for _ in 1:Graphs.nv(g)]
    @inbounds for mc in cliques
        for c in mc
            union!(edge_sets[c], mc)
        end
    end
    ne = 0
    @inbounds for (i, edges) in enumerate(edge_sets)
        g.fadjlist[i] = sort!(collect(edges))
        ne += (length(edges) * (length(edges) +1)) >> 1
    end
    g.ne = ne
    return cliques
end

extend_graphs!(g::Graphs.SimpleGraph{I}, ::Val{:chordal_cliques}) where {I<:Integer} = chordal_cliques!(g)

extend_graphs!(ts::SparsityTermCliques, indices::AbstractArray{<:Integer}) =
    extend_graphs!.(@view(ts.logic.graphs[indices]), ts.chordal_completion ? Val(:chordal_cliques) : Val(:cliques))