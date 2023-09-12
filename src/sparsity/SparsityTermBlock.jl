export SparsityTermBlock

struct SparsityTermBlock <: SparsityTerm
    logic::SparsityTermLogic

    @doc """
        SparsityTermBlock(problem::PolyOptProblem)

    Analyze the term sparsity of the problem.
    [Term sparsity](https://doi.org/10.1137/19M1307871) is a recent iterative sparsity analysis that groups terms with shared
    supports. Its last iteration will give the same value as the original problem, although it may still be smaller.
    Often, even the uniterated analysis already gives the same bound as the dense problem.
    The terms are grouped based on connected components of a graph; this can be improved by using chordal cliques, which will
    lead to even smaller problem sizes, but typically also worse bounds.

    See also [`poly_problem`](@ref), [`sparse_optimize`](@ref), [`SparsityTermCliques`](@ref),
    [`SparsityCorrelativeTerm`](@ref).

        SparsityTermBlock(term_sparsity::SparsityTerm)
    
    Re-interprets a different kind of term sparsity pattern as term sparsity. Note that the data between `term_sparsity` and
    the resulting object will be shared - `sparse_groupings` will always give back the result of the last iteration with the
    type that the pattern had _at the time of iteration_.

    See also [`SparsityTerm`](@ref).
    """
    function SparsityTermBlock(problem::PolyOptProblem, args...)
        return new(SparsityTermLogic(problem, args..., :block))
    end

    function SparsityTermBlock(term_sparsity::SparsityTermBlock)
        return new(term_sparsity)
    end
end

sparse_iterate!(ts::SparsityTermBlock; kwargs...) = isnothing(sparse_iterate!(ts.logic, :block; kwargs...)) ? nothing : ts

function extend_graphs!(g::Graphs.SimpleGraph{I}, ::Val{:block}) where {I<:Integer}
    ne = 0
    # connected components are not overlapping
    connections = Graphs.connected_components(g)
    @inbounds for cc in connections
        @assert(issorted(cc))
        for c in cc
            if length(g.fadjlist[c]) != length(cc)
                g.fadjlist[c] = copy(cc) # this will include self-loops, but this is easier and doesn't bother us
            end
        end
        # the edges are bidirectional, but count only once
        ne += (length(cc) * (length(cc) +1)) >> 1
    end
    g.ne = ne
    return connections
end

extend_graphs!(ts::SparsityTermBlock, indices::AbstractArray{<:Integer}) =
    extend_graphs!.(@view(ts.logic.graphs[indices]), Val(:block))