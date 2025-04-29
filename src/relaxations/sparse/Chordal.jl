using CliqueTrees, Graphs
using CliqueTrees: EliminationAlgorithm

"""
    chordal_completion!(G::Graphs.SimpleGraph; alg::EliminationAlgorithm=MF())

Augment `G` by a chordal completion using a greedy minimal fill-in, and also return a perfect elimination ordering.
This is a more efficient implementation of `ChordalGraph.jl/GreedyOrder` (for the "MF" case).

See also [`ChordalGraph.jl`](https://github.com/wangjie212/ChordalGraph)

"""
function chordal_completion!(graph::SimpleGraph; alg::EliminationAlgorithm=MF())
    # compute a tree decomposition using
    # the given algorithm
    order, tree = cliquetree(graph; alg)
    filledgraph = FilledGraph(tree)
    
    # triangulate graph
    for v in vertices(filledgraph)
        for w in neighbors(filledgraph, v)
            add_edge!(graph, order[v], order[w])
        end
    end
    
    return graph, order
end

"""
    chordal_cliques!(G::Graphs.SimpleGraph; alg::EliminationAlgorithm=MF())

Make the given graph chordal, and then calculate its maximal cliques.
This is almost the same implementation as in `ChordalGraph.jl/chordal_cliques!`.

See also [`chordal_completion!`](@ref), [`ChordalGraph.jl`](https://github.com/wangjie212/ChordalGraph)
"""
function chordal_cliques!(graph::SimpleGraph; alg::EliminationAlgorithm=MF())
    # compute a tree decomposition using
    # the given algorithm
    order, tree = cliquetree(graph; alg)
    filledgraph = FilledGraph(tree)
    
    # triangulate graph
    for v in vertices(filledgraph)
        for w in neighbors(filledgraph, v)
            add_edge!(graph, v, w)
        end
    end
    
    # compute maximal cliques
    cliques = Vector{Vector{Int}}(undef, length(tree))
    
    for (i, clique) in enumerate(tree)
        cliques[i] = sort!(order[clique])
    end

    return cliques
end
