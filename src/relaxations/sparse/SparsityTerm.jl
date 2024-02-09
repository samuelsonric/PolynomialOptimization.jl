export SparsityTerm

# Here, we provide the basic utilities for term sparsity-like methods. In particular, this is the representation of the problem
# by means of a term sparsity graph and a way to perform the iteration.
# However, an important part of all iterations is the extension of the graphs itself, which is not implemented in the basic
# logic class, but must instead be done by its encapulators (which should be subclasses, but Julia doesn't allow this).
mutable struct SparsityTermLogic{P<:POProblem,U<:AbstractVector,S<:Vector{<:AbstractVector},B<:Vector{<:AbstractVector}}
    problem::P
    union_supports::U
    polynomial_supports::S
    bases::B
    graphs::Vector{Graphs.SimpleGraph{Int}}
    cache::Union{Vector{Vector{Vector{Int}}},Nothing}

    function SparsityTermLogic(problem::POProblem{P,M}, union_supports::AbstractVector{M}, kind::Symbol) where {P,M}
        polynomial_supports = [monomials(variables(problem.objective), [0]),
                               monomials.(getfield.(problem.constraints, :constraint))...]
        bases = [problem.basis, getfield.(problem.constraints, :basis)...]
        result = new{typeof(problem),typeof(union_supports),typeof(polynomial_supports),typeof(bases)}(
            problem, union_supports, polynomial_supports, bases,
            Vector{Graphs.SimpleGraph}(undef, length(problem.constraints) +1)
        )
        # In the zeroth step, the objective graph is just the term sparsity graph, and all the other graphs are empty.
        # But note that due to the choice union_supports[1] = [1], the generic supports_to_graphs! routine works also to
        # generate the term sparsity graph.
        # In fact, the zeroth step is never accessed directly; instead, the first graph relevant for groupings is step 1.
        # However, the objective graph will not change in the first step apart from extension, and the constraint graphs will
        # be exactly the result of supports_to_graphs!.
        supports_to_graphs!(result, 1:length(problem.constraints)+1)
        result.cache = extend_graphs!.(result.graphs, Val(kind))
        return result
    end
end

function SparsityTermLogic(problem::POProblem{P,M}, kind::Symbol) where {P,M}
    if problem.complex
        union_supports = merge_monomial_vectors([monomials(problem.objective),
            monomials.(getfield.(problem.constraints, :constraint))...,
            squarebasis(problem.basis[isreal.(problem.basis)], problem.gröbner_basis)])
    else
        union_supports = merge_monomial_vectors([monomials(problem.objective),
            monomials.(getfield.(problem.constraints, :constraint))...,
            squarebasis(problem.basis, problem.gröbner_basis)])
    end
    return SparsityTermLogic(problem, union_supports, kind)
end

sparse_problem(ts::SparsityTermLogic) = ts.problem

@inline function unsafe_insorted_monomials(supp, mon)
    # assume: supp is sorted in ascending order
    l, u = 1, length(supp)
    @inbounds while l ≤ u
        m = (l + u) >> 1
        val = supp[m]
        if val < mon
            l = m +1
        elseif val > mon
            u = m -1
        else
            return true
        end
    end
    return false
end

function supports_to_graphs!(ts::SparsityTermLogic, indices::AbstractVector{Int})
    problem = sparse_problem(ts)
    supp = ts.union_supports
    if problem.complex
        for k in indices
            basis = ts.bases[k] # better do bounds checking here, this is an uncontrolled parameter
            graph = Graphs.SimpleGraph(length(basis))
            @inbounds for var1I in 1:length(basis)
                var1 = basis[var1I]
                var1conj = conj(var1)
                for var2I in (var1 == var1conj ? var1I+1 : 1):length(basis)
                    var2 = basis[var2I]
                    for supp_constr in ts.polynomial_supports[k]
                        check = monomials(rem(supp_constr * var1conj * var2, problem.gröbner_basis))
                        for mon in check
                            if unsafe_insorted_monomials(supp, mon)
                                Graphs.add_edge!(graph, Graphs.Edge(var1I, var2I))
                                @goto foundcp
                            end
                        end
                    end
                    @label foundcp
                end
            end
            ts.graphs[k] = graph
        end
    else
        for k in indices
            basis = ts.bases[k] # better do bounds checking here, this is an uncontrolled parameter
            graph = Graphs.SimpleGraph(length(basis))
            # Note: The paper on chordal term sparsity assumes (claiming "without loss of generality," but rather "without
            # justification") that 1 ∈ supp(everything). Neither the (block) term sparsity nor the correlative/term sparsity
            # paper follow this assumption, and we also don't do it. However, note that TSSOS always considers the constant
            # term to be present if a Newton basis is calculated in the unconstrained case (but this can be turned off setting
            # the parameter feasible=true in tssos_first).
            @inbounds for var1I in 1:length(basis)
                var1 = basis[var1I]
                for var2I in var1I+1:length(basis)
                    var2 = basis[var2I]
                    for supp_constr in ts.polynomial_supports[k]
                        check = monomials(rem(supp_constr * var1 * var2, problem.gröbner_basis))
                        for mon in check
                            if unsafe_insorted_monomials(supp, mon)
                                Graphs.add_edge!(graph, Graphs.Edge(var1I, var2I))
                                @goto found
                            end
                        end
                    end
                    @label found
                end
            end
            @inbounds ts.graphs[k] = graph
        end
    end
    ts.cache = nothing
    return ts
end

function extend_graphs! end

function groupings(ts::SparsityTermLogic)
    return [[basis[i] for i in clique] for (basis, clique) in zip(ts.bases, ts.cache)], [ts.problem.variables]
end

function supports(ts::SparsityTermLogic{P}) where {PP,M,P<:POProblem{PP,M}}
    # Note: Our implementation follows takes into account _all_ graphs (including those from constraints). The TSSOS
    # implementation instead takes into account only the support of the objective (which, apart from a typo in the chordal
    # TSSOS paper, does not seem to be well-founded; in particular, their CS-TSSOS implementation also considers all graphs).
    problem = sparse_problem(ts)
    if problem.complex
        if problem.gröbner_basis isa EmptyGröbnerBasis
            @inbounds return monomial_vector([
                mon
                for (polynomial_support, graph, basis) in zip(ts.polynomial_supports, ts.graphs, ts.bases)
                for (poly_supp, edge) in Iterators.product(polynomial_support, Graphs.edges(graph))
                for mon::M in (poly_supp * basis[edge.src] * conj(basis[edge.dst]),
                               poly_supp * basis[edge.dst] * conj(basis[edge.src]))
            ])
        else
            @inbounds return merge_monomial_vectors([
                monomials(rem(mon, problem.gröbner_basis))
                for (polynomial_support, graph, basis) in zip(ts.polynomial_supports, ts.graphs, ts.bases)
                for (poly_supp, edge) in Iterators.product(polynomial_support, Graphs.edges(graph))
                for mon::M in (poly_supp * basis[edge.src] * conj(basis[edge.dst]),
                               poly_supp * basis[edge.dst] * conj(basis[edge.src]))
            ])
        end
    else
        if problem.gröbner_basis isa EmptyGröbnerBasis
            @inbounds return monomial_vector(M[
                poly_supp * basis[edge.src] * basis[edge.dst]
                for (polynomial_support, graph, basis) in zip(ts.polynomial_supports, ts.graphs, ts.bases)
                for (poly_supp, edge) in Iterators.product(polynomial_support, Graphs.edges(graph))
            ])
        else
            @inbounds return merge_monomial_vectors([
                monomials(rem(poly_supp * basis[edge.src] * basis[edge.dst], problem.gröbner_basis))
                for (polynomial_support, graph, basis) in zip(ts.polynomial_supports, ts.graphs, ts.bases)
                for (poly_supp, edge) in Iterators.product(polynomial_support, Graphs.edges(graph))
            ])
        end
    end
end

function iterate!(ts::SparsityTermLogic, kind::Symbol=:block; objective::Bool=true,
    zero::Union{Bool,<:AbstractSet{<:Integer}}=true, nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true,
    psd::Union{Bool,<:AbstractSet{<:Integer}}=true)
    problem = sparse_problem(ts)
    new_supports = supports(ts)
    # Even if new_supports is the same as ts.union_supports, we might run into the unlikely scenario that the user changed the
    # type of term sparsity, so that now extending with a different type might lead to a different result despite having the
    # same support.
    old_cache = ts.cache
    ts.union_supports = new_supports
    indices = poly_structure_indices(problem, objective, zero, nonneg, psd)
    supports_to_graphs!(ts, indices)
    ts.cache = extend_graphs!.(@view(ts.graphs[indices]), Val(kind))
    return old_cache == ts.cache ? nothing : ts
end

"""
    SparsityTerm

This is an abstract superclass for different types of term sparsity that just differ in the way their underlying graph
representation is extended (resp. converted into groupings).

See also [`SparsityTermBlock`](@ref), [`SparsityTermCliques`](@ref).
"""
abstract type SparsityTerm <: AbstractSPOProblem end

sparse_problem(stl::SparsityTerm) = sparse_problem(stl.logic)

groupings(stl::SparsityTerm) = groupings(stl.logic)

supports(stl::SparsityTerm) = supports(stl.logic)

default_solution_method(::SparsityTerm) = :heuristic