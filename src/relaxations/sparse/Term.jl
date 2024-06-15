mutable struct SparsityTerm{
    I<:Integer,
    P<:Problem{<:(SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}} where {Nr,Nc})},
    PG<:RelaxationGroupings,
    U<:AbstractSet{I},
    S<:SimpleMonomialVector,
    G<:RelaxationGroupings,
} <: AbstractRelaxationSparse{P}
    const problem::P
    const parentgroupings::PG
    support_union::Set{I}
    const localizing_supports::Vector{S}
    const graphs::Vector{Graphs.SimpleGraph{Int}}
    groupings::G
    const method::Symbol

    function SparsityTerm(relaxation::AbstractRelaxation{P},
        support_union::AbstractSet{I}; method::Symbol, verbose::Bool=false) where
        {Nr,Nc,I<:Integer,MV<:SimpleMonomialVector{Nr,Nc,I},P<:Problem{<:SimplePolynomial{<:Any,Nr,Nc,MV}}}
        problem = poly_problem(relaxation)
        @verbose_info("Generating localizing supports")
        localizing_supports = Vector{MV}(undef, 1 + length(problem.constr_zero) + length(problem.constr_nonneg) +
            length(problem.constr_psd))
        @inbounds localizing_supports[1] = monomials(problem.prefactor)
        i = 2
        for constrs in (problem.constr_zero, problem.constr_nonneg, problem.constr_psd)
            for constr in constrs
                @inbounds localizing_supports[i] = monomials(constr)
                i += 1
            end
        end
        everything = Set{Int}(1:length(localizing_supports))
        @verbose_info("Converting supports into graphs")
        graphtime = @elapsed begin
            parent = groupings(relaxation)
            graphs = Vector{Graphs.SimpleGraph{Int}}(undef, length(parent.obj) + sum(length, parent.zeros, init=0) +
                sum(length, parent.nonnegs, init=0) + sum(length, parent.psds, init=0))
            _supports_to_graphs!(graphs, support_union, localizing_supports, parent, everything)
        end
        @verbose_info("Obtained graphs in ", graphtime, " seconds. Generating groupings.")
        grouptime = @elapsed begin
            newgroupings = _extend_graphs!(Val(method), parent, graphs, everything)
        end
        @verbose_info("Generated new groupings in ", grouptime, " seconds; intersecting with old.")
        intersecttime = @elapsed begin
            gr = intersect(newgroupings, parent)
        end
        @verbose_info("Obtained intersection in ", intersecttime, " seconds")

        new{I,P,typeof(parent),typeof(support_union),eltype(localizing_supports),typeof(gr)}(
            problem, parent, support_union, localizing_supports, graphs, gr, method
        )
    end
end

# Only a helper so that we can cheaply obtain the monomial union without having to construct the square basis.
struct _SquareBasis{M<:SimpleMonomial,B<:AbstractVector{M}}
    parent::B
end

Base.IteratorSize(::Type{<:_SquareBasis{M,B}}) where {M<:SimpleMonomial,B<:AbstractVector{M}} = Base.IteratorSize(B)
Base.IteratorEltype(::Type{<:_SquareBasis{M}}) where {M} = Base.HasEltype()
Base.eltype(::Type{<:_SquareBasis{M}}) where {M} = SimplePolynomials._get_I(M)
Base.length(b::_SquareBasis{<:SimpleMonomial{<:Any,0}}) = length(b.parent)
Base.length(b::_SquareBasis) = count(isreal, b.parent) # assuming that no conjugates occur, a monomial with complex variables
                                                       # cannot be real-valued
function Base.iterate(b::_SquareBasis{M}, state=missing) where {I<:Integer,M<:SimpleMonomial{<:Any,<:Any,I}}
    result = ismissing(state) ? iterate(b.parent) : iterate(b.parent, state)
    local parent_mon, parent_state
    while true
        isnothing(result) && return nothing
        parent_mon, parent_state = result
        real_mon = true
        for (v, _) in parent_mon # faster than isreal(parent_mon), as we can assume that no conjugates are present
            if !isreal(v)
                real_mon = false
                break
            end
        end
        real_mon && break
        result = iterate(b.parent, parent_state)
    end
    return monomial_index(parent_mon, parent_mon), parent_state
end

function SparsityTerm(relaxation::AbstractRelaxation{P}; method::Symbol, verbose::Bool=false) where
    {Nr,Nc,I<:Integer,P<:Problem{<:SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}}}}
    # Let 𝒜 = supp(obj) ∪ supp(constrs)
    # support_union corresponds to 𝒮. For real-valued problems, the initialization is
    # 𝒮 = 𝒜 ∪ 2grouping(obj), for complex-valued problems, only 𝒜. There is no good explanation given for why we even use
    # 2grouping(obj) - it seems to be good for convergence, it might be related to the Hankel structure... So what to do with
    # mixed problems? Two immediate possibilities:
    # - form the union with all monomials that are squared in the real variables, provided they have at least one real variable
    #   present
    # - same, but only if the monomials don't have a complex variable
    # Here, we choose the second approach (just because we have to choose).
    @verbose_info("Generating initial support union")
    prob = poly_problem(relaxation)
    parent = groupings(relaxation)
    supptime = @elapsed begin
        # When a Problem is constructed using poly_problem, all polynomials are converted from MP interface; therefore, they
        # will all internally use ExponentsAll as the exponents backend. We can therefore directly access the index.
        # However, just to be on the very safe side (maybe someone somehow constructs a Problem directly, or we want to change
        # the implementation later?), we don't rely on this and use monomial_index instead. This will convert the index to
        # ExponentsAll, but has a short path for recognizing when it already is given in this way, so it can be optimized to
        # just grab the field value if everything is as it should be.
        support_union = Set{I}(Iterators.map(monomial_index, monomials(prob.objective)))
        union!(support_union, Iterators.map(monomial_index, monomials(prob.prefactor)))
        # ^ this is for the minimal value that is subtracted from the objective
        for constrs in (prob.constr_zero, prob.constr_nonneg, prob.constr_psd)
            for constr in constrs
                union!(support_union, Iterators.map(monomial_index, monomials(constr)))
            end
        end
        if !iszero(Nr)
            for grouping in parent.obj
                union!(support_union, _SquareBasis(grouping))
            end
        end
    end
    @verbose_info("Generated support union in ", supptime, " seconds")
    return SparsityTerm(relaxation, support_union; method, verbose)
end

function _supports_to_graphs!(graphs::Vector{Graphs.SimpleGraph{Int}}, support_union::AbstractSet{I},
    localizing_supports::(Vector{MV} where {MV<:SimpleMonomialVector}), parent::RelaxationGroupings,
    indices::AbstractSet{Int}) where {I<:Integer}
    ipoly = 1
    igroup = 1
    for constrs in ((parent.obj,), parent.zeros, parent.nonnegs, parent.psds)
        for constr_groupings in constrs
            if ipoly ∈ indices
                localizing_support = localizing_supports[ipoly]
                for grouping in constr_groupings
                    graphs[igroup] = graph = Graphs.SimpleGraph(length(grouping))
                    for (exp2, g₂) in enumerate(grouping)
                        r = isreal(g₂)
                        for (exp1, g₁) in zip(Iterators.countfrom(r ? exp2 +1 : 1), @view(grouping[(r ? exp2 +1 : 1):end]))
                            Graphs.has_edge(graph, exp1, exp2) && continue
                            for supp_constr in localizing_support
                                if monomial_index(g₁, supp_constr, SimpleConjMonomial(g₂)) ∈ support_union
                                    Graphs.add_edge!(graph, Graphs.Edge(exp1, exp2))
                                    break
                                end
                            end
                        end
                    end
                    igroup += 1
                end
            end
            ipoly += 1
        end
    end
    return
end
_supports_to_graphs!(relaxation::SparsityTerm, indices) = _supports_to_graphs!(relaxation.graphs,
    relaxation.support_union, relaxation.localizing_supports, relaxation.parentgroupings, indices)

function _extend_graphs!(::Val{:block}, g::Graphs.SimpleGraph)
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

function _extend_graphs!(::Val{:cliques}, g::Graphs.SimpleGraph{Int})
    # cliques can be overlapping
    cliques = sort!(sort!.(Graphs.maximal_cliques(g)))
    edge_sets = [Set{Int}() for _ in 1:Graphs.nv(g)]
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

_extend_graphs!(::Val{:chordal_cliques}, g::Graphs.SimpleGraph) = chordal_cliques!(g)

@eval function _extend_graphs!(@nospecialize(method::Val), parent::RelaxationGroupings, g::Vector{G},
        indices::AbstractSet{Int}) where {G<:Graphs.SimpleGraph}
    igroup = 1
    newobj = FastVec{Base.promote_op(view, eltype(parent.obj), Vector{Int})}()
    1 ∈ indices && for grouping in parent.obj
        cliques = _extend_graphs!(method, g[igroup]) # enable bounds checking
        for clique in cliques
            push!(newobj, @view(grouping[clique]))
        end
        igroup += 1
    end
    ipoly = 2
    $((quote
        $(Symbol(:new, name)) = Vector{Vector{Base.promote_op(view, eltype(eltype(parent.$name)), Vector{Int})}}(
            undef, length(parent.$name)
        )
        for (i, constr_groupings) in enumerate(parent.$name)
            if ipoly ∈ indices
                newgroup = FastVec{eltype(eltype($(Symbol(:new, name))))}()
                for grouping in constr_groupings
                    cliques = _extend_graphs!(method, g[igroup]) # enable bounds checking
                    for clique in cliques
                        push!(newgroup, @view(grouping[clique]))
                    end
                    igroup += 1
                end
                @inbounds $(Symbol(:new, name))[i] = finish!(newgroup)
            else
                @inbounds $(Symbol(:new, name))[i] = constr_groupings
            end
        end
    end for name in (:zeros, :nonnegs, :psds))...)

    return RelaxationGroupings(finish!(newobj), newzeros, newnonnegs, newpsds, parent.var_cliques)
end

function _extend_graphs!(relaxation::SparsityTerm, method::Symbol, indices)
    newgroupings = _extend_graphs!(Val(method), relaxation.parentgroupings, relaxation.graphs, indices)
    if newgroupings != relaxation.groupings
        relaxation.groupings = newgroupings
        return relaxation
    else
        return nothing
    end
end

function _iterate_supports(parent::RelaxationGroupings, localizing_supports::Vector{MV}, g::Vector{G}) where
    {I<:Integer,Nc,MV<:SimpleMonomialVector{<:Any,Nc,I},G<:Graphs.SimpleGraph}
    support_union = Set{I}()
    ipoly = 1
    igroup = 1
    @inbounds for constrs in ((parent.obj,), parent.zeros, parent.nonnegs, parent.psds)
        for constr_groupings in constrs
            localizing_support = localizing_supports[ipoly]
            for grouping in constr_groupings
                graph = g[igroup]
                sizehint!(support_union, length(support_union) +
                    (iszero(Nc) ? 1 : 2) * Graphs.ne(graph) * length(localizing_support))
                for e in Graphs.edges(graph)
                    for α in localizing_support
                        push!(support_union, monomial_index(grouping[Graphs.src(e)], α,
                                                            SimpleConjMonomial(grouping[Graphs.dst(e)])))
                        if !iszero(Nc) && Graphs.src(e) != Graphs.dst(e) # or noncommutative
                            push!(support_union, monomial_index(grouping[Graphs.dst(e)], α,
                                SimpleConjMonomial(grouping[Graphs.src(e)])))
                        end
                    end
                end
                # TODO: check the complex case
                igroup += 1
            end
            ipoly += 1
        end
    end
    return support_union
end
_iterate_supports!(relaxation::SparsityTerm) = relaxation.support_union =
    _iterate_supports(relaxation.parentgroupings, relaxation.localizing_supports, relaxation.graphs)

"""
    SparsityTermBlock(relaxation::AbstractProblem; verbose::Bool=false)

Analyze the term sparsity of the problem.
[Term sparsity](https://doi.org/10.1137/19M1307871) is a recent iterative sparsity analysis that groups terms with shared
supports. Its last iteration will give the same optimal value as the original problem, although it may still be smaller.
Often, even the uniterated analysis already gives the same bound as the dense problem.
The terms are grouped based on connected components of a graph; this can be improved by using the smallest chordal extension
(see [`SparsityTermChordal`](@ref)), which will lead to even smaller problem sizes, but typically also worse bounds.

If correlative and term sparsity are to be used together, use [`SparsityCorrelativeTerm`](@ref) instead of
nesting the sparsity objects.
"""
SparsityTermBlock(args...; kwargs...) = SparsityTerm(args...; method=:block, kwargs...)

"""
    SparsityTermChordal(relaxation::AbstractProblem; chordal_completion=true, verbose=false)

Analyze the term sparsity of the problem using chordal cliques.
[Chordal term sparsity](https://doi.org/10.1137/20M1323564) is a recent iterative sparsity analysis that groups terms
with shared supports. Even in its last iteration, it may give strictly smaller values than the dense problem.
The basis elements are grouped in terms of chordal cliques of the term sparsity graph. This uses maximal cliques; as obtaining
maximal cliques of an arbitrary graph is not efficient, the graph is extended to a chordal graph if `chordal_completion` is
`true` using a heuristic. Disabling the chordal completion can lead to smaller problem sizess.

If correlative and term sparsity are to be used together, use [`SparsityCorrelativeTerm`](@ref) instead of
nesting the sparsity objects.
"""
SparsityTermChordal(args...; chordal_completion::Bool=true, kwargs...) =
    SparsityTerm(args...; method=chordal_completion ? :chordal_cliques : :cliques, kwargs...)

function _jointindices(problem::Problem, objective::Bool, zero::Union{Bool,<:AbstractSet{<:Integer}},
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}, psd::Union{Bool,<:AbstractSet{<:Integer}})
    indices = Set{Int}()
    objective && push!(indices, 1)
    if zero === true
        union!(indices, 2:length(problem.constr_zero)+1)
    else
        union!(indices, Iterators.map(Base.Fix1(+, 1), zero))
    end
    δ = 1 + length(problem.constr_zero)
    if nonneg === true
        union!(indices, δ+1:length(problem.constr_nonneg)+δ)
    else
        union!(indices, Iterators.map(Base.Fix2(+, δ), nonneg))
    end
    δ += length(problem.constr_nonneg)
    if psd === true
        union!(indices, δ+1:length(problem.constr_psd)+δ)
    else
        union!(indices, Iterators.map(Base.Fix2(+, δ), psd))
    end
    return indices
end

function iterate!(relaxation::SparsityTerm; method::Symbol=relaxation.method, objective::Bool=true,
    zero::Union{Bool,<:AbstractSet{<:Integer}}=true, nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true,
    psd::Union{Bool,<:AbstractSet{<:Integer}}=true)
    problem = poly_problem(relaxation)
    _iterate_supports!(relaxation)
    # Even if new_supports is the same as ts.support_union, we might run into the unlikely scenario that the user changed the
    # type of term sparsity, so that now extending with a different type might lead to a different result despite having the
    # same support.
    indices = _jointindices(problem, objective, zero, nonneg, psd)
    _supports_to_graphs!(relaxation, indices)
    return _extend_graphs!(relaxation, method, indices)
end

default_solution_method(::SparsityTerm) = :heuristic