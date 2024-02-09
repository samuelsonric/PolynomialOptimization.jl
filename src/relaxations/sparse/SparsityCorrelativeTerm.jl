export TermMode, tm_none, tm_block, tm_cliques, tm_chordal_cliques, SparsityCorrelativeTerm

@enum TermMode tm_none = 0 tm_block = 1 tm_cliques = 2 tm_chordal_cliques = 3
const term_mode_to_static = (Val(:block), Val(:cliques), Val(:chordal_cliques))

@doc "
    @enum TermMode tm_none tm_block tm_cliques tm_chordal_cliques

Specifies which kind of completion procedure is used for the iteration of the individual term sparsity patterns of the cliques.
Valid values are `tm_none` ([`SparsityNone`](@ref)), `tm_block` ([`SparsityTermBlock`](@ref)),
`tm_chordal_cliques` ([`SparsityTermCliques`](@ref)), and `tm_cliques` ([`SparsityTermCliques`](@ref) with
`chordal_completion = false`).
" TermMode

struct SparsityCorrelativeTerm <: AbstractSPOProblem
    problem::POProblem
    cliques::Vector{AbstractSPOProblem}
    constraint_locations::Vector{Pair{Int,Int}}
    index_maps::Vector{Base.ImmutableDict{Int,Int}}

    @doc """
        SparsityCorrelativeTerm(problem::POProblem; clique_chordal_completion=true, term_mode=tm_block)

    Analyze both the [correlative as well as the term sparsity](http://arxiv.org/abs/2005.02828v2) of the problem.
    This is the most versatile kind of sparsity analysis, combining the effects of correlative sparsity with term analysis per
    clique. `clique_chordal_completion` controls whether the correlative sparsity analysis works on a chordal completion of the
    correlative sparsity graph or on the graph itself; `term_mode` allows to determine whether block completion (`tm_block`),
    chordal cliques (`tm_chordal_cliques`) or just cliques (`tm_cliques`) should be used for the term sparsity analysis. It is
    not permitted to use `tm_none` here, as this would completely disable the term sparsity; use `SparsityCorrelative` for
    this purpose.

    See also [`poly_problem`](@ref), [`poly_optimize`](@ref), [`SparsityCorrelative`](@ref), [`SparsityTermBlock`](@ref),
    [`SparsityTermCliques`](@ref), [`TermMode`](@ref).

        SparsityCorrelativeTerm(correlative_sparsity::SparsityCorrelative, term_modes::AbstractVector{TermMode})

    This form allows to first create the correlative sparsity pattern of an optimization problem by hand, then to individually
    specify with kind of term sparsity (if any, as `tm_none` is also allowed here) should be applied to the individual cliques.

    See also [`SparsityCorrelative`](@ref), [`TermMode`](@ref).
    """
    function SparsityCorrelativeTerm(correlative_sparsity::SparsityCorrelative;
        term_modes::Union{TermMode,<:AbstractVector{TermMode}}, verbose::Bool=false)
        (if term_modes isa TermMode
            term_modes != tm_none
        else
            !all(term_modes .== tm_none)
        end) || error("Term modes cannot all be none; use SparsityCorrelative instead.")
        problem = sparse_problem(correlative_sparsity)
        # This method is relatively slow. To counter the effects, we expand a lot of more idiomatic Julia broadcasts into
        # explicit loops, as we don't need to allocate all these temporaries...
        # ef_vars = problem.complex ? ordinary_variable ∘ effective_variables : effective_variables
        # This is essentially SparsityCorrelative.groupings, but without partitioning the full basis, only the
        # constraint bases.
        @verbose_info("Obtaining cliques according to correlative sparsity pattern")
        correlative_groupings, cliques, (variable_groupings, constraint_groups) = groupings(correlative_sparsity;
            verbose, return_assignments=true)
        if term_modes isa TermMode
            term_modes = fill(term_modes, length(cliques))
        elseif length(term_modes) != length(cliques)
            error("The number of term modes specified does not match the number of cliques present in the sparsity pattern")
        end
        @verbose_info("Determining variable groupings and inequality assignments")
        constr_locations = Vector{Pair{Int,Int}}(undef, length(problem.constraints))
        per_group = ones(Int, length(cliques)) # first index is always 2 for inequalities, since 1 is objective
        for i in 1:length(constraint_groups)
            @inbounds per_group[constraint_groups[i]] += 1
            @inbounds constr_locations[i] = Pair{Int,Int}(constraint_groups[i], per_group[constraint_groups[i]])
        end

        index_maps = Vector{Base.ImmutableDict{Int,Int}}(undef, length(cliques))
        sparse_cliques = Vector{AbstractSPOProblem}(undef, length(cliques))
        problems = Vector{typeof(problem)}(undef, length(cliques))
        @verbose_info("Generating subproblems and subbases")
        gentime = @elapsed begin
            Threads.@threads for group_idx in 1:length(cliques)
                @inbounds var_vec = cliques[group_idx]
                constr_indices_mask = constraint_groups .== (group_idx,)
                @inbounds constr_indices = (2:length(problem.constraints)+1)[constr_indices_mask]
                @inbounds index_maps[group_idx] = Base.ImmutableDict(1 => 1, (outer => inner + 1
                                                                              for (inner, outer) in enumerate(constr_indices))...)
                @inbounds prob = typeof(problem)(
                    problem.objective,
                    problem.prefactor,
                    var_vec,
                    Dict(var_vec[i] => i for i in 1:length(var_vec)),
                    problem.degree,
                    correlative_groupings[1][group_idx],
                    [PolyOptConstraint{polynomial_type(problem),monomial_type(problem)}(constr.type, constr.constraint,
                                                                                      correlative_groupings[i][1])
                     for (constr, i) in zip(@view(problem.constraints[constr_indices_mask]), constr_indices)],
                    subbasis(problem.gröbner_basis, variable_groupings[group_idx]),
                    problem.complex,
                )
                problems[group_idx] = prob
            end
        end
        @verbose_info("Generation completed in ", gentime, " seconds. Constructing union supports")
        gentime = @elapsed begin
            # we want a canonical order of the cliques at all times, so that they can be reliably addressed in the iteration
            shuffle = sortperm(problems, by=x -> sort(x.variables, rev=true), rev=true)
            ishuffle = invperm(shuffle)
            # with this order, we can appropriately associate the desired term modes and construct the subpatterns
            if problem.complex
                union_supports = merge_monomial_vectors([monomials(problem.objective),
                    monomials.(getfield.(problem.constraints, :constraint))...,
                    squarebasis(problem.basis[isreal.(problem.basis)], problem.gröbner_basis)])
            else
                union_supports = merge_monomial_vectors([monomials(problem.objective),
                    monomials.(getfield.(problem.constraints, :constraint))...,
                    squarebasis(problem.basis, problem.gröbner_basis)])
            end
        end
        @verbose_info("Got supports in ", gentime, " seconds. Generating sparsity subpatterns")
        @inbounds gentime = @elapsed begin
            Threads.@threads for group_idx in 1:length(cliques)
                prob = problems[shuffle[group_idx]]
                @inbounds if term_modes[group_idx] == tm_none
                    sparse_cliques[group_idx] = SparsityNone(prob)
                else
                    if term_modes[group_idx] == tm_block
                        sparse_cliques[group_idx] = SparsityTermBlock(prob, union_supports)
                    elseif term_modes[group_idx] == tm_chordal_cliques
                        sparse_cliques[group_idx] = SparsityTermCliques(prob, union_supports, chordal_completion=true)
                    elseif term_modes[group_idx] == tm_cliques
                        sparse_cliques[group_idx] = SparsityTermCliques(prob, union_supports, chordal_completion=false)
                    else
                        @assert(false)
                    end
                end
            end
        end
        @verbose_info("Generation completed in ", gentime, " seconds. Gathering locations")
        @inbounds index_maps = index_maps[shuffle]
        for i in 1:length(constraint_groups)
            @inbounds constr_locations[i] = Pair{Int,Int}(ishuffle[constr_locations[i].first],
                constr_locations[i].second)
        end
        @verbose_info("Assembling everything")
        @inbounds return new(typeof(problem)(
                problem.objective, problem.prefactor, problem.variables, problem.var_map, problem.degree,
                monomial_vector(union((sparse_problem(ts).basis for ts in sparse_cliques)...)),
                [PolyOptConstraint{polynomial_type(problem),monomial_type(problem)}(
                    constr.type, constr.constraint, sparse_problem(sparse_cliques[loc[1]]).constraints[loc[2]-1].basis
                 ) for (constr, loc) in zip(problem.constraints, constr_locations)],
                problem.gröbner_basis,
                problem.complex
            ), sparse_cliques, constr_locations, index_maps)
    end
end

function SparsityCorrelativeTerm(problem::POProblem; clique_chordal_completion::Bool=true,
    term_mode::TermMode=tm_block, verbose::Bool=false)
    term_mode != tm_none || error("Term mode cannot be none; use SparsityCorrelative instead")
    return SparsityCorrelativeTerm(SparsityCorrelative(problem, chordal_completion=clique_chordal_completion);
        term_modes=term_mode, verbose)
end

sparse_problem(cts::SparsityCorrelativeTerm) = cts.problem

function Base.show(io::IO, m::MIME"text/plain", x::SparsityCorrelativeTerm)
    print(io, typeof(x), " with ", length(x.cliques), " cliques(s)")
    for (i, cl) in enumerate(x.cliques)
        print(io, "\n=========\nClique #", i, "\n")
        show(io, m, cl)
    end
end

function groupings(cts::SparsityCorrelativeTerm)
    clique_groupings = first.(groupings.(cts.cliques))
    @inbounds return ([union(map(x -> x[1], clique_groupings)...),
                       (clique_groupings[loc[1]][loc[2]] for loc in cts.constraint_locations)...],
                      map(x -> sparse_problem(x).variables, cts.cliques))
end

"""
    iterate!(cts::SparsityCorrelativeTerm; term_modes=default, cliques=true, kwargs...)

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
function iterate!(cts::SparsityCorrelativeTerm;
    @nospecialize(term_modes::Union{Missing,TermMode,<:AbstractVector{<:Union{Missing,TermMode}},Pair{<:Integer,TermMode},<:AbstractVector{Pair{<:Integer,TermMode}},AbstractDict{<:Integer,TermMode}}=missing),
    objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true, nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true,
    psd::Union{Bool,<:AbstractSet{<:Integer}}=true, cliques::Union{Bool,<:AbstractSet{<:Integer}}=true, verbose::Bool=false)
    problem = sparse_problem(cts)

    function convert_mode!(i, to::TermMode)
        # convert the mode of clique i. Note that by setting a mode to None, this will erase all iteration information, so
        # re-setting it to
        if to == tm_none
            if !(cts.cliques[i] isa SparsityNone)
                cts.cliques[i] = SparsityNone(sparse_problem(cts.cliques[i]))
            end
        elseif to == tm_block
            if cts.cliques[i] isa SparsityNone
                cts.cliques[i] = SparsityTermBlock(sparse_problem(cts.cliques[i]))
            elseif cts.cliques[i] isa SparsityTermCliques
                cts.cliques[i] = cts.cliques[i].term_sparsity
            end
        elseif to == tm_chordal_cliques
            if cts.cliques[i] isa SparsityNone
                cts.cliques[i] = SparsityTermCliques(sparse_problem(cts.cliques[i]), chordal_completion=true)
            elseif cts.cliques[i] isa SparsityTermBlock
                cts.cliques[i] = SparsityTermCliques(cts.cliques[i], chordal_completion=true)
            elseif cts.cliques[i] isa SparsityTermCliques && !cts.cliques[i].chordal_completion
                cts.cliques[i] = SparsityTermCliques(cts.cliques[i].term_sparsity, chordal_completion=true)
            end
        elseif to == tm_cliques
            if cts.cliques[i] isa SparsityNone
                cts.cliques[i] = SparsityTermCliques(sparse_problem(cts.cliques[i]), chordal_completion=false)
            elseif cts.cliques[i] isa SparsityTermBlock
                cts.cliques[i] = SparsityTermCliques(cts.cliques[i], chordal_completion=false)
            elseif cts.cliques[i] isa SparsityTermCliques && cts.cliques[i].chordal_completion
                cts.cliques[i] = SparsityTermCliques(cts.cliques[i].term_sparsity, chordal_completion=false)
            end
        else
            @assert(false)
        end
    end

    @verbose_info("Obtaining term modes")
    if term_modes isa TermMode
        for i in length(cts.cliques)
            convert_mode!(i, term_modes)
        end
    elseif term_modes isa AbstractVector{<:Union{Missing,TermMode}}
        if length(term_modes) != length(cts.cliques)
            if cliques !== true && length(term_modes) == length(cliques)
                for (cl, tm) in zip(sort!(collect(cliques)), term_modes)
                    ismissing(tm) || convert_mode!(cl, tm)
                end
            else
                error("The number of term modes specified does not match the number of cliques present in the sparsity pattern")
            end
        else
            for (i, tm) in enumerate(term_modes)
                ismissing(tm) || convert_mode!(i, tm)
            end
        end
    elseif term_modes isa Pair{<:Integer,TermMode}
        (term_modes.first ≥ 1 && term_modes.first ≤ length(cts.cliques)) || error("Invalid term mode index")
        convert_mode!(term_modes.first, to)
    elseif term_modes isa AbstractVector{Pair{<:Integer,TermMode}}
        for p in term_modes
            (p.first ≥ 1 && p.first ≤ length(cts.cliques)) || error("Invalid term mode index")
            convert_mode!(p.first, p.second)
        end
    elseif term_modes isa AbstractDict{<:Integer,TermMode}
        for (i, tm) in term_modes
            (p.first ≥ 1 && p.first ≤ length(cts.cliques)) || error("Invalid term mode index")
            convert_mode(i, tm)
        end
    end
    @verbose_info("Generating new union supports")
    gentime = @elapsed begin
        new_union_supports = supports(cts)
    end
    @verbose_info("Generation finished in ", gentime, " seconds. Iterating individual cliques.")
    indices = poly_structure_indices(problem, objective, zero, nonneg, psd)
    finished = true
    gentime = @elapsed begin
        Threads.@threads for (i, (ts, index_map)) in collect(enumerate(zip(cts.cliques, cts.index_maps)))
            if ts isa SparsityTerm
                ts.logic.union_supports = new_union_supports
                if cliques === true || i ∈ cliques
                    # we need to map the "global" indices to local indices. Note that objective is 1, inequalities start at 2
                    # in indices, but the bit array nonneg_indices only indexes the constraints
                    loc_indices = [index_map[k] for k in indices if haskey(index_map, k)]
                    old_cache = ts.logic.cache
                    supports_to_graphs!(ts.logic, loc_indices)
                    ts.logic.cache = extend_graphs!(ts, loc_indices)
                    old_cache == ts.logic.cache || (finished = false)
                end
            end
        end
    end
    @verbose_info("Individual iterations finished in ", gentime, " seconds. Checking for fixed point.")
    return finished ? nothing : cts
end

function supports(cts::SparsityCorrelativeTerm)
    # basically just a threaded merge_monomial_vectors(supports.(cts.cliques))
    supports = Vector{typeof(cts.problem.basis)}(undef, length(cts.cliques))
    Threads.@threads for i in 1:length(cts.cliques) # enumerate doesn't work with threads
        @inbounds supports[i] = supports(cts.cliques[i])
    end
    return merge_monomial_vectors(supports)
end

default_solution_method(::SparsityCorrelativeTerm) = :heuristic