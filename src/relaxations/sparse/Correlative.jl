export RelaxationSparsityCorrelative

struct RelaxationSparsityCorrelative{P<:POProblem,G<:RelaxationGroupings} <: AbstractRelaxationSparse{P}
    problem::P
    groupings::G

    @doc """
        RelaxationSparsityCorrelative(relaxation::AbstractPORelaxation; chordal_completion=true, verbose::Bool=false)

    Analyze the correlative sparsity of a problem.
    [Correlative sparsity](https://doi.org/10.1137/050623802) is a rough sparsity analysis that groups the variables into
    cliques based on the constraints in which they appear (or the terms of the objective in which they appear). By default,
    the correlative sparsity graph is completed to a chordal graph before the cliques are determined, which guarantees that the
    maximal cliques can be determined quickly; however, this may degrade the sparsity and it may be favorable not to carry out
    the completion.

    If correlative and term sparsity are to be used together, use [`RelaxationSparsityCorrelativeTerm`](@ref) instead of
    nesting the sparsity objects.
    """
    function RelaxationSparsityCorrelative(relaxation::AbstractPORelaxation{P}; chordal_completion::Bool=true,
        verbose::Bool=false) where {Nr,Nc,P<:POProblem{<:SimplePolynomial{<:Any,Nr,Nc}}}
        problem = poly_problem(relaxation)
        @verbose_info("Constructing correlative sparsity graph")
        g = Graphs.SimpleGraph(Nr + Nc)
        # objective: check which pairs of exponents appear together in a term
        # effective objective due to polynomial optimization: it is actually checked whether
        # prefactor * (obj - lowerbound) is SOS. The prefactor * obj multiplication was already done during problem
        # construction, but prefactor * lowerbound is still missing.
        for poly in (problem.objective, problem.prefactor)
            for term in poly
                mon = monomial(term)
                for (i, (var1, _)) in enumerate(mon)
                    var1o = ordinary_variable(var1)
                    for (var2, _) in Iterators.take(mon, i -1)
                        Graphs.add_edge!(g, Graphs.Edge(var1o.index, ordinary_variable(var2).index))
                    end
                end
            end
        end
        parent = groupings(relaxation)
        # This parent grouping will put some constraints on the form of the prefactor(s). With correlative sparsity, we
        # additionally enforce that no variable occurs in a grouping that is not already present in the constraint.
        for (constrs, parentgroupings) in ((problem.constr_zero, parent.zeros), (problem.constr_nonneg, parent.nonnegs),
                                           (problem.constr_psd, parent.psds))
            for (constr, groupings) in zip(constrs, parentgroupings)
                vars = Set(effective_variables(constr))
                for grouping in groupings, var_gr in effective_variables(grouping)
                    var_gro = ordinary_variable(var_gr)
                    if var_gr ∈ vars
                        for var ∈ vars
                            Graphs.add_edge!(g, Graphs.Edge(var_gro.index, ordinary_variable(var).index))
                        end
                    end
                end
            end
        end
        @verbose_info("Determining cliques")
        gentime = @elapsed(cliques = chordal_completion ? chordal_cliques!(g) : Graphs.maximal_cliques(g))
        @verbose_info("Obtained ", length(cliques), " clique", length(cliques) > 1 ? "s" : "", " in ", gentime,
            " seconds. Generating groupings.")
        # The correlative iterator could potentially be made even smaller by determining all the multideg boundaries, but would
        # this be worth the effort?
        T = SimplePolynomials._get_p(parent)
        minmultideg = zeros(T, Nr + 2Nc)
        newobj = Vector{LazyMonomials{Nr,Nc,T,MonomialIterator{Vector{T},T}}}(undef, length(cliques))
        newzero = [similar(newobj) for _ in 1:length(problem.constr_zero)]
        newnonneg = [similar(newobj) for _ in 1:length(problem.constr_nonneg)]
        newpsd = [similar(newobj) for _ in 1:length(problem.constr_psd)]
        parentmaxobjdeg = T(maximum(maxdegree, parent.obj))
        parentmaxzerodeg = T.(maximum.(maxdegree, parent.zeros, init=zero(T)))
        parentmaxnonnegdeg = T.(maximum.(maxdegree, parent.nonnegs, init=zero(T)))
        parentmaxpsddeg = T.(maximum.(maxdegree, parent.psds, init=zero(T)))
        @inbounds for (i, clique) in enumerate(cliques)
            maxmultideg = zeros(T, Nr + 2Nc)
            fill!(@view(maxmultideg[clique]), parentmaxobjdeg)
            newobj[i] = LazyMonomials{Nr,Nc}(zero(T):parentmaxobjdeg; minmultideg, maxmultideg, powers=ownpowers)
            for (parentdeg, news) in ((parentmaxzerodeg, newzero), (parentmaxnonnegdeg, newnonneg), (parentmaxpsddeg, newpsd))
                for (maxdeg, newel) in zip(parentdeg, news)
                    maxmultideg = zeros(T, Nr + 2Nc)
                    fill!(@view(maxmultideg[clique]), maxdeg)
                    newel[i] = LazyMonomials{Nr,Nc}(zero(T):maxdeg; minmultideg, maxmultideg, powers=ownpowers)
                end
            end
        end
        @verbose_info("Generated new groupings; intersecting with old.")
        gentime = @elapsed(gr = intersect(
            RelaxationGroupings(newobj, newzero, newnonneg, newpsd, map.(SimpleVariable{Nr,Nc}, cliques)),
            parent
        ))
        @verbose_info("Obtained intersection in ", gentime, " seconds")

        return new{P,typeof(gr)}(problem, gr)
    end
end

iterate!(::RelaxationSparsityCorrelative; objective::Bool=true, zero::Union{Bool,<:AbstractSet{<:Integer}}=true,
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}=true, psd::Union{Bool,<:AbstractSet{<:Integer}}=true) = nothing

default_solution_method(::RelaxationSparsityCorrelative) = :mvhankel