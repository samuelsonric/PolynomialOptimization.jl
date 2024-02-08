export sparse_groupings, sparse_iterate!

"""
    AbstractSPOProblem <: AbstractPOProblem

This is the general abstract type for any kind of sparse analysis of a polynomial optimization problem.
Its concrete types can be used for analyzing and optimizing the problem.

See also [`poly_problem`](@ref), [`POProblem`](@ref), [`poly_optimize`](@ref).
"""
abstract type AbstractSPOProblem{P<:SimplePolynomial,Prob<:POProblem{P}} <: AbstractPOProblem{P} end

"""
    SparseGroupings

Contains information about how the elements in a certain (sparse) polynomial optimization problem combine.
Groupings are contained in the fields `obj`, `zero`, `nonneg`, and `psd`.
The field `var_cliques` contains a list of sets of variables, each corresponding to a variable clique in the total problem. In
the complex case, only the declared variables are returned, not their conjugates.
"""
struct SparseGroupings{MV,V}
    obj::Vector{MV}
    zero::Vector{Vector{MV}}
    nonneg::Vector{Vector{MV}}
    psd::Vector{Vector{MV}}
    var_cliques::Vector{V}
end

"""
    sparse_groupings(state::AbstractSPOProblem)

Analyze the current state and return the bases and cliques as indicated by its sparsity in a [`SparseGroupings`](@ref) struct.
"""
function sparse_groupings end

"""
    sparse_iterate!(state::AbstractSPOProblem; objective=true, zero=true, nonneg=true, psd=true)

Iterate the sparsity, which will lead to a more dense representation and might give better bounds at the expense of a more
costly optimization. Return `nothing` if the iterations converged (`state` did not change any more), else return the new state.

The keyword arguments allow to restrict the iteration to certain elements. This does not necessarily mean that the bases
associated to other elements will not change as well to keep consistency; but their own contribution will not be considered.
The parameters `nonneg` and `psd` may either be `true` (to iterate all those constraints) or a set of integers that refer to
the indices of the constraints, as they were originally given to [`poly_problem`](@ref).
"""
function sparse_iterate! end

# internal function
function sparse_supports end

function Base.show(io::IO, m::MIME"text/plain", x::AbstractSPOProblem)
    groupings = sparse_groupings(x)
    sort!.(groupings.var_cliques)
    sort!(groupings.var_cliques)
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

function merge_cliques(problem::AbstractPOProblem, groupings::SparseGroupings{MV}) where {MV}
    max_power = _get_p(MV)
    representation = problem.basis isa SimpleDenseMonomialVector ? :dense : :sparse
    vars = filter(∘(!, isconj), variables(MV))
    let cout=merge_cliques(Set.(groupings.obj))
        empty!(groupings.obj)
        sizehint!(groupings.obj, length(cout))
        for coutᵢ in cout
            push!(groupings.obj, SimpleMonomialVector(coutᵢ, max_power, representation, vars))
        end
    end
    for constr in (groupings.zero, groupings.nonneg, groupings.psd)
        for (i, cin) in enumerate(constr)
            @inbounds constr[i] = SimpleMonomialVector.(merge_cliques!(Set.(cin)), max_power, representation, (vars,))
        end
    end
    return constr
end

struct _DummyMonomial
    degree::Int
end

MultivariatePolynomials.degree(d::_DummyMonomial) = d.degree

function truncate_basis(v::SimpleMonomialVector, maxdeg::Integer)
    idx = searchsortedlast(v, _DummyMonomial(maxdeg), by=degree)
    if idx < firstindex(v)
        return @view(v[begin:end])
    else
        return @view(v[1:idx])
    end
end

include("./SparsityNone.jl")
