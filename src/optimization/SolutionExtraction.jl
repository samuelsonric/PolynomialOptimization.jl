export poly_solutions, poly_solution_badness, poly_all_solutions

struct MonomialMissing <: Exception end

struct PolynomialSolutions{R<:Real,V<:Union{R,Complex{R}},Nr,Nc,Var<:SimpleVariable{Nr,Nc},has_length}
    cliques::Vector{Vector{Var}}
    solutions_cl::Vector{Union{Missing,Matrix{V}}}
    δ::R
    verbose::Bool
end

function Base.show(io::IO, ::MIME"text/plain", s::PolynomialSolutions)
    println(io, "Solution iterator for a polynomial optimization problem")
    if isone(length(s.cliques))
        print(io, "Number of solutions: ", size(s.solutions_cl[1], 2))
    else
        println(io, "Number of solutions: unknown")
        println(io, "Number of cliques: ", length(s.cliques))
        for (i, (clique, clique_sol)) in enumerate(zip(s.cliques, s.solutions_cl))
            println(io, "Clique #", i, ": ", length(clique.variables), " variable")
            isone(length(clique)) || print(io, "s")
            print(io, ", ", size(clique_sol, 2), " subsolution")
            isone(size(clique_sol, 2)) || print(io, "s")
        end
        println(io, "Maximally allowed discrepancy between same variables in different cliques: ", s.δ)
        print(io, "Verbose display while assembling solutions: ", s.verbose ? "enabled" : "disabled")
    end
end

"""
    poly_solutions(result::POResult, ϵ=1e-6, δ=1e-3; verbose=false)

Performs a multivariate Hankel decomposition of the full moment matrix that was obtained via optimization of the problem, using
the [best currently known algorithm](https://doi.org/10.1016/j.laa.2017.04.015). This method is not deterministic due to a
random sampling, hence negligible deviations are expected from run to run.
The parameter `ϵ` controls the bound below which singular values are regarded as zero.
The parameter `δ` controls up to which threshold supposedly identical numerical values for the same variables from different
cliques must match.
Note that for sparsity patterns different from no sparsity and [`SparsityCorrelative`](@ref), this method not be able to
deliver results, although it might give partial results for [`SparsityCorrelativeTerm`](@ref) if some of the cliques did not
have a term sparsity pattern. Consider using [`poly_solutions_heuristic`](@ref) in such a case.
This function returns an iterator.

See also [`poly_optimize`](@ref), [`poly_optimize`](@ref), [`poly_solutions_heuristic`](@ref).
"""
function poly_solutions(result::POResult{Rx,V}, ϵ::R=R(1 // 1_000_000), δ::R=R(1 // 1_000); verbose::Bool=false) where
    {Nr,Nc,Rx<:AbstractPORelaxation{<:POProblem{<:SimplePolynomial{<:Any,Nr,Nc}}},R<:Real,V<:Union{R,Complex{R}}}
    @verbose_info("Preprocessing for decomposition")
    relaxation = result.relaxation
    moments = result.moments
    nvars = Nr + Nc
    deg = 2relaxation.degree
    # potentially scale the moment matrix
    λ = maximum(abs, @view(moments[monomial_count(deg -2, nvars)+1:monomial_count(deg -1, nvars)])) /
        maximum(abs, @view(moments[monomial_count(deg -1, nvars)+1:end]))
    if λ > ϵ
        for d in 0:deg
            rmul!(@view(moments[monomial_count(d -1, nvars)+1:monomial_count(d, nvars)]), λ^d)
        end
    end
    # for each variable clique, we can perform the original decomposition algorithm
    cliques = groupings(result.relaxation).var_cliques
    solutions_cl = FastVec{Union{Matrix{V},Missing}}(buffer=length(cliques))
    @verbose_info("Starting solution extraction per clique")
    extraction_time = @elapsed begin
        for (i, clique) in enumerate(cliques)
            @verbose_info("Investigating clique ", clique)
            a1 = basis(relaxation, i)
            a2 = @view(a1[1:min(searchsortedfirst(a1, _DummyMonomial(relaxation.degree), by=degree) -1, length(a1))])
            if !isreal(relaxation)
                # this is the transpose of what we'd get with moment_matrix, but this is not important. Since the monomials
                # in the matrix will be multiplied by variables (which are the un-conjugated ones), we must make sure that
                # the un-conjugated ones are of the smaller degree.
                a1 = conj(a1)
            end
            try
                result = poly_solutions_scaled(moments, a1, a2, clique, ϵ)
                λ > ϵ && rmul!(result, inv(λ))
                unsafe_push!(solutions_cl, result)
                @verbose_info("Potential solutions:\n", result)
            catch e
                if e isa MonomialMissing
                    unsafe_push!(solutions_cl, missing)
                    @verbose_info("Not all monomials were present to allow for a solution extraction in this clique")
                else
                    rethrow()
                end
            end
        end
    end
    # undo the scaling
    if λ > ϵ
        λ = inv(λ)
        for d in 0:deg
            rmul!(@view(moments[monomial_count(d -1, nvars)+1:monomial_count(d, nvars)]), λ^d)
        end
    end
    # now we combine all the solutions. In principle, this is Iterators.product, but we only look for compatible entries (as
    # variables in the cliques can overlap).
    @verbose_info("Found all potential individual solutions in ", extraction_time, " seconds. Building iterator.")
    return PolynomialSolutions{R,V,Nr,Nc,iszero(Nc) ? SimpleRealVariable{Nr,Nc} :
                                        (iszero(Nr) ? SimpleComplexVariable{Nr,Nc} : <:SimpleVariable{Nr,Nc}),
                               isone(length(cliques))}(
        cliques,
        finish!(solutions_cl),
        δ,
        verbose
    )
end

function Base.iterate(iter::PolynomialSolutions)
    length(iter.cliques) == 0 && return nothing
    # we first fabricate an artificial state whose only purpose is to be iterated in the very next step
    state = ones(Int, length(iter.cliques))
    @inbounds state[1] = 0
    return iterate(iter, state)
end

variable_index(v::SimpleRealVariable) = v.index
variable_index(v::SimpleComplexVariable{Nr}) where {Nr} = (@assert(!v.isconj); Nr + v.index)

function Base.iterate(iter::PolynomialSolutions{R,V,Nr,Nc}, state::Vector{Int}) where {R<:Real,V<:Union{R,Complex{R}},Nr,Nc}
    verbose = iter.verbose
    # first, we should increase the current state by one
    hasnext = false
    @inbounds for j in 1:length(state)
        if !ismissing(iter.solutions_cl[j]) && state[j] < size(iter.solutions_cl[j], 2)
            state[j] += 1
            fill!(@view(state[1:j-1]), 1)
            hasnext = true
            break
        end
    end
    hasnext || return nothing
    # then, try to get the solution for this state
    solution = Vector{V}(undef, Nr + Nc)
    @label whiletrue
        fill!(solution, V(NaN))
        @inbounds for (j, (variables, subsolutions)) in enumerate(zip(iter.cliques, iter.solutions_cl))
            ismissing(subsolutions) && continue
            for (var, val) in zip(variables, @view(subsolutions[:, state[j]]))
                var_idx = variable_index(var)
                old_val = solution[var_idx]
                if isnan(old_val)
                    solution[var_idx] = val
                elseif abs(old_val - val) > iter.δ
                    @verbose_info("Skipped a solution: Discrepancy ", abs(old_val - val), " > δ = ", iter.δ)
                    @inbounds for j in 1:length(state)
                        if !ismissing(iter.solutions_cl[j]) && state[j] < size(iter.solutions_cl[j], 2)
                            state[j] += 1
                            fill!(@view(state[1:j-1]), 1)
                            @goto whiletrue
                        end
                    end
                    return nothing
                end
            end
        end
    return solution, state
end

Base.IteratorSize(::Type{PolynomialSolutions{R,V,Nr,Nc,<:SimpleVariable{Nr,Nc},true} where {R<:Real,V<:Union{R,Complex{R}},Nr,Nc}}) =
    Base.HasLength()
Base.IteratorSize(::Type{PolynomialSolutions{R,V,Nr,Nc,<:SimpleVariable{Nr,Nc},false} where {R<:Real,V<:Union{R,Complex{R}},Nr,Nc}}) =
    Base.SizeUnknown()
Base.IteratorEltype(::Type{PolynomialSolutions}) = Base.HasEltype()
Base.eltype(::Type{<:PolynomialSolutions{R,V}}) where {R<:Real,V<:Union{R,Complex{R}}} = Vector{V}
function Base.length(iter::PolynomialSolutions{R,V,Nr,Nc,<:SimpleVariable{Nr,Nc},true}) where {R<:Real,V<:Union{R,Complex{R}},Nr,Nc}
    @inbounds cl = iter.solutions_cl[1]  # true = there is exactly one clique
    if ismissing(cl)
        return 0
    else
        return size(iter.solutions_cl[1], 2)
    end
end

function poly_solutions_scaled(moments::FullMonomialVector{V}, a1, a2, variables, ϵ::R) where {R<:Real,V<:Union{R,Complex{R}}}
    Hd1d2 = moment_matrix(moments, Val(:throw), a1, a2)
    UrSrVrbar = svd!(Hd1d2)
    ϵ *= UrSrVrbar.S[1]
    numEntries = findfirst(<(ϵ), UrSrVrbar.S)
    if numEntries === nothing
        Ur, Sr, Vrbar = UrSrVrbar.U, UrSrVrbar.S, UrSrVrbar.V
    else
        @inbounds Ur, Sr, Vrbar = UrSrVrbar.U[:, 1:numEntries-1], UrSrVrbar.S[1:numEntries-1], UrSrVrbar.V[:, 1:numEntries-1]
    end
    Ms = Vector{Matrix{V}}(undef, length(variables))
    for i in 1:length(variables)
        @inbounds Ms[i] = (Ur' ./ Sr) * moment_matrix(moments, Val(:throw), a1, a2, variables[i]) * Vrbar
    end
    v = eigvecs(sum((2rand(V) - one(V)) * Mi for Mi in Ms))
    len = size(Sr, 1)
    result = Matrix{V}(undef, length(variables), len)
    for j in 1:len
        @inbounds vj = @view(v[:, j])
        for i in 1:length(variables)
            @inbounds Mv = Ms[i] * vj
            den = zero(R)
            vre = zero(R)
            if V <: Complex
                vim = zero(R)
            end
            for (Mvᵢ, vjᵢ) in zip(Mv, vj)
                vre += real(Mvᵢ) * real(vjᵢ) + imag(Mvᵢ) * imag(vjᵢ)
                if V <: Complex
                    vim += real(vjᵢ) * imag(Mvᵢ) - real(Mvᵢ) * imag(vjᵢ)
                end
                den += abs2(vjᵢ)
            end
            if V <: Complex
                @inbounds result[i, j] = V(vre, vim) / den
            else
                @inbounds result[i, j] = vre / den
            end
        end
    end
    # we delete duplicates (which might arise due to floating point inefficiencies)
    keep = fill(true, len)
    redo = false
    @inbounds for i in 1:len -1
        keep[i] || continue
        for j in i+1:len
            keep[j] || continue
            @views if norm(result[:, i] - result[:, j]) < ϵ
                keep[j] = false
                redo = true
            end
        end
    end
    if redo
        result = keepcol!(result, keep)
    end
    return result
end

"""
    poly_solution_badness(result::POResult, solution)

Determines the badness of a solution by comparing the value of the objective with the value according to the optimization given
in `result`, and also by checking the violation of the constraints.
The closer the return value is to zero, the better. If the return value is too large, `solution` probably has nothing to do
with the actual solution.

See also [`poly_optimize`](@ref), [`poly_optimize`](@ref), [`poly_solutions`](@ref), [`poly_all_solutions`](@ref).
"""
function poly_solution_badness(result::POResult, solution::Vector)
    # check whether we can certify optimality
    any(isnan, solution) && return Inf
    problem = poly_problem(result)
    violation = abs(problem.objective(solution) - result.objective)
    for constr in problem.constr_zero
        new_violation = abs(constr(solution))
        new_violation > violation && (violation = new_violation)
    end
    for constr in problem.constr_nonneg
        new_violation = -real(constr(solution)) # they are real, but they might not know it yet
        new_violation > violation && (violation = new_violation)
    end
    for constr in problem.constr_psd
        new_violation = -eigmin(Symmetric((solution,) .|> constr))
        new_violation > violation && (violation = new_violation)
    end
    return violation
end

default_solution_method(result::POResult) = default_solution_method(result.relaxation)

"""
    poly_all_solutions(result::POResult, ϵ=1e-6, δ=1e-3; verbose=false, rel_threshold=100, abs_threshold=Inf, method::Symbol)

Obtains a vector of all the solutions to a previously optimized problem; then iterates over all of them and grades and sorts
them by their badness. Every solution of the returned vector is a tuple that first contains the optimal point and second the
badness at this point. Solutions that are `rel_threshold` times worse than the best solution or worse than `abs_threshold` will
be dropped from the result.
The currently accepted methods are
- `:mvhankel` to perform a multivariate Hankel decomposition as in [`poly_optimize`](@ref), default for dense problems and
  those with correlative sparsity only.
- `:heuristic` to perform a heuristic method as in [`poly_solutions_heuristic`](@ref), default for problems with term sparsity.

See also [`poly_optimize`](@ref), [`poly_optimize`](@ref), [`poly_solutions`](@ref), [`poly_solution_badness`](@ref),
[`poly_solutions_heuristic`](@ref).
"""
function poly_all_solutions(result::POResult, ϵ::R=1e-6, δ::R=1e-3; verbose::Bool=false,
    rel_threshold::Float64=100., abs_threshold::Float64=Inf, method::Symbol=default_solution_method(result)) where {R<:Real}
    if method === :heuristic
        sol_itr = poly_solutions_heuristic(result; verbose)
    elseif method === :mvhankel
        sol_itr = poly_solutions(result, ϵ, δ; verbose)
    else
        error("Unknown solution extraction method: $method")
    end
    solutions_fv = FastVec{Tuple{Vector{eltype(result)},Float64}}()
    Base.haslength(sol_itr) && sizehint!(solutions_fv, length(sol_itr))
    for solution in sol_itr
        push!(solutions_fv, (solution, poly_solution_badness(result, solution)))
    end
    solutions = finish!(solutions_fv)
    isempty(solutions) && return solutions
    sort!(solutions, by=last)
    # cut off solutions that are too bad (maybe if the list grows too fast, we could do this during construction?)
    threshold = min(rel_threshold * solutions[1][2], abs_threshold)
    return filter(x -> x[2] ≤ threshold, solutions)
end