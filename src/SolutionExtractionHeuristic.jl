export poly_solutions_heuristic

"""
    poly_solutions_heuristic(problem::Union{PolyOptProblem,AbstractSparsity}; verbose=false)

Apply a heuristic solution extraction mechanism to a solved polynomial optimization problem. If there is only a single
solution, the heuristic will always work (but is much faster than a rank factorization). It may in certain cases also work if
there are multiple solutions if the individual components only differ by their signs/phases. This method can also be applied in
the presence of term sparsity, where [`poly_solutions`](@ref) fails.
This function gives a vector of solutions (not an iterator, as [`poly_solutions`](@ref) does).

See also [`poly_optimize`](@ref), [`sparse_optimize`](@ref), [`poly_solutions`](@ref).
"""
function poly_solutions_heuristic(problem::PolyOptProblem; verbose::Bool=false)
    mons = last_moments(problem)
    # This whole procedure works under the assumption that there be a single unique solution, or that are multiple solutions
    # which may differ in the signs of arguments which only show up with even exponents. This may not necessarily be the case
    # then, when analyzing the problem, set the perturbation parameter to some small value, which should lift all degeneracies.
    # However, this will also slightly modify the problem, changing the optimal value (and its point) a bit. For small
    # perturbations, the solution is close enough to provide ideal starting points for another optimizer.
    solution, unknown_values, unknown_phases = solutions_heuristic_simple(problem, mons)
    if verbose
        length(unknown_values) > 0 && @info("The values of the following variables are still unknown: " *
            join(problem.variables[unknown_values], ", "))
        length(unknown_phases) > 0 && @info("The signs/phases of the following variables are still unknown: " *
            join(problem.variables[unknown_phases], ", "))
        flush(stdout)
    end
    # Sign ambiguities may lead to multiple solutions
    solutions = solutions_heuristic_phases!(problem, mons, solution, unknown_phases)
    @verbose_info("Extracted all signs/phases, found ", length(solutions), " possible solution(s)")
    # Every solution is now filled with everything we can get from the optimization. All that remains is
    # filling the rest by exploiting the ideal.
    @inbounds if length(unknown_values) > 0
        # TODO: make more efficient, both sets are ordered
        sol_variables = setdiff(1:length(problem.variables), unknown_values)
        for solution in solutions
            solution_map = problem.variables[sol_variables] => solution[sol_variables]
            for i in unknown_values
                # note this may propagate NaNs if we had issues before
                try
                    solution[i] = subs(rem(problem.variables[i], problem.gröbner_basis), solution_map)
                catch e
                    e isa InexactError || rethrow(e)
                end
            end
        end
        @verbose_info("Filled in all missing variables from equality constraints")
    end
    return solutions
end
poly_solutions_heuristic(state::AbstractSparsity; kwargs...) = poly_solutions_heuristic(sparse_problem(state); kwargs...)

@inline function solutions_heuristic_simple(problem::PolyOptProblem{P,M,V}, mons::Dict{M,Float64}) where {P,M,V}
    # In this part, we extract the parts of the solution by looking at the monomials that are powers of variables.
    solution = fill(NaN, length(problem.variables))
    unknown_signs = Int[]
    unknown_values = Int[]
    zero_checks = Set{V}()
    for (i, var) in enumerate(problem.variables)
        mon = monomial(var)
        if insorted(mon, problem.basis)
            # the monomial is present in the original basis, so we try to reconstruct it by searching for its powers. We will
            # favor even powers (even over the variable itself), as they are not affected by multiple sign-symmetric solutions
            for pow in 2:2:2problem.degree
                mon2 = mon^pow
                if haskey(mons, mon2)
                    val = mons[mon2]
                    if abs(val) <= 1e-7 # almost zero
                        solution[i] = 0.0
                        @goto done_assignment
                    elseif val >= 0
                        solution[i] = val^(1 / pow)
                        push!(unknown_signs, i)
                        @goto done_assignment
                    end
                    # else something bad happened - probably the solution was not unique. We will proceed and try to get it by
                    # using remaining powers, but the result will probably be useless.
                end
            end
            for pow in 1:2:2problem.degree
                mon2 = mon^pow
                if haskey(mons, mon2)
                    val = mons[mon2]
                    solution[i] = abs(val)^(1 / pow)
                    if val < 0
                        solution[i] = -solution[i]
                    end
                    @goto done_assignment
                end
            end
            push!(zero_checks, var)
        else
            # the monomial is not even present in our original basis. Assuming that the user did not provide an insufficient
            # basis manually, this can only happen if we are working modulo an ideal; then, all the remaining variables should
            # be sufficient to deduce what we need to know.
            push!(unknown_values, i)
        end
        @label done_assignment
    end
    zero_checks!(problem, mons, solution, zero_checks)
    return solution, unknown_values, unknown_signs
end

@inline function solutions_heuristic_simple(problem::PolyOptProblem{P,M,V},
    mons::Dict{MonomialComplexContainer{M},Float64}) where {P,M,V}
    # In this part, we extract the parts of the solution by looking at the monomials that are powers of variables.
    solution = fill(Complex(NaN, NaN), length(problem.variables))
    unknown_phases = Dict{Int,Set{Float64}}()
    unknown_values = Int[]
    zero_checks = Set{V}()
    for (i, var) in enumerate(problem.variables)
        mon = monomial(var)
        if isreal(var)
            # even a complex-valued optimization hierarchy might feature real variables
            if insorted(mon, problem.basis)
                # the monomial is present in the original basis, so we try to reconstruct it by searching for its powers. We
                # will favor even powers (even over the variable itself), as they are not affected by multiple sign-symmetric
                # solutions
                for pow in 2:2:2problem.degree
                    mon2 = mon^pow
                    if haskey(mons, MonomialComplexContainer(mon2, true))
                        val = mons[MonomialComplexContainer(mon2, true)]
                        if abs(val) <= 1e-7 # almost zero
                            solution[i] = 0.0
                            @goto done_assignment
                        elseif val >= 0
                            solution[i] = val^(1 / pow)
                            unknown_phases[i] = Set{Float64}(0.)
                            @goto done_assignment
                        end
                        # else something bad happened - probably the solution was not unique. We will proceed and try to get
                        # it by using remaining powers, but the result will probably be useless.
                    end
                end
                for pow in 1:2:2problem.degree
                    mon2 = mon^pow
                    if haskey(mons, MonomialComplexContainer(mon2, true))
                        val = mons[MonomialComplexContainer(mon2, true)]
                        solution[i] = abs(val)^(1 / pow)
                        if val < 0
                            solution[i] = -solution[i]
                        end
                        @goto done_assignment
                    end
                end
                push!(zero_checks, var)
            else
                # the monomial is not even present in our original basis. Assuming that the user did not provide an
                # insufficient basis manually, this can only happen if we are working modulo an ideal; then, all the remaining
                # variables should be sufficient to deduce what we need to know.
                push!(unknown_values, i)
            end
            @label done_assignment
        elseif haskey(mons, MonomialComplexContainer(mon, true)) && haskey(mons, MonomialComplexContainer(mon, false))
            solution[i] = Complex(mons[MonomialComplexContainer(mon, true)], mons[MonomialComplexContainer(mon, false)])
        elseif insorted(mon, problem.basis)
            # the monomial itself does not occur in our sparse optimization list, but it is present in the original basis, so
            # we try to reconstruct it by searching for its powers. The phase reconstruction is ambiguous.
            abs_val = NaN
            phase_candidates = Set{Float64}()
            for pow in 2:2problem.degree
                mon2 = mon^pow
                if haskey(mons, MonomialComplexContainer(mon2, true)) && haskey(mons, MonomialComplexContainer(mon2, false))
                    val = Complex(mons[MonomialComplexContainer(mon2, true)], mons[MonomialComplexContainer(mon2, false)])
                    # we have candidates for the phase
                    if isnan(abs_val)
                        abs_val = abs(val)^(1 / pow)
                        this_phase = atan(imag(val), real(val)) / pow
                        ω = 2π / pow
                        phase_candidates = Set{Float64}(this_phase + j * ω for j in 0:pow-1)
                    else
                        # consistency check: all powers should give rise to the same magnitude
                        if abs(abs_val - abs(val)^(1 / pow)) > 1e-7
                            abs_val = NaN
                            break
                        end
                        # now check which phase candidates still survive. This is tricky given the floating-point inaccuracy.
                        new_candidates = Set{Float64}()
                        pow_phase = atan(imag(val), real(val))
                        for old_candidate in phase_candidates
                            if abs(mod(old_candidate * pow, 2π) - pow_phase) < 1e-7
                                new_candiates |= old_candiate
                            end
                        end
                        phase_candidates = new_candidates
                    end
                    # even if we end up with just one phase candidate, we continue the consistency check with higher powers
                end
            end
            if isnan(abs_val) || length(phase_candidates) == 0
                push!(zero_checks, var)
            elseif length(phase_candidates) == 1
                solution[i] = abs_val * exp(1im * first(phase_candidates))
            else
                solution[i] = abs_val
                unknown_phases[i] = phase_candidates
            end
        else
            # the monomial is not even present in our original basis. Assuming that the user did not provide an insufficient
            # basis manually, this can only happen if we are working modulo an ideal; then, all the remaining variables should
            # be sufficient to deduce what we need to know.
            push!(unknown_values, i)
        end
    end
    zero_checks!(problem, mons, solution, zero_checks)
    return solution, unknown_values, unknown_phases
end

@inline function zero_checks!(problem::PolyOptProblem{P,M,V}, mons::Dict{X,Float64}, solution::AbstractVector,
    zero_checks::Set{V}) where {P,M,V,X<:Union{M,MonomialComplexContainer{M}}}
    if length(zero_checks) > 0
        # check whether this value is zero
        for (mon_cnt, mon_val) in mons
            mon = X <: MonomialComplexContainer{M} ? mon_cnt.mon : mon_cnt
            abs(mon_val) ≥ 1e-7 && continue
            found = missing
            for (mon_var, var_exp) in zip(variables(mon), exponents(mon))
                mon_var = ordinary_variable(mon_var)
                if mon_var ∈ zero_checks
                    var_exp == 0 && continue
                    if ismissing(found)
                        found = mon_var
                    else
                        found = missing
                        break
                    end
                else
                    var_exp == 0 && continue
                    var_idx = searchsortedfirst(problem.variables, mon_var, rev=true)
                    val = solution[var_idx]
                    if abs(val) < 1e-7
                        # does not help, there's another zero variable
                        found = missing
                        break
                    end
                end
            end
            if !ismissing(found)
                var_idx = searchsortedfirst(problem.variables, found, rev=true)
                solution[var_idx] = 0
                setdiff!(zero_checks, (found,))
            end
        end
        # Let the user do something on their own. It might for example be the case that powers of var could be calculated from
        # the ideal or enough other information is available to somehow get its value.
        length(zero_checks) > 0 && @warn("Could not discern value for " * string(zero_checks))
    end
end

@inline function solutions_heuristic_phases!(problem::PolyOptProblem{P,M}, mons::Dict{M,Float64}, solution::Vector{Float64},
    unknown_positions::Vector{Int}) where {P,M}
    # step 1: which signs can we find out for sure?
    while length(unknown_positions) > 0
        still_unknown_positions = similar(unknown_positions, 0)
        for i in unknown_positions
            # let's go through all the other monomials that are available and that contain this variable.
            search_for = problem.variables[i]
            for (mon, mon_val) in mons
                # If the value attributes to this monomial is zero, we cannot tell anything about the sign.
                abs(mon_val) < 1e-7 && continue
                # for this monomial, check: is the factor contributed by all variables other than the one under scrutiny
                # positive?
                restpos = true
                # does our relevant variable occur in the monomial at all?
                found = false
                for (var, exp) in zip(variables(mon), exponents(mon))
                    if var == search_for
                        @assert !found
                        iseven(exp) && break # does not help if this monomial is sign-agnostic
                        found = true
                    elseif isodd(exp)
                        # the value of this particular other variable is only relevant if its sign can contribute due to an odd
                        # exponent
                        var_idx = searchsortedfirst(problem.variables, var, rev=true)
                        val = solution[var_idx]
                        if val < 0.0
                            restpos = !restpos
                        elseif isnan(val) || insorted(var_idx, unknown_positions)
                            # but if the value or at least its sign is unknown, we cannot infer anything (note unknown values
                            # with even exponents are not a problem)
                            found = false
                            break
                        end
                    end
                end
                if found
                    if (restpos && mon_val < 0) || (!restpos && mon_val > 0)
                        solution[i] = -solution[i]
                    end
                    @goto done_signfix
                end
            end
            push!(still_unknown_positions, i)
            @label done_signfix
        end
        if length(unknown_positions) == length(still_unknown_positions)
            break
        else
            unknown_positions = still_unknown_positions
        end
    end
    # are we done?
    if length(unknown_positions) == 0
        return [solution]
    end
    # step 2: are still issues, because we cannot assign some signs definitely. We will try all possible combinations.
    # However, any single choice may lead to eliminating lots of other possibilities, so we don't check for all tuples.
    first_sol = solutions_heuristic_phases!(problem, mons, copy(solution), unknown_positions[2:end])
    solution[unknown_positions[1]] = -solution[unknown_positions[1]]
    second_sol = solutions_heuristic_phases!(problem, mons, solution, unknown_positions[2:end])
    return [first_sol; second_sol]
end

# this should rather be called ...Phases, but if we keep the same name, we don't need discriminating logic in extract_solution
@inline function solutions_heuristic_phases!(problem::PolyOptProblem{P,M}, mons::Dict{MonomialComplexContainer{M},Float64},
    solution::Vector{ComplexF64}, unknown_phases::Dict{Int,Set{Float64}}) where {P,M}
    # step 1: which phases can we find out for sure?
    while length(unknown_phases) > 0
        len_before = length(unknown_phases)
        for (i, candidates) in unknown_phases
            # let's go through all the other monomials that are available and that contain this variable.
            search_for = problem.variables[i]
            for (mon_cnt, mon_cnt_val) in mons
                !mon_cnt.re && continue # to avoid going over duplicates, we only work with the real part containers
                if haskey(mons, MonomialComplexContainer(mon_cnt.mon, false))
                    mon = mon_cnt.mon
                    mon_val = Complex(mon_cnt_val, mons[MonomialComplexContainer(mon, false)])
                else
                    continue # we need both real and imaginary part values, else there is no phase
                end
                # If the value attributes to this monomial is zero, we cannot tell anything about the phase.
                abs(mon_val) < 1e-7 && continue
                # does our relevant variable occur in the monomial at all?
                this_exponent = 0
                # for this monomial, we would like to have the phases of all other variables already fixed; but even if there
                # are multiple candidates, they may yield the same value (mod 2π) after exponentiation.
                rest_phase = 0.0
                for (var, exp) in zip(variables(mon), exponents(mon))
                    if var == search_for
                        @assert(this_exponent == 0)
                        this_exponent = exp
                    else
                        # the value of this particular other variable is only relevant if its sign can contribute
                        # due to an odd exponent
                        var_idx = searchsortedfirst(problem.variables, var, rev=true)
                        val = solution[var_idx]
                        if isnan(val)
                            # this will only happen if we know nothing about the variable, which will effectively prevent us
                            # from drawing any conclusion about the monomial
                            this_exponent = 0
                            break
                        elseif haskey(unknown_phases, var_idx)
                            # the phase of this variable might be ambiguous, but check whether we can uniquely determine the
                            # phase of the variable to the given power
                            exp_phases = mod.(unknown_phases[var_idx] .* exp, 2π)
                            exp_phase = first(exp_phases)
                            for compare in exp_phases
                                if abs(compare - exp_phase) ≥ 1e-7
                                    this_exponent = 0
                                    @goto failed
                                end
                            end
                            # indeed this works!
                            rest_phase += exp_phase
                            continue
                            @label failed
                            break
                        else
                            # the variable is known, so we also know its phase
                            rest_phase += atan(imag(val), real(val))
                        end
                    end
                end
                if this_exponent > 0
                    mon_phase = atan(imag(mon_val), real(mon_val))
                    solution[i] *= exp(1im * (mon_phase - rest_phase) / this_exponent)
                    delete!(unknown_phases, i)
                    break
                end
            end
        end
        length(unknown_phases) == len_before && break
    end
    # are we done?
    if length(unknown_phases) == 0
        return [solution]
    end
    # step 2: are still issues, because we cannot assign some signs definitely. We will try all possible combinations.
    # However, any single choice may lead to eliminating lots of other possibilities, so we don't check for all tuples.
    # Find the variable with the least possible combinations.
    candidate_i, candidate_phases = first(unknown_phases)
    for (k, v) in unknown_phases
        if length(v) < length(candidate_phases)
            candidate_i, candidate_phases = k, v
        end
    end
    delete!(unknown_phases, candidate_i)
    solutions = Vector{ComplexF64}[]
    for candidate_phase in candidate_phases
        append!(solutions, solutions_heuristic_phases!(problem, mons, copy(solution), copy(unknown_phases)))
    end
    return solutions
end