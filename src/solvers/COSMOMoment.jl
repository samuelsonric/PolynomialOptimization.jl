mutable struct COSMOStateMoment{M}
    variable_mappings::AbstractDict{M,Int}
    constraints::Vector{Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Float64},COSMO.AbstractConvexSet{Float64}}}
end

get_variable!(csm::COSMOStateMoment{M}, mon::M) where {M} = get!(csm.variable_mappings, mon, length(csm.variable_mappings) + 1)

function moment_matrix!(csm::COSMOStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    # we do not distinguish on the cone sizes; the projection algorithm is already optimized to take special care of
    # 1x1 and 2x2 matrices.
    rows = FastVec{Int}()
    cols = FastVec{Int}()
    vals = FastVec{Float64}()
    i = 1
    for exp2 in 1:lg
        for exp1 in 1:lg # COSMO is slightly more efficient for the full version
            @inbounds constr_terms = mergewith(+, (get_variable!(csm, monomial(term)) => coefficient(term)
                                                   for mon_constr in constraint
                                                   for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                Int, Float64)
            append!(rows, Iterators.repeated(i, length(constr_terms)))
            append!(cols, keys(constr_terms))
            append!(vals, values(constr_terms))
            i += 1
        end
    end
    push!(csm.constraints, (finish!(rows), finish!(cols), finish!(vals), fill(0.0, i - 1), COSMO.PsdCone(lg * lg)))
    return
end

function moment_matrix!(csm::COSMOStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    lg = length(grouping)
    dim = lg * block_size
    rows = FastVec{Int}()
    cols = FastVec{Int}()
    vals = FastVec{Float64}()
    i = 1
    for exp2 in 1:lg
        for block_j in 1:block_size
            for exp1 in 1:lg
                for block_i in 1:block_size
                    @inbounds constr_terms = mergewith(+, (get_variable!(csm, monomial(term)) => coefficient(term)
                                                           for mon_constr in constraint[block_i, block_j]
                                                           for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                        Int32, Float64)
                    append!(rows, Iterators.repeated(i, length(constr_terms)))
                    append!(cols, keys(constr_terms))
                    append!(vals, values(constr_terms))
                    i += 1
                end
            end
        end
    end
    push!(csm.constraints, (finish!(rows), finish!(cols), finish!(vals), fill(0.0, i - 1), COSMO.PsdCone(dim * dim)))
    return
end

function moment_matrix_eq!(csm::COSMOStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    rows = FastVec{Int}()
    cols = FastVec{Int}()
    vals = FastVec{Float64}()
    i = 1
    for exp2 in 1:lg
        for exp1 in 1:exp2
            @inbounds constr_terms = mergewith(+, (get_variable!(csm, monomial(term)) => coefficient(term)
                                                   for mon_constr in constraint
                                                   for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                Int, Float64)
            append!(rows, Iterators.repeated(i, length(constr_terms)))
            append!(cols, keys(constr_terms))
            append!(vals, values(constr_terms))
            i += 1
        end
    end
    push!(csm.constraints, (finish!(rows), finish!(cols), finish!(vals), fill(0.0, i - 1), COSMO.ZeroSet((lg * (lg +1)) >> 1)))
    return
end

function sparse_optimize(::Val{:COSMOMoment}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, customize::Function=(csm, obj) -> obj,
    parameters...) where {P,M,V}
    @assert(!problem.complex)
    model = COSMO.Model()
    setup_time = @elapsed begin
        !haskey(parameters, :decompose) && (parameters = (decompose=false, parameters...))
        settings = COSMO.Settings(; verbose=verbose, verbose_timing=verbose, parameters...)

        csm = COSMOStateMoment(Dict{M,Int}(),
            Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Float64},COSMO.AbstractConvexSet{Float64}}[])

        # Riesz functional of objective mod ideal. Note that during problem setup, the objective (which includes prefactor) was
        # already taken modulo the ideal.
        if isone(problem.prefactor)
            # fix constant term to 1
            push!(csm.constraints, ([1], [get_variable!(csm, constant_monomial(problem.objective))], [1.0], [-1.0],
                COSMO.ZeroSet(1)))
        else
            push!(csm.constraints, (fill(1, nterms(problem.prefactor)), get_variable!.((csm,), monomials(problem.prefactor)),
                Float64.(coefficients(problem.prefactor)), [-1.0], COSMO.ZeroSet(1)))
        end
        objective = (get_variable!.((csm,), monomials(problem.objective)), Float64.(coefficients(problem.objective)))

        # moment matrix
        for grouping in groupings[1]
            moment_matrix!(csm, problem.gröbner_basis, sort(grouping, by=degree),
                polynomial(constant_monomial(problem.objective)))
        end
        # localizing matrices
        for (groupings, constr) in zip(Iterators.drop(groupings, 1), problem.constraints)
            if constr.type == pctNonneg || constr.type == pctPSD
                for grouping in groupings
                    moment_matrix!(csm, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                end
            elseif constr.type == pctEqualityNonneg
                for grouping in groupings
                    let sg = sort(grouping, by=degree)
                        moment_matrix!(csm, problem.gröbner_basis, sg, constr.constraint)
                        moment_matrix!(csm, problem.gröbner_basis, sg, -constr.constraint)
                    end
                end
            elseif constr.type == pctEqualityGröbner
                for grouping in groupings
                    moment_matrix_eq!(csm, EmptyGröbnerBasis{P}(), sort(grouping, by=degree), constr.constraint)
                end
            elseif constr.type == pctEqualitySimple
                for grouping in groupings
                    moment_matrix_eq!(csm, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                end
            else
                @assert(false)
            end
        end

        objective = customize(csm, objective)

        num_vars = length(csm.variable_mappings)
        COSMO.assemble!(model, spzeros(num_vars, num_vars), sparsevec(objective..., num_vars),
            map(x -> COSMO.Constraint(sparse(x[1], x[2], x[3], maximum(x[1]), num_vars), x[4], x[5]), csm.constraints),
            settings=settings)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    res = COSMO.optimize!(model)
    status = res.status
    value = res.obj_val
    @verbose_info("Optimization complete")

    empty!(problem.last_moments)
    sizehint!(problem.last_moments, length(csm.variable_mappings))
    for (mon, i) in csm.variable_mappings
        push!(problem.last_moments, mon => res.x[i])
    end
    return status, value
end