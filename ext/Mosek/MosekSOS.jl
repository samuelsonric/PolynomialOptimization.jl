mutable struct StateSOS{M}
    task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
    num_constrs::Int32
    constraint_mappings::Dict{M,Int32}
    sparse_extraction::Dict{Int,Vector{Int64}}
end

get_constraint!(state::StateSOS{M}, mon::M) where {M} =
    get!(state.constraint_mappings, mon) do
        if state.num_constrs == length(state.constraint_mappings)
            appendcons(state.task, 20)
            state.num_constrs += 20
        end
        return length(state.constraint_mappings) + 1
    end

@inline function sos_matrix_scalar!(state::StateSOS, what::Dict{Int32,Float64})
    # if we multiply by zero, we do not need this variable at all
    state.num_vars += 1
    task = state.task
    appendvars(task, 1)
    putvarbound(task, state.num_vars, MSK_BK_LO, 0.0, Inf)
    putacol(task, state.num_vars, collect(keys(what)), collect(values(what)))
end

@inline function sos_matrix_quadratic!(state::StateSOS, what::Vector{Dict{Int32,Float64}})
    task = state.task
    appendvars(task, 3)
    putvarboundslice(task, state.num_vars + Int32(1), state.num_vars + Int32(4),
        [MSK_BK_LO, MSK_BK_FR, MSK_BK_LO], [0.0, -Inf, 0.0], [Inf, Inf, Inf])
    appendcone(task, MSK_CT_RQUAD, 0.0, state.num_vars .+ Int32[1, 3, 2])
    for i in 1:3
        state.num_vars += 1
        putacol(task, state.num_vars, collect(keys(what[i])),
            i == 1 ? 0.5 .* collect(values(what[i])) : collect(values(what[i])))
    end
end

@inline function sos_matrix_psd!(state::StateSOS, lg::Int, matrix_data::Dict{Int32,SparseVector{Float64}})
    task = state.task
    items = (lg * (lg + 1)) >> 1
    sparse_extraction = get!(state.sparse_extraction, lg) do
        return appendsparsesymmatlist(
            task,
            fill(Int32(lg), items),
            ones(Int64, items),
            vcat((i:lg for i in 1:lg)...),
            vcat((fill(Int32(i), lg - i + 1) for i in 1:lg)...),
            ones(items)
        )
    end
    appendbarvars(task, [lg])
    state.num_bar_vars += 1

    alphas = accumulate(+, map(nnz, values(matrix_data)))
    @inbounds putbaraijlist(
        task,
        collect(keys(matrix_data)),
        fill(state.num_bar_vars, length(matrix_data)),
        [0; alphas[1:end-1]],
        alphas,
        vcat([sparse_extraction[rowvals(md)] for md in values(matrix_data)]...),
        vcat(map(nonzeros, values(matrix_data))...)
    )
end

function sos_matrix!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    if lg == 1
        @inbounds sos_matrix_scalar!(state, mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                                        for mon_constr in constraint
                                                        for term in rem(grouping[1] * grouping[1] * mon_constr, gröbner_basis)),
            Int32, Float64))
    elseif lg == 2
        # rotated quadratic cone: 2 * x[1] * x[2] ≥ x[3]^2
        # and the 2x2 PSD should look like y[1] * y[3] ≥ y[2]^2
        @inbounds sos_matrix_quadratic!(state, [mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                                            for mon_constr in constraint
                                                            for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
            Int32, Float64) for exp2 in 1:2 for exp1 in exp2:2])
    else
        items = (lg * (lg + 1)) >> 1
        matrix_data = Dict{Int32,SparseVector{Float64}}()
        i = 1
        for exp2 in 1:lg
            for exp1 in exp2:lg
                sqr = grouping[exp1] * grouping[exp2]
                for mon_constr in constraint
                    @inbounds for term in rem(sqr * mon_constr, gröbner_basis)
                        @inbounds get!(
                            () -> spzeros(items),
                            matrix_data,
                            get_constraint!(state, monomial(term))
                        )[i] += Float64(coefficient(term)) # strict triangle is doubled automatically by Mosek
                    end
                end
                i += 1
            end
        end
        sos_matrix_psd!(state, lg, matrix_data)
    end
    return
end

function sos_matrix!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return sos_matrix!(state, gröbner_basis, grouping, constraint[1, 1])
    end
    lg = length(grouping)
    if lg == 1 && block_size == 2
        sqr = grouping[1] * grouping[1]
        @inbounds sos_matrix_quadratic!(state, [mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                                  for mon_constr in constraint
                                                  for term in rem(sqr * constraint[exp1, exp2], gröbner_basis)),
            Int32, Float64) for exp2 in 1:2 for exp1 in exp2:2])
    else
        dim = lg * block_size
        items = (dim * (dim + 1)) >> 1

        matrix_data = Dict{Int32,SparseVector{Float64}}()
        i = 1
        @inbounds for exp2 in 1:lg
            for block_j in 1:block_size
                for exp1 in exp2:lg
                    sqr = grouping[exp1] * grouping[exp2]
                    for block_i in (exp1 == exp2 ? block_j : 1):block_size
                        for mon_constr in constraint[block_i, block_j]
                            for term in rem(sqr * mon_constr, gröbner_basis)
                                get!(
                                    () -> spzeros(items),
                                    matrix_data,
                                    get_constraint!(state, monomial(term))
                                )[i] += Float64(coefficient(term))
                            end
                        end
                        i += 1
                    end
                end
            end
        end
        sos_matrix_psd!(state, dim, matrix_data)
    end
    return
end

function sos_matrix_eq!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    task = state.task
    lg = length(grouping)
    items = (lg * (lg +1)) >> 1
    appendvars(task, items)
    putvarboundsliceconst(task, state.num_vars + Int32(1), state.num_vars + Int32(items +1), MSK_BK_FR, -Inf, Inf)
    for exp2 in 1:lg
        for exp1 in exp2:lg
            state.num_vars += 1
            @inbounds what = mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                           for mon_constr in constraint
                                           for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                                       Int32, Float64)
            putacol(task, state.num_vars, collect(keys(what)), collect(values(what)))
        end
    end
    return
end

function PolynomialOptimization.sparse_optimize(::Val{:MosekSOS}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, customize::Function=(state) -> nothing,
    parameters...) where {P,M,V}
    @assert(!problem.complex)
    maketask() do task
        setup_time = @elapsed begin
            verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                putparam(task, string(k), v)
            end

            putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)

            state = StateSOS{M}(task, zero(Int32), zero(Int32), zero(Int32), Dict{M,Int32}(), Dict{Int,Vector{Int64}}())

            # SOS term for objective
            for grouping in groupings[1]
                sos_matrix!(state, problem.gröbner_basis, sort(grouping, by=degree),
                    polynomial(constant_monomial(problem.objective)))
            end
            # localizing matrices
            for (groupings, constr) in zip(Iterators.drop(groupings, 1), problem.constraints)
                if constr.type == pctNonneg || constr.type == pctPSD
                    for grouping in groupings
                        sos_matrix!(state, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualityNonneg
                    for grouping in groupings
                        let sg = sort(grouping, by=degree)
                            sos_matrix!(state, problem.gröbner_basis, sg, constr.constraint)
                            sos_matrix!(state, problem.gröbner_basis, sg, -constr.constraint)
                        end
                    end
                elseif constr.type == pctEqualityGröbner
                    for grouping in groupings
                        sos_matrix_eq!(state, EmptyGröbnerBasis{P}(), sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualitySimple
                    for grouping in groupings
                        sos_matrix_eq!(state, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                else
                    @assert(false)
                end
            end

            # add lower bound
            state.num_vars += 1
            appendvars(task, 1)
            putvarbound(task, state.num_vars, MSK_BK_FR, -Inf, Inf)
            putcj(task, state.num_vars, 1.0)
            if isone(problem.prefactor)
                putaij(task, get_constraint!(state, constant_monomial(problem.objective)), state.num_vars, 1.0)
            else
                putacol(task, state.num_vars, map(x -> get_constraint!(state, monomial(x)), problem.prefactor),
                    Float64.(coefficients(problem.prefactor)))
            end

            # fix all constraints to zero (objective constraints will be overwritten later)
            putconboundsliceconst(task, 1, length(state.constraint_mappings) + 1, MSK_BK_FX, 0.0, 0.0)

            # add objective (which is already mod gröbner_basis)
            boundkeys = fill(MSK_BK_FX, length(problem.objective))
            bounds = map(Float64 ∘ coefficient, problem.objective)
            putconboundlist(task, map(x -> get_constraint!(state, monomial(x)), problem.objective), boundkeys, bounds,
                bounds)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        customize(state)

        optimize(task)
        status = getsolsta(task, MSK_SOL_ITR)
        value = getprimalobj(task, MSK_SOL_ITR)
        @verbose_info("Optimization complete")

        empty!(problem.last_moments)
        sizehint!(problem.last_moments, length(state.constraint_mappings))
        mon_vals = gety(task, MSK_SOL_ITR)
        for (mon, i) in state.constraint_mappings
            push!(problem.last_moments, mon => mon_vals[i])
        end
        return status, value
    end
end