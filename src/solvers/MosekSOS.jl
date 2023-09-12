mutable struct MosekStateSOS{M}
    task::Mosek.MSKtask
    num_vars::Int32
    num_bar_vars::Int32
    num_constrs::Int32
    constraint_mappings::Dict{M,Int32}
    sparse_extraction::Dict{Int,Vector{Int64}}
end

get_constraint!(msk::MosekStateSOS{M}, mon::M) where {M} =
    get!(msk.constraint_mappings, mon) do
        if msk.num_constrs == length(msk.constraint_mappings)
            Mosek.appendcons(msk.task, 20)
            msk.num_constrs += 20
        end
        return length(msk.constraint_mappings) + 1
    end

@inline function sos_matrix_scalar!(msk::MosekStateSOS, what::Dict{Int32,Float64})
    # if we multiply by zero, we do not need this variable at all
    msk.num_vars += 1
    task = msk.task
    Mosek.appendvars(task, 1)
    Mosek.putvarbound(task, msk.num_vars, Mosek.MSK_BK_LO, 0.0, Inf)
    Mosek.putacol(task, msk.num_vars, collect(keys(what)), collect(values(what)))
end

@inline function sos_matrix_quadratic!(msk::MosekStateSOS, what::Vector{Dict{Int32,Float64}})
    task = msk.task
    Mosek.appendvars(task, 3)
    Mosek.putvarboundslice(task, msk.num_vars + Int32(1), msk.num_vars + Int32(4),
        [Mosek.MSK_BK_LO, Mosek.MSK_BK_FR, Mosek.MSK_BK_LO], [0.0, -Inf, 0.0], [Inf, Inf, Inf])
    Mosek.appendcone(task, Mosek.MSK_CT_RQUAD, 0.0, msk.num_vars .+ Int32[1, 3, 2])
    for i in 1:3
        msk.num_vars += 1
        Mosek.putacol(task, msk.num_vars, collect(keys(what[i])),
            i == 1 ? 0.5 .* collect(values(what[i])) : collect(values(what[i])))
    end
end

@inline function sos_matrix_psd!(msk::MosekStateSOS, lg::Int, matrix_data::Dict{Int32,SparseVector{Float64}})
    task = msk.task
    items = (lg * (lg + 1)) >> 1
    sparse_extraction = get!(msk.sparse_extraction, lg) do
        return Mosek.appendsparsesymmatlist(
            task,
            fill(Int32(lg), items),
            ones(Int64, items),
            vcat((i:lg for i in 1:lg)...),
            vcat((fill(Int32(i), lg - i + 1) for i in 1:lg)...),
            ones(items)
        )
    end
    Mosek.appendbarvars(task, [lg])
    msk.num_bar_vars += 1

    alphas = accumulate(+, map(nnz, values(matrix_data)))
    @inbounds Mosek.putbaraijlist(
        task,
        collect(keys(matrix_data)),
        fill(msk.num_bar_vars, length(matrix_data)),
        [0; alphas[1:end-1]],
        alphas,
        vcat([sparse_extraction[rowvals(md)] for md in values(matrix_data)]...),
        vcat(map(nonzeros, values(matrix_data))...)
    )
end

function sos_matrix!(msk::MosekStateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    if lg == 1
        @inbounds sos_matrix_scalar!(msk, mergewith(+, (get_constraint!(msk, monomial(term)) => coefficient(term)
                                                        for mon_constr in constraint
                                                        for term in rem(grouping[1] * grouping[1] * mon_constr, gröbner_basis)),
            Int32, Float64))
    elseif lg == 2
        # rotated quadratic cone: 2 * x[1] * x[2] ≥ x[3]^2
        # and the 2x2 PSD should look like y[1] * y[3] ≥ y[2]^2
        @inbounds sos_matrix_quadratic!(msk, [mergewith(+, (get_constraint!(msk, monomial(term)) => coefficient(term)
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
                            get_constraint!(msk, monomial(term))
                        )[i] += Float64(coefficient(term)) # strict triangle is doubled automatically by Mosek
                    end
                end
                i += 1
            end
        end
        sos_matrix_psd!(msk, lg, matrix_data)
    end
    return
end

function sos_matrix!(msk::MosekStateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return sos_matrix!(msk, gröbner_basis, grouping, constraint[1, 1])
    end
    lg = length(grouping)
    if lg == 1 && block_size == 2
        sqr = grouping[1] * grouping[1]
        @inbounds sos_matrix_quadratic!(msk, [mergewith(+, (get_constraint!(msk, monomial(term)) => coefficient(term)
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
                                    get_constraint!(msk, monomial(term))
                                )[i] += Float64(coefficient(term))
                            end
                        end
                        i += 1
                    end
                end
            end
        end
        sos_matrix_psd!(msk, dim, matrix_data)
    end
    return
end

function sos_matrix_eq!(msk::MosekStateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    task = msk.task
    lg = length(grouping)
    items = (lg * (lg +1)) >> 1
    Mosek.appendvars(task, items)
    Mosek.putvarboundsliceconst(task, msk.num_vars + Int32(1), msk.num_vars + Int32(items +1), Mosek.MSK_BK_FR, -Inf, Inf)
    for exp2 in 1:lg
        for exp1 in exp2:lg
            msk.num_vars += 1
            @inbounds what = mergewith(+, (get_constraint!(msk, monomial(term)) => coefficient(term)
                                           for mon_constr in constraint
                                           for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                                       Int32, Float64)
            Mosek.putacol(task, msk.num_vars, collect(keys(what)), collect(values(what)))
        end
    end
    return
end

function sparse_optimize(::Val{:MosekSOS}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, customize::Function=(msk) -> nothing,
    parameters...) where {P,M,V}
    @assert(!problem.complex)
    Mosek.maketask() do task
        setup_time = @elapsed begin
            verbose && Mosek.putstreamfunc(task, Mosek.MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                Mosek.putparam(task, string(k), v)
            end

            Mosek.putobjsense(task, Mosek.MSK_OBJECTIVE_SENSE_MAXIMIZE)

            msk = MosekStateSOS{M}(task, zero(Int32), zero(Int32), zero(Int32), Dict{M,Int32}(), Dict{Int,Vector{Int64}}())

            # SOS term for objective
            for grouping in groupings[1]
                sos_matrix!(msk, problem.gröbner_basis, sort(grouping, by=degree),
                    polynomial(constant_monomial(problem.objective)))
            end
            # localizing matrices
            for (groupings, constr) in zip(Iterators.drop(groupings, 1), problem.constraints)
                if constr.type == pctNonneg || constr.type == pctPSD
                    for grouping in groupings
                        sos_matrix!(msk, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualityNonneg
                    for grouping in groupings
                        let sg = sort(grouping, by=degree)
                            sos_matrix!(msk, problem.gröbner_basis, sg, constr.constraint)
                            sos_matrix!(msk, problem.gröbner_basis, sg, -constr.constraint)
                        end
                    end
                elseif constr.type == pctEqualityGröbner
                    for grouping in groupings
                        sos_matrix_eq!(msk, EmptyGröbnerBasis{P}(), sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualitySimple
                    for grouping in groupings
                        sos_matrix_eq!(msk, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                else
                    @assert(false)
                end
            end

            # add lower bound
            msk.num_vars += 1
            Mosek.appendvars(task, 1)
            Mosek.putvarbound(task, msk.num_vars, Mosek.MSK_BK_FR, -Inf, Inf)
            Mosek.putcj(task, msk.num_vars, 1.0)
            if isone(problem.prefactor)
                Mosek.putaij(task, get_constraint!(msk, constant_monomial(problem.objective)), msk.num_vars, 1.0)
            else
                Mosek.putacol(task, msk.num_vars, map(x -> get_constraint!(msk, monomial(x)), problem.prefactor),
                    Float64.(coefficients(problem.prefactor)))
            end

            # fix all constraints to zero (objective constraints will be overwritten later)
            Mosek.putconboundsliceconst(task, 1, length(msk.constraint_mappings) + 1, Mosek.MSK_BK_FX, 0.0, 0.0)

            # add objective (which is already mod gröbner_basis)
            boundkeys = fill(Mosek.MSK_BK_FX, length(problem.objective))
            bounds = map(Float64 ∘ coefficient, problem.objective)
            Mosek.putconboundlist(task, map(x -> get_constraint!(msk, monomial(x)), problem.objective), boundkeys, bounds,
                bounds)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        customize(msk)

        Mosek.optimize(task)
        status = Mosek.getsolsta(task, Mosek.MSK_SOL_ITR)
        value = Mosek.getprimalobj(task, Mosek.MSK_SOL_ITR)
        @verbose_info("Optimization complete")

        empty!(problem.last_moments)
        sizehint!(problem.last_moments, length(msk.constraint_mappings))
        mon_vals = Mosek.gety(task, Mosek.MSK_SOL_ITR)
        for (mon, i) in msk.constraint_mappings
            push!(problem.last_moments, mon => mon_vals[i])
        end
        return status, value
    end
end