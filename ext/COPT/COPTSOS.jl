mutable struct COPTProb
    ptr::Ptr{copt_prob}
    finalize_called::Bool

    function COPTProb(env::Env)
        p_ptr = Ref{Ptr{copt_prob}}(C_NULL)
        _check_ret(env, COPT_CreateProb(env.ptr, p_ptr))
        prob = new(p_ptr[], false)
        finalizer(prob) do p
            p.finalize_called = true
            if p.ptr != C_NULL
                COPT_DeleteProb(Ref(p.ptr))
                p.ptr = C_NULL
            end
        end
        return prob
    end
end

Base.unsafe_convert(::Type{Ptr{copt_prob}}, prob::COPTProb) = prob.ptr

mutable struct StateSOS{M}
    prob::COPTProb
    num_vars::Cint
    num_psd_vars::Cint
    num_symmat::Cint
    constraint_mappings::Dict{M,Cint}
    linear_constrs::FastVec{Union{Missing,Tuple{FastVec{Cint},FastVec{Cdouble}}}}
    psd_constrs::FastVec{Union{Missing,Tuple{FastVec{Cint},FastVec{Cint}}}}
end

get_constraint!(state::StateSOS{M}, mon::M) where {M} = get!(state.constraint_mappings, mon, length(state.constraint_mappings))

@inline function sos_matrix_scalar!(state::StateSOS, what::Dict{Cint,Cdouble})
    lin = state.linear_constrs
    # if we multiply by zero, we do not need this variable at all
    maxind = maximum(keys(what), init=zero(Cint)) +1
    if maxind > length(lin)
        append!(lin, Iterators.repeated(missing, maxind - length(lin)))
    end
    @inbounds for (i, v) in what
        ind = Int(i +1)
        if ismissing(lin[ind])
            lin[ind] = (FastVec{Cint}(), FastVec{Cdouble}())
        end
        push!(lin[ind][1], state.num_vars)
        push!(lin[ind][2], v)
    end
    _check_ret(copt_env, COPT_AddCol(state.prob, zero(Cdouble), zero(Cint), C_NULL, C_NULL, COPT_CONTINUOUS, zero(Cdouble),
        COPT_INFINITY, C_NULL))
    state.num_vars += 1
    return
end

const copt_refZero = Ref(Cint(0))
const copt_refThree = Ref(Cint(3))
const copt_refConequad = Ref(Cint(COPT_CONE_RQUAD))
const copt_quadRowMatBeg = Cint[0, 2, 4]
const copt_quadRowMatCnt = Cint[2, 2, 2]
const copt_quadRowMatIdx = Cint[0, 3, 2, 4, 3, 5] # will be modified
const copt_quadRowMatElem = Cdouble[1., -1., 1., -1., 1., -1.]
const copt_quadRowSense = fill(COPT_EQUAL, 3)
const copt_quadRowBound = zeros(Cdouble, 3)

@inline function sos_matrix_quadratic!(state::StateSOS, what::NTuple{3,Dict{Cint,Cdouble}})
    lin = state.linear_constrs
    maxind = max(maximum.(keys.(what), init=zero(Cint))...) +1
    if maxind > length(lin)
        append!(lin, Iterators.repeated(missing, maxind - length(lin)))
    end
    @inbounds for (k, whatₖ) in enumerate(what)
        for (i, v) in whatₖ
            ind = Int(i +1)
            if ismissing(lin[ind])
                lin[ind] = (FastVec{Cint}(), FastVec{Cdouble}())
            end
            push!(lin[ind][1], state.num_vars)
            push!(lin[ind][2], k == 1 ? .5v : v)
        end
        state.num_vars += 1
    end
    # we must be careful: COPT does not allow a variable to occur both in a SOC and a PSD constraint (without any documentation
    # of this fact or any checks, the problem will just not solve correctly). So we need to create three additional auxiliaries
    # for the cone and set them equal to the previously used variables.
    _check_ret(copt_env, COPT_AddCols(state.prob, Cint(6), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        Cdouble[0., -COPT_INFINITY, 0., 0., -COPT_INFINITY, 0.], C_NULL, C_NULL))
    @inbounds begin
        copt_quadRowMatIdx[1] = state.num_vars - Cint(3)
        copt_quadRowMatIdx[2] = state.num_vars
        copt_quadRowMatIdx[3] = state.num_vars - Cint(2)
        copt_quadRowMatIdx[4] = state.num_vars + Cint(1)
        copt_quadRowMatIdx[5] = state.num_vars - Cint(1)
        copt_quadRowMatIdx[6] = state.num_vars + Cint(2)
    end
    _check_ret(copt_env, COPT_AddRows(state.prob, 3, copt_quadRowMatBeg, copt_quadRowMatCnt, copt_quadRowMatIdx,
        copt_quadRowMatElem, copt_quadRowSense, copt_quadRowBound, C_NULL, C_NULL))
    state.num_vars += 3
    _check_ret(copt_env, COPT_AddCones(state.prob, one(Cint), copt_refConequad, copt_refZero, copt_refThree,
        [state.num_vars-Cint(3), state.num_vars-Cint(1), state.num_vars-Cint(2)]))
    return
end

@inline function sos_matrix_psd!(state::StateSOS, lg::Int, matrix_data::Dict{Cint,Dict{Tuple{Cint,Cint},Cdouble}})
    psd = state.psd_constrs
    maxind = maximum(keys(matrix_data)) +1
    if maxind > length(psd)
        append!(psd, Iterators.repeated(missing, maxind - length(psd)))
    end
    @inbounds for (i, matrixᵢ) in matrix_data
        _check_ret(copt_env, COPT_AddSymMat(state.prob, lg, length(matrixᵢ), first.(keys(matrixᵢ)), last.(keys(matrixᵢ)),
            collect(values(matrixᵢ))))
        ind = Int(i +1)
        if ismissing(psd[ind])
            psd[ind] = (FastVec{Cint}(), FastVec{Cint}())
        end
        push!(psd[ind][1], state.num_psd_vars)
        push!(psd[ind][2], state.num_symmat)
        state.num_symmat += 1
    end
    _check_ret(copt_env, COPT_AddPSDCol(state.prob, lg, C_NULL))
    state.num_psd_vars += 1
    return
end

function sos_matrix!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    if lg == 1
        @inbounds sos_matrix_scalar!(state,
            mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                          for mon_constr in constraint
                          for term in rem(grouping[1] * grouping[1] * mon_constr, gröbner_basis)),
            Cint, Cdouble))
    elseif lg == 2
        # rotated quadratic cone: 2 * x[1] * x[2] ≥ x[3]^2
        # and the 2x2 PSD should look like y[1] * y[3] ≥ y[2]^2
        @inbounds sos_matrix_quadratic!(state,
            Tuple(mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                for mon_constr in constraint
                                for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                            Cint, Cdouble) for exp2 in 1:2 for exp1 in exp2:2))
    else
        matrix_data = Dict{Cint,Dict{Tuple{Cint,Cint},Cdouble}}()
        for exp2 in 1:lg
            for exp1 in exp2:lg
                sqr = grouping[exp1] * grouping[exp2]
                key = (exp1 - one(Cint), exp2 - one(Cint))
                for mon_constr in constraint
                    @inbounds for term in rem(sqr * mon_constr, gröbner_basis)
                        matrixᵢ = get!(
                            () -> Dict{Tuple{Cint,Cint},Cdouble}(),
                            matrix_data,
                            get_constraint!(state, monomial(term))
                        )
                        matrixᵢ[key] = get(matrixᵢ, key, zero(Cdouble)) + Cdouble(coefficient(term))
                        # strict triangle is doubled automatically by COPT
                    end
                end
            end
        end
        sos_matrix_psd!(state, lg, matrix_data)
    end
    return
end

function sos_matrix!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    block_size == 1 && @inbounds return sos_matrix!(state, gröbner_basis, grouping, constraint[1, 1])
    lg = length(grouping)
    if lg == 1 && block_size == 2
        let sqr=grouping[1] * grouping[1]
            @inbounds sos_matrix_quadratic!(state,
                Tuple(mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                    for mon_constr in constraint
                                    for term in rem(sqr * constraint[exp1, exp2], gröbner_basis)),
                                Cint, Cdouble) for exp2 in 1:2 for exp1 in exp2:2))
        end
    else
        dim = lg * block_size

        matrix_data = Dict{Cint,Dict{Tuple{Cint,Cint},Cdouble}}()
        col = zero(Cint)
        @inbounds for exp2 in 1:lg
            for block_j in 1:block_size
                row = col
                for exp1 in exp2:lg
                    sqr = grouping[exp1] * grouping[exp2]
                    for block_i in (exp1 == exp2 ? block_j : 1):block_size
                        for mon_constr in constraint[block_i, block_j]
                            @inbounds for term in rem(sqr * mon_constr, gröbner_basis)
                                matrixᵢ = get!(
                                    () -> Dict{Tuple{Cint,Cint},Cdouble}(),
                                    matrix_data,
                                    get_constraint!(state, monomial(term))
                                )
                                matrixᵢ[(row, col)] = get(matrixᵢ, (row, col), zero(Cdouble)) + Cdouble(coefficient(term))
                            end
                        end
                        row += one(Cint)
                    end
                end
                col += one(Cint)
            end
        end
        sos_matrix_psd!(state, dim, matrix_data)
    end
    return
end

function sos_matrix_eq!(state::StateSOS{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lin = state.linear_constrs
    lg = length(grouping)
    items = (lg * (lg +1)) >> 1
    for exp2 in 1:lg
        for exp1 in exp2:lg
            @inbounds what = mergewith(+, (get_constraint!(state, monomial(term)) => coefficient(term)
                                           for mon_constr in constraint
                                           for term in rem(grouping[exp1] * grouping[exp2] * mon_constr, gröbner_basis)),
                                       Cint, Cdouble)
            maxind = maximum(keys(what), init=zero(Cint)) +1
            if maxind > length(lin)
                append!(lin, Iterators.repeated(missing, maxind - length(lin)))
            end
            @inbounds for (i, v) in what
                ind = Int(i +1)
                if ismissing(lin[ind])
                    lin[ind] = (FastVec{Cint}(), FastVec{Cdouble}())
                end
                push!(lin[ind][1], state.num_vars)
                push!(lin[ind][2], v)
            end
            state.num_vars += 1
        end
    end
    _check_ret(copt_env, COPT_AddCols(state.prob, items, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        fill(-COPT_INFINITY, items), C_NULL, C_NULL))
    return
end

function PolynomialOptimization.sparse_optimize(::Val{:COPTSOS}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, customize::Function=(copt) -> nothing,
    parameters...) where {P,M,V}
    @assert(!problem.complex)
    prob = COPTProb(copt_env)
    setup_time = @elapsed begin
        _check_ret(copt_env, COPT_SetIntParam(prob, COPT_INTPARAM_LOGTOCONSOLE, Cint(verbose)))
        for (k, v) in parameters
            if v isa Integer
                _check_ret(copt_env, COPT_SetIntParam(prob, k, Cint(v)))
            elseif v isa AbstractFloat
                _check_ret(copt_env, COPT_SetDblParam(prob, k, Cdouble(v)))
            else
                throw(ArgumentError("Parameter $k is not of type Integer or AbstractFloat"))
            end
        end

        _check_ret(copt_env, COPT_SetObjSense(prob, COPT_MAXIMIZE))
        lin = FastVec{Union{Missing,Tuple{FastVec{Cint},FastVec{Cdouble}}}}()
        psd = FastVec{Union{Missing,Tuple{FastVec{Cint},FastVec{Cint}}}}()

        state = StateSOS{M}(prob, zero(Cint), zero(Cint), zero(Cint), Dict{M,Cint}(), lin, psd)

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
        _check_ret(copt_env, COPT_AddCol(prob, 1., 0, C_NULL, C_NULL, COPT_CONTINUOUS, -COPT_INFINITY,
            COPT_INFINITY, C_NULL))
        @inbounds if isone(problem.prefactor)
            ind = Int(get_constraint!(state, constant_monomial(problem.objective)) +1)
            if ind > length(lin)
                append!(lin, Iterators.repeated(missing, ind - length(lin)))
            end
            if ismissing(lin[ind])
                lin[ind] = (FastVec{Cint}(), FastVec{Cdouble}())
            end
            push!(lin[ind][1], state.num_vars)
            push!(lin[ind][2], 1.)
        else
            what = map(x -> get_constraint!(state, monomial(x)), problem.prefactor)
            maxind = maximum(what, init=zero(Cint)) +1
            if maxind > length(lin)
                append!(lin, Iterators.repeated(missing, maxind - length(lin)))
            end
            for (i, v) in zip(what, coefficients(problem.prefactor))
                ind = Int(i +1)
                if ismissing(lin[ind])
                    lin[ind] = (FastVec{Cint}(), FastVec{Cdouble}())
                end
                push!(lin[ind][1], state.num_vars)
                push!(lin[ind][2], Cdouble(v))
            end
        end
        state.num_vars += 1

        # now we can add all our constraints. This was not possible before, as constraints that involve both PSD matrices and
        # linear variables must be added with a single function call. Had we added them directly in AddCol(s), there would be
        # no way of coupling linear and PSD variables.
        # We also must fix the value (zero if they don't appear in the objective, else the coefficient) right here.
        objcoeffs = Dict{Cint,Cdouble}(((get_constraint!(state, monomial(x)), Cdouble(coefficient(x)))
                                        for x in problem.objective))
        ll, lp = length(lin), length(psd)
        for i in 1:max(ll, lp)
            linc = i ≤ ll ? lin[i] : missing
            psdc = i ≤ lp ? psd[i] : missing
            # fix all constraints to zero (objective constraints will be overwritten later)
            _check_ret(copt_env, COPT_AddPSDConstr(
                prob,
                ismissing(linc) ? zero(Cint) : Cint(length(linc[1])),
                ismissing(linc) ? C_NULL : linc[1],
                ismissing(linc) ? C_NULL : linc[2],
                ismissing(psdc) ? zero(Cint) : Cint(length(psdc[1])),
                ismissing(psdc) ? C_NULL : psdc[1],
                ismissing(psdc) ? C_NULL : psdc[2],
                COPT_EQUAL,
                get(objcoeffs, Cint(i -1), zero(Cdouble)), zero(Cdouble),
                C_NULL
            ))
        end
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")

    customize(state)

    _check_ret(copt_env, COPT_Solve(prob))
    status = Ref{Cint}()
    _check_ret(copt_env, COPT_GetIntAttr(prob, COPT_INTATTR_LPSTATUS, status))
    value = Ref{Cdouble}()
    _check_ret(copt_env, COPT_GetDblAttr(prob, COPT_DBLATTR_LPOBJVAL, value))
    @verbose_info("Optimization complete")

    empty!(problem.last_moments)
    if status[] ∈ (COPT_LPSTATUS_OPTIMAL, COPT_LPSTATUS_IMPRECISE)
        # we'll get an exception if we try to query the solution for an unsolved problem
        sizehint!(problem.last_moments, length(state.constraint_mappings))
        mon_vals = Vector{Cdouble}(undef, length(state.constraint_mappings))
        _check_ret(copt_env, @ccall libcopt.COPT_GetPSDSolution(
            prob::Ptr{copt_prob},
            C_NULL::Ptr{Cdouble}, C_NULL::Ptr{Cdouble}, mon_vals::Ptr{Cdouble}, C_NULL::Ptr{Cdouble})::Cint
        )
        for (mon, i) in state.constraint_mappings
            push!(problem.last_moments, mon => mon_vals[i+1])
        end
    end
    return status[], value[]
end