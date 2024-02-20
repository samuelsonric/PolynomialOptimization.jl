mutable struct COPTProb
    ptr::Ptr{copt_prob}
    finalize_called::Bool

    function COPTProb(env::Env)
        p_ptr = Ref{Ptr{copt_prob}}(C_NULL)
        _check_ret(env, COPT_CreateProb(env.ptr, p_ptr))
        problem = new(p_ptr[], false)
        finalizer(problem) do p
            p.finalize_called = true
            if p.ptr != C_NULL
                COPT_DeleteProb(Ref(p.ptr))
                p.ptr = C_NULL
            end
        end
        return problem
    end
end

Base.unsafe_convert(::Type{Ptr{copt_prob}}, problem::COPTProb) = problem.ptr

# In COPT, the only way to mix "ordinary" with matrix variables is to address them in the same call to AddPSDConstr; so we need
# to cache everything that we want to do with these variables.
mutable struct StateSOS
    problem::COPTProb
    num_vars::Cint
    num_psd_vars::Cint
    num_symmat::Cint
    linstart::Cint
    constrs::Dict{FastKey{Int},Tuple{Tuple{FastVec{Cint},FastVec{Cdouble}},
                                     Tuple{FastVec{Cint},FastVec{Cint}}}}
end

get_constr(state::StateSOS, constr::Int) = get!(() -> ((FastVec{Cint}(), FastVec{Cdouble}()),
                                                       (FastVec{Cint}(), FastVec{Cint}())), state.constrs, FastKey(constr))
get_constr(state::StateSOS, constr::FastKey{Int}) = get!(() -> ((FastVec{Cint}(), FastVec{Cdouble}()),
                                                                (FastVec{Cint}(), FastVec{Cint}())), state.constrs, constr)

function PolynomialOptimization.sos_solver_add_scalar!(state::StateSOS, indices::AbstractVector{Int},
    values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    @inbounds for (i, v) in zip(indices, values)
        (idx, val), _ = get_constr(state, i)
        push!(idx, state.num_vars)
        push!(val, v)
    end
    _check_ret(copt_env, COPT_AddCol(state.problem, 0., 0, C_NULL, C_NULL, COPT_CONTINUOUS, 0., COPT_INFINITY, C_NULL))
    state.num_vars += 1
    return
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, indices₊::AbstractVector{Int},
    values₊::AbstractVector{Float64}, rest_free::Tuple{AbstractVector{Int},AbstractVector{Float64}}...)
    for (indices_free, values_free) in ((indices₊, values₊), rest_free...)
        for (i, v) in zip(indices_free, values_free)
            (idx, val), _ = get_constr(state, i)
            push!(idx, state.num_vars)
            push!(val, v)
        end
        state.num_vars += 1
    end
    _check_ret(copt_env, COPT_AddCols(state.problem, 1 + length(rest_free), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        StackVec(0., ntuple(_ -> -COPT_INFINITY, Val(length(rest_free)))...), C_NULL, C_NULL))
    return
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, indices₁::AbstractVector{Int},
    values₁::AbstractVector{Float64}, indices₂::AbstractVector{Int}, values₂::AbstractVector{Float64},
    rest::Tuple{AbstractVector{Int},AbstractVector{Float64}}...)
    @assert(length(indices₁) == length(values₁) && length(indices₂) == length(values₂) &&
        all(x -> length(x[1]) == length(x[2]), rest))
    @inbounds for (k, (indices, values)) in enumerate(((indices₁, values₁), (indices₂, values₂), rest...))
        for (i, v) in zip(indices, values)
            (idx, val), _ = get_constr(state, i)
            push!(idx, state.num_vars)
            push!(val, isone(k) ? 2v : v)
        end
        state.num_vars += 1
    end
    rhsdim = length(rest)
    conedim = 2 + rhsdim
    if COPT._COPT_VERSION < v"7.0.4"
        # Bug in COPT < 7.0.4: a variable must not occur both in a SOC and a PSD constraint, else the problem will be solved
        # incorrectly (without any notice). So in this case, we need additional auxiliaries for the cone
        _check_ret(copt_env, COPT_AddCols(state.problem, 2conedim, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            StackVec(0., 0., ntuple(_ -> -COPT_INFINITY, Val(rhsdim))...,
                     0., 0., ntuple(_ -> -COPT_INFINITY, Val(rhsdim))...), C_NULL, C_NULL))
        _check_ret(copt_env, COPT_AddRows(state.problem, conedim,
            StackVec(ntuple(i -> Cint(2(i -1)), Val(conedim))...), # rowMatBeg
            StackVec(ntuple(_ -> Cint(2), Val(conedim))...), # rowMatCnt
            StackVec(ntuple(let nv=state.num_vars, conedim=conedim
                               i -> Cint(isodd(i) ? nv - conedim + (i >> 1) : nv + (i >> 1) -1)
                            end, Val(2conedim))...), # rowMatIdx
            StackVec(ntuple(i -> isodd(i) ? 1. : -1., Val(2conedim))...), # rowMatElem,
            StackVec(ntuple(_ -> COPT_EQUAL, Val(conedim))...), # quadRowSense
            StackVec(ntuple(_ -> 0., Val(conedim))...), # quadRowBound
            C_NULL, C_NULL))
        state.num_vars += conedim
    else
        _check_ret(copt_env, COPT_AddCols(state.problem, conedim, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            StackVec(0., 0., ntuple(_ -> -COPT_INFINITY, Val(rhsdim))...), C_NULL, C_NULL))
    end
    _check_ret(copt_env, COPT_AddCones(state.problem, 1, Ref(Cint(COPT_CONE_RQUAD)), Ref(Cint(0)), Ref(Cint(conedim)),
        StackVec(ntuple(let nv=state.num_vars-conedim-1
                            i -> Cint(nv + i)
                        end, Val(conedim))...)))
    return
end

PolynomialOptimization.sos_solver_supports_quadratic(::StateSOS) = true

function PolynomialOptimization.sos_solver_add_psd!(state::StateSOS, dim::Int,
    data::Dict{FastKey{Int},<:Tuple{AbstractVector{Cint},AbstractVector{Cint},AbstractVector{Float64}}})
    @inbounds for (i, matrixᵢ) in data
        _check_ret(copt_env, COPT_AddSymMat(state.problem, dim, length(matrixᵢ[1]), matrixᵢ[1], matrixᵢ[2], matrixᵢ[3]))
        _, (idx, val) = get_constr(state, i)
        push!(idx, state.num_psd_vars)
        push!(val, state.num_symmat)
        state.num_symmat += 1
    end
    _check_ret(copt_env, COPT_AddPSDCol(state.problem, dim, C_NULL))
    state.num_psd_vars += 1
    return
end

PolynomialOptimization.sos_solver_psd_indextype(::StateSOS) = Tuple{Cint,Cint}, :L, zero(Cint)

function PolynomialOptimization.sos_solver_add_free_prepare!(state::StateSOS, num::Int)
    _check_ret(copt_env, COPT_AddCols(state.problem, num, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        fill(-COPT_INFINITY, num), C_NULL, C_NULL))
    prev = state.num_vars
    state.num_vars += num
    return prev
end

function PolynomialOptimization.sos_solver_add_free!(state::StateSOS, eqstate::Cint, indices::AbstractVector{Int},
    values::AbstractVector{Float64}, obj::Bool)
    @inbounds for (i, v) in zip(indices, values)
        (idx, val), _ = get_constr(state, i)
        push!(idx, eqstate)
        push!(val, v)
    end
    obj && _check_ret(copt_env, COPT_SetColObj(state.problem, 1, Ref(eqstate), Ref(1.)))
    return eqstate + one(Cint)
end

function PolynomialOptimization.sos_solver_fix_constraints!(state::StateSOS, indices::Vector{Int}, values::Vector{Float64})
    len = length(indices)
    @assert(len == length(values))
    # this is almost the end - we can now add all our PSD constraints
    sort_along!(indices, values)
    COPT_GetIntAttr(state.problem, COPT_INTATTR_ROWS, pointer_from_objref(state) + fieldoffset(typeof(state), 5))
    for (i, ((linidx, linval), (psdidx, psdval))) in state.constrs
        constr_idx = searchsorted(indices, convert(Int, i))
        if isempty(constr_idx)
            constr_val = 0.
        else
            @inbounds constr_val = values[constr_idx[1]]
        end
        # If we call AddPSDConstr with isempty(psdidx), COPT will automatically turn this into an ordinary linear constraint.
        # Therefore, we cannot get its dual values in the "usual" way. This makes everything quite cumbersome (as the whole
        # issue that COPT_SetPSDElem does not allow to change the linear terms in PSD constraints, which is why we need this
        # whole dictionary caching), we must discriminate between "true" and "scalarized" PSD constraints (and take care of
        # the potential additional linear constraints introduced before due to the quadratic bug - this is why we store the
        # number of linear constraints before this step in linstart).
        # However, maybe COPT will change this behavior some day. To make sure our approach still works, we manually call
        # AddRow instead of AddPSDConstr when appropriate.
        if isempty(psdidx)
            _check_ret(copt_env, COPT_AddRow(
                state.problem,
                length(linidx), linidx, linval,
                COPT_EQUAL, constr_val, 0., C_NULL
            ))
        else
            _check_ret(copt_env, COPT_AddPSDConstr(
                state.problem,
                length(linidx), linidx, linval,
                length(psdidx), psdidx, psdval,
                COPT_EQUAL, constr_val, 0., C_NULL
            ))
        end
    end
    return
end

function PolynomialOptimization.poly_optimize(::Val{:COPTSOS}, relaxation::AbstractPORelaxation{<:POProblem{P}},
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=(state) -> nothing, parameters=()) where {P}
    setup_time = @elapsed begin
        task = COPTProb(copt_env)
        _check_ret(copt_env, COPT_SetIntParam(task, COPT_INTPARAM_LOGTOCONSOLE, Cint(verbose)))
        for (k, v) in parameters
            if v isa Integer
                _check_ret(copt_env, COPT_SetIntParam(task, k, Cint(v)))
            elseif v isa AbstractFloat
                _check_ret(copt_env, COPT_SetDblParam(task, k, Cdouble(v)))
            else
                throw(ArgumentError("Parameter $k is not of type Integer or AbstractFloat"))
            end
        end

        _check_ret(copt_env, COPT_SetObjSense(task, COPT_MAXIMIZE))

        # Let's put some overestimate on the number of monomials
        state = StateSOS(task, zero(Cint), zero(Cint), zero(Cint), zero(Cint),
            Dict{FastKey{Int},Tuple{Tuple{FastVec{Cint},FastVec{Cdouble}},
                                    Tuple{FastVec{Cint},FastVec{Cint}}}}())

        PolynomialOptimization.sos_setup!(state, relaxation, groupings)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")

    customize(state)

    _check_ret(copt_env, COPT_Solve(task))
    status = Ref{Cint}()
    _check_ret(copt_env, COPT_GetIntAttr(task, COPT_INTATTR_LPSTATUS, status))
    value = Ref{Cdouble}()
    _check_ret(copt_env, COPT_GetDblAttr(task, COPT_DBLATTR_LPOBJVAL, value))
    @verbose_info("Optimization complete, extracting solution")

    if status[] ∈ (COPT_LPSTATUS_OPTIMAL, COPT_LPSTATUS_IMPRECISE)
        linconstrs = Ref{Cint}()
        _check_ret(
            copt_env,
            COPT_GetIntAttr(task, COPT_INTATTR_ROWS, linconstrs)
        )
        psdconstrs = Ref{Cint}()
        _check_ret(
            copt_env,
            COPT_GetIntAttr(task, COPT_INTATTR_PSDCONSTRS, psdconstrs)
        )
        if linconstrs[] > state.linstart
            lin_duals = Vector{Cdouble}(undef, linconstrs[])
            _check_ret(
                copt_env,
                COPT_GetLpSolution(task, C_NULL, C_NULL, lin_duals, C_NULL)
            )
        end
        if psdconstrs[] > 0
            psd_duals = Vector{Cdouble}(undef, psdconstrs[])
            _check_ret(
                copt_env,
                @ccall libcopt.COPT_GetPSDSolution(
                    task::Ptr{copt_prob},
                    C_NULL::Ptr{Cdouble},
                    C_NULL::Ptr{Cdouble},
                    psd_duals::Ptr{Cdouble},
                    C_NULL::Ptr{Cdouble}
                )::Cint
            )
        end
        max_mons = monomial_count(2degree(relaxation), nvariables(relaxation.objective))
        if 3((linconstrs[] - state.linstart) + psdconstrs[]) < max_mons
            inds = convert.(Int, keys(state.constrs))
            vals = FastVec{Float64}(buffer=(linconstrs[] - state.linstart + psdconstrs[]))
            psdi = 1
            lini = state.linstart +1
            for (_, (havepsd, _)) in values(state.constrs)
                if isempty(havepsd)
                    unsafe_push!(vals, lin_duals[lini])
                    lini += 1
                else
                    unsafe_push!(vals, psd_duals[psdi])
                    psdi += 1
                end
            end
            sort_along!(inds, vals)
            solution = SparseVector(max_mons, inds, finish!(vals))
        else
            solution = fill(NaN, max_mons)
            psdi = 1
            lini = state.linstart +1
            for (i, (_, (havepsd, _))) in state.constrs
                if isempty(havepsd)
                    @inbounds solution[convert(Int, i)] = lin_duals[lini]
                    lini += 1
                else
                    @inbounds solution[convert(Int, i)] = psd_duals[psdi]
                    psdi += 1
                end
            end
        end
        @verbose_info("Solution data extraction complete")
        return status[], value[], MomentVector(relaxation, solution)
    else
        @verbose_info("Solution data extraction complete")
        return status[], value[], MomentVector(relaxation, Cdouble[])
    end
end