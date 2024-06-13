mutable struct StateMoment{K<:Integer}
    problem::COPTProb
    num_solver_vars::Cint # total number of variables available in the solver
    num_used_vars::Cint # number of variables already used for something (might include scratch variables)
    num_symmat::Cint
    const mon_to_coptvar::Dict{FastKey{K},Cint}
end

function append_vars!(state::StateMoment, key)
    if state.num_used_vars == state.num_solver_vars
        newnum = overallocation(state.num_solver_vars + one(Cint))
        Δ = newnum - state.num_solver_vars
        _check_ret(copt_env, COPT_AddCols(state.problem, Δ, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            fill(-COPT_INFINITY, Δ), C_NULL, C_NULL))
        state.num_solver_vars += Δ
    end
    return state.mon_to_coptvar[FastKey(key)] = let uv=state.num_used_vars
        state.num_used_vars += one(Cint)
        uv
    end
end

@inline function Solver.mindex(state::StateMoment{K}, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {K,Nr,Nc}
    idx = monomial_index(monomials...)
    dictidx = Base.ht_keyindex(state.mon_to_coptvar, FastKey(idx))
    @inbounds return (dictidx < 0 ?
        # split this into its own function so that we can inline the good part and call the more complicated appending
        append_vars!(state, idx) :
        state.mon_to_coptvar.vals[dictidx]
    )::Cint
end

Solver.supports_quadratic(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeMatrixCartesian(:L, Cint(0))
# While COPT expect the lower triangle, we calculate things with linear indexing, and the conversion is easier of upper tri.

function Solver.add_constr_nonnegative!(state::StateMoment, indices::AbstractVector{Cint}, values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    _check_ret(copt_env, COPT_AddRow(state.problem, length(indices), indices, values, 0, 0., COPT_INFINITY, C_NULL))
    return
end

@generated function Solver.add_constr_quadratic!(state::StateMoment,
    indvals::Tuple{AbstractVector{Cint},AbstractVector{Float64}}...)
    N = length(indvals)
    quote
        # COPT does not support an arbitrary-content quadratic cone. Either we put the cone into the form <x, Qx> ≤ b or we
        # need to create more variables that wrap this functionality.
        if state.num_solver_vars < state.num_used_vars + $N
            newnum = overallocation(state.num_solver_vars + Cint($N))
            Δ = newnum - state.num_solver_vars
            lb = fill(-COPT_INFINITY, Δ)
            @inbounds lb[1] = 0.0
            @inbounds lb[2] = 0.0
            _check_ret(copt_env, COPT_AddCols(state.problem, Δ, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                lb, C_NULL, C_NULL))
            state.num_solver_vars += Δ
        end
        rowMatCnt = StackVec($((:(Cint(length(indvals[$i][1]) +1)) for i in 1:N)...),)
        rowMatBeg = StackVec(zero(Cint), $((Expr(:call, :+, (:(rowMatCnt[$j]) for j in 1:i)...) for i in 1:N-1)...))
        rowMatIdx = vcat($((x for i in 1:N for x in (:(indvals[$i][1]), :(state.num_used_vars + $(Cint(i -1)))))...),)
        rowMatElem = vcat($((x for i in 1:N for x in (:(indvals[$i][2]), :(-1.0)))...),)
        _check_ret(copt_env, COPT_AddRows(state.problem, $N, rowMatBeg, rowMatCnt, rowMatIdx, rowMatElem, C_NULL,
            StackVec($((:(0.0) for _ in 1:N)...),), StackVec($((:(0.0) for _ in 1:N)...),), C_NULL))
        _check_ret(copt_env, COPT_AddCones(state.problem, 1, Ref(Cint(COPT_CONE_RQUAD)), Ref(Cint(0)), Ref(Cint($N)),
            StackVec($((:(state.num_used_vars + $i) for i in Cint(0):Cint(N -1))...),)))
        state.num_used_vars += $N
        return
    end
end

function Solver.add_constr_psd!(state::StateMoment, dim::Int, data::PSDMatrixCartesian{Cint,Float64})
    colIdx = data.rowind    # We can do variable piracy: those vectors have more than sufficient length, and we will only write
    symMatIdx = data.colind # to positions that were already traveled.
    start = state.num_symmat
    j = 1
    @inbounds for (midx, (symrows, symcols, vals)) in data
        _check_ret(copt_env, COPT_AddSymMat(state.problem, dim, length(symrows), symrows, symcols, vals))
        colIdx[j] = midx
        symMatIdx[j] = state.num_symmat
        state.num_symmat += 1
        j += 1
    end
    # It appears (undocumented) that we can use -1 to specify that there is no constant term present. Other ways of enforcing
    # this don't work:
    # - creation of a zero-element symmat fails
    # - creation of a one-element symmat with just a zero entry works; but as soon as a quadratic constraint is present, the
    #   solver fails with Invalid Data.
    _check_ret(copt_env, COPT_AddLMIConstr(state.problem, dim, j -1, colIdx, symMatIdx, Cint(-1), C_NULL))
end

function Solver.add_constr_fix!(state::StateMoment, ::Nothing, indices::AbstractVector{Cint}, values::AbstractVector{Float64},
    rhs::Float64)
    @assert(length(indices) == length(values))
    _check_ret(copt_env, COPT_AddRow(state.problem, length(indices), indices, values, 0, rhs, rhs, C_NULL))
    return
end

Solver.fix_objective!(state::StateMoment, indices::AbstractVector{Cint}, values::AbstractVector{Float64}) =
    _check_ret(copt_env, COPT_ReplaceColObj(state.problem, length(indices), indices, values))

function Solver.poly_optimize(::Val{:COPTMoment}, relaxation::AbstractPORelaxation{<:POProblem{P}},
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=_ -> nothing, parameters=()) where {P}
    setup_time = @elapsed begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))

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

        _check_ret(copt_env, COPT_SetObjSense(task, COPT_MINIMIZE))

        state = StateMoment(task, zero(Cint), zero(Cint), zero(Cint), Dict{FastKey{K},Cint}())
        moment_setup!(state, relaxation, groupings)
        customize(state)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")

    _check_ret(copt_env, COPT_Solve(task))
    status = Ref{Cint}()
    _check_ret(copt_env, COPT_GetIntAttr(task, COPT_INTATTR_LPSTATUS, status))
    value = Ref{Cdouble}()
    _check_ret(copt_env, COPT_GetDblAttr(task, COPT_DBLATTR_LPOBJVAL, value))
    @verbose_info("Optimization complete, extracting solution")

    if status[] ∈ (COPT_LPSTATUS_OPTIMAL, COPT_LPSTATUS_IMPRECISE)
        x = Vector{Cdouble}(undef, state.num_solver_vars)
        _check_ret(copt_env, COPT_GetLpSolution(task, x, C_NULL, C_NULL, C_NULL))
        x = resize!(x, state.num_used_vars)

        max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
        # In any case, our variables will not be in the proper order. First figure this out.
        # Dict does not preserve the insertion order. While we could use OrderedDict instead, we need to do lots of insertions
        # and lookups and only once access the elements in the insertion order; so it is probably better to do the sorting
        # once.
        mon_pos = convert.(K, keys(state.mon_to_coptvar))
        var_vals = collect(values(state.mon_to_coptvar))
        sort_along!(var_vals, mon_pos)
        # Now mon_pos is in insertion order. There might still not be a 1:1 correspondence between x and mon_pos, as slack
        # variables due to the quadratic constraints could be present.
        if length(var_vals) < length(x)
            var_vals .+= one(Cint)
            @inbounds x = x[var_vals] # remove the slack
        end
        # Now we have the 1:1 correspondence, but we want the actual monomial order.
        sort_along!(mon_pos, x)
        # Finally, x is ordered!
        if length(x) == max_mons # dense case
            solution = x
        elseif 3length(mon_pos) < max_mons
            solution = SparseVector(max_mons, mon_pos, x)
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), x)
        end
        @verbose_info("Solution data extraction complete")
        return status[], value[], MomentVector(relaxation, solution)
    else
        @verbose_info("Solution data extraction complete")
        return status[], value[], MomentVector(relaxation, Cdouble[])
    end
end