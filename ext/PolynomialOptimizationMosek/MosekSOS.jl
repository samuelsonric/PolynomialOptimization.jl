mutable struct StateSOS{K<:Integer} <: AbstractAPISolver{K,Int32,Float64}
    const task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
    num_solver_cons::Int32 # total number of constraints available in the solver
    num_used_cons::Int32 # number of constraints already used for something (might include scratch constraints)
    const mon_to_solver::Dict{FastKey{K},Int32}

    StateSOS{K}(task::Mosek.Task) where {K<:Integer} = new{K}(
        task, zero(Int32), zero(Int32), zero(Int32), zero(Int32), Dict{FastKey{K},Int32}()
    )
end

function Base.append!(state::StateSOS, key)
    if state.num_used_cons == state.num_solver_cons
        newcon = overallocation(state.num_solver_cons + one(state.num_solver_cons))
        appendcons(state.task, newcon - state.num_solver_cons)
        state.num_solver_cons = newcon
    end
    return state.mon_to_solver[FastKey(key)] = let uc=state.num_used_cons
        state.num_used_cons += one(state.num_used_cons)
        uc
    end
end

Solver.supports_rotated_quadratic(::StateSOS) = true

Solver.psd_indextype(::StateSOS) = PSDIndextypeMatrixCartesian(:L, zero(Int32))

function Solver.add_constr_slack!(state::StateSOS, num::Int)
    if state.num_used_cons + num > state.num_solver_cons
        newnum = overallocation(state.num_used_cons + Int32(num))
        appendcons(state.task, newnum - state.num_solver_cons)
        state.num_solver_cons = newnum
    end
    result = state.num_used_cons:state.num_used_cons+Int32(num -1)
    state.num_used_cons += num
    return result
end

function Solver.add_var_nonnegative!(state::StateSOS, indvals::Indvals{Int32,Float64})
    appendvars(state.task, 1)
    id = state.num_vars
    Mosek.@MSK_putacol(state.task.task, id, length(indvals), indvals.indices, indvals.values)
    Mosek.@MSK_putvarbound(state.task.task, id, MSK_BK_LO.value, 0., Inf)
    state.num_vars += one(state.num_vars)
    return id
end

function Solver.add_var_nonnegative!(state::StateSOS, indvals::IndvalsIterator{Int32,Float64})
    N = length(indvals)
    appendvars(state.task, N)
    id = state.num_vars
    curvar = id
    for indval in indvals
        Mosek.@MSK_putacol(state.task.task, curvar, length(indval), indval.indices, indval.values)
        curvar += one(curvar)
    end
    Mosek.@MSK_putvarboundsliceconst(state.task.task, id, curcon, MSK_BK_LO.value, 0., Inf)
    state.num_vars = curvar
    return id:(curvar - one(curvar))
end

function Solver.add_var_quadratic!(state::StateSOS, indvals::IndvalsIterator{Int32,Float64})
    conedim = length(indvals)
    rhsdim = conedim -2
    appendvars(state.task, conedim)
    id = state.num_vars
    Mosek.@MSK_putvarbound(state.task.task, id, MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putvarboundsliceconst(state.task.task, id + one(id), id + Int32(conedim), MSK_BK_FR.value, -Inf, Inf)
    Mosek.@MSK_appendconeseq(state.task.task, MSK_CT_QUAD.value, 0., conedim, id)
    curvar = id
    for indval in indvals
        Mosek.@MSK_putacol(state.task.task, curvar, length(indval), indval.indices, indval.values)
        curvar += one(curvar)
    end
    state.num_vars = curvar
    return id:(curvar - one(curvar))
end

function Solver.add_var_rotated_quadratic!(state::StateSOS, indvals::IndvalsIterator{Int32,Float64})
    conedim = length(indvals)
    rhsdim = conedim -2
    appendvars(state.task, conedim)
    id = state.num_vars
    Mosek.@MSK_putvarboundsliceconst(state.task.task, id, id + Int32(2), MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putvarboundsliceconst(state.task.task, id + Int32(2), id + Int32(conedim), MSK_BK_FR.value, -Inf, Inf)
    Mosek.@MSK_appendconeseq(state.task.task, MSK_CT_RQUAD.value, 0., conedim, id)
    curvar = id
    for indval in indvals
        Mosek.@MSK_putacol(state.task.task, curvar, length(indval), indval.indices, indval.values)
        curvar += one(curvar)
    end
    state.num_vars = curvar
    return id:(curvar - one(curvar))
end

function Solver.add_var_psd!(state::StateSOS, dim::Int, data::PSDMatrixCartesian{Int32,Float64})
    Mosek.@MSK_appendbarvars(state.task.task, 1, Ref(Int32(dim)))
    outidx = Ref{Int64}()
    oneref = Ref{Float64}(1.)
    @inbounds for (midx, (symrows, symcols, vals)) in data
        Mosek.@MSK_appendsparsesymmat(state.task.task, dim, length(symrows), symrows, symcols, vals, outidx)
        Mosek.@MSK_putbaraij(state.task.task, midx, state.num_bar_vars, 1, outidx, oneref)
    end
    state.num_bar_vars += one(state.num_bar_vars)
    return state.num_bar_vars - one(state.num_bar_vars)
end

function Solver.add_var_free_prepare!(state::StateSOS, num::Int)
    appendvars(state.task, num)
    Mosek.@MSK_putvarboundsliceconst(state.task.task, state.num_vars, state.num_vars + num, MSK_BK_FR.value, -Inf, Inf)
    return
end

function Solver.add_var_free!(state::StateSOS, ::Nothing, indvals::Indvals{Int32,Float64}, obj::Float64)
    Mosek.@MSK_putacol(state.task.task, state.num_vars, length(indvals), indvals.indices, indvals.values)
    iszero(obj) || Mosek.@MSK_putcj(state.task.task, state.num_vars, obj)
    state.num_vars += 1
    return
end

function Solver.fix_constraints!(state::StateSOS, indvals::Indvals{Int32,Float64})
    len = length(indvals)
    Mosek.@MSK_putconboundsliceconst(state.task.task, 0, state.num_used_cons, MSK_BK_FX, 0., 0.)
    Mosek.@MSK_putconboundlist(state.task.task, len, indvals.indices, fill(MSK_BK_FX.value, len), indvals.values,
        indvals.values)
    return
end

function Solver.poly_optimize(::Val{:MosekSOS}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation, verbose::Bool=false, customize::Base.Callable=(state) -> nothing, parameters...)
    task = Mosek.Task(msk_global_env::Env)
    setup_time = @elapsed begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))

        verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
        for (k, v) in parameters
            putparam(task, string(k), string(v))
        end
        putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)

        state = StateSOS{K}(task)
        sos_setup!(state, relaxation, groupings; representation)
        customize(state)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")

    optimize(task)
    @verbose_info("Optimization complete")

    return state, getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR)
end

function Solver.extract_moments(relaxation::AbstractRelaxation, state::StateSOS)
    y = Vector{Float64}(undef, length(state.mon_to_solver))
    Mosek.@MSK_getyslice(state.task.task, MSK_SOL_ITR.value, 0, length(y), y)
    return MomentVector(relaxation, y, state)
end