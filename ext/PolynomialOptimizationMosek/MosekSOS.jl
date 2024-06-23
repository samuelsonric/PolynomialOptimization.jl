mutable struct StateSOS{K<:Integer} <: AbstractAPISolver{K}
    const task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
    num_cons::Int32
    const mon_to_solver::Dict{FastKey{K},Int32}

    StateSOS{K}(task::Mosek.Task) where {K<:Integer} = new{K}(
        task, zero(Int32), zero(Int32), zero(Int32), Dict{FastKey{K},Int32}()
    )
end

function Base.append!(state::StateSOS, key)
    if state.num_cons == length(state.mon_to_solver)
        newcon = overallocation(state.num_cons + one(state.num_cons))
        appendcons(state.task, newcon - state.num_cons)
        state.num_cons = newcon
    end
    return state.mon_to_solver[FastKey(key)] = Int32(length(state.mon_to_solver))
end

Solver.supports_quadratic(::StateSOS) = SOLVER_QUADRATIC_RSOC

Solver.psd_indextype(::StateSOS) = PSDIndextypeMatrixCartesian(:L, zero(Int32))

function Solver.add_var_nonnegative!(state::StateSOS, indvals::AbstractIndvals{Int32,Float64})
    task = state.task.task
    Mosek.@MSK_appendvars(task, 1)
    Mosek.@MSK_putvarbound(task, state.num_vars, MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putacol(task, state.num_vars, length(indvals), indvals.indices, indvals.values)
    state.num_vars += 1
    return
end

function Solver.add_var_quadratic!(state::StateSOS, indvals::AbstractIndvals{Int32,Float64}...)
    conedim = length(indvals)
    rhsdim = conedim -2
    appendvars(state.task, conedim)
    # TODO: check boxing
    Mosek.@MSK_putvarboundslice(state.task.task, state.num_vars, state.num_vars + Int32(conedim),
        StackVec(MSK_BK_LO.value, MSK_BK_LO.value, ntuple(_ -> MSK_BK_FR.value, Val(rhsdim))...),
        StackVec(0., 0., ntuple(_ -> -Inf, Val(rhsdim))...), StackVec(ntuple(_ -> Inf, Val(conedim))...))
    varidx = StackVec(ntuple(let nv=state.num_vars -1
                                i -> Int32(nv + i)
                             end, Val(conedim)))
    Mosek.@MSK_appendcone(state.task.task, MSK_CT_RQUAD.value, 0., conedim, varidx)
    for indval in indvals
        Mosek.@MSK_putacol(state.task.task, state.num_vars, length(indval), indval.indices, indval.values)
        state.num_vars += 1
    end
    return
end

function Solver.add_var_psd!(state::StateSOS, dim::Int, data::PSDMatrixCartesian{Int32,Float64})
    Mosek.@MSK_appendbarvars(state.task.task, 1, Ref(Int32(dim)))
    outidx = Ref{Int64}()
    oneref = Ref{Float64}(1.)
    @inbounds for (midx, (symrows, symcols, vals)) in data
        Mosek.@MSK_appendsparsesymmat(state.task.task, dim, length(symrows), symrows, symcols, vals, outidx)
        Mosek.@MSK_putbaraij(state.task.task, midx, state.num_bar_vars, 1, outidx, oneref)
    end
    state.num_bar_vars += 1
    return
end

function Solver.add_var_free_prepare!(state::StateSOS, num::Int)
    appendvars(state.task, num)
    Mosek.@MSK_putvarboundsliceconst(state.task.task, state.num_vars, state.num_vars + num, MSK_BK_FR.value, -Inf, Inf)
    return
end

function Solver.add_var_free!(state::StateSOS, ::Nothing, indvals::AbstractIndvals{Int32}, obj::Float64)
    Mosek.@MSK_putacol(state.task.task, state.num_vars, length(indvals), indvals.indices, indvals.values)
    iszero(obj) || Mosek.@MSK_putcj(state.task.task, state.num_vars, obj)
    state.num_vars += 1
    return
end

function Solver.fix_constraints!(state::StateSOS, indvals::AbstractIndvals{Int32,Float64})
    len = length(indvals)
    Mosek.@MSK_putconboundsliceconst(state.task.task, 0, length(state.mon_to_solver), MSK_BK_FX, 0., 0.)
    Mosek.@MSK_putconboundlist(state.task.task, len, indvals.indices, fill(MSK_BK_FX.value, len), indvals.values,
        indvals.values)
    return
end

function Solver.poly_optimize(::Val{:MosekSOS}, relaxation::AbstractRelaxation,
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=(state) -> nothing, parameters...)
    task = Mosek.Task(msk_global_env::Env)
    try
        setup_time = @elapsed begin
            K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))

            verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                putparam(task, string(k), string(v))
            end
            putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)

            state = StateSOS{K}(task)
            sos_setup!(state, relaxation, groupings)
            customize(state)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        optimize(task)
        @verbose_info("Optimization complete, retrieving moments")

        y = Vector{Float64}(undef, length(state.mon_to_solver))
        Mosek.@MSK_getyslice(task.task, MSK_SOL_ITR.value, 0, length(y), y)
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR), MomentVector(relaxation, y, state)
    finally
        deletetask(task)
    end
end