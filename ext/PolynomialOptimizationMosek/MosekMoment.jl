mutable struct StateMoment{K<:Integer} <: APISolver{K}
    const task::Mosek.Task
    num_vars::Int32
    num_cons::Int32
    num_afes::Int64
    const mon_to_solver::Dict{FastKey{K},Int32}

    StateMoment{K}(task::Mosek.Task) where {K<:Integer} = new{K}(
        task, zero(Int32), zero(Int32), zero(Int64), Dict{FastKey{K},Int32}()
    )
end

function Base.append!(state::StateMoment, key)
    if state.num_vars == length(state.mon_to_solver)
        newnum = overallocation(state.num_vars + one(state.num_vars))
        appendvars(state.task, newnum - state.num_vars)
        state.num_vars = newnum
    end
    return state.mon_to_solver[FastKey(key)] = Int32(length(state.mon_to_solver))
end

Solver.supports_quadratic(::StateMoment) = SOLVER_QUADRATIC_RSOC

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L)

function Solver.add_constr_nonnegative!(state::StateMoment, indvals::AbstractIndvals{Int32,Float64})
    appendafes(state.task, 1)
    Mosek.@MSK_putafefrow(state.task.task, state.num_afes, length(indvals), indvals.indices, indvals.values)
    Mosek.@MSK_appendacc(state.task.task, 0, 1, StackVec(state.num_afes), StackVec(0.0))
    state.num_afes += 1
    return
end

function Solver.add_constr_quadratic!(state::StateMoment, indvals::AbstractIndvals{Int32,Float64}...)
    conedim = length(indvals)
    appendafes(state.task, conedim)
    for indval in indvals
        Mosek.@MSK_putafefrow(state.task.task, state.num_afes, length(indval), indval.indices, indval.values)
        state.num_afes += 1
    end
    if length(indvals) == 3
        Mosek.@MSK_appendaccseq(state.task.task, 1, 3, state.num_afes -3, C_NULL)
    elseif length(indvals) == 4
        Mosek.@MSK_appendaccseq(state.task.task, 2, 4, state.num_afes -4, C_NULL)
    else
        error("Unsupported length of quadratic cone")
    end
    return
end

function Solver.add_constr_psd!(state::StateMoment, dim::Int, data::PSDVector{Int32,Float64})
    n = trisize(dim)
    appendafes(state.task, n)
    curafe = state.num_afes
    # putafefrowlist would require us to create collect(state.num_afes:state.num_afes+length(data)) for the afeidx parameter
    # and accumulate Base.index_lengths(data) for ptrrow
    for (row, val) in data
        @capture(Mosek.@MSK_putafefrow(state.task.task, $curafe, length(row), row, val))
        curafe += 1
    end
    # for sure, the number of PSD cones will not be the bottleneck, so let's save ourselves the trouble of caching already
    # existing domains
    cone = Ref{Int64}()
    Mosek.@MSK_appendsvecpsdconedomain(state.task.task, n, cone)
    Mosek.@MSK_appendaccseq(state.task.task, cone[], n, state.num_afes, C_NULL)
    state.num_afes = curafe
    return
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    appendcons(state.task, num)
    # most of the constraints will be zero, so we can save some calls if we all set them to zero and only change the few
    # nonzero ones
    Mosek.@MSK_putconboundsliceconst(state.task.task, state.num_cons, state.num_cons + num, MSK_BK_FX, 0.0, 0.0)
    return
end

function Solver.add_constr_fix!(state::StateMoment, ::Nothing, indvals::AbstractIndvals{Int32,Float64}, rhs::Float64)
    Mosek.@MSK_putarow(state.task.task, state.num_cons, length(indvals), indvals.indices, indvals.values)
    iszero(rhs) || Mosek.@MSK_putconbound(state.task.task, state.num_cons, MSK_BK_FX, rhs, rhs)
    state.num_cons += 1
    return
end

function Solver.fix_objective!(state::StateMoment, indvals::AbstractIndvals{Int32,Float64})
    Mosek.@MSK_putclist(state.task.task, length(indvals), indvals.indices, indvals.values)
    return
end

function Solver.poly_optimize(::Val{:MosekMoment}, relaxation::AbstractRelaxation,
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=_ -> nothing, parameters...)
    task = Mosek.Task(msk_global_env::Env)
    try
        setup_time = @elapsed begin
            K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))

            verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                putparam(task, string(k), string(v))
            end
            putobjsense(task, MSK_OBJECTIVE_SENSE_MINIMIZE)
            # just set up some domains, this is cheap and we don't have to query them afterwards
            appendrplusdomain(task, 1)
            appendrquadraticconedomain(task, 3)
            appendrquadraticconedomain(task, 4)

            state = StateMoment{K}(task)
            moment_setup!(state, relaxation, groupings)
            Mosek.@MSK_putvarboundsliceconst(task.task, 0, length(state.mon_to_solver), MSK_BK_FR.value, -Inf, Inf)
            customize(state)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        optimize(task)
        @verbose_info("Optimization complete, retrieving moments")

        x = Vector{Float64}(undef, length(state.mon_to_solver))
        Mosek.@MSK_getxxslice(task.task, MSK_SOL_ITR.value, 0, length(x), x)
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR), MomentVector(relaxation, x, state)
    finally
        deletetask(task)
    end
end