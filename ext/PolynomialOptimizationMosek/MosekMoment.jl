mutable struct StateMoment{K<:Integer}
    const task::Mosek.Task
    num_vars::Int32
    num_cons::Int32
    num_afes::Int64
    const mon_to_mskvar::Dict{FastKey{K},Int32}
end

function append_vars!(state::StateMoment, key)
    if state.num_vars == length(state.mon_to_mskvar)
        newnum = overallocation(state.num_vars + one(state.num_vars))
        appendvars(state.task, newnum - state.num_vars)
        state.num_vars = newnum
    end
    return state.mon_to_mskvar[FastKey(key)] = Int32(length(state.mon_to_mskvar))
end

@inline function Solver.mindex(state::StateMoment, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc}
    idx = monomial_index(monomials...)
    dictidx = Base.ht_keyindex(state.mon_to_mskvar, FastKey(idx))
    @inbounds return (dictidx < 0 ?
        # split this into its own function so that we can inline the good part and call the more complicated appending
        append_vars!(state, idx) :
        state.mon_to_mskvar.vals[dictidx]
    )::Int32
end

Solver.supports_quadratic(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L)

function Solver.add_constr_nonnegative!(state::StateMoment, indices::AbstractVector{Int32}, values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    appendafes(state.task, 1)
    Mosek.@MSK_putafefrow(state.task.task, state.num_afes, length(indices), indices, values)
    Mosek.@MSK_appendacc(state.task.task, 0, 1, StackVec(state.num_afes), StackVec(0.0))
    state.num_afes += 1
    return
end

function Solver.add_constr_quadratic!(state::StateMoment, indvals::Tuple{AbstractVector{Int32},AbstractVector{Float64}}...)
    @assert(all(x -> length(x[1]) == length(x[2]), indvals))
    conedim = length(indvals)
    appendafes(state.task, conedim)
    for (indices, values) in indvals
        Mosek.@MSK_putafefrow(state.task.task, state.num_afes, length(indices), indices, values)
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

function Solver.add_constr_fix!(state::StateMoment, ::Nothing, indices::AbstractVector{Int32},
    values::AbstractVector{Float64}, rhs::Float64)
    @assert(length(indices) == length(values))
    Mosek.@MSK_putarow(state.task.task, state.num_cons, length(indices), indices, values)
    iszero(rhs) || Mosek.@MSK_putconbound(state.task.task, state.num_cons, MSK_BK_FX, rhs, rhs)
    state.num_cons += 1
    return
end

Solver.fix_objective!(state::StateMoment, indices::AbstractVector{Int32}, values::AbstractVector{Float64}) =
    Mosek.@MSK_putclist(state.task.task, length(indices), indices, values)

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

            state = StateMoment(task, zero(Int32), zero(Int32), zero(Int64), Dict{FastKey{K},Int32}())
            moment_setup!(state, relaxation, groupings)
            Mosek.@MSK_putvarboundsliceconst(task.task, 0, length(state.mon_to_mskvar), MSK_BK_FR.value, -Inf, Inf)
            customize(state)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        optimize(task)
        @verbose_info("Optimization complete, retrieving moments")

        max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
        x = Vector{Float64}(undef, length(state.mon_to_mskvar))
        Mosek.@MSK_getxxslice(task.task, MSK_SOL_ITR.value, 0, length(x), x)
        # In any case, our variables will not be in the proper order. First figure this out.
        # Dict does not preserve the insertion order. While we could use OrderedDict instead, we need to do lots of insertions
        # and lookups and only once access the elements in the insertion order; so it is probably better to do the sorting
        # once.
        mon_pos = convert.(K, keys(state.mon_to_mskvar))
        var_vals = collect(values(state.mon_to_mskvar))
        sort_along!(var_vals, mon_pos)
        # Now mon_pos is in insertion order, i.e., there is a 1:1 correspondence between x and mon_pos.
        # However, we want the actual monomial order.
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
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR), MomentVector(relaxation, solution)
    finally
        deletetask(task)
    end
end