mutable struct StateSOS{K<:Integer}
    const task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
    num_cons::Int32
    const mon_to_mskcon::Dict{FastKey{K},Int32}
end

function append_cons!(state::StateSOS, key)
    if state.num_cons == length(state.mon_to_mskcon)
        newcon = overallocation(state.num_cons + one(state.num_cons))
        appendcons(state.task, newcon - state.num_cons)
        state.num_cons = newcon
    end
    return state.mon_to_mskcon[FastKey(key)] = Int32(length(state.mon_to_mskcon))
end

@inline function Solver.mindex(state::StateSOS, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc}
    idx = monomial_index(monomials...)
    dictidx = Base.ht_keyindex(state.mon_to_mskcon, FastKey(idx))
    @inbounds return (dictidx < 0 ?
        # split this into its own function so that we can inline the good part and call the more complicated appending
        append_cons!(state, idx) :
        state.mon_to_mskcon.vals[dictidx]
    )::Int32
end

Solver.supports_quadratic(::StateSOS) = true

Solver.psd_indextype(::StateSOS) = PSDIndextypeMatrixCartesian(:L, zero(Int32))

function Solver.add_var_nonnegative!(state::StateSOS, indices::AbstractVector{Int32}, values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    task = state.task.task
    Mosek.@MSK_appendvars(task, 1)
    Mosek.@MSK_putvarbound(task, state.num_vars, MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putacol(task, state.num_vars, length(indices), indices, values)
    state.num_vars += 1
    return
end

function Solver.add_var_quadratic!(state::StateSOS, indvals::Tuple{AbstractVector{Int32},AbstractVector{Float64}}...)
    @assert(all(x -> length(x[1]) == length(x[2]), indvals))
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
    for (indices, values) in indvals
        Mosek.@MSK_putacol(state.task.task, state.num_vars, length(indices), indices, values)
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

function Solver.add_var_free!(state::StateSOS, ::Nothing, indices::AbstractVector{Int32}, values::AbstractVector{Float64},
    obj::Float64)
    @assert(length(indices) == length(values))
    Mosek.@MSK_putacol(state.task.task, state.num_vars, length(indices), indices, values)
    iszero(obj) || Mosek.@MSK_putcj(state.task.task, state.num_vars, obj)
    state.num_vars += 1
    return
end

function Solver.fix_constraints!(state::StateSOS, indices::Vector{Int32}, values::Vector{Float64})
    len = length(indices)
    @assert(len == length(values))
    Mosek.@MSK_putconboundsliceconst(state.task.task, 0, length(state.mon_to_mskcon), MSK_BK_FX, 0., 0.)
    Mosek.@MSK_putconboundlist(state.task.task, len, indices, fill(MSK_BK_FX.value, len), values, values)
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

            state = StateSOS(task, zero(Int32), zero(Int32), zero(Int32), Dict{FastKey{K},Int32}())
            sos_setup!(state, relaxation, groupings)
            customize(state)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        optimize(task)
        @verbose_info("Optimization complete, retrieving moments")

        max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
        y = Vector{Float64}(undef, length(state.mon_to_mskcon))
        Mosek.@MSK_getyslice(task.task, MSK_SOL_ITR.value, 0, length(y), y)
        # In any case, our variables will not be in the proper order. First figure this out.
        # Dict does not preserve the insertion order. While we could use OrderedDict instead, we need to do lots of insertions
        # and lookups and only once access the elements in the insertion order; so it is probably better to do the sorting
        # once.
        mon_pos = convert.(K, keys(state.mon_to_mskcon))
        var_vals = collect(values(state.mon_to_mskcon))
        sort_along!(var_vals, mon_pos)
        # Now mon_pos is in insertion order, i.e., there is a 1:1 correspondence between x and mon_pos.
        # However, we want the actual monomial order.
        sort_along!(mon_pos, y)
        # Finally, y is ordered!
        if length(y) == max_mons # dense case
            solution = y
        elseif 3length(mon_pos) < max_mons
            solution = SparseVector(max_mons, mon_pos, y)
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), y)
        end
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR), MomentVector(relaxation, solution)
    finally
        deletetask(task)
    end
end