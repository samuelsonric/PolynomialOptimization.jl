mutable struct StateSOS
    const task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
    num_constrs::Int32
    const constr_map::Dict{FastKey{UInt},Int32}
end

function append_constrs!(state, monidx, conjmonidx)
    # We assign new indices contiguously, but we always create conjugate pairs at the same time, so that we can maintain the
    # canonical order as required.
    solveridx₁ = Int32(length(state.constr_map))::Int32
    solveridx₂ = conjmonidx == monidx ? solveridx₁ : solveridx₁ + one(Int32)
    if solveridx₂ ≥ state.num_constrs
        # unused constraints are quite cheap (though not entirely for free), so we'll liberally overestimate to reduce the
        # number of calls
        new_num = FastVector.overallocation(solveridx₂ +1)
        Mosek.@MSK_appendcons(state.task.task, new_num - state.num_constrs)
        state.num_constrs = new_num
    end
    if conjmonidx == monidx
        state.constr_map[FastKey(monidx)] = solveridx₁
        return solveridx₁
    else
        reidx, imidx = minmax(monidx, conjmonidx)
        state.constr_map[FastKey(reidx)] = solveridx₁
        state.constr_map[FastKey(imidx)] = solveridx₂
        return monidx < conjmonidx ? solveridx₁ : solveridx₂
    end
end

@inline function Solver.mindex(state::StateSOS, monomials::SimpleMonomialOrConj...)
    idx = monomial_index(monomials...)
    dictidx = Base.ht_keyindex(state.constr_map, FastKey(idx))
    if dictidx < 0
        # split this into its own function so that we can inline the good part and call the more complicated appending
        return append_constrs!(state, idx, monomial_index(SimpleConjMonomial.(reverse(monomials))...))
    else
        @inbounds return state.constr_map.vals[dictidx]
    end
end

function Solver.add_nonnegative!(state::StateSOS, indices::AbstractVector{Int32}, values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    task = state.task.task
    Mosek.@MSK_appendvars(task, 1)
    Mosek.@MSK_putvarbound(task, state.num_vars, MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putacol(task, state.num_vars, length(indices), indices, values)
    state.num_vars += 1
    return
end

function Solver.add_quadratic!(state::StateSOS, index₊::Int32, value₊::Float64, rest_free::Tuple{Int32,Float64}...)
    task = state.task.task
    frdim = length(rest_free)
    Mosek.@MSK_appendvars(task, 1 + frdim)
    Mosek.@MSK_putvarboundslice(task, state.num_vars, state.num_vars + Int32(frdim),
        StackVec(MSK_BK_LO.value, ntuple(_ -> MSK_BK_FR.value, Val(frdim))...),
        StackVec(0., ntuple(_ -> -Inf, Val(frdim))...),
        StackVec(ntuple(_ -> Inf, Val(frdim +1))...))
    Mosek.@MSK_putaijlist(task, frdim +1, StackVec(index₊, ntuple(i -> rest_free[i][1], Val(frdim))...),
        StackVec(ntuple(let nv=state.num_vars -1
                            i -> Int32(nv + i)
                        end, Val(frdim +1))...),
        StackVec(value₊, ntuple(i -> rest_free[i][2], Val(frdim))...))
    state.num_vars += 1 + frdim
    return
end

function Solver.add_quadratic!(state::StateSOS, indices₊::AbstractVector{Int32}, values₊::AbstractVector{Float64},
    rest_free::Tuple{AbstractVector{Int32},AbstractVector{Float64}}...)
    task = state.task.task
    frdim = length(rest_free)
    Mosek.@MSK_appendvars(task, 1 + frdim)
    Mosek.@MSK_putvarboundslice(task, state.num_vars, state.num_vars + Int32(frdim),
        StackVec(MSK_BK_LO.value, ntuple(_ -> MSK_BK_FR.value, Val(frdim))...),
        StackVec(0., ntuple(_ -> -Inf, Val(frdim))...),
        StackVec(ntuple(_ -> Inf, Val(frdim +1))...))
    Mosek.@MSK_putacol(task, state.num_vars, length(indices₊), indices₊, values₊)
    state.num_vars += 1
    for (indicesᵣ, valuesᵣ) in rest_free
        Mosek.@MSK_putacol(task, state.num_vars, length(indicesᵣ), indicesᵣ, valuesᵣ)
        state.num_vars += 1
    end
    return
end

function Solver.add_quadratic!(state::StateSOS, index₁::Int32, value₁::Float64, index₂::Int32, value₂::Float64,
    rest::Tuple{Int32,Float64}...)
    task = state.task.task
    rhsdim = length(rest)
    conedim = 2 + rhsdim
    Mosek.@MSK_appendvars(task, conedim)
    Mosek.@MSK_putvarboundslice(task, state.num_vars, state.num_vars + Int32(conedim),
        StackVec(MSK_BK_LO.value, MSK_BK_LO.value, ntuple(_ -> MSK_BK_FR.value, Val(rhsdim))...),
        StackVec(0., 0., ntuple(_ -> -Inf, Val(rhsdim))...), StackVec(ntuple(_ -> Inf, Val(conedim))...))
    varidx = StackVec(ntuple(let nv=state.num_vars -1
                                i -> Int32(nv + i)
                             end, Val(conedim)))
    Mosek.@MSK_appendcone(task, MSK_CT_RQUAD.value, 0., conedim, varidx)
    Mosek.@MSK_putaijlist(task, conedim, StackVec(index₁, index₂, ntuple(i -> rest[i][1], Val(rhsdim))...), varidx,
        StackVec(2value₁, value₂, ntuple(i -> rest[i][2], Val(rhsdim))...))
    state.num_vars += conedim
    return
end

function Solver.add_quadratic!(state::StateSOS, indices₁::AbstractVector{Int32}, values₁::AbstractVector{Float64},
    indices₂::AbstractVector{Int32}, values₂::AbstractVector{Float64},
    rest::Tuple{AbstractVector{Int32},AbstractVector{Float64}}...)
    @assert(length(indices₁) == length(values₁) && length(indices₂) == length(values₂) &&
        all(x -> length(x[1]) == length(x[2]), rest))
    task = state.task.task
    rhsdim = length(rest)
    conedim = 2 + rhsdim
    Mosek.@MSK_appendvars(task, conedim)
    Mosek.@MSK_putvarboundslice(task, state.num_vars, state.num_vars + Int32(conedim),
        StackVec(MSK_BK_LO.value, MSK_BK_LO.value, ntuple(_ -> MSK_BK_FR.value, Val(rhsdim))...),
        StackVec(0., 0., ntuple(_ -> -Inf, Val(rhsdim))...), StackVec(ntuple(_ -> Inf, Val(conedim))...))
    varidx = StackVec(ntuple(let nv=state.num_vars -1
                                 i -> Int32(nv + i)
                             end, Val(conedim)))
    Mosek.@MSK_appendcone(task, MSK_CT_RQUAD.value, 0., conedim, varidx)
    rmul!(values₁, 2)
    Mosek.@MSK_putacol(task, state.num_vars, length(indices₁), indices₁, values₁)
    Mosek.@MSK_putacol(task, state.num_vars + Int32(1), length(indices₂), indices₂, values₂)
    state.num_vars += 2
    for (indicesᵣ, valuesᵣ) in rest
        Mosek.@MSK_putacol(task, state.num_vars, length(indicesᵣ), indicesᵣ, valuesᵣ)
        state.num_vars += 1
    end
    return
end

Solver.supports_quadratic(::StateSOS) = true

function Solver.add_psd!(state::StateSOS, dim::Int,
    data::Dict{FastKey{Int32},<:Tuple{AbstractVector{Int32},AbstractVector{Int32},AbstractVector{Float64}}})
    task = state.task.task
    Mosek.@MSK_appendbarvars(task, 1, Ref(Int32(dim)))
    total = 0
    for (rows, _, _) in values(data)
        len = length(rows)
        total += len
    end
    subi = FastVec{Int32}(buffer=total)
    subj = fill(state.num_bar_vars, total)
    subk = FastVec{Int32}(buffer=total)
    subl = FastVec{Int32}(buffer=total)
    valijkl = FastVec{Float64}(buffer=total)
    for (constr, (rows, cols, vals)) in data
        unsafe_append!(subi, Iterators.repeated(constr, length(rows)))
        unsafe_append!(subk, rows)
        unsafe_append!(subl, cols)
        unsafe_append!(valijkl, vals)
    end
    Mosek.@MSK_putbarablocktriplet(task, total, subi, subj, subk, subl, valijkl)

#=
    items = trisize(dim)
    sparse_extraction = get!(state.sparse_extraction, dim) do
        subi = Vector{Int32}(undef, dim * (dim +1) ÷ 2)
        subj = similar(subi)
        idx = 1
        for j in 0:dim-1
            for i in j:dim-1
                @inbounds subi[idx] = i
                @inbounds subj[idx] = j
                idx += 1
            end
        end
        ret = Vector{Int64}(undef, items)
        Mosek.@MSK_appendsparsesymmatlist(
            task,
            items,
            fill(Int32(dim), items),
            ones(Int64, items),
            subi,
            subj,
            ones(items),
            ret
        )
        return ret
    end
    alphas = Vector{Int64}(undef, length(data) +1)
    @inbounds alphas[1] = lastval = 0
    for (i, v) in zip(Iterators.countfrom(2), values(data))
        @inbounds alphas[i] = (lastval += length(v[1]))
    end
    @inbounds matidx = Vector{Int64}(undef, alphas[end])
    weights = similar(matidx, Float64)
    let idx=0
        for (i, v) in values(data)
            l = length(i)
            for (j, iⱼ) in enumerate(i)
                @inbounds matidx[idx+j] = sparse_extraction[iⱼ]
            end
            @inbounds copyto!(weights, idx +1, v, 1, l)
            idx += l
        end
    end
    Mosek.@MSK_putbaraijlist(
        task,
        length(data),
        collect(keys(data)),
        fill(state.num_bar_vars, length(data)),
        alphas,
        pointer(alphas, 2),
        matidx,
        weights
    )
=#
    state.num_bar_vars += 1
    return
end

Solver.psd_indextype(::StateSOS) = PSDIndextypeMatrixCartesian(Int32, :L, zero(Int32))

function Solver.add_free_prepare!(state::StateSOS, num::Int)
    task = state.task.task
    Mosek.@MSK_appendvars(task, num)
    Mosek.@MSK_putvarboundsliceconst(task, state.num_vars, state.num_vars + num, MSK_BK_FR.value, -Inf, Inf)
    prev = state.num_vars
    state.num_vars += num
    return prev
end

function Solver.add_free!(state::StateSOS, eqstate::Int32, indices::AbstractVector{Int32}, values::AbstractVector{Float64},
    obj::Bool)
    @assert(length(indices) == length(values))
    Mosek.@MSK_putacol(state.task.task, eqstate, length(indices), indices, values)
    obj && Mosek.@MSK_putcj(state.task.task, eqstate, 1.)
    return eqstate + one(Int32)
end

function Solver.add_free_finalize!(state::StateSOS, eqstate::Int32)
    rem = Int32(eqstate):state.num_vars-one(Int32)
    if !isempty(rem)
        Mosek.@MSK_removevars(state.task.task, length(rem), collect(rem))
        state.num_vars -= length(rem)
    end
    return
end

function Solver.fix_constraints!(state::StateSOS, indices::Vector{Int32}, values::Vector{Float64})
    len = length(indices)
    @assert(len == length(values))
    Mosek.@MSK_putconboundsliceconst(state.task.task, 0, length(state.constr_map), MSK_BK_FX, 0., 0.)
    Mosek.@MSK_putconboundlist(state.task.task, len, indices, fill(MSK_BK_FX.value, len), values, values)
    return
end

function Solver.poly_optimize(::Val{:MosekSOS}, relaxation::AbstractRelaxation,
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=(state) -> nothing, parameters=())
    task = Mosek.Task(msk_global_env::Env)
    try
        setup_time = @elapsed begin
            verbose && putstreamfunc(task, MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                if k isa AbstractString
                    putparam(task, string(k), string(v))
                elseif v isa Integer
                    putintparam(task, k, v)
                elseif v isa AbstractFloat
                    putdouparam(task, k, v)
                elseif v isa AbstractString
                    putstrparam(task, k, string(v))
                end
            end
            putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE)
            state = StateSOS(task, zero(Int32), zero(Int32), zero(Int32), Dict{FastKey{Int32},Int32}())
            sos_setup!(state, relaxation, groupings)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        customize(state)

        optimize(task) # TODO: maybe we want to delete the empty constraints before the optimization starts? Benchmark, as this
                       # would make solution extraction much harder.
        @verbose_info("Optimization complete, extracting solution")

        max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
        mon_pos = convert.(UInt, keys(state.constr_map))
        mon_val = resize!(gety(task, MSK_SOL_ITR), length(mon_pos))
        sort_along!(mon_pos, mon_val) # we also sort in the dense case; this improves cache times in the assignment
        if 3length(mon_pos) < max_mons
            solution = SparseVector(max_mons, mon_pos, mon_val)
        elseif length(mon_pos) == max_mons # dense case
            solution = mon_val
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), mon_val)
        end
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR), MomentVector(relaxation, solution)
    finally
        deletetask(task)
    end
end