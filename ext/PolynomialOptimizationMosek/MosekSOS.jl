mutable struct StateSOS
    task::Mosek.Task
    num_vars::Int32
    num_bar_vars::Int32
end

function PolynomialOptimization.sos_solver_add_scalar!(state::StateSOS, indices::AbstractVector{Int32},
    values::AbstractVector{Float64})
    @assert(length(indices) == length(values))
    task = state.task.task
    Mosek.@MSK_appendvars(task, 1)
    Mosek.@MSK_putvarbound(task, state.num_vars, MSK_BK_LO.value, 0., Inf)
    Mosek.@MSK_putacol(task, state.num_vars, length(indices), indices, values)
    state.num_vars += 1
    return
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, index₊::Int32, value₊::Float64,
    rest_free::Tuple{Int32,Float64}...)
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

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, indices₊::AbstractVector{Int32},
    values₊::AbstractVector{Float64}, rest_free::Tuple{AbstractVector{Int32},AbstractVector{Float64}}...)
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

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, index₁::Int32, value₁::Float64, index₂::Int32,
    value₂::Float64, rest::Tuple{Int32,Float64}...)
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

function PolynomialOptimization.sos_solver_add_quadratic!(state::StateSOS, indices₁::AbstractVector{Int32},
    values₁::AbstractVector{Float64}, indices₂::AbstractVector{Int32}, values₂::AbstractVector{Float64},
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

PolynomialOptimization.sos_solver_supports_quadratic(::StateSOS) = true

function PolynomialOptimization.sos_solver_add_psd!(state::StateSOS, dim::Int,
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

PolynomialOptimization.sos_solver_psd_indextype(::StateSOS) = Tuple{Int32,Int32}, :L, zero(Int32)

function PolynomialOptimization.sos_solver_add_free_prepare!(state::StateSOS, num::Int)
    task = state.task.task
    Mosek.@MSK_appendvars(task, num)
    Mosek.@MSK_putvarboundsliceconst(task, state.num_vars, state.num_vars + num, MSK_BK_FR.value, -Inf, Inf)
    prev = state.num_vars
    state.num_vars += num
    return prev
end

function PolynomialOptimization.sos_solver_add_free!(state::StateSOS, eqstate::Int32, indices::AbstractVector{Int32},
    values::AbstractVector{Float64}, obj::Bool)
    @assert(length(indices) == length(values))
    Mosek.@MSK_putacol(state.task.task, eqstate, length(indices), indices, values)
    obj && Mosek.@MSK_putcj(state.task.task, eqstate, 1.)
    return eqstate + one(Int32)
end

function PolynomialOptimization.sos_solver_fix_constraints!(state::StateSOS, indices::Vector{Int32}, values::Vector{Float64})
    len = length(indices)
    @assert(len == length(values))
    Mosek.@MSK_putconboundlist(state.task.task, len, indices, fill(MSK_BK_FX.value, len), values, values)
    return
end

function PolynomialOptimization.poly_optimize(::Val{:MosekSOS}, relaxation::AbstractPORelaxation{<:POProblem{P}},
    groupings::RelaxationGroupings; verbose::Bool=false, customize::Function=(state) -> nothing, parameters=()) where {P}
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
            # We don't know which monomials will be required in the end due to sparsity. We could keep accumulating all the
            # potential monomials, increasing the task as we go.
            # However, we know a clear upper bound on the number of monomials: It is given by the monomial space of twice the
            # degree. Every monomial corresponds to a constraint, and it is extremely cheap to allocate a huge number of
            # constraints in Mosek, so we just create it.
            concount = monomial_count(2degree(relaxation), nvariables(relaxation.objective))
            appendcons(task, concount)
            putconboundsliceconst(task, 1, concount +1, MSK_BK_FX, 0., 0.)

            state = StateSOS(task, zero(Int32), zero(Int32))

            PolynomialOptimization.sos_setup!(state, relaxation, groupings)
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        customize(state)

        optimize(task) # TODO: maybe we want to delete the empty constraints before the optimization starts? Benchmark, as this
                       # would make solution extraction much harder.
        @verbose_info("Optimization complete")
        return getsolsta(task, MSK_SOL_ITR), getprimalobj(task, MSK_SOL_ITR),
            sos_solution(monomial_type(P), gety(task, MSK_SOL_ITR))
    finally
        deletetask(task)
    end
end