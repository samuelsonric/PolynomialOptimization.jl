mutable struct StateMoment{K<:Integer,V<:Real} <: AbstractSolver{K,V}
    num_con::Int
    const psd_dim::FastVec{Int}
    const psds::FastVec{Tuple{FastVec{Int},FastVec{Int},FastVec{V}}}
    const nonnegs::Tuple{FastVec{Int},FastVec{Int},FastVec{V}}
    const b::Tuple{FastVec{Int},FastVec{V}}
    const mon_eq::Dict{FastKey{K},Tuple{Int,Int,Int}}
    c::Indvals{K,V}
    info::Vector{<:Vector{<:Tuple{Symbol,Any}}}

    StateMoment{K,V}() where {K<:Integer,V<:Real} = new{K,V}(
        0,
        FastVec{Int}(),
        FastVec{Tuple{FastVec{Int},FastVec{Int},FastVec{V}}}(),
        (FastVec{Int}(), FastVec{Int}(), FastVec{V}()),
        (FastVec{Int}(), FastVec{V}()),
        Dict{FastKey{K},Tuple{Int,Int,Int}}(),
    )
end

Solver.issuccess(::Val{:LoraineMoment}, status::Int) = status == 1

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L, true) # we actually need :F, but it is easier this way

Solver.prepend_fix(::StateMoment) = false

@counter_atomic(StateMoment, :psd)

Solver.add_var_slack!(state::StateMoment, num::Int) = error("Not implemented")

@inline function addconstr!(state::StateMoment{K,V}, indval::Indvals{K,V}, num_con::Int, ::Val{checkfree}=Val(false)) where {K,V,checkfree}
    @inbounds for (k, v) in indval
        if checkfree && !haskey(state.mon_eq, FastKey(k))
            # We did not see this monomial before in a PSD/nonnegative constraint, but now we need it in a fixed constraint.
            # This can happen due to basis reduction and still lead to a solvable problem if there are multiple well-balanced
            # fixed constraints. We would need to create free variables, which we don't have in Loraine; so create two
            # nonnegatives instead.
            refₘ = -length(state.nonnegs[1]) -1
            refᵢ = 1
            refⱼ = 0 # just to have it
            state.mon_eq[FastKey(k)] = (refₘ, 1, 0)
            push!(state.nonnegs[1], length(state.nonnegs[2]) +1, length(state.nonnegs[2]) +1)
        else
            refₘ, refᵢ, refⱼ = state.mon_eq[FastKey(k)]
        end
        if refₘ > 0 # likely
            dimₘ = state.psd_dim[refₘ]
            cooₘᵢ, cooₘⱼ, cooₘᵥ = state.psds[refₘ]
            if refᵢ == refⱼ
                push!(cooₘᵢ, num_con)
                push!(cooₘⱼ, (refⱼ -1) * dimₘ + refᵢ)
                push!(cooₘᵥ, v)
            else
                push!(cooₘᵢ, num_con, num_con)
                push!(cooₘⱼ, (refⱼ -1) * dimₘ + refᵢ, (refᵢ -1) * dimₘ + refⱼ)
                push!(cooₘᵥ, .5v, .5v)
            end
        else # quite unlikely (would require size-1 moment matrix or the case above), so that we do inefficient insertions
            insertpos = state.nonnegs[1][-refₘ+1]
            if checkfree && refᵢ == 1
                # put one at the end, the other at the beginning; doesn't matter
                state.nonnegs[1][-refₘ+1] += 1
                state.nonnegs[1][-refₘ+2:end] .+= 2
                Base._growat!(state.nonnegs[2], insertpos, 2)
                Base._growat!(state.nonnegs[3], insertpos, 2)
                state.nonnegs[2][insertpos+1] = state.nonnegs[2][insertpos] = num_con
                state.nonnegs[3][insertpos+1] = -(state.nonnegs[3][insertpos] = v)
            else
                state.nonnegs[1][-refₘ+1:end] .+= 1
                insert!(state.nonnegs[2], insertpos, num_con)
                insert!(state.nonnegs[3], insertpos, v)
            end
        end
    end
    return
end

function Solver.add_constr_nonnegative!(state::StateMoment{K,V}, indvals::IndvalsIterator{K,V}) where {K,V}
    # We directly construct the A_lin matrix as CSC data, as every entry here corresponds to a variable that is highly
    # unlikely to be used later again unless we have a size-1 moment (not localizing) matrix. In this very unexpected case, we
    # have to do inefficient insertions, but this will probably never happen, so we can save the time of converting COO to CSC.
    colptr, rowval, nzval = state.nonnegs
    mon_eq = state.mon_eq
    req_elems = length(rowval) + length(indvals)
    prepare_push!(colptr, length(indvals))
    prepare_push!(rowval, length(indvals))
    prepare_push!(nzval, length(indvals))
    num_con = state.num_con
    @inbounds for indval in indvals
        unsafe_push!(colptr, length(rowval) +1)
        if isone(length(indval))
            k, v = first(indval)
            if !haskey(mon_eq, FastKey(k)) # unlikely
                @assert(isone(v))
                state.mon_eq[FastKey(k)] = (-length(colptr), 0, 0)
                continue
            end
        end
        num_con += 1
        req_elems -= 1

        # first this element
        unsafe_push!(rowval, num_con)
        unsafe_push!(nzval, -1.)
        # then the others
        addconstr!(state, indval, num_con)
        # check if the unlikely case happened: then, the previous command will have led to insertions into rowval and nzval.
        # As a consequence, our buffer might not be large enough any more.
        δ = length(rowval) + req_elems
        prepare_push!(rowval, δ)
        prepare_push!(nzval, δ)
    end
    state.num_con = num_con
    return
end

function Solver.add_constr_psd!(state::StateMoment{K,V}, dim::Int, data::IndvalsIterator{K,V}) where {K,V}
    push!(state.psd_dim, dim)
    psd_index = Int(length(state.psd_dim))
    mon_eq = state.mon_eq
    dsq = dim^2
    cooᵢ = FastVec{Int}(buffer=dsq + 2length(rowvals(data))) # very conservative, as we requested :L but need :F
    cooⱼ = similar(cooᵢ)
    cooᵥ = similar(cooᵢ, V)
    push!(state.psds, (cooᵢ, cooⱼ, cooᵥ))
    col = 1
    row = 0
    num_con = state.num_con
    @inbounds for indval in data
        if (row += 1) > dim
            col += 1
            row = col
        end

        if isone(length(indval))
            k, v = first(indval)
            if !haskey(mon_eq, FastKey(k))
                @assert(isone(v))
                state.mon_eq[FastKey(k)] = (psd_index, row, col)
                continue
            end
        end
        num_con += 1

        # first this element
        unsafe_push!(cooᵢ, num_con)
        unsafe_push!(cooⱼ, (col -1) * dim + row)
        if row == col
            unsafe_push!(cooᵥ, -1.)
        else
            unsafe_push!(cooᵢ, num_con)
            unsafe_push!(cooⱼ, (row -1) * dim + col)
            unsafe_push!(cooᵥ, -.5, -.5)
        end
        # then the others
        addconstr!(state, indval, num_con)
    end
    state.num_con = num_con
    return
end

function Solver.add_constr_fix!(state::StateMoment{K,V}, ::Nothing, indvals::Indvals{K,V}, rhs::V) where {K,V}
    num_con = (state.num_con += 1)
    addconstr!(state, indvals, num_con, Val(true))
    @inbounds if !iszero(rhs)
        push!(state.b[1], state.num_con)
        push!(state.b[2], rhs)
    end
    return
end

function Solver.fix_objective!(state::StateMoment{K,V}, indvals::Indvals{K,V}) where {K,V}
    state.c = indvals
    return
end

function Solver.poly_optimize(::Val{:LoraineMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation, verbose::Bool=false, customize=(state) -> nothing, parameters...)
    setup_time = @elapsed @inbounds begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        V = real(coefficient_type(poly_problem(relaxation).objective))

        state = StateMoment{K,V}()
        state.info = moment_setup!(state, relaxation, groupings; representation)
        customize(state)

        # Now we must put create our objective data by converting the indices into tuples. This is not very efficient - but do
        # we want to allocate the data in COO form first, then convert? Hopefully the objective is not so dense.
        # Note that Loraine has abstract types in the vector specifications, which we need to match unfortunately.
        C = Vector{SparseMatrixCSC{V,Int}}(undef, length(state.psds))
        c_lin_data = (FastVec{Int}(), FastVec{V}())
        try
            for (k, v) in state.c
                refₘ, refᵢ, refⱼ = state.mon_eq[FastKey(k)]
                if refₘ > 0
                    if !isassigned(C, refₘ)
                        C[refₘ] = spzeros(V, Int, state.psd_dim[refₘ], state.psd_dim[refₘ])
                    end
                    spₘ = C[refₘ]
                    # we came from the lower triangle, so row ≥ col. Insert the smaller one first.
                    insert!(spₘ.rowval, spₘ.colptr[refⱼ+1], refᵢ)
                    if refᵢ == refⱼ
                        insert!(spₘ.nzval, spₘ.colptr[refⱼ+1], v)
                        spₘ.colptr[refⱼ+1:end] .+= 1
                    else
                        @assert(refᵢ > refⱼ)
                        insert!(spₘ.nzval, spₘ.colptr[refⱼ+1], v / 2)
                        spₘ.colptr[refⱼ+1:refᵢ] .+= 1
                        insert!(spₘ.rowval, spₘ.colptr[refᵢ+1] +1, refⱼ)
                        insert!(spₘ.nzval, spₘ.colptr[refᵢ+1] +1, v / 2)
                        spₘ.colptr[refᵢ+1:end] .+= 2
                    end
                else
                    push!(c_lin_data[1], -refₘ)
                    push!(c_lin_data[2], v)
                end
            end
        catch e
            if e isa KeyError
                # This can happen if the objective contains monomials that were not present before due to a smaller basis. In
                # principle, we should add a free variable for each of those monomials; however, these variables will never
                # occur anywhere else. Therefore, the problem is naturally unbounded (unless it is infeasible, which we simply
                # disregard here) and we don't need to solve it.
                @verbose_info("Detected unbounded objective during problem construction; skipping solver")
                return missing, Loraine.STATUS_INFEASIBLE_OR_UNBOUNDED, typemin(V)
            else
                rethrow(e)
            end
        end
        # It may very well happen that the objective is just from a single variable
        for i in eachindex(C)
            if !isassigned(C, i)
                C[i] = spzeros(V, Int, state.psd_dim[i], state.psd_dim[i])
            end
        end
        c_lin = SparseVector(length(state.nonnegs[1]), finish!.(c_lin_data)...)

        A = Vector{SparseMatrixCSC{V,Int}}(undef, length(state.psd_dim))
        for i in 1:length(state.psd_dim)
            A[i] = SparseArrays.sparse!(finish!.(state.psds[i])..., state.num_con, state.psd_dim[i]^2)
        end
        push!(state.nonnegs[1], length(state.nonnegs[2]) +1)
        A_lin = SparseMatrixCSC(state.num_con, length(state.nonnegs[1]) -1, finish!.(state.nonnegs)...)

        b = zeros(V, state.num_con)
        for (i, v) in zip(state.b...)
            b[i] = v
        end

        @verbose_info("Initializing solver")
        solver = Loraine.Solver(Loraine.Model(; A, C, b, A_lin, c_lin, coneDims=finish!(state.psd_dim), check=false);
            verb=verbose ? Loraine.VERBOSITY_FULL : Loraine.VERBOSITY_NONE, parameters...)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    solver_time = @elapsed Loraine.solve!(solver)

    @verbose_info("Optimization complete in ", solver_time, " seconds")

    return (state, solver), solver.status, sum(v * solver.y[i] for (i, v) in zip(state.b...), init=0.)
end

function Solver.extract_moments(relaxation::AbstractRelaxation{<:Problem{<:SimplePolynomial{<:Any,Nr,Nc}}},
    (state, solver)::Tuple{StateMoment{K},Loraine.Solver}) where {Nr,Nc,K<:Integer}
    # Dilemma: Do we want to calculate the full matrix product U Uᵀ (or one triangle) and extract the relevant entries, which
    # would be favorable if we have a large moment matrix that contains everything? Or do we want to calculate the entries
    # on-the-fly doing vector products, which in total is not cache-optimal, but would be better if we only need few entries
    # from the matrix, perhaps because it represents a moment sub-matrix where a lot of entries have already been obtained by
    # other matrices? Let's do the first: We assume that this solver is used for large moment matrices, and we assume that
    # there won't be too much overlap between sparse groupings.
    y = Vector{V}(undef, length(state.mon_eq))
    mon_pos = convert.(K, keys(state.mon_eq))
    mm = solver.X
    mm_lin = solver.X_lin
    @inbounds for (j, (psdᵢ, row, col)) in enumerate(values(state.mon_eq))
        y[j] = psdᵢ > 0 ? mm[psdᵢ][row, col] : mm_lin[-psdᵢ]
    end
    sort_along!(mon_pos, y)
    max_mons = mon_pos[end]
    if length(y) == max_mons
        solution = y
    elseif 3length(y) < max_mons
        solution = SparseVector(max_mons, mon_pos, y)
    else
        solution = fill(NaN, max_mons)
        copyto!(@view(solution[mon_pos]), y)
    end
    return MomentVector(relaxation, ExponentsAll{Nr+2Nc,K}(), solution)
end

Solver.extract_sos(::AbstractRelaxation, (_, solver)::Tuple{StateMoment,Loraine.Solver}, ::Val{:nonnegative},
    index::AbstractUnitRange, ::Nothing) = @view(solver.S_lin[index])

Solver.extract_sos(::AbstractRelaxation, (_, solver)::Tuple{StateMoment,Loraine.Solver}, ::Val{:psd},
    index::Integer, ::Nothing) = solver.S[index]