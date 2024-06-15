mutable struct StateMoment{I<:Integer,K<:Integer}
    const minusAcoo_zeropsd::Tuple{FastVec{I},FastVec{K},FastVec{Float64}} # all zero-indexed
    const minusAcoo_nonneg::Tuple{FastVec{I},FastVec{K},FastVec{Float64}}
    const minusAcoo_soc::Tuple{FastVec{I},FastVec{K},FastVec{Float64}}
    const b_zero::Tuple{FastVec{Int},FastVec{Float64}} # except for this one
    const c::Ref{Tuple{Vector{K},Vector{Float64}}}
    numzero::I
    lenzero::I
    const socsizes::FastVec{I}
    const psdsizes::FastVec{I}
end

Solver.mindex(::StateMoment{<:Any,K}, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {K,Nr,Nc} =
    (monomial_index(monomials...) - one(K))::K

Solver.supports_quadratic(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L)

function Solver.add_constr_nonnegative!(state::StateMoment{I,K}, indices::AbstractVector{K}, values::AbstractVector{Float64}) where {I,K}
    @assert(length(indices) == length(values))
    prepare_push!(state.minusAcoo_nonneg[1], length(indices))
    v = isempty(state.minusAcoo_nonneg[1]) ? zero(I) : last(state.minusAcoo_nonneg[1]) + one(I)
    for _ in 1:length(indices)
        unsafe_push!(state.minusAcoo_nonneg[1], v)
    end
    append!(state.minusAcoo_nonneg[2], indices)
    append!(state.minusAcoo_nonneg[3], values)
end

function Solver.add_constr_quadratic!(state::StateMoment{I,K},
    (indices₁, values₁)::Tuple{AbstractVector{K},AbstractVector{Float64}},
    (indices₂, values₂)::Tuple{AbstractVector{K},AbstractVector{Float64}},
    rest::Tuple{AbstractVector{K},AbstractVector{Float64}}...) where {I,K}
    @assert(length(indices₁) == length(values₁) && length(indices₂) == length(values₂) &&
        all(t -> length(t[1]) == length(t[2]), rest))
    # SCS doesn't support the rotated quadratic cone, so we'll have to do it ourselves. The trafo is
    # y = [1/sqrt(2) 1/sqrt(2) 1 ... 1; 1/sqrt(2) -1/sqrt(2) 1 ... 1; 0 0 id] x
    len = 2length(indices₁) + 2length(indices₂) + sum(∘(length, first), rest, init=0)
    prepare_push!(state.minusAcoo_soc[1], len)
    prepare_push!(state.minusAcoo_soc[2], len)
    prepare_push!(state.minusAcoo_soc[3], len)
    isqrt2 = inv(sqrt(2.0))
    rmul!(values₁, isqrt2)
    rmul!(values₂, isqrt2)
    row = isempty(state.minusAcoo_soc[1]) ? zero(I) : last(state.minusAcoo_soc[1]) + one(I)
    lastend = length(state.minusAcoo_soc[2])
    for (i, v) in zip(indices₁, values₁)
        unsafe_push!(state.minusAcoo_soc[1], row)
        unsafe_push!(state.minusAcoo_soc[2], i)
        unsafe_push!(state.minusAcoo_soc[3], v)
    end
    # our indices are guaranteed to be unique - but not when we join multiple together
    @inbounds searchregion = @view(state.minusAcoo_soc[2][lastend+1:end])
    for (i, v) in zip(indices₂, values₂)
        idx = findfirst(isequal(i), searchregion)
        if isnothing(idx)
            unsafe_push!(state.minusAcoo_soc[1], row)
            unsafe_push!(state.minusAcoo_soc[2], i)
            unsafe_push!(state.minusAcoo_soc[3], v)
        else
            @inbounds state.minusAcoo_soc[3][idx+lastend] += v
        end
    end
    row += one(I)
    lastend = length(state.minusAcoo_soc[2])
    for (i, v) in zip(indices₁, values₁)
        unsafe_push!(state.minusAcoo_soc[1], row)
        unsafe_push!(state.minusAcoo_soc[2], i)
        unsafe_push!(state.minusAcoo_soc[3], v)
    end
    @inbounds searchregion = @view(state.minusAcoo_soc[2][lastend+1:end])
    for (i, v) in zip(indices₂, values₂)
        idx = findfirst(isequal(i), searchregion)
        if isnothing(idx)
            unsafe_push!(state.minusAcoo_soc[1], row)
            unsafe_push!(state.minusAcoo_soc[2], i)
            unsafe_push!(state.minusAcoo_soc[3], -v)
        else
            @inbounds state.minusAcoo_soc[3][idx+lastend] -= v
        end
    end
    for (r, _) in rest
        row += one(I)
        for _ in 1:length(r)
            unsafe_push!(state.minusAcoo_soc[1], row)
        end
    end
    append!(state.minusAcoo_soc[2], first.(rest)...)
    append!(state.minusAcoo_soc[3], last.(rest)...)
    push!(state.socsizes, length(rest) +2)
end

function Solver.add_constr_psd!(state::StateMoment{I,K}, dim::Int, data::PSDVector{K,Float64}) where {I,K}
    prepare_push!(state.minusAcoo_zeropsd[1], length(rowvals(data)))
    v = isempty(state.minusAcoo_zeropsd[1]) ? zero(I) : last(state.minusAcoo_zeropsd[1])
    for l in Base.index_lengths(data)
        v += one(I)
        for _ in 1:l
            unsafe_push!(state.minusAcoo_zeropsd[1], v)
        end
    end
    append!(state.minusAcoo_zeropsd[2], rowvals(data))
    append!(state.minusAcoo_zeropsd[3], nonzeros(data))
    push!(state.psdsizes, dim)
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # Those are all just lower bounds, as the number of constraints is multiplied by the individual index count. But better
    # than nothing.
    prepare_push!(state.minusAcoo_zeropsd[1], num)
    prepare_push!(state.minusAcoo_zeropsd[2], num)
    prepare_push!(state.minusAcoo_zeropsd[3], num)
    return
end

function Solver.add_constr_fix!(state::StateMoment{I,K}, ::Nothing, indices::AbstractVector{K},
    values::AbstractVector{Float64}, rhs::Float64) where {I,K}
    v = isempty(state.minusAcoo_zeropsd[1]) ? zero(I) : last(state.minusAcoo_zeropsd[1]) + one(I)
    prepare_push!(state.minusAcoo_zeropsd[1], length(indices))
    for _ in 1:length(indices)
        unsafe_push!(state.minusAcoo_zeropsd[1], v)
    end
    append!(state.minusAcoo_zeropsd[2], indices)
    append!(state.minusAcoo_zeropsd[3], values)
    if !iszero(rhs)
        push!(state.b_zero[1], v +1)
        push!(state.b_zero[2], rhs)
    end
    state.numzero += one(I)
    state.lenzero += length(indices)
    return
end

Solver.fix_objective!(state::StateMoment{<:Any,K}, indices::AbstractVector{K}, values::AbstractVector{Float64}) where {K} =
    state.c[] = (indices, values)

function consolidate_vec(vec::AbstractVector, callback)
    # vec is a sorted vector with duplicates. Call the callback with every last unique index in coo.
    i = 1
    remaining = length(vec)
    index = 1
    @inbounds while !iszero(remaining)
        cur = vec[i]
        # skip over duplicates
        while remaining > 1 && vec[i+1] == cur
            i += 1; remaining -= 1
        end
        @inline callback(index, i)
        i += 1; remaining -= 1
        index += 1
    end
    return index -1 # return the count
end

function Solver.poly_optimize(::Val{:SCSMoment}, relaxation::AbstractRelaxation,
    groupings::RelaxationGroupings; verbose::Bool=false, dense::Bool=!isone(poly_problem(relaxation).prefactor),
    customize::Function=_ -> nothing, linear_solver::Type{<:LinearSolver}=SCS.DirectSolver, parameters...)
    setup_time = @elapsed begin
        I = scsint_t(linear_solver)
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        state = StateMoment(
            (FastVec{I}(), FastVec{K}(), FastVec{Float64}()),
            (FastVec{I}(), FastVec{K}(), FastVec{Float64}()),
            (FastVec{I}(), FastVec{K}(), FastVec{Float64}()),
            (FastVec{Int}(), FastVec{Float64}()),
            Ref{Tuple{Vector{K},Vector{Float64}}}(), 0, 0,
            FastVec{I}(), FastVec{I}()
        )

        moment_setup!(state, relaxation, groupings)
        customize(state)

        # We now must merge the noneg and the soc constraints into the zeropsd COO at the appropriate position
        Acoo = finish!.(state.minusAcoo_zeropsd)
        nzero = state.numzero
        npsd = isempty(Acoo[1]) ? 0 : last(Acoo[1]) - nzero +1
        lenzero = state.lenzero
        lenpsd = length(Acoo[1]) - lenzero
        lennonneg = length(state.minusAcoo_nonneg[1])
        nnonneg = iszero(lennonneg) ? 0 : last(state.minusAcoo_nonneg[1]) +1
        lensoc = length(state.minusAcoo_soc[1])
        nsoc = iszero(lensoc) ? 0 : last(state.minusAcoo_soc[1]) +1
        @inbounds if !iszero(lennonneg + lensoc)
            # calculate the shifts
            # TODO: this shifts all the PSD data to the end, then inserts the remaining ones in between. As we sort the data
            # afterwards anyway, we wouldn't really have to do the shift. Check if it is faster to rely on sorting.
            # merge all the row data (note that we would then have to adjust the detection of the number of rows in b).
            resize!(Acoo[1], length(Acoo[1]) + lennonneg + lensoc)
            @views Acoo[1][end:-1:lenzero+lennonneg+lensoc+1] .= Acoo[1][lenzero+lenpsd:-1:lenzero+1] .+ (nnonneg + nsoc)
            if iszero(nzero)
                copyto!(Acoo[1], 1, state.minusAcoo_nonneg[1], 1, lennonneg)
            else
                @views Acoo[1][lenzero+1:lenzero+lennonneg] .= state.minusAcoo_nonneg[1] .+ nzero
            end
            if iszero(nzero + nnonneg)
                copyto!(Acoo[1], 1, state.minusAcoo_soc[1], 1, lensoc)
            else
                @views Acoo[1][lenzero+lennonneg+1:lenzero+lennonneg+lensoc] .= state.minusAcoo_soc[1] .+ (nzero + nnonneg)
            end
            # merge the remaining data
            resize!(Acoo[2], length(Acoo[2]) + lennonneg + lensoc)
            copyto!(Acoo[2], lenzero + lennonneg + lensoc +1, Acoo[2], lenzero +1, lenpsd)
            copyto!(Acoo[2], lenzero +1, state.minusAcoo_nonneg[2], 1, lennonneg)
            copyto!(Acoo[2], lenzero + lennonneg +1, state.minusAcoo_soc[2], 1, lensoc)

            resize!(Acoo[3], length(Acoo[3]) + lennonneg + lensoc)
            copyto!(Acoo[3], lenzero + lennonneg + lensoc +1, Acoo[3], lenzero +1, lenpsd)
            copyto!(Acoo[3], lenzero +1, state.minusAcoo_nonneg[3], 1, lennonneg)
            copyto!(Acoo[3], lenzero + lennonneg +1, state.minusAcoo_soc[3], 1, lensoc)
        end
        rmul!(@view(Acoo[3][lenzero+1:end]), -1.0) # this must happen before we sort the columns

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Ccoo = state.c[]
        m = last(Acoo[1]) +1
        b = zeros(Float64, m)
        copy!(@view(b[state.b_zero[1]]), state.b_zero[2])

        sort_along!(Acoo[2], Acoo[1], Acoo[3], relevant=2) # sort according to col indices, but keep row indices in order
                                                           # (sort_along is in general not order-preserving, so use row indices
                                                           # as secondary option)
        sort_along!(Ccoo[1], Ccoo[2])
        # how many distinct monomials do we have?
        moncount = consolidate_vec(Acoo[2], (_, _) -> nothing)
        # We only need to construct the colptr - Acoo[1] is already the correct rowval and Acoo[3] the correct nonzeros.
        Acolptr = Vector{I}(undef, moncount +1)
        @inbounds Acolptr[1] = 0
        c = zeros(Float64, moncount)
        ic = Ref(1)
        consolidate_vec(Acoo[2], @capture((index, ia) -> @inbounds begin
            colidx = $Acoo[2][ia]
            if $ic[] ≤ length($Ccoo[1]) && Ccoo[1][ic[]] == colidx
                $c[index] = Ccoo[2][ic[]]
                ic[] += 1
            end
            $Acolptr[index+1] = ia
        end))

        _socsizes = Base.@_gc_preserve_begin state.socsizes
        _psdsizes = Base.@_gc_preserve_begin state.psdsizes
        _Acoo = Base.@_gc_preserve_begin Acoo
        _Acolptr = Base.@_gc_preserve_begin Acolptr
        scs_cone = ScsCone{I}(
            nzero, nnonneg, C_NULL, C_NULL, zero(I), pointer(state.socsizes), length(state.socsizes),
            pointer(state.psdsizes), length(state.psdsizes), zero(I), zero(I), C_NULL, zero(I)
        )
        scs_A = ScsMatrix{I}(pointer(Acoo[3]), pointer(Acoo[1]), pointer(Acolptr), m, moncount)
        _scs_A = Base.@_gc_preserve_begin scs_A
        _b = Base.@_gc_preserve_begin b
        _c = Base.@_gc_preserve_begin c
        scs_data = ScsData{I}(
            m, moncount, pointer_from_objref(scs_A), C_NULL, pointer(b), pointer(c)
        )
        scs_settings = ScsSettings(linear_solver)
        for (k, v) in parameters
            setproperty!(scs_settings, k, v)
        end
        scs_settings.verbose = verbose
        scs_work = scs_init(linear_solver, scs_data, scs_cone, scs_settings)
        x = Vector{Float64}(undef, moncount)
        y = Vector{Float64}(undef, m)
        s = Vector{Float64}(undef, m)
        _y = Base.@_gc_preserve_begin y
        _s = Base.@_gc_preserve_begin s
        scs_solution = ScsSolution(pointer(x), pointer(y), pointer(s))
        scs_info = ScsInfo{I}()
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    status = scs_solve(linear_solver, scs_work, scs_solution, scs_info, false)
    scs_finish(linear_solver, scs_work)
    Base.Libc.flush_cstdio()
    @verbose_info("Optimization complete, retrieving moments")

    Base.@_gc_preserve_end _socsizes
    Base.@_gc_preserve_end _psdsizes
    Base.@_gc_preserve_end _Acoo
    Base.@_gc_preserve_end _Acolptr
    Base.@_gc_preserve_end _scs_A
    Base.@_gc_preserve_end _b
    Base.@_gc_preserve_end _c
    Base.@_gc_preserve_end _y
    Base.@_gc_preserve_end _s

    max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
    if length(x) == max_mons # dense case
        solution = x
    else
        # We need to build the vector of monomial indices.
        mon_pos = Vector{Int}(undef, moncount)
        consolidate_vec(Acoo[2], @capture((index, ia) -> @inbounds begin
            $mon_pos[index] = $Acoo[2][ia] +1
        end))
        if 3length(x) < max_mons
            solution = SparseVector(max_mons, mon_pos, x)
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), x)
        end
    end
    return status, scs_info.pobj, MomentVector(relaxation, solution)
end