mutable struct StateMoment{I<:Integer,K<:Integer,Offset} <: AbstractSparseMatrixSolver{I,K,Float64}
    const minusAcoo_zeropsd::SparseMatrixCOO{I,K,Float64,Offset}
    const minusAcoo_nonneg::SparseMatrixCOO{I,K,Float64,Offset}
    const minusAcoo_soc::SparseMatrixCOO{I,K,Float64,Offset}
    const b_zero::Tuple{FastVec{Int},FastVec{Float64}} # this is one-indexed
    const c::Ref{Tuple{Vector{K},Vector{Float64}}}
    numzero::I
    lenzero::I
    const socsizes::FastVec{I}
    const psdsizes::FastVec{I}

    StateMoment{I,K}() where {I<:Integer,K<:Integer} = new{I,K,zero(I)}(
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        (FastVec{Int}(), FastVec{Float64}()),
        Ref{Tuple{Vector{K},Vector{Float64}}}(), 0, 0,
        FastVec{I}(), FastVec{I}()
    )
end

Solver.supports_quadratic(::StateMoment) = SOLVER_QUADRATIC_SOC

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L)

function Solver.add_constr_nonnegative!(state::StateMoment{<:Integer,K}, indvals::AbstractIndvals{K,Float64}) where {K}
    append!(state.minusAcoo_nonneg, indvals)
    return
end

function Solver.add_constr_quadratic!(state::StateMoment{<:Integer,K}, indvals::AbstractIndvals{K,Float64}...) where {K}
    append!(state.minusAcoo_soc, indvals...)
    push!(state.socsizes, length(indvals))
    return
end

function Solver.add_constr_psd!(state::StateMoment{<:Integer,K}, dim::Int, data::PSDVector{K,Float64}) where {K}
    append!(state.minusAcoo_zeropsd, data)
    push!(state.psdsizes, dim)
    return
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # Those is just a lower bound, as the number of constraints is multiplied by the individual index count. But better than
    # nothing.
    prepare_push!(state.minusAcoo_zeropsd, num)
    return
end

function Solver.add_constr_fix!(state::StateMoment{<:Integer,K}, ::Nothing, indvals::AbstractIndvals{K,Float64}, rhs::Float64) where {K}
    v = append!(state.minusAcoo_zeropsd, indvals)
    if !iszero(rhs)
        push!(state.b_zero[1], v +1)
        push!(state.b_zero[2], rhs)
    end
    state.numzero += one(I)
    state.lenzero += length(indvals)
    return
end

function Solver.fix_objective!(state::StateMoment{<:Any,K}, indvals::AbstractIndvals{K,Float64}) where {K}
    state.c[] = (indvals.indices, indvals.values)
    return
end

function Solver.poly_optimize(::Val{:SCSMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    verbose::Bool=false, customize::Base.Callable=_ -> nothing, linear_solver::Type{<:LinearSolver}=SCS.DirectSolver,
    parameters...)
    setup_time = @elapsed begin
        I = scsint_t(linear_solver)
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        state = StateMoment{I,K}()

        moment_setup!(state, relaxation, groupings)
        customize(state)

        # We now must merge the noneg and the soc constraints into the zeropsd COO at the appropriate position
        Acoo = state.minusAcoo_zeropsd
        nzero = state.numzero
        lenzero = state.lenzero
        nnonneg = size(state.minusAcoo_nonneg, 1)
        lennonneg = length(state.minusAcoo_nonneg)
        nsoc = size(state.minusAcoo_soc, 1)
        lensoc = length(state.minusAcoo_soc)
        npsd = size(Acoo, 1) - nzero
        lenpsd = length(Acoo) - lenzero
        @inbounds if !iszero(lennonneg + lensoc)
            # calculate the shifts
            # TODO: this shifts all the PSD data to the end, then inserts the remaining ones in between. As we sort the data
            # afterwards anyway, we wouldn't really have to do the shift. Check if it is faster to rely on sorting.
            # merge all the row data (note that we would then have to adjust the detection of the number of rows in b).
            resize!(Acoo.rowinds, length(Acoo) + lennonneg + lensoc)
            @views Acoo.rowinds[end:-1:lenzero+lennonneg+lensoc+1] .= Acoo.rowinds[lenzero+lenpsd:-1:lenzero+1] .+ (nnonneg + nsoc)
            if iszero(nzero)
                copyto!(Acoo.rowinds, 1, state.minusAcoo_nonneg.rowinds, 1, lennonneg)
            else
                @views Acoo.rowinds[lenzero+1:lenzero+lennonneg] .= state.minusAcoo_nonneg.rowinds .+ nzero
            end
            if iszero(nzero + nnonneg)
                copyto!(Acoo.rowinds, 1, state.minusAcoo_soc.rowinds, 1, lensoc)
            else
                @views Acoo.rowinds[lenzero+lennonneg+1:lenzero+lennonneg+lensoc] .= state.minusAcoo_soc.rowinds .+ (nzero + nnonneg)
            end
            # merge the remaining data
            resize!(Acoo.moninds, length(Acoo.moninds) + lennonneg + lensoc)
            copyto!(Acoo.moninds, lenzero + lennonneg + lensoc +1, Acoo.moninds, lenzero +1, lenpsd)
            copyto!(Acoo.moninds, lenzero +1, state.minusAcoo_nonneg.moninds, 1, lennonneg)
            copyto!(Acoo.moninds, lenzero + lennonneg +1, state.minusAcoo_soc.moninds, 1, lensoc)

            resize!(Acoo.nzvals, length(Acoo.nzvals) + lennonneg + lensoc)
            copyto!(Acoo.nzvals, lenzero + lennonneg + lensoc +1, Acoo.nzvals, lenzero +1, lenpsd)
            copyto!(Acoo.nzvals, lenzero +1, state.minusAcoo_nonneg.nzvals, 1, lennonneg)
            copyto!(Acoo.nzvals, lenzero + lennonneg +1, state.minusAcoo_soc.nzvals, 1, lensoc)
        end
        rmul!(@view(Acoo.nzvals[lenzero+1:end]), -1.0) # this must happen before we sort the columns

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Ccoo = state.c[]
        m = size(Acoo, 1)
        b = zeros(Float64, m)
        copy!(@view(b[state.b_zero[1]]), state.b_zero[2])

        moncount, (Acolptr, Arowind, Anzval), c = coo_to_csc!(Acoo, Ccoo)

        _socsizes = Base.@_gc_preserve_begin state.socsizes
        _psdsizes = Base.@_gc_preserve_begin state.psdsizes
        _Acolptr = Base.@_gc_preserve_begin Acolptr
        _Arowval = Base.@_gc_preserve_begin Arowind
        _Anzval = Base.@_gc_preserve_begin Anzval
        scs_cone = ScsCone{I}(
            nzero, nnonneg, C_NULL, C_NULL, zero(I), pointer(state.socsizes), length(state.socsizes),
            pointer(state.psdsizes), length(state.psdsizes), zero(I), zero(I), C_NULL, zero(I)
        )
        scs_A = ScsMatrix{I}(pointer(Anzval), pointer(Arowind), pointer(Acolptr), m, moncount)
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
    Base.@_gc_preserve_end _Acolptr
    Base.@_gc_preserve_end _Arowval
    Base.@_gc_preserve_end _Anzval
    Base.@_gc_preserve_end _scs_A
    Base.@_gc_preserve_end _b
    Base.@_gc_preserve_end _c
    Base.@_gc_preserve_end _y
    Base.@_gc_preserve_end _s

    return status, scs_info.pobj, MomentVector(relaxation, x, Acoo)
end