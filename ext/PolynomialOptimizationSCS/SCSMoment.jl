mutable struct StateMoment{I<:Integer,K<:Integer,Offset} <: AbstractSparseMatrixSolver{I,K,Float64}
    const minusAcoo_zero::SparseMatrixCOO{I,K,Float64,Offset}
    const minusAcoo_nonneg::SparseMatrixCOO{I,K,Float64,Offset}
    const minusAcoo_soc::SparseMatrixCOO{I,K,Float64,Offset}
    const minusAcoo_psd::SparseMatrixCOO{I,K,Float64,Offset}
    const b_zero::Tuple{FastVec{Int},FastVec{Float64}} # this is one-indexed
    const socsizes::FastVec{I}
    const psdsizes::FastVec{I}
    slack::K
    c::Tuple{Vector{K},Vector{Float64}}

    StateMoment{I,K}() where {I<:Integer,K<:Integer} = new{I,K,zero(I)}(
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        SparseMatrixCOO{I,K,Float64,zero(I)}(),
        (FastVec{Int}(), FastVec{Float64}()),
        FastVec{I}(), FastVec{I}(), K <: Signed ? -one(K) : typemax(K)
    )
end

Solver.supports_quadratic(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:L)

function Solver.add_constr_nonnegative!(state::StateMoment{<:Integer,K}, indvals::IndvalsIterator{K,Float64}) where {K}
    return append!(state.minusAcoo_nonneg, indvals)
end

function Solver.add_constr_quadratic!(state::StateMoment{<:Integer,K}, indvals::IndvalsIterator{K,Float64}) where {K}
    push!(state.socsizes, length(indvals))
    return append!(state.minusAcoo_soc, indvals)
end

function Solver.add_constr_psd!(state::StateMoment{<:Integer,K}, dim::Int, data::IndvalsIterator{K,Float64}) where {K}
    push!(state.psdsizes, dim)
    return append!(state.minusAcoo_psd, data)
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # Those is just a lower bound, as the number of constraints is multiplied by the individual index count. But better than
    # nothing.
    prepare_push!(state.minusAcoo_zero, num)
    return
end

function Solver.add_constr_fix!(state::StateMoment{<:Integer,K}, ::Nothing, indvals::Indvals{K,Float64}, rhs::Float64) where {K}
    v = append!(state.minusAcoo_zero, indvals)
    if !iszero(rhs)
        push!(state.b_zero[1], v +1)
        push!(state.b_zero[2], rhs)
    end
    return
end

function Solver.fix_objective!(state::StateMoment{<:Any,K}, indvals::Indvals{K,Float64}) where {K}
    state.c = (indvals.indices, indvals.values)
    return
end

function Solver.poly_optimize(::Val{:SCSMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation, verbose::Bool=false, customize::Base.Callable=_ -> nothing,
    linear_solver::Type{<:LinearSolver}=SCS.DirectSolver, parameters...)
    setup_time = @elapsed begin
        I = scsint_t(linear_solver)
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        state = StateMoment{I,K}()

        moment_setup!(state, relaxation, groupings; representation)
        customize(state)

        # We now must merge the all constraints together in an appropriate order
        Acoo = state.minusAcoo_zero
        nzero = size(Acoo, 1)
        lenzero = length(Acoo)
        nnonneg = size(state.minusAcoo_nonneg, 1)
        lennonneg = length(state.minusAcoo_nonneg)
        nsoc = size(state.minusAcoo_soc, 1)
        lensoc = length(state.minusAcoo_soc)
        npsd = size(state.minusAcoo_psd, 1)
        lenpsd = length(state.minusAcoo_psd)
        if iszero(nzero) && iszero(nnonneg) && iszero(nsoc)
            Acoo = state.minusAcoo_psd
        else
            prepare_push!(Acoo, lennonneg + lensoc + lenpsd)
            δ = nzero
            for (newA, newδ) in ((state.minusAcoo_nonneg, nnonneg),
                                 (state.minusAcoo_soc, nsoc),
                                 (state.minusAcoo_psd, npsd))
                @simd for rowind in newA.rowinds
                    unsafe_push!(Acoo.rowinds, rowind + δ)
                end
                append!(Acoo.moninds, newA.moninds)
                @simd for nzval in newA.nzvals
                    unsafe_push!(Acoo.nzvals, -nzval)
                end
                δ += newδ
            end
        end

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Ccoo = state.c
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

    return (Val(:SCSMoment), x, state.slack, Acoo), status, scs_info.pobj
end

Solver.extract_moments(relaxation::AbstractRelaxation, (_, x, slack, Acoo)::Tuple{Val{:SCSMoment},Vararg}) =
    MomentVector(relaxation, x, slack, Acoo)