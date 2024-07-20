struct StateMoment{K<:Integer,V<:Real} <: AbstractSparseMatrixSolver{Int,K,V}
    Acoo::SparseMatrixCOO{Int,K,V,1}
    b::Tuple{FastVec{Int},FastVec{V}}
    q::Ref{Tuple{Vector{K},Vector{V}}}
    cones::FastVec{Clarabel.SupportedCone}
    slack::K

    StateMoment{K,V}() where {K<:Integer,V<:Real} = new{K,V}(
        SparseMatrixCOO{Int,K,V,1}(),
        (FastVec{Int}(), FastVec{V}()),
        Ref{Tuple{Vector{K},Vector{V}}}(),
        FastVec{Clarabel.SupportedCone}(),
        K <: Signed ? -one(K) : typemax(K)
    )
end

Solver.supports_quadratic(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:U)

function Solver.add_constr_nonnegative!(state::StateMoment{K,V}, indvals::IndvalsIterator{K,V}) where {K,V}
    append!(state.Acoo, indvals)
    push!(state.cones, Clarabel.NonnegativeConeT(1))
    return
end

function Solver.add_constr_quadratic!(state::StateMoment{K,V}, indvals::IndvalsIterator{K,V}) where {K,V}
    append!(state.Acoo, indvals)
    push!(state.cones, Clarabel.SecondOrderConeT(length(indvals)))
    return
end

function Solver.add_constr_psd!(state::StateMoment{K,V}, dim::Int, data::IndvalsIterator{K,V}) where {K,V}
    append!(state.Acoo, data)
    push!(state.cones, Clarabel.PSDTriangleConeT(dim))
    return
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # This is just a lower bound, as the number of constraints is multiplied by the individual index count. But better than
    # nothing.
    prepare_push!(state.Acoo, num)
    push!(state.cones, Clarabel.ZeroConeT(num))
    return
end

function Solver.add_constr_fix!(state::StateMoment{K,V}, ::Nothing, indvals::Indvals{K,V}, rhs::V) where {K,V}
    v = append!(state.Acoo, indvals)
    if !iszero(rhs)
        push!(state.b[1], v)
        push!(state.b[2], -rhs)
    end
    return
end

function Solver.fix_objective!(state::StateMoment{K,V}, indvals::Indvals{K,V}) where {K,V}
    state.q[] = (indvals.indices, indvals.values)
    return
end

function Solver.poly_optimize(::Val{:ClarabelMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation, verbose::Bool=false, customize::Base.Callable=_ -> nothing, parameters...)
    setup_time = @elapsed begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        V = real(coefficient_type(poly_problem(relaxation).objective))
        state = StateMoment{K,V}()

        moment_setup!(state, relaxation, groupings; representation)
        customize(state)

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Qcoo = state.q[]
        b = zeros(V, size(state.Acoo, 1))
        copy!(@view(b[state.b[1]]), state.b[2])

        moncount, (Acolptr, Arowind, Anzval), q = coo_to_csc!(state.Acoo, Qcoo)
        solver = Clarabel.Solver()
        Clarabel.setup!(solver,
            spzeros(V, moncount, moncount), # P
            q,
            SparseMatrixCSC{V,Int}(length(b), moncount, Acolptr, Arowind, rmul!(Anzval, -one(V))), # A
            b, # b
            finish!(state.cones), # cones
            Clarabel.Settings(; verbose, parameters...)
        )
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    solution = Clarabel.solve!(solver)
    status = solution.status
    value = solution.obj_val
    @verbose_info("Optimization complete, retrieving moments")

    return status, value, MomentVector(relaxation, solution.x, state.slack, state.Acoo)
end