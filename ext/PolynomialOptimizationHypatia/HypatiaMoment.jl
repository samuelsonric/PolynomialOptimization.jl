struct StateMoment{K<:Integer,V<:Real} <: AbstractSparseMatrixSolver{Int,K,V}
    Acoo::SparseMatrixCOO{Int,K,V,1}
    b::Tuple{FastVec{Int},FastVec{V}}
    minusGcoo::SparseMatrixCOO{Int,K,V,1}
    c::Ref{Tuple{Vector{K},Vector{V}}}
    cones::FastVec{Cones.Cone{V}}

    StateMoment{K,V}() where {K<:Integer,V<:Real} = new{K,V}(
        SparseMatrixCOO{Int,K,V,1}(),
        (FastVec{Int}(), FastVec{V}()),
        SparseMatrixCOO{Int,K,V,1}(),
        Ref{Tuple{Vector{K},Vector{V}}}(),
        FastVec{Cones.Cone{V}}()
    )
end

Solver.supports_quadratic(::StateMoment) = SOLVER_QUADRATIC_RSOC

Solver.supports_complex_psd(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:U)

function Solver.add_constr_nonnegative!(state::StateMoment{K,V}, indvals::AbstractIndvals{K,V}) where {K,V}
    append!(state.minusGcoo, indvals)
    push!(state.cones, Cones.Nonnegative{V}(1))
    return
end

function Solver.add_constr_quadratic!(state::StateMoment{K,V}, indvals::AbstractIndvals{K,V}...) where {K,V}
    append!(state.minusGcoo, indvals...)
    push!(state.cones, Cones.EpiPerSquare{V}(length(indvals)))
    return
end

function Solver.add_constr_psd!(state::StateMoment{K,V}, dim::Int, data::PSDVector{K,V}) where {K,V}
    append!(state.minusGcoo, data)
    push!(state.cones, Cones.PosSemidefTri{V,V}(trisize(dim)))
    return
end

function Solver.add_constr_psd_complex!(state::StateMoment{K,V}, dim::Int, data::PSDVector{K,V}) where {K,V}
    append!(state.minusGcoo, data)
    push!(state.cones, Cones.PosSemidefTri{V,Complex{V}}(dim^2))
    return
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # This is just a lower bound, as the number of constraints is multiplied by the individual index count. But better than
    # nothing.
    prepare_push!(state.Acoo, num)
    return
end

function Solver.add_constr_fix!(state::StateMoment{K,V}, ::Nothing, indvals::AbstractIndvals{K,V}, rhs::V) where {K,V}
    v = append!(state.Acoo, indvals)
    if !iszero(rhs)
        push!(state.b[1], v)
        push!(state.b[2], rhs)
    end
    return
end

function Solver.fix_objective!(state::StateMoment{K,V}, indvals::AbstractIndvals{K,V}) where {K,V}
    state.c[] = (indvals.indices, indvals.values)
    return
end

function Solver.poly_optimize(::Val{:HypatiaMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    verbose::Bool=false, dense::Bool=!isone(poly_problem(relaxation).prefactor), customize::Base.Callable=_ -> nothing,
    parameters...)
    setup_time = @elapsed begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        V = real(coefficient_type(poly_problem(relaxation).objective))
        state = StateMoment{K,V}()

        moment_setup!(state, relaxation, groupings)
        customize(state)

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Ccoo = state.c[]
        b = zeros(V, size(state.Acoo, 1))
        copyto!(@view(b[state.b[1]]), state.b[2])
        h = zeros(V, size(state.minusGcoo, 1))

        moncount, (Acolptr, Arowind, Anzval), (Gcolptr, Growind, Gnzval), c = coo_to_csc!(state.Acoo, state.minusGcoo, Ccoo)
        model = Models.Model{V}(
            c, # c
            SparseMatrixCSC{V,Int}(length(b), moncount, Acolptr, Arowind, Anzval), # A
            b, # b
            SparseMatrixCSC{V,Int}(length(h), moncount, Gcolptr, Growind, rmul!(Gnzval, -one(V))), # G
            h, # h
            finish!(state.cones) # cones
        )
        if !dense
            # for lots of smaller constraints, a sparse solver is much better. However, all non-QRCholDenseSystemSolver
            # types also require to turn off reduction (else, we just get completely wrong results), and performing a dense
            # preprocessing also defeats the purpose of a sparse solver.
            parameters = (syssolver=get(() -> Solvers.SymIndefSparseSystemSolver{Float64}(), parameters, :syssolver),
                preprocess=get(parameters, :preprocess, false), reduce=get(parameters, :reduce, false), parameters...)
        end
        solver = Solvers.Solver{V}(; verbose, parameters...)
        Solvers.load(solver, model)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    Solvers.solve(solver)
    status = Solvers.get_status(solver)
    value = Solvers.get_primal_obj(solver)
    @verbose_info("Optimization complete, retrieving moments")

    return status, value, MomentVector(relaxation, Solvers.get_x(solver), state.Acoo, state.minusGcoo)
end