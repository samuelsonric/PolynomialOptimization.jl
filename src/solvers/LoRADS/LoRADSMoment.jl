mutable struct StateMoment{K<:Integer} <: AbstractSparseMatrixSolver{LoRADSInt,K,Cdouble}
    # while this is in the end an AbstractAPISolver, the add_ functions are matrix-based
    const A::Vector{Tuple{Vector{LoRADSInt},Vector{LoRADSInt},Vector{Cdouble}}}
    A_lin::Tuple{Vector{LoRADSInt},Vector{LoRADSInt},Vector{Cdouble}}
    b::Vector{Cdouble}
    info::Vector{<:Vector{<:Tuple{Symbol,Any}}}
    data

    StateMoment{K}() where {K<:Integer} = new{K}(
        Tuple{Vector{LoRADSInt},Vector{LoRADSInt},Vector{Cdouble}}[]
    )
end

Solver.issuccess(::Val{:LoRADSMoment}, status::LoRADS.Status) =
    status ∈ (LoRADS.STATUS_PRIMAL_DUAL_OPTIMAL, LoRADS.STATUS_PRIMAL_OPTIMAL)

Solver.psd_indextype(::StateMoment) = PSDIndextypeCOOVectorized(:L, 2., zero(LoRADSInt))

@counter_atomic(StateMoment, :psd)

function Solver.add_var_nonnegative!(state::StateMoment, m::Int, n::Int,
    data::SparseMatrixCOO{LoRADSInt,LoRADSInt,Cdouble,zero(LoRADSInt)},
    obj::Tuple{FastVec{LoRADSInt},FastVec{Cdouble}})
    @assert(!isdefined(state, :A_lin))
    data.rows .+= one(LoRADSInt)
    append!(data.rows, Iterators.repeated(zero(LoRADSInt), length(obj[1])))
    append!(data.cols, obj[1])
    append!(data.vals, obj[2])
    state.A_lin = coo_to_csr!(m +1, data)
    return
end

function Solver.add_var_psd!(state::StateMoment, m::Int, dim::LoRADSInt,
    data::SparseMatrixCOO{LoRADSInt,LoRADSInt,Cdouble,zero(LoRADSInt)},
    obj::Union{Nothing,Tuple{FastVec{LoRADSInt},FastVec{Cdouble}}})
    data.rows .+= one(LoRADSInt)
    if !isnothing(obj)
        append!(data.rows, Iterators.repeated(zero(LoRADSInt), length(obj[1])))
        append!(data.cols, obj[1])
        append!(data.vals, obj[2])
    end
    push!(state.A, coo_to_csr!(m +1, data))
    return
end

function Solver.fix_constraints!(state::StateMoment, m::Int, indvals::Indvals{LoRADSInt,Cdouble})
    state.b = b = zeros(Cdouble, m)
    for (i, v) in indvals
        b[i+1] = v
    end
    return
end

function Solver.poly_optimize(::Val{:LoRADSMoment}, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation, verbose::Bool=false, customize=(state) -> nothing, precision::Union{Nothing,<:Real}=nothing, parameters...)
    if isnothing(precision)
        params = LoRADS.Params(; parameters...)
    else
        params = LoRADS.Params(; phase2Tol=precision, phase1Tol=.01precision, parameters...)
    end
    setup_time = @elapsed @inbounds begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))

        state = StateMoment{K}()
        primal_data = primal_moment_setup!(state, relaxation, groupings; verbose)
        ismissing(primal_data) && return missing, LoRADS.STATUS_UNKNOWN, -Inf
        isempty(primal_data[2].psd_dim) && error("LoRADS requires at least one semidefinite variable")
        state.info, state.data = primal_data
        customize(state)

        @verbose_info("Initializing solver")
        solver = LoRADS.Solver()
        num_con, num_nonneg, psd_dim = primal_data[2].num_con, primal_data[2].num_nonneg, primal_data[2].psd_dim
        LoRADS.init_solver(solver, num_con, psd_dim, num_nonneg)
        LoRADS.set_dual_objective(solver, state.b)
        if isdefined(state, :A_lin)
            LoRADS.set_lp_cone(solver, num_con, num_nonneg, state.A_lin...) # this will copy all the A data
            empty!.(state.A_lin)
        end
        for (i, dim, data) in zip(Iterators.countfrom(zero(LoRADSInt)), psd_dim, state.A)
            LoRADS.set_cone(solver, i, LoRADS.conedata_to_userdata(LoRADS.CONETYPE_DENSE_SDP, num_con, dim, data...))
        end
        LoRADS.preprocess(solver, psd_dim)
        empty!(state.A) # the data was transferred to an internal buffer in preprocess
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    solver_time = @elapsed begin
        LoRADS.solve(solver, params, psd_dim; verbose)
    end

    @verbose_info("Optimization complete in ", solver_time, " seconds")

    return (state, solver), solver.AStatus, solver.pObjVal
end

struct GetX
    solver::LoRADS.Solver
    mm::Vector{Matrix{Cdouble}}
    trinumbers::Vector{Vector{Int}}

    function GetX(solver::LoRADS.Solver, dims::AbstractVector{I}) where {I<:Integer}
        mm = Vector{Matrix{Cdouble}}(undef, length(dims))
        trinumbers = similar(mm, Vector{Int})
        trinumbers_lookup = Dict{I,Vector{Int}}()
        @inbounds for (i, dim) in enumerate(dims)
            trinumbers[i] = get!(trinumbers_lookup, dim) do
                tn = Vector{Int}(undef, dim)
                tn[1] = 0
                rest = dim
                for i in 2:dim
                    tn[i] = tn[i-1] + rest
                    rest -= 1
                end
                tn
            end
        end
        new(solver, mm, trinumbers)
    end
end

@inline function Base.getindex(x::GetX, i::Integer)
    @inbounds begin
        if !isassigned(x.mm, i)
            x.mm[i] = LoRADS.get_X(x.solver, i)
        end
        return GetXI(x.solver, x.mm[i], x.trinumbers[i])
    end
end

struct GetXI
    solver::LoRADS.Solver
    mm::Matrix{Cdouble}
    trinumbers::Vector{Int}
end

@inline function Base.getindex(x::GetXI, idx::Integer)
    @inbounds begin
        col = searchsortedlast(x.trinumbers, idx)
        row = idx - x.trinumbers[col] + col
        return x.mm[row, col]
    end
end

Solver.extract_moments(relaxation::AbstractRelaxation, (state, solver)::Tuple{StateMoment,LoRADS.Solver}) =
    # Dilemma: Do we want to calculate the full matrix product U Uᵀ (or one triangle) and extract the relevant entries, which
    # would be favorable if we have a large moment matrix that contains everything? Or do we want to calculate the entries
    # on-the-fly doing vector products, which in total is not cache-optimal, but would be better if we only need few entries
    # from the matrix, perhaps because it represents a moment sub-matrix where a lot of entries have already been obtained by
    # other matrices? Let's do the first: We assume that this solver is used for large moment matrices, and we assume that
    # there won't be too much overlap between sparse groupings.
    MomentVector(relaxation, state.data, GetX(solver, state.data.psd_dim),
        LoRADS.get_Xlin(solver))

# While the information is there, it can be very bad. Always look at the reported dual infeasibility - if it too large, the
# SOS decomposition will be useless. Note that this is not part of the termination criteria.
Solver.extract_sos(::AbstractRelaxation, (_, solver)::Tuple{StateMoment,LoRADS.Solver}, ::Val,
    index::AbstractUnitRange, ::Nothing) = LoRADS.get_Slin(solver, index)

Solver.extract_sos(::AbstractRelaxation, (_, solver)::Tuple{StateMoment,LoRADS.Solver}, ::Val{:psd}, index::Integer,
    ::Nothing) = LoRADS.get_S(solver, index)