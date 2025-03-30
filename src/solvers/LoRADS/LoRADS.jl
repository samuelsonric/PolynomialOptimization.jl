# This is an interface to some of the functions of the experimental LoRADS solver, https://github.com/COPT-Public/LoRADS,
# tightly integrated with the PolynomialOptimization framework
module LoRADS

using LinearAlgebra: chkstride1, Transpose, Symmetric, rmul!
using SparseArrays, Preferences, Printf
using LinearAlgebra.BLAS: axpy!, syrk!
using StandardPacked: tpttr!
using ...PolynomialOptimization: @assert, @inbounds, @verbose_info

const solverlib = @load_preference("lorads-solver", "")
const LoRADSInt = @load_preference("lorads-int", Int64)

havefree::Bool = false

function __init__()
    !isempty(solverlib) && let dl=Libc.dlopen(solverlib, throw_error=false)
        if isnothing(dl)
            @warn("The LoRADS library is configured to $solverlib, but it could not be opened. Call \
                   `PolynomialOptimization.Solvers.LoRADS.set_solverlib` to change the configuration; set it to an empty \
                   value to disable the solver.")
        else
            global havefree
            havefree = !isnothing(Libc.dlsym(dl, :FREE, throw_error=false))
            havefree || @warn("The unpatched version of the LoRADS library is used. This is not recommended.")
            Libc.dlclose(dl)
        end
    end
    return
end

"""
    set_solverlib(path)

Changes the path to the LoRADS library, which takes effect after restarting Julia.
"""
function set_solverlib(path::String)
    if isfile(path)
        @set_preferences!("lorads-solver" => path)
        @info("New path to solver library set; restart you Julia session for this change to take effect!")
        return
    else
        throw(ArgumentError("Library not found: $path"))
    end
end

include("./Enums.jl")
include("./DataTypes.jl")

"""
    init_solver(solver, nConstrRows, coneDims::AbstractVector{LoRADSInt}, nLpCols)

Initializes a fresh `Solver` object with `nConstrRows` constraints, positive semidefinite variables of side dimension
`coneDims` (a vector of integers), and `nLpCols` scalar nonnegative variables.
"""
function init_solver(solver::Solver, nConstrRows::LoRADSInt, coneDims::AbstractVector{LoRADSInt},
    nLpCols::LoRADSInt)
    chkstride1(coneDims)
    @ccall solverlib.LORADSInitSolver(solver::Ptr{Cvoid}, nConstrRows::LoRADSInt, length(coneDims)::LoRADSInt,
        coneDims::Ptr{LoRADSInt}, nLpCols::LoRADSInt)::Cvoid
    solver.timeSolveStart = get_timestamp()
    return solver
end

function cleanup(solver::Solver)
    @ccall solverlib.LORADSDestroyADMMVars(solver::Ptr{Cvoid})::Cvoid
    @ccall solverlib.LORADSDestroyALMVars(solver::Ptr{Cvoid})::Cvoid
    global havefree
    solver.hisRecT = 0
    if solver.sparsitySDPCoeff != C_NULL
        if havefree
            @ccall solverlib.FREE(solver.sparsitySDPCoeff::Ptr{Cvoid})::Cvoid
        else
            @ccall solverlib.REALLOC(
                (pointer_from_objref(solver) + fieldoffset(Solver, Base.fieldindex(Solver, :sparsitySDPCoeff)))::Ptr{Ptr{Cvoid}},
                zero(LoRADSInt)::LoRADSInt,
                zero(LoRADSInt)::LoRADSInt
            )::Cvoid
        end
        solver.sparsitySDPCoeff = C_NULL # since zero-length calloc is implementation-defined
    end
    if solver.var.rankElem != C_NULL
        if havefree
            @ccall solverlib.FREE(solver.var.rankElem::Ptr{Cvoid})::Cvoid
        else
            @ccall solverlib.REALLOC(
                (pointer_from_objref(solver.var) + fieldoffset(Variable, Base.fieldindex(Variable, :rankElem)))::Ptr{Ptr{LoRADSInt}},
                zero(LoRADSInt)::LoRADSInt,
                zero(LoRADSInt)::LoRADSInt
            )::Cvoid
        end
        solver.var.rankElem = C_NULL
    end
    @ccall solverlib.destroyPreprocess(solver::Ptr{Cvoid})::Cvoid
    @ccall solverlib.LORADSDestroyConeData(solver::Ptr{Cvoid})::Cvoid
    @ccall solverlib.LORADSDestroySolver(solver::Ptr{Cvoid})::Cvoid
    solver.nCones = 0
    solver.nLpCols = 0
    return solver
end

"""
    set_dual_objective(solver, dObj::AbstractVector{Cdouble})

Sets the dual objective, i.e., the right-hand side of the constraints, in an initialized `Solver` object.
"""
function set_dual_objective(solver::Solver, dObj::AbstractVector{Cdouble})
    chkstride1(dObj)
    @ccall solverlib.LORADSSetDualObjective(solver::Ptr{Cvoid}, dObj::Ptr{Cdouble})::Cvoid
    return solver
end

"""
    conedata_to_userdata(cone::ConeType, nConstrRows, dim,
        coneMatBeg::AbstractVector{LoRADSInt}, coneMatIdx::AbstractVector{LoRADSInt},
        coneMatElem::AbstractVector{Cdouble})

Allocates a new user data objects and sets its conic data. This consists of a cone type (only `CONETYPE_DENSE_SDP` and
`CONETYPE_SPARSE_SDP` are supported), the number of rows (which is the same as the number of constraints in the solver)
and the side dimension of the semidefinite variable, followed by the constraint matrices in zero-indexed CSR format. Every row
corresponds to the vectorized lower triangle of the column of a constraint matrix. The zeroth row is the coefficient matrix for
the objective.
Therefore, `nConstrRows +1 = length(coneMatBeg) -1` should hold (`+1` for the objective; `-1` for CSR).

The returned userdata pointer should be assigned to a solver, which will take care of freeing the allocated data. Note that the
vectors passed to this function must be preserved until the [`preprocess`](@ref) function was called, after which they can be
freed.

See also [`set_cone`](@ref), [`init_cone_data`](@ref).
"""
function conedata_to_userdata(cone::ConeType, nConstrRows::Integer, dim::Integer, coneMatBeg::AbstractVector{LoRADSInt},
    coneMatIdx::AbstractVector{LoRADSInt}, coneMatElem::AbstractVector{Cdouble})
    chkstride1(coneMatBeg)
    chkstride1(coneMatIdx)
    chkstride1(coneMatElem)
    nConstrRows == length(coneMatBeg) -2 ||
        throw(ArgumentError("The number of constraint rows is not compatible with the given matrix."))
    (length(coneMatIdx) == length(coneMatElem) && iszero(coneMatBeg[begin]) && coneMatBeg[end] == length(coneMatIdx) &&
        issorted(coneMatBeg)) || throw(ArgumentError("Error in the CSR format"))
    result = Ref{Ptr{Cvoid}}()
    @ccall solverlib.LUserDataCreate(result::Ref{Ptr{Cvoid}})::Cvoid
    @ccall solverlib.LUserDataSetConeData(result[]::Ptr{Cvoid}, cone::Cint, nConstrRows::LoRADSInt, dim::LoRADSInt,
        coneMatBeg::Ptr{LoRADSInt}, coneMatIdx::Ptr{LoRADSInt}, coneMatElem::Ptr{Cdouble})::Cvoid
    return result[]
end

"""
    set_cone(solver, iCone, userCone)

Sets the `iCone`th cone to the data previously defined using [`conedata_to_userdata`](@ref).

See also [`init_cone_data`](@ref).
"""
function set_cone(solver::Solver, iCone::Integer, userCone::Ptr{Cvoid})
    @ccall solverlib.LORADSSetCone(solver::Ptr{Cvoid}, iCone::LoRADSInt, userCone::Ptr{Cvoid})::Cvoid
    return solver
end

"""
    set_lp_cone(solver, nConstrRows, nLpCols, lpMatBeg::AbstractVector{LoRADSInt},
        lpMatIdx::AbstractVector{LoRADSInt}, lpMatElem::AbstractVector{Cdouble})

Set the data of the constraint matrix for the linear variables according to the CSR data specified in the parameters.

!!! warning
    This function is not exported on the original code release and can therefore not be used. However, only the patched version
    should be used, as it fixes heap corruption errors that can arise during the optimization.
"""
function set_lp_cone(solver::Solver, nConstrRows::Integer, nLpCols::Integer, lpMatBeg::AbstractVector{LoRADSInt},
    lpMatIdx::AbstractVector{LoRADSInt}, lpMatElem::AbstractVector{Cdouble})
    chkstride1(lpMatBeg)
    chkstride1(lpMatIdx)
    chkstride1(lpMatElem)
    nConstrRows == length(lpMatBeg) -2 ||
        throw(ArgumentError("The number of constraint rows is not compatible with the given matrix."))
    (length(lpMatIdx) == length(lpMatElem) && iszero(lpMatBeg[begin]) && lpMatBeg[end] == length(lpMatIdx) &&
        issorted(lpMatBeg)) || throw(ArgumentError("Error in the CSR format"))
    @ccall solverlib.LORADSSetLpCone(solver.lpCone::Ptr{Cvoid}, nConstrRows::LoRADSInt, nLpCols::LoRADSInt, lpMatBeg::Ptr{LoRADSInt},
        lpMatIdx::Ptr{LoRADSInt}, lpMatElem::Ptr{Cdouble})::Cvoid
    return solver
end

@doc raw"""
    init_cone_data(solver, coneMat, coneDims, lpMat)

Initializes the solver for a problem in the form
```math
   \min \vec a_0 \cdot \vec x + \sum_j \langle\operatorname{mat}(G_{j, 0}), Z_j\rangle \\
   \text{such that} \\
   x_i \geq 0 \ \forall i \\
   Z_j \succeq 0 \ \forall j \\
   \vec a_k \cdot \vec x - \sum_j \langle\operatorname{mat}(G_{j, k}, Z_j)\rangle = c_k \ \forall k
```
with the following representation in the variables:
- `1 ≤ j ≤ length(coneDims) = length(coneMat)`
- `coneMat` is a vector of matrices, `lpMat` is a matrix. They should be in CSR storage, where the row index (starting at 0 for
  the objective, then `k` for the `k`th constraint) is the constraint. Since CSR is not natively supported by Julia, the
  transpose of a `SparseMatrixCSC{Cdouble,LoRADSInt}` is expected.
- `mat` makes the unscaled lower triangle into a full matrix

This is a convenience function that does the job of [`conedata_to_userdata`](@ref), [`set_cone`](@ref), and
[`preprocess`](@ref) in one step. However, note that it is more efficient to call these functions individually.
"""
function init_cone_data(solver::Solver, coneMat::AbstractVector{Transpose{Cdouble,SparseMatrixCSC{Cdouble,LoRADSInt}}},
    coneDims::AbstractVector{LoRADSInt}, lpMat::Union{Nothing,Transpose{Cdouble,SparseMatrixCSC{Cdouble,LoRADSInt}}})
    chkstride1(coneDims)
    if isempty(coneMat)
        isnothing(lpMat) && throw(ArgumentError("No data"))
        nConstrs = size(lpMat, 1)
    else
        nConstrs = size(first(coneMat), 1)
    end
    length(coneDims) == length(coneMat) || throw(ArgumentError("Lengths of cones are inconsistent"))
    coneMatElem = Vector{Ptr{Cdouble}}(undef, length(coneMat))
    coneMatBeg = similar(coneMatElem, Ptr{LoRADSInt})
    coneMatIdx = similar(coneMatElem, Ptr{LoRADSInt})
    @inbounds for (i, cm) in enumerate(coneMat)
        size(cm, 1) == nConstrs || throw(ArgumentError("Number of constraints inconsistent"))
        # cm is the transpose of CSC, i.e., CSR. Each row in cm is a constraint, each column an entry in the lower triangle
        coneDims[i] * (coneDims[i] +1) ÷ 2 == size(cm, 2) ||
            throw(ArgumentError(lazy"Length of cone $i is wrong (is $(coneDims[i])), should be $((isqrt(1 + 8size(cm, 2)) -1) ÷ 2)"))
        colptr = SparseArrays.getcolptr(parent(cm))
        colptr .-= one(LoRADSInt)
        rowval = SparseArrays.getrowval(parent(cm))
        rowval .-= one(LoRADSInt)
        coneMatBeg[i] = pointer(colptr)
        coneMatIdx[i] = pointer(rowval)
        coneMatElem[i] = pointer(SparseArrays.getnzval(parent(cm)))
    end
    if !isnothing(lpMat)
        size(lpMat, 1) == nConstrs || throw(ArgumentError("Number of constraints inconsistent"))
        nLpCols = size(lpMat, 2)
        LpMatBeg = SparseArray.getcolptr(lpMat)
        LpMatIdx = SparseArray.getrowval(lpMat)
        LpMatElem = SparseArray.getnzval(lpMat)
        LpMatBeg .-= one(LoRADSInt)
        LpMatIdx .-= one(LoRADSInt)
    else
        nLpCols = 0
        LpMatBeg = C_NULL
        LpMatIdx = C_NULL
        LpMatElem = C_NULL
    end
    sdpDatas = similar(coneMatElem, Ptr{Cvoid})
    @ccall solverlib.LORADSInitConeData(
        solver::Ptr{Cvoid},
        sdpDatas::Ptr{Ptr{Cvoid}},
        coneMatElem::Ptr{Ptr{Cdouble}},
        coneMatBeg::Ptr{Ptr{LoRADSInt}},
        coneMatIdx::Ptr{Ptr{LoRADSInt}},
        coneDims::Ptr{LoRADSInt},
        nConstrs::LoRADSInt,
        length(coneMat)::LoRADSInt,
        nLpCols::LoRADSInt,
        LpMatBeg::Ptr{LoRADSInt},
        LpMatIdx::Ptr{LoRADSInt},
        LpMatElem::Ptr{Cdouble}
    )::Cvoid
    # sdpDatas is filled. Every cone in solver has a field usrData that contains the same pointer. And when the cones are
    # destroyed when finalizing solver, this userdata is also freed (just the struct that holds the references, not the arrays
    # themselves). So we can just forget about our sdpDatas.
    preprocess(solver, coneDims)
    # Now all the data was converted to internal data. We don't need any of them any more.
    @inbounds for (i, cm) in enumerate(coneMat)
        SparseArrays.getcolptr(parent(cm)) .+= one(LoRADSInt)
        SparseArrays.getrowval(parent(cm)) .+= one(LoRADSInt)
    end
    if !isnothing(lpMat)
        LpMatBeg .+= one(LoRADSInt)
        LpMatIdx .+= one(LoRADSInt)
    end
    return solver
end

"""
    preprocess(solver, coneDims::AbstractVector{LoRADSInt})

Invokes the preprocessor. This should be called after all cones were set up, after which their original data may be reused or
destroyed.

See also [`conedata_to_userdata`](@ref), [`set_cone`](@ref).
"""
function preprocess(solver::Solver, coneDims::AbstractVector{LoRADSInt})
    chkstride1(coneDims)
    @ccall solverlib.LORADSPreprocess(solver::Ptr{Cvoid}, coneDims::Ptr{LoRADSInt})::Cvoid
    return solver
end

function determine_rank(solver::Solver, coneDims::AbstractVector{LoRADSInt}, timesRank::Real)
    chkstride1(coneDims)
    @ccall solverlib.LORADSDetermineRank(solver::Ptr{Cvoid}, coneDims::Ptr{LoRADSInt}, timesRank::Cdouble)::Cvoid
    return solver
end

function detect_max_cut_prob(solver::Solver, coneDims::AbstractVector{LoRADSInt})
    chkstride1(coneDims)
    maxCut = Ref{LoRADSInt}()
    @ccall solverlib.detectMaxCutProb(solver::Ptr{Cvoid}, coneDims::Ptr{LoRADSInt}, maxCut::Ref{LoRADSInt})::Cvoid
    return maxCut[]
end

function detect_sparsity_sdp_coeff(solver::Solver)
    @ccall solverlib.detectSparsitySDPCoeff(solver::Ptr{Cvoid})::Cvoid
    return solver
end

function init_alm_vars(solver::Solver, coneDims::AbstractVector{LoRADSInt}, nLpCols::Integer, lbfgsHis::Integer)
    chkstride1(coneDims)
    @ccall solverlib.LORADSInitALMVars(
        solver::Ptr{Cvoid},
        solver.var.rankElem::Ptr{LoRADSInt},
        coneDims::Ptr{LoRADSInt},
        length(coneDims)::LoRADSInt,
        nLpCols::LoRADSInt,
        lbfgsHis::LoRADSInt
    )::Cvoid
    solver.hisRecT = lbfgsHis
    return solver
end

function init_admm_vars(solver::Solver, coneDims::AbstractVector{LoRADSInt}, nLpCols::Integer)
    chkstride1(coneDims)
    @ccall solverlib.LORADSInitADMMVars(
        solver::Ptr{Cvoid},
        solver.var.rankElem::Ptr{LoRADSInt},
        coneDims::Ptr{LoRADSInt},
        length(coneDims)::LoRADSInt,
        nLpCols::LoRADSInt
    )::Cvoid
    return solver
end

function initial_solver_state(params::Params, solver::Solver)
    alm_state = ALMState()
    admm_state = ADMMState()
    sdpConst = SDPConst()
    @ccall solverlib.initial_solver_state(
        params::Ptr{Cvoid},
        solver::Ptr{Cvoid},
        alm_state::Ptr{Cvoid},
        admm_state::Ptr{Cvoid},
        sdpConst::Ptr{Cvoid}
    )::Cvoid
    return alm_state, admm_state, sdpConst
end

alm_optimize(params::Params, solver::Solver, alm_iter_state::ALMState, rho_update_factor::Real, timeSolveStart::Real) =
    @ccall solverlib.LORADS_ALMOptimize(params::Ptr{Cvoid}, solver::Ptr{Cvoid}, alm_iter_state::Ptr{Cvoid},
        rho_update_factor::Cdouble, timeSolveStart::Cdouble)::Retcode

function alm_to_admm(solver::Solver, params::Params, alm_state::ALMState, admm_state::ADMMState)
    @ccall solverlib.LORADS_ALMtoADMM(solver::Ptr{Cvoid}, params::Ptr{Cvoid}, alm_state::Ptr{Cvoid},
        admm_state::Ptr{Cvoid})::Cvoid
    return solver
end

admm_optimize(params::Params, solver::Solver, admm_iter_sate::ADMMState, iter_celling::Integer, timeSolveStart::Real) =
    @ccall solverlib.LORADSADMMOptimize(params::Ptr{Cvoid}, solver::Ptr{Cvoid}, admm_iter_sate::Ptr{Cvoid},
        iter_celling::LoRADSInt, timeSolveStart::Cdouble)::Retcode

function calculate_dual_infeasibility(solver::Solver)
    @ccall solverlib.calculate_dual_infeasibility_solver(solver::Ptr{Cvoid})::Cvoid
    return solver
end

function reopt(params::Params, solver::Solver, alm_state::ALMState, admm_state::ADMMState, reopt_param::Real,
    alm_iter::Integer, admm_iter::Integer, timeSolveStart::Real, admm_bad_iter_flag::Integer, reopt_level::Integer)
    bad_iter_flag = Ref{Cint}(admm_bad_iter_flag)
    @ccall solverlib.reopt(params::Ptr{Cvoid}, solver::Ptr{Cvoid}, alm_state::Ptr{Cvoid}, admm_state::Ptr{Cvoid},
        reopt_param::Ref{Cdouble}, alm_iter::Ref{LoRADSInt}, admm_iter::Ref{LoRADSInt}, timeSolveStart::Cdouble,
        bad_iter_flag::Ref{Cint}, reopt_level::Cint)::Cdouble # returns @elapsed, not interesting
    return bad_iter_flag[]
end

function averageUV(solver::Solver)
    solver.nLpCols > 0 &&
        @ccall solverlib.averageUVLP(
            solver.var.uLp::Ptr{LpDense},
            solver.var.vLp::Ptr{LpDense},
            solver.var.rLp::Ptr{LpDense}
        )::Cvoid
    for iCone in one(LoRADSInt):solver.nCones
        @ccall solverlib.averageUV(
            unsafe_load(solver.var.U, iCone)::Ptr{SDPDense},
            unsafe_load(solver.var.V, iCone)::Ptr{SDPDense},
            unsafe_load(solver.var.R, iCone)::Ptr{SDPDense}
        )::Cvoid
    end
    return solver
end

function copyRToV(solver::Solver)
    if solver.nLpCols > 0
        @ccall solverlib.copyRtoVLP(
            solver.var.rLp::Ptr{LpDense},
            solver.var.vLp::Ptr{LpDense},
            solver.var.R::Ptr{Ptr{LpDense}},
            solver.var.V::Ptr{Ptr{LpDense}},
            solver.nCones::LoRADSInt
        )::Cvoid
    else
        @ccall solverlib.copyRtoV(
            solver.var.rLp::Ptr{LpDense},
            solver.var.vLp::Ptr{LpDense},
            solver.var.R::Ptr{Ptr{LpDense}},
            solver.var.V::Ptr{Ptr{LpDense}},
            solver.nCones::LoRADSInt
        )::Cvoid
    end
    return solver
end

function end_program(solver::Solver)
    @ccall solverlib.LORADSEndProgram(solver::Ptr{Cvoid})::Cvoid
    return solver
end

get_timestamp() = @ccall solverlib.LUtilGetTimeStamp()::Cdouble

clear_usr_data(coneMatBeg::Ptr{Ptr{LoRADSInt}}, coneMatIdx::Ptr{Ptr{LoRADSInt}}, coneMatElem::Ptr{Ptr{Cdouble}},
    nBlks::Integer, blkDims::Ptr{LoRADSInt}, rowRHS::Ptr{Cdouble}, lpMatBeg::Ptr{LoRADSInt}, lpMatIdx::Ptr{LoRADSInt},
    lpMatElem::Ptr{Cdouble}, SDPDatas::Ptr{Ptr{Cvoid}}) =
    @ccall solverlib.LORADSClearUsrData(
        coneMatBeg::Ptr{Ptr{LoRADSInt}}, coneMatIdx::Ptr{Ptr{LoRADSInt}}, coneMatElem::Ptr{Ptr{Cdouble}},
        nBlks::LoRADSInt, blkDims::Ptr{LoRADSInt},
        rowRHS::Ptr{Cdouble}, lpMatBeg::Ptr{LoRADSInt}, lpMatIdx::Ptr{LoRADSInt}, lpMatElem::Ptr{Cdouble},
        SDPDatas::Ptr{Ptr{Cvoid}}
    )::Cvoid

"""
    load_sdpa(fn)

Loads a problem from a file `fn` in SDPA format and returns a preprocessed `Solver` instance, a vector containing the cone
dimensions, and the number of nonnegative scalar variables.

!!! warning
    This function will produce memory leaks. The data that is allocated by the LoRADS library cannot be freed by Julia, as no
    corresponding functions are exported. Only use it for quick tests, not in production.
"""
function load_sdpa(fn::AbstractString)
    isempty(solverlib) &&
        error("LoRADS is not configured. Call `PolynomialOptimization.Solvers.LoRADS.set_solverlib` to specify the location \
               of the solver library")
    fn_str = Vector{UInt8}(fn)
    fn_str[end] == 0x00 || push!(fn_str, 0x00)
    pnConstrs = Ref{LoRADSInt}()
    pnBlks = Ref{LoRADSInt}()
    pblkDims = Ref{Ptr{LoRADSInt}}()
    prowRHS = Ref{Ptr{Cdouble}}()
    pconeMatBeg = Ref{Ptr{Ptr{LoRADSInt}}}()
    pconeMatIdx = Ref{Ptr{Ptr{LoRADSInt}}}()
    pconeMatElem = Ref{Ptr{Ptr{Cdouble}}}()
    pnLpCols = Ref{LoRADSInt}()
    pLpMatBeg = Ref{Ptr{LoRADSInt}}()
    pLpMatIdx = Ref{Ptr{LoRADSInt}}()
    pLpMatElem = Ref{Ptr{Cdouble}}()

    ret = @ccall solverlib.AReadSDPA(fn_str::Ptr{Cvoid}, pnConstrs::Ptr{Cvoid}, pnBlks::Ptr{Cvoid}, pblkDims::Ptr{Cvoid},
        prowRHS::Ptr{Cvoid}, pconeMatBeg::Ptr{Cvoid}, pconeMatIdx::Ptr{Cvoid}, pconeMatElem::Ptr{Cvoid},
        Ref{LoRADSInt}()::Ptr{Cvoid}, pnLpCols::Ptr{Cvoid}, pLpMatBeg::Ptr{Cvoid}, pLpMatIdx::Ptr{Cvoid},
        pLpMatElem::Ptr{Cvoid}, Ref{LoRADSInt}()::Ptr{Cvoid})::Retcode
    if ret != RETCODE_OK
        clear_usr_data(pconeMatBeg[], pconeMatIdx[], pconeMatElem[], pnBlks[], pblkDims[], prowRHS[], pLpMatBeg[],
            pLpMatIdx[], pLpMatElem[], C_VOID)
        error("Reading SDPA data failed with retcode $ret")
    end

    coneDims = unsafe_wrap(Vector{LoRADSInt}, pblkDims[], pnBlks[], own=false)
    rowRHS = unsafe_wrap(Vector{Cdouble}, prowRHS[], pnConstrs[], own=false)

    solver = Solver()
    init_solver(solver, pnConstrs[], coneDims, pnLpCols[])
    set_dual_objective(solver, rowRHS)
    sdpDatas = Vector{Ptr{Cvoid}}(undef, pnBlks[])
    @ccall solverlib.LORADSInitConeData(
        solver::Ptr{Cvoid},
        sdpDatas::Ptr{Ptr{Cvoid}},
        pconeMatElem[]::Ptr{Ptr{Cdouble}},
        pconeMatBeg[]::Ptr{Ptr{LoRADSInt}},
        pconeMatIdx[]::Ptr{Ptr{LoRADSInt}},
        pblkDims[]::Ptr{LoRADSInt},
        pnConstrs[]::LoRADSInt,
        pnBlks[]::LoRADSInt,
        pnLpCols[]::LoRADSInt,
        pLpMatBeg[]::Ptr{LoRADSInt},
        pLpMatIdx[]::Ptr{LoRADSInt},
        pLpMatElem[]::Ptr{Cdouble}
    )::Cvoid
    preprocess(solver, coneDims)
    GC.@preserve sdpDatas begin
        clear_usr_data(pconeMatBeg[], pconeMatIdx[], pconeMatElem[], pnBlks[], pblkDims[], prowRHS[], pLpMatBeg[],
            pLpMatIdx[], pLpMatElem[], pointer(sdpDatas))
    end
    return solver, coneDims
end

"""
    solve(solver, params, coneDims)

Solves a preprocessed `Solver` instance.
"""
function solve(solver::Solver, params::Params, coneDims::AbstractVector{LoRADSInt}; verbose::Bool=false)
    isempty(solverlib) &&
        error("LoRADS is not configured. Call `PolynomialOptimization.Solvers.LoRADS.set_solverlib` to specify the location of the solver library")
    chkstride1(coneDims)

    prep = @elapsed begin
        determine_rank(solver, coneDims, params.timesLogRank)
        # detect_sparsity_sdp_coeff(solver)
        init_alm_vars(solver, coneDims, solver.nLpCols, params.lbfgsListLength)
        init_admm_vars(solver, coneDims, solver.nLpCols)

        alm_state, admm_state, sdpConst = initial_solver_state(params, solver)
    end
    @verbose_info("Solver prepared in ", prep, " seconds")

    function dual_inf end
    let solver=solver, admm_state=admm_state
        function dual_inf()
            dual_inf_time = @elapsed calculate_dual_infeasibility(solver)
            @verbose_info("Calculate dual infeasibility in ", dual_inf_time, " seconds")

            local dfℓ₁ = solver.dimacError.dualfeasible_l1
            local dfℓ₁cObj = dfℓ₁ * (1 + solver.cObjNrm1)
            local cvℓ₁ = solver.dimacError.constrvio_l1
            local cvℓ₁bRHS = cvℓ₁ * (1 + solver.bRHSNrm1)

            admm_state.l_1_dual_infeasibility = dfℓ₁
            admm_state.l_inf_dual_infeasibility = dfℓ₁cObj / (1 + solver.cObjNrmInf)
            admm_state.l_2_dual_infeasibility = dfℓ₁cObj/ (1 + solver.cObjNrm2)
            admm_state.primal_dual_gap = solver.dimacError.pdgap
            admm_state.l_1_primal_infeasibility = cvℓ₁
            admm_state.l_inf_primal_infeasibility = cvℓ₁bRHS / (1 + solver.bRHSNrmInf)
            admm_state.l_2_primal_infeasibility = cvℓ₁bRHS / (1 + solver.bRHSNrm2)

            @verbose_info("Dual infeasibility: ℓ₁ = ", dfℓ₁, ", ℓ_∞ = ", admm_state.l_inf_dual_infeasibility, ", ℓ₂ = ",
                admm_state.l_2_dual_infeasibility)

            return
        end
    end

    reopt_param::Cdouble = 5
    reopt_alm_iter::LoRADSInt = 3
    reopt_admm_iter::LoRADSInt = params.highAccMode ? 1000 : 50
    alm_reopt_min_iter::LoRADSInt = 3
    admm_reopt_min_iter::LoRADSInt = 50
    admm_bad_iter_flag::Cint = 0

    alm_optimize(params, solver, alm_state, params.maxALMIter, solver.timeSolveStart)
    if get_timestamp() - solver.timeSolveStart > params.timeSecLimit
        solver.AStatus = STATUS_TIME_LIMIT
        @goto END_SOLVING
    end

    a2a_time = @elapsed alm_to_admm(solver, params, alm_state, admm_state)
    @verbose_info("Converted ALM to ADMM in ", a2a_time, " seconds")

    admm_bad_iter_flag = admm_optimize(params, solver, admm_state, params.maxADMMIter, solver.timeSolveStart) == RETCODE_BAD_ITER
    # if admm_state.primal_dual_gap ≥ params.phase1Tol
    #     params.phase1Tol = max(.1params.phase1Tol, params.phase2Tol)
    # end
    if verbose
        dual_inf()
        if get_timestamp() - solver.timeSolveStart > params.timeSecLimit
            solver.AStatus = STATUS_TIME_LIMIT
            @goto END_SOLVING
        end
        println("After initial solver results:")
        @sprintf("Objective function
                 ==================
                 1. Primal Objective: %10.6e
                 2. Dual Objective:   %10.6e
                 Dimacs Errors:
                 ==============
                 1. Constraint Violation (ℓ₁): %10.6e
                 2. Dual Infeasibility (ℓ₁):   %10.6e
                 3. Primal-Dual Gap:           %10.6e
                 4. Constraint Violation (∞):  %10.6e
                 5. Dual Infeasibility (∞):    %10.6e",
            solver.pObjVal, solver.dObjVal,
            solver.dimacError.constrvio_l1, solver.dimacError.dualfeasible_l1, solver.dimacError.pdgap,
            solver.dimacError.constrvio_l1 * (1 + solver.bRHSNrm1) / (1 + solver.bRHSNrmInf),
            solver.dimacError.dualfeasible_l1 * (1 + solver.cObjNrm1) / (1 + solver.cObjNrmInf))
    end
    if params.reoptLevel ≥ 1 &&
        (alm_state.primal_dual_gap > params.phase2Tol || alm_state.l_1_primal_infeasibility > params.phase2Tol) &&
        (admm_state.primal_dual_gap > params.phase2Tol || admm_state.l_1_primal_infeasibility > params.phase2Tol)
        @verbose_info("Reoptimization parameter: ", reopt_param)
        reopt_time = @elapsed begin
            admm_bad_iter_flag = reopt(params, solver, alm_state, admm_state, reopt_param, alm_reopt_min_iter,
                admm_reopt_min_iter, solver.timeSolveStart, admm_bad_iter_flag, 1)
        end
        @verbose_info("Reoptimization in ", reopt_time, " seconds")
        if get_timestamp() - solver.timeSolveStart > params.timeSecLimit
            solver.AStatus = STATUS_TIME_LIMIT
            @goto END_SOLVING
        end
    end

    dual_inf()

    params.reoptLevel ≥ 2 &&
        for dual_cnt in 0:1
            admm_state.l_1_dual_infeasibility > params.phase2Tol || admm_state.primal_dual_gap > params.phase2Tol ||
                admm_state.l_1_primal_infeasibility > params.phase2Tol || break
            # if admm_state.primal_dual_gap ≥ 10params.phase2Tol
            #     params.phase1Tol = max(.1params.phase1Tol, params.phase2Tol)
            # end
            params.highAccMode || admm_state.l_1_dual_infeasibility > 5params.phase2Tol ||
                admm_state.primal_dual_gap > 5params.phase2Tol || admm_state.l_1_primal_infeasibility > params.phase2Tol ||
                break
            @verbose_info("Reoptimization parameter: ", reopt_param)
            reopt_time = @elapsed begin
                admm_bad_iter_flag = reopt(params, solver, alm_state, admm_state, reopt_param, reopt_alm_iter,
                    reopt_admm_iter, solver.timeSolveStart, admm_bad_iter_flag, 2)
                copyRToV(averageUV(solver))
            end
            @verbose_info("Reoptimization in ", reopt_time, " seconds")

            dual_inf()

            if get_timestamp() - solver.timeSolveStart > params.timeSecLimit
                solver.AStatus = STATUS_TIME_LIMIT
                @goto END_SOLVING
            end
        end

    if admm_state.primal_dual_gap ≤ 5params.phase2Tol && admm_state.l_1_primal_infeasibility ≤ params.phase2Tol
        solver.AStatus = admm_state.l_1_dual_infeasibility ≤ 5params.phase2Tol ?
                            STATUS_PRIMAL_DUAL_OPTIMAL : STATUS_PRIMAL_OPTIMAL
    else
        solver.AStatus = STATUS_MAXITER
    end

    @label END_SOLVING

    verbose && end_program(solver)

    return solver
end

"""
    get_X(solver, i)

Returns the `i`th PSD solution matrix ``X_i``. The result will be a freshly allocated symmetric view of a dense matrix.

!!! warning
    This method may only be called once per `i`. All further calls with the same `i` will give wrong output, as the internal
    solver data is modified.
"""
function get_X(solver::Solver, i::Integer)
    # The solver doesn't contain the primal matrix in full form, but in a low-rank factorization.
    V = unsafe_load(unsafe_load(solver.var.V, i))
    # Since the ADMM approach uses a U Vᵀ, V = U rewrite with additional constraints, even then it is not just U Uᵀ. So the
    # recommended reconstruction is Û = (U + V)/2, X = Û Ûᵀ.
    U = unsafe_load(unsafe_load(solver.var.U, i))
    @assert U.nRows == V.nRows && U.rank == V.rank
    axpy!(V.nRows * V.rank, one(Cdouble), U.matElem, 1, V.matElem, 1)

    result = Matrix{Cdouble}(undef, V.nRows, V.nRows)
    Vwrap = unsafe_wrap(Array, V.matElem, (V.nRows, V.rank), own=false)
    syrk!('L', 'N', .25, Vwrap, false, result)
    return Symmetric(result, :L)
end

"""
    get_Xlin(solver)

Returns the linear solution vector ``x``. The result will be a vector backed by internal solver data and will be invalidated if
the solver is destroyed. Copy it if desired.

!!! warning
    This method may only be called once. All further calls will give wrong output, as the internal solver data is modified.
"""
function get_Xlin(solver::Solver)
    v = unsafe_load(solver.var.vLp)
    vWrap = unsafe_wrap(Array, v.matElem, v.nLpCols, own=false)
    u = unsafe_load(solver.var.uLp)
    @assert u.nLpCols == v.nLpCols
    uWrap = unsafe_wrap(Array, u.matElem, u.nLpCols, own=false)
    vWrap .= .25 .* (uWrap .+ vWrap) .^ 2
    return vWrap
end

"""
    get_S(solver, i)

Returns the `i`th slack variable for the PSD solution matrix ``S_i``. The result will be a freshly allocated symmetric view of
a dense matrix.
"""
function get_S(solver::Solver, i::Integer)
    cone = unsafe_load(unsafe_load(solver.SDPCones, i))
    # There's no dual data available, we have to construct it ourselves
    dim = unsafe_load(cone.sdp_slack_var).nSDPCol
    negLambd = unsafe_wrap(Array, solver.var.dualVar, solver.nRows, own=false)
    rmul!(negLambd, -1)
    S = zeros(Cdouble, dim, dim)
    slack_data = Ref(SDPCoeffDense(dim, pointer(S)))
    slack_var = Ref(SDPCoeff(dim, SDP_COEFF_DENSE, Base.unsafe_convert(Ptr{SDPCoeffDense}, slack_data)))
    slack_ptr = Base.unsafe_convert(Ptr{SDPCoeff}, slack_var)
    GC.@preserve slack_data slack_var begin
        @ccall $(cone.addObjCoeff)(cone.coneData::Ptr{Cvoid}, slack_ptr::Ptr{SDPCoeff})::Cvoid
        @ccall $(cone.sdpDataWSum)(cone.coneData::Ptr{Cvoid}, negLambd::Ptr{Cdouble}, slack_ptr::Ptr{SDPCoeff})::Cvoid
    end
    rmul!(negLambd, -1)
    rmul!(S, inv(solver.scaleObjHis))
    return Symmetric(S, :L)
end

"""
    get_Slin(solver[, i::AbstractUnitRange])

Returns the slack variables to the nonnegative variables of index `i` (by default, all). The result will be a freshly allocated
vector.
"""
function get_Slin(solver::Solver, i::AbstractUnitRange=1:solver.nLpCols)
    cone = unsafe_load(solver.lpCone)
    # There's no dual data available, we have to construct it ourselves
    s = Vector{Cdouble}(undef, length(i))
    unsafe_copyto!(pointer(s), unsafe_load(cone.coneData).objMatElem + (first(i) -1) * sizeof(Cdouble), length(s))
    # ^ so that we don't have to call objCoeffSum in a loop
    negLambd = unsafe_wrap(Array, solver.var.dualVar, solver.nRows, own=false)
    rmul!(negLambd, -1)
    for idx in i
        ccall(cone.lpDataWSum, Cvoid, (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, LoRADSInt), cone.coneData, negLambd, s, idx -1)
    end
    rmul!(negLambd, -1)
    return rmul!(s, inv(solver.scaleObjHis))
end

end