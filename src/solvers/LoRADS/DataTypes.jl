@kwdef mutable struct Params
    fname::Ptr{Cchar} = C_NULL
    initRho::Cdouble = 0.0
    rhoMax::Cdouble = 5000.0
    rhoCellingALM::Cdouble = 1e8
    rhoCellingADMM::Cdouble = 5000.0 * 200
    maxALMIter::LoRADSInt = 200
    maxADMMIter::LoRADSInt = 10_000
    timesLogRank::Cdouble = 2.0
    rhoFreq::LoRADSInt = 5
    rhoFactor::Cdouble = 1.2
    ALMRhoFactor::Cdouble = 2.0
    phase1Tol::Cdouble = 1e-3
    phase2Tol::Cdouble = 1e-5
    timeSecLimit::Cdouble = 3600.0
    heuristicFactor::Cdouble = 1.0
    lbfgsListLength::LoRADSInt = 2
    endTauTol::Cdouble = 1e-16
    endALMSubTol::Cdouble = 1e-10
    l2Rescaling::Bool = false
    reoptLevel::LoRADSInt = 2
    dyrankLevel::LoRADSInt = 2
    highAccMode::Bool = false
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::Params) = pointer_from_objref(x)

struct SDPDense
    rank::LoRADSInt
    nRows::LoRADSInt
    matElem::Ptr{Cdouble}
end

struct LpDense
    nLpCols::LoRADSInt
    matElem::Ptr{Cdouble}
end

struct SparseVec
    nnz::LoRADSInt
    nnzIdx::Ptr{LoRADSInt}
    val::Ptr{Cdouble}
end

struct DenseVec
    nnz::LoRADSInt
    val::Ptr{Cdouble}
end

struct Vec
    type::VecType
    data::Ptr{Union{SparseVec,DenseVec}}
    add::Ptr{Cvoid}
    zero::Ptr{Cvoid}
end

struct SDPCoeff
    nSDPCol::LoRADSInt
    dataType::SDPCoeffType
    dataMat::Ptr{Cvoid}
end

struct SDPCoeffZero
    # In a zero matrix. There is nothing but an integer recording matrix dimension
    nSDPCol::LoRADSInt
end

struct SDPCoeffSparse
    # In a sparse matrix, we adopt the triplet format i, j, x
    nSDPCol::LoRADSInt
    nTriMatElem::LoRADSInt
    triMatCol::Ptr{LoRADSInt}
    triMatRow::Ptr{LoRADSInt}
    triMatElem::Ptr{Cdouble}
    # rowCol2NnzIdx::Ptr{Ptr{LoRADSInt}}
    nnzIdx2ResIdx::Ptr{LoRADSInt}
end

struct SDPCoeffDense
    # In a dense matrix, we store an n * (n + 1) / 2 array in packed format
    nSDPCol::LoRADSInt
    dsMatElem::Ptr{Cdouble}
    # rowCol2NnzIdx::Ptr{Ptr{LoRADSInt}}
    fullMat::Ptr{Cdouble} # UVt full matrix, not dataMat full version
end

struct SDPCone
    cone::ConeType
    sdp_coeff_w_sum::Ptr{SDPCoeff} # only two type: sparse and dense
    sdp_obj_sum::Ptr{SDPCoeff} # sdp_coeff data + obj only two type: sparse and dense
    sdp_slack_var::Ptr{SDPCoeff}
    UVt_w_sum::Ptr{SDPCoeff}
    UVt_obj_sum::Ptr{SDPCoeff}

    usrData::Ptr{Cvoid}
    coneData::Ptr{Cvoid}

    sdp_coeff_w_sum_sp_ratio::Cdouble
    nConstr::LoRADSInt
end

mutable struct Variable
    # Variables SDPCone
    U::Ptr{Ptr{SDPDense}} # admm variable, and lbfgs descent direction D
    V::Ptr{Ptr{SDPDense}} # admm variable only
    #ifdef DUAL_U_V
    #   S::Ptr{Ptr{SDPDense}} # admm dual variable
    #endif
    R::Ptr{Ptr{SDPDense}} # average variable for storage and ALM variable
    Grad::Ptr{Ptr{SDPDense}} # grad of R

    # Variable LpCone
    rLp::Ptr{LpDense}
    uLp::Ptr{LpDense}
    vLp::Ptr{LpDense}
    #ifdef DUAL_U_V
    #   sLp::Ptr{LpDense}
    #endif
    gradLp::Ptr{LpDense}

    # ALM lbfgs and ADMM Variables
    dualVar::Ptr{Cdouble}

    # Auxiliary variable
    constrVal::Ptr{Ptr{Vec}}    # constraint violation [iCone][iConstr]
    constrValSum::Ptr{Cdouble}  # constraint violation [iConstr]
    ARDSum::Ptr{Cdouble}        # q1 in line search of ALM
    ADDSum::Ptr{Cdouble}        # q2 in line search of ALM
    bLinSys::Ptr{Ptr{Cdouble}}  # for solving linear system, bInLinSys[iCone]
    rankElem::Ptr{LoRADSInt}    # all rank
    M1temp::Ptr{Cdouble}        # M1 for solving linear system
    bestDualVar::Ptr{Cdouble}
    M2temp::Ptr{Ptr{SDPDense}}  # M2 for solving linear system
    Dtemp::Ptr{Cdouble}
    constrValLp::Ptr{Ptr{Vec}}

    Variable() = new(C_NULL, C_NULL, C_NULL, C_NULL,
                     C_NULL, C_NULL, C_NULL, C_NULL,
                     C_NULL,
                     C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL)
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::Variable) = pointer_from_objref(x)

struct LpConeData
    nRow::LoRADSInt # constraint number
    nCol::LoRADSInt # dim of Lp

    # obj coeff, full
    objMatElem::Ptr{Cdouble}

    # constraint coeff
    rowMatBeg::Ptr{LoRADSInt}
    rowMatIdx::Ptr{LoRADSInt}
    rowMatElem::Ptr{LoRADSInt}

    lpCol::Ptr{Ptr{Cvoid}}
    nrm2Square::Ptr{Cdouble}
end

struct LpCone
    nCol::LoRADSInt
    coneData::Ptr{LpConeData}
    coneObjNrm1::Ptr{Cvoid}
    coneObjNrm2Square::Ptr{Cvoid}
    destroyConeData::Ptr{Cvoid}
    coneView::Ptr{Cvoid}
    coneAUV::Ptr{Cvoid}
    coneAUV2::Ptr{Cvoid}
    objAUV::Ptr{Cvoid}
    coneObjNrmInf::Ptr{Cvoid}
    lpDataWSum::Ptr{Cvoid}
    objCoeffSum::Ptr{Cvoid}
    scalObj::Ptr{Cvoid}
end

struct DIMACError
    data::Ptr{Cdouble}

    DIMACError() = new(C_NULL)
end

function Base.getproperty(d::DIMACError, f::Symbol)
    if f === :constrvio_l1
        return unsafe_load(d.data)
    elseif f === :pdgap
        return unsafe_load(d.data::Ptr{Cdouble}, 2)
    elseif f === :dualfeasible_l1
        return unsafe_load(d.data::Ptr{Cdouble}, 3)
    end
    return getfield(d, f)
end

Base.propertynames(d::Type{DIMACError}) = (:constrvio_l1, :pdgap, :dualfeasible_l1)

mutable struct Solver
    # User data
    nRows::LoRADSInt # constraint number
    rowRHS::Ptr{Cdouble} # b of Ax = b

    # Cones
    nCones::LoRADSInt # sdp cones block number
    SDPCones::Ptr{Ptr{SDPCone}}
    CGLinsys::Ptr{Cvoid}

    # variable
    var::Variable

    # Auxiliary variable LpCone (SDPCone but rank is 1, dim is 1)
    lpCone::Ptr{LpCone}
    nLpCols::LoRADSInt

    # ALM lbfgs Variables
    hisRecT::LoRADSInt
    lbfgsHis::Ptr{Cvoid}   # record difference of primal variable history and gradient history, lbfgsHis[iCone]

    # Monitor
    nIterCount::LoRADSInt
    cgTime::Cdouble
    cgIter::LoRADSInt
    checkSolTimes::LoRADSInt
    #traceSum::Cdouble

    # Convergence criterion
    pObjVal::Cdouble
    dObjVal::Cdouble
    pInfeas::Cdouble
    dInfeas::Cdouble
    cObjNrm1::Cdouble
    cObjNrm2::Cdouble
    cObjNrmInf::Cdouble
    bRHSNrm1::Cdouble
    bRHSNrmInf::Cdouble
    bRHSNrm2::Cdouble
    dimacError::DIMACError
    constrVio::Ptr{Cdouble}

    # Starting time
    dTimeBegin::Cdouble

    # scale
    cScaleFactor::Cdouble
    bScaleFactor::Cdouble

    # check exit bm
    rank_max::Ptr{LoRADSInt}
    sparsitySDPCoeff::Ptr{Cdouble}
    overallSparse::Cdouble
    nnzSDPCoeffSum::LoRADSInt
    SDPCoeffSum::LoRADSInt
    AStatus::Status

    scaleObjHis::Cdouble

    timeSolveStart::Cdouble # own addition

    @doc """
        Solver()

    Creates a new empty LoRADS solver object. This has to be initialized by called [`init_solver`](@ref).
    """
    function Solver()
        result = new(
            zero(LoRADSInt), C_NULL,
            zero(LoRADSInt), C_NULL, C_NULL,
            Variable(),
            C_NULL, zero(LoRADSInt),
            zero(LoRADSInt), C_NULL,
            zero(LoRADSInt), zero(Cdouble), zero(LoRADSInt), zero(LoRADSInt),
            zero(Cdouble), zero(Cdouble), zero(Cdouble), zero(Cdouble), zero(Cdouble), zero(Cdouble), zero(Cdouble),
                zero(Cdouble), zero(Cdouble), zero(Cdouble), DIMACError(), C_NULL,
            zero(Cdouble),
            zero(Cdouble), zero(Cdouble),
            C_NULL, C_NULL, zero(Cdouble), zero(LoRADSInt), zero(LoRADSInt), STATUS_UNKNOWN,
            zero(Cdouble),
            zero(Cdouble)
        )
        finalizer(cleanup, result)
        result
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::Solver) = pointer_from_objref(x)

mutable struct SDPConst
    l_1_norm_c::Cdouble
    l_2_norm_c::Cdouble
    l_inf_norm_c::Cdouble
    l_1_norm_b::Cdouble
    l_2_norm_b::Cdouble
    l_inf_norm_b::Cdouble

    SDPConst() = new()
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::SDPConst) = pointer_from_objref(x)

mutable struct ALMState
    is_rank_updated::Bool
    outerIter::LoRADSInt
    innerIter::LoRADSInt
    rho::Cdouble
    l_inf_primal_infeasibility::Cdouble
    l_1_primal_infeasibility::Cdouble
    l_2_primal_infeasibility::Cdouble
    primal_dual_gap::Cdouble
    primal_objective_value::Cdouble
    dual_objective_value::Cdouble
    l_inf_dual_infeasibility::Cdouble
    l_1_dual_infeasibility::Cdouble
    l_2_dual_infeasibility::Cdouble
    tau::Cdouble

    ALMState() = new()
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::ALMState) = pointer_from_objref(x)

mutable struct ADMMState
    iter::LoRADSInt
    nBlks::LoRADSInt
    cg_iter::LoRADSInt
    rho::Cdouble
    l_1_dual_infeasibility::Cdouble
    l_inf_dual_infeasibility::Cdouble
    l_1_primal_infeasibility::Cdouble
    l_inf_primal_infeasibility::Cdouble
    l_2_primal_infeasibility::Cdouble
    l_2_dual_infeasibility::Cdouble
    primal_objective_value::Cdouble
    dual_objective_value::Cdouble
    primal_dual_gap::Cdouble

    ADMMState() = new()
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::ADMMState) = pointer_from_objref(x)