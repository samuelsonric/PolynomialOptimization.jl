@enum Retcode begin
    RETCODE_OK = 0
    RETCODE_TIME_OUT = 1
    RETCODE_EXIT = 3
    RETCODE_NUM_ERR = 4
    RETCODE_BAD_ITER = 8
end

@enum Status begin
    STATUS_UNKNOWN
    STATUS_PRIMAL_DUAL_OPTIMAL
    STATUS_PRIMAL_OPTIMAL
    STATUS_MAXITER
    STATUS_TIME_LIMIT
end

"""
    ConeType

The type of the current cone. Only `CONETYPE_DENSE_SDP` and `CONETYPE_SPARSE_SDP` are implemented.
"""
@enum ConeType begin
    CONETYPE_UNKNOWN # not implement
    CONETYPE_LP # A' * y <= c */ // isolated into lp_cone
    CONETYPE_BOUND # y <= u
    CONETYPE_SCALAR_BOUND
    CONETYPE_DENSE_SDP
    CONETYPE_SPARSE_SDP
end

@enum VecType VECTYPE_DENSE VECTYPE_SPARSE

@enum SDPCoeffType SDP_COEFF_ZERO SDP_COEFF_SPARSE SDP_COEFF_DENSE