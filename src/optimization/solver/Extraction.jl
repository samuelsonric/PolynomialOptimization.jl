
"""
    extract_moments(relaxation::AbstractRelaxation, state)

Extracts a [`MomentVector`](@ref) from a solved relaxation. The `state` parameter is the first return value of the
[`poly_optimize`](@ref) call by the solver. This function is only called once for each result; the output is cached.
"""
function extract_moments end

extract_moments(relaxation::AbstractRelaxation, ::Missing) = MomentVector(relaxation, missing)
