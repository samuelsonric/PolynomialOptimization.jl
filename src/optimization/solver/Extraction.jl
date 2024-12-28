export extract_moments, extract_info, extract_sos_prepare, extract_sos

"""
    extract_moments(relaxation::AbstractRelaxation, state)

Extracts a [`MomentVector`](@ref) from a solved relaxation. The `state` parameter is the first return value of the
[`poly_optimize`](@ref) call by the solver. This function is only called once for each result; the output is cached.
"""
function extract_moments end

extract_moments(relaxation::AbstractRelaxation, ::Missing) = MomentVector(relaxation, missing)

"""
    extract_info(state)

Returns the internal information on the problem that was given back by [`moment_setup!`](@ref) or [`sos_setup!`](@ref). There
is a default implementation that returns the value of the property `info` on `state` (which also works if the original state is
wrapped as the first element of a tuple).
"""
function extract_info end

extract_info(state::Tuple{AbstractSolver,Vararg}) = state[1].info
extract_info(state::AbstractSolver) = state.info
extract_info(::Missing) =
    throw(ArgumentError("The optimization did not run successfully; no solution information is available"))

"""
    extract_sos_prepare(relaxation::AbstractRelaxation, state)

Prepares for one or multiple calls of [`extract_sos`](@ref). The return value will be passed to the function as an argument.
This is particularly relevant for solvers which don't allow the extraction of subsets of the data, but only the whole vector:
Retrieve it here, then pass it as an output.
The default implementation does nothing.
"""
extract_sos_prepare(::AbstractRelaxation, _) = nothing

"""
    extract_sos(relaxation::AbstractRelaxation, state, ::Val{type},
        index::Union{<:Integer,<:AbstractUnitRange}, rawstate) where {type}

Extracts data that contains the raw solver information about the SOS data contained in the result. For moment optimizations,
this corresponds to the dual data (except for the type `slack`, which is primal information); for SOS optimizations, this is
the primal data (except for the type `slack`, which is dual information). `rawstate` is the return value of the preceding call
to [`extract_sos_prepare`](@ref) (by default, `nothing`). Note that the SOS data may be queried in any order, partially or
completely.

The parameters `type` and `index` indicates which constraint/variable the data corresponds to. `type` is a symbol, `index` is
the index or range of indices within constraints of the same type.

The return value of this function should be a scalar, vector, or matrix, depending on what data was requested. The following
relations should hold:

| `type`               | `index`   | result type              |
| -------------------: | :-------: | :----------------------- |
| `:fix` (moment only) | `Int`     | vector                   |
| `:free` (SOS only)   | `Int`     | vector                   |
| `:nonnegative`       | `Int`     | vector                   |
| `:quadratic`         | `Int`     | vector                   |
| `:rotated_quadratic` | `Int`     | vector                   |
| `:psd`               | `Int`     | vector[^5] or matrix     |
| `:psd_complex`       | `Int`     | vector[^5][^6] or matrix |
| `:dd`                | `Int`     | vector[^5] or matrix     |
| `:dd_complex`        | `Int`     | vector[^5][^6] or matrix |
| `:lnorm`             | `Int`     | vector                   |
| `:lnorm_complex`     | `Int`     | vector[^6]               |
| `:sdd`               | `Int`     | vector                   |
| `:sdd_complex`       | `Int`     | vector[^6]               |
| `:slack`             | range[^4] | vector                   |

Note that to obtain the slacks, for implementations of [`AbstractSparseMatrixSolver`](@ref), the helper function
[`get_slack`](@ref) is useful.

This table is based on the default implementation of [`addtocounter!`](@ref), whose return type is actually what enters into
the `index` parameter. If this is overwritten for some types, then the type of `index` may differ.

See also [`@counter_is_scalar`](@ref).

[^4]: It is guaranteed that during the construction of the problem, the corresponding `add...` functions for this cone were all
      called consecutively, with no other cones interspersed. For nonnegative variables or constraints, regardless if they were
      called with the vector version or one-by-one, each constraint/variable will be exactly one entry, even if the solver
      might internally subsume consecutive nonnegative variables in a single cone.
[^5]: If the return type is a vector, [`psd_indextype`](@ref) should be defined on `state`, and it must return a
      [`PSDIndextypeVector`](@ref). However, the scaling operation on the off-diagonals is now inverted: To go from the vector
      of the triangle to the full matrix, off-diagonals must be scaled by ``\\frac{1}{\\sqrt2}``.
[^6]: Complex values can be treated either by returning a vector of `Complex` element type, or by returning a real-valued
      vector where the diagonals (PSD/DD/SDD)/first elements (``\\ell``-norm) have a single entry and off-diagonals two.
"""
function extract_sos end