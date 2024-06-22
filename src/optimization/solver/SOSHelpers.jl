export sos_add_matrix!, sos_add_equality!, sos_setup!

struct SOSWrapper{S}
    state::S
end

mindex(state::SOSWrapper, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc} = mindex(state.state, monomials...)

supports_quadratic(state::SOSWrapper) = supports_quadratic(state.state)
supports_complex_psd(state::SOSWrapper) = supports_complex_psd(state.state)
psd_indextype(state::SOSWrapper) = psd_indextype(state.state)

add_constr_nonnegative!(state::SOSWrapper, args...) = add_var_nonnegative!(state.state, args...)
add_constr_quadratic!(state::SOSWrapper, args...) = add_var_quadratic!(state.state, args...)
add_constr_psd!(state::SOSWrapper, args...) = add_var_psd!(state.state, args...)
add_constr_psd_complex!(state::SOSWrapper, args...) = add_var_psd_complex!(state.state, args...)

add_constr_fix_prepare!(state::SOSWrapper, num::Int) = add_var_free_prepare!(state.state, num)
add_constr_fix!(state::SOSWrapper, args...) = add_var_free!(state.state, args...)
add_constr_fix_finalize!(state::SOSWrapper, constrstate) = add_var_free_finalize!(state.state, constrstate)

fix_objective!(state::SOSWrapper, args...) = fix_constraints!(state.state, args...)

"""
    sos_add_matrix!(state, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}})

Parses a SOS constraint with a basis given in `grouping` (this might also be a partial basis due to sparsity), premultiplied by
`constraint` (which may be the unit polynomial for the SOS cone membership) and calls the appropriate solver functions to set
up the problem structure.

To make this function work for a solver, implement the following low-level primitives:
- [`mindex`](@ref)
- [`add_var_nonnegative!`](@ref)
- [`add_var_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to
  [`SOLVER_QUADRATIC_SOC`](@ref SolverQuadratic) or [`SOLVER_QUADRATIC_RSOC`](@ref SolverQuadratic))
- [`add_var_psd!`](@ref)
- [`add_var_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)

Usually, this function does not have to be called explicitly; use [`sos_setup!`](@ref) instead.

See also [`sos_add_equality!`](@ref).
"""
sos_add_matrix!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial}, constraint::SimplePolynomial) =
    moment_add_matrix!(SOSWrapper(state), grouping, constraint)

"""
    sos_add_equality!(state, grouping::SimpleMonomialVector, constraint::SimplePolynomial)

Parses a polynomial equality constraint for sums-of-squares and calls the appropriate solver functions to set up the problem
structure. `grouping` contains the basis that will be squared in the process to generate the prefactor.

To make this function work for a solver, implement the following low-level primitives:
- [`add_var_free_prepare!`](@ref) (optional)
- [`add_var_free!`](@ref) (required)
- [`add_var_free_finalize!`](@ref) (optional)

Usually, this function does not have to be called explicitly; use [`sos_setup!`](@ref) instead.

See also [`sos_add_matrix!`](@ref).
"""
sos_add_equality!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial}, constraint::SimplePolynomial) =
    moment_add_equality!(SOSWrapper(state), grouping, constraint)

"""
    sos_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings)

Sets up all the necessary SOS matrices, free variables, objective, and constraints of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`mindex`](@ref)
- [`add_var_nonnegative!`](@ref)
- [`add_var_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to
  [`SOLVER_QUADRATIC_SOC`](@ref SolverQuadratic) or [`SOLVER_QUADRATIC_RSOC`](@ref SolverQuadratic))
- [`add_var_psd!`](@ref)
- [`add_var_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_var_free_prepare!`](@ref) (optional)
- [`add_var_free_fix!`](@ref)
- [`add_var_free_finalize!`](@ref) (optional)
- [`fix_constraints!`](@ref)

!!! warning "Indices"
    The constraint indices used in all solver functions directly correspond to the indices given back by [`mindex`](@ref).
    However, in a sparse problem there may be far fewer indices present; therefore, when the problem is finally given to the
    solver, care must be taken to eliminate all unused indices.

!!! info "Order"
    This function is guaranteed to set up the free variables first, then followed by all the others. However, the order of
    nonnegative, quadratic, and PSD variables is undefined (depends on the problem).

See also [`moment_setup!`](@ref), [`sos_add_matrix!`](@ref), [`sos_add_equality!`](@ref).
"""
sos_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings) =
    moment_setup!(SOSWrapper(state), relaxation, groupings)