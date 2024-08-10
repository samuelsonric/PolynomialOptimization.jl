export sos_add_matrix!, sos_add_equality!, sos_setup!

struct SOSWrapper{S}
    state::S
end

mindex(state::SOSWrapper, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc} = mindex(state.state, monomials...)

supports_rotated_quadratic(state::SOSWrapper) = supports_rotated_quadratic(state.state)
supports_quadratic(state::SOSWrapper) = supports_quadratic(state.state)
supports_lnorm(state::SOSWrapper) = supports_lnorm(state.state)
supports_lnorm_complex(state::SOSWrapper) = supports_lnorm_complex(state.state)
supports_psd_complex(state::SOSWrapper) = supports_psd_complex(state.state)
supports_dd(state::SOSWrapper) = supports_dd(state.state)
supports_dd_complex(state::SOSWrapper) = supports_dd_complex(state.state)
psd_indextype(state::SOSWrapper) = psd_indextype(state.state)

# both necessary for disambiguation
add_constr_nonnegative!(state::SOSWrapper, indvals::Indvals) = add_var_nonnegative!(state.state, indvals)
add_constr_nonnegative!(state::SOSWrapper, indvals::IndvalsIterator) = add_var_nonnegative!(state.state, indvals)
add_constr_rotated_quadratic!(state::SOSWrapper, indvals::IndvalsIterator) = add_var_rotated_quadratic!(state.state, indvals)
add_constr_quadratic!(state::SOSWrapper, indvals::IndvalsIterator) = add_var_quadratic!(state.state, indvals)
add_constr_psd!(state::SOSWrapper, dim::Int, data) = add_var_psd!(state.state, dim, data)
add_constr_psd_complex!(state::SOSWrapper, dim::Int, data) = add_var_psd_complex!(state.state, dim, data)
add_constr_linf!(state::SOSWrapper, indvals::IndvalsIterator) = add_var_l1!(state.state, indvals)
add_constr_linf_complex!(state::SOSWrapper, indvals::IndvalsIterator) = add_var_l1_complex!(state.state, indvals)
add_constr_dddual!(state::SOSWrapper, dim::Int, indvals::IndvalsIterator) = add_var_dd!(state.state, dim, indvals)
add_constr_dddual_complex!(state::SOSWrapper, dim::Int, indvals::IndvalsIterator) =
    add_var_dd_complex!(state.state, dim, indvals)

add_constr_fix_prepare!(state::SOSWrapper, num::Int) = add_var_free_prepare!(state.state, num)
add_constr_fix!(state::SOSWrapper, args...) = add_var_free!(state.state, args...)
add_constr_fix_finalize!(state::SOSWrapper, constrstate) = add_var_free_finalize!(state.state, constrstate)

add_var_slack!(state::SOSWrapper, num::Int) = add_constr_slack!(state.state, num)

fix_objective!(state::SOSWrapper, indvals::Indvals) = fix_constraints!(state.state, indvals)

"""
    sos_add_matrix!(state, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}},
        representation::RepresentationMethod=RepresentationPSD())

Parses a SOS constraint with a basis given in `grouping` (this might also be a partial basis due to sparsity), premultiplied by
`constraint` (which may be the unit polynomial for the SOS cone membership) and calls the appropriate solver functions to set
up the problem structure according to `representation`.

To make this function work for a solver, implement the following low-level primitives:
- [`mindex`](@ref)
- [`add_var_nonnegative!`](@ref)
- [`add_var_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_var_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_var_l1!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_var_l1_complex!`](@ref) (optional, then set [`supports_lnorm_complex`](@ref) to `true`)
- [`add_var_psd!`](@ref)
- [`add_var_psd_complex!`](@ref) (optional, then set [`supports_psd_complex`](@ref) to `true`)
- [`add_var_dd!`](@ref) (optional, then set [`supports_dd`](@ref) to `true`)
- [`add_var_dd_complex!`](@ref) (optional, then set [`supports_dd_complex`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_constr_slack!`](@ref)

Usually, this function does not have to be called explicitly; use [`sos_setup!`](@ref) instead.

See also [`sos_add_equality!`](@ref).
"""
sos_add_matrix!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial}, constraint::SimplePolynomial,
    representation::RepresentationMethod=RepresentationPSD()) =
    moment_add_matrix!(SOSWrapper(state), grouping, constraint, representation)

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
    sos_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings[; representation])

Sets up all the necessary SOS matrices, free variables, objective, and constraints of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`mindex`](@ref)
- [`add_var_nonnegative!`](@ref)
- [`add_var_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_var_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_var_l1!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_var_l1_complex!`](@ref) (optional, then set [`supports_lnorm_complex`](@ref) to `true`)
- [`add_var_psd!`](@ref)
- [`add_var_psd_complex!`](@ref) (optional, then set [`supports_psd_complex`](@ref) to `true`)
- [`add_var_dd!`](@ref) (optional, then set [`supports_dd`](@ref) to `true`)
- [`add_var_dd_complex!`](@ref) (optional, then set [`supports_dd_complex`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_var_free_prepare!`](@ref) (optional)
- [`add_var_free!`](@ref)
- [`add_var_free_finalize!`](@ref) (optional)
- [`fix_constraints!`](@ref)
- [`add_constr_slack!`](@ref)

!!! warning "Indices"
    The constraint indices used in all solver functions directly correspond to the indices given back by [`mindex`](@ref).
    However, in a sparse problem there may be far fewer indices present; therefore, when the problem is finally given to the
    solver, care must be taken to eliminate all unused indices. The functionality provided by [`AbstractAPISolver`](@ref) and
    [`AbstractSparseMatrixSolver`](@ref) already takes care of this.

!!! info "Order"
    This function is guaranteed to set up the free variables first, then followed by all the others. However, the order of
    nonnegative, quadratic, ``\\ell_1`` norm, and PSD variables is undefined (depends on the problem).

!!! info "Representation"
    This function may also be used to describe simplified cones such as the (scaled) diagonally dominant one. The
    `representation` parameter can be used to define a representation that is employed for the individual groupings. This may
    either be an instance of a [`RepresentationMethod`](@ref) - which requires the method to be independent of the dimension of
    the grouping - or a callable. In the latter case, it will be passed as a first parameter an identifier[^3] of the current
    conic variable, and as a second parameter the side dimension of its matrix. The method must then return a
    [`RepresentationMethod`](@ref) instance.

See also [`moment_setup!`](@ref), [`sos_add_matrix!`](@ref), [`sos_add_equality!`](@ref),
[`RepresentationMethod`](@ref).
"""
sos_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings;
    representation::Union{<:RepresentationMethod,<:Base.Callable}=RepresentationPSD()) =
    moment_setup!(SOSWrapper(state), relaxation, groupings; representation)