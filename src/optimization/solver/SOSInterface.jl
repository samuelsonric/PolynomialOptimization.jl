export add_var_nonnegative!, add_var_rotated_quadratic!, add_var_quadratic!, add_var_l1!, add_var_l1_complex!, add_var_psd!,
    add_var_psd_complex!, add_var_free_prepare!, add_var_free!, add_var_free_finalize!, fix_constraints!, add_constr_slack!

function add_var_nonnegative! end

"""
    add_var_nonnegative!(state, indvals::Indvals)

Add a nonnegative decision variable to the solver and put its value into the linear constraints (rows in the linear constraint
matrix) indexed according to `indvals`.
Falls back to the vector-valued version if not implemented.

See also [`Indvals`](@ref).
"""
add_var_nonnegative!(state, indvals::Indvals) =
    add_var_nonnegative!(state, IndvalsIterator(indvals.indices, indvals.values, StackVec(length(indvals))))

"""
    add_var_nonnegative!(state, indvals::IndvalsIterator)

Add multiple nonnegative decision variables to the solver and put their values into the linear constraints (rows in the linear
constraint matrix) indexed according to the entries in `indvals`.
Falls back to calling the scalar-valued version multiple times if not implemented.

See also [`IndvalsIterator`](@ref).
"""
function add_var_nonnegative!(state, iv::IndvalsIterator)
    for indvals in iv
        add_var_nonnegative!(state, indvals)
    end
    return
end

function add_var_quadratic! end

@doc raw"""
    add_var_quadratic!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds decision variables in a quadratic cone to the solver and put their values into the linear constraints (rows in the linear
constraint matrix), indexed according to `indvals`. The `N = length(indvals)` variables will satisfy ``x_1 \geq 0``,
``x_1^2 \geq \sum_{i = 2}^N x_i^2``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. If the scaled
    diagonally dominant representation is requested, `indvals` can have any length.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) returns `true` for the given state.
    If (rotated) quadratic constraints are unsupported, a fallback to a 2x2 PSD variable is used.
"""
add_var_quadratic!(::Any, ::IndvalsIterator{<:Any,Real})

function add_var_rotated_quadratic! end

@doc raw"""
    add_var_rotated_quadratic!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds decision variables in a rotated quadratic cone to the solver and put their values into the linear constraints (rows in
the linear constraint matrix), indexed according to `indvals`. The `N = length(indvals)` variables will satisfy
``x_1, x_2 \geq 0``, ``2x_1 x_2 \geq \sum_{i = 3}^N x_i^2``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. If the scaled
    diagonally dominant representation is requested, `indvals` can have any length.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) returns `true` for the given state.
    If (rotated) quadratic constraints are unsupported, a fallback to a 2x2 PSD variable is used.
"""
add_var_rotated_quadratic!(::Any, ::IndvalsIterator{<:Any,Real})

function add_var_psd! end

"""
    add_var_psd!(state, dim::Int, data::PSDMatrixCartesian{T,V}) where {T,V<:Real}

Add a PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the return value
of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear constraint
matrix) indicated by the keys when iterating through `data`, which are of the type returned by [`mindex`](@ref), at positions
and with coefficients given by their values.
Note that if [`add_var_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_var_psd_complex!`](@ref) must be implemented.
"""
add_var_psd!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Real})

"""
    add_var_psd!(state, dim::Int, data::IndvalsIterator{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_var_psd_complex!`](@ref) must be implemented.
"""
add_var_psd!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

function add_var_psd_complex! end

"""
    add_var_psd_complex!(state, dim::Int, data::PSDMatrixCartesian) where {T,V<:Complex}

Add a Hermitian PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the
return value of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear
constraint matrix) indicated by the keys when iterating through `data`, which are of the type returned by [`mindex`](@ref),
at positions and with coefficients given by their values. The real part of the coefficient corresponds to the coefficient in
front of the real part of the matrix entry, the imaginary part is the coefficient for the imaginary part of the matrix entry.
Note that if [`add_var_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_var_psd_complex!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Complex})

"""
    add_var_psd_complex!(state, dim::Int, data::IndvalsIterator{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_var_psd_complex!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

function add_var_l1! end

@doc raw"""
    add_var_l1!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds decision variables in a ℓ₁ norm cone to the solver and put their values into the linear constraints (rows in the linear
constraint matrix), indexed according to the `indvals`. The `N = length(indvals)` variables will satisfy
``x_1 \geq \sum_{i = 2}^N \lvert x_i\rvert``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! warning
    This function will only be called if [`supports_l1`](@ref) returns `true` for the given state.
    If ℓ₁ norm cones are unsupported, a fallback to multiple nonnegative variables with slack constraints will be
    used (see [`add_constr_slack!`](@ref)).
"""
add_var_l1!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_var_l1_complex! end

@doc raw"""
    add_var_l1_complex!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Same as [`add_var_l1!`](@ref), but now two successive items in `indvals` are interpreted as determining the real and
imaginary part of a component of the ℓ₁ norm variable.

!!! warning
    This function will only be called if [`supports_complex_l1`](@ref) returns `true` for the given state.
    If complex-valued ℓ₁ norm cones are unsupported, a fallback to multiple nonnegative and quadratic variables with slack
    constraints and will be used (see [`add_constr_slack!`](@ref)).
"""
add_var_l1_complex!(::Any, ::IndvalsIterator{<:Any,<:Real})

"""
    add_var_free_prepare!(state, num::Int)

Prepares to add exactly `num` free variables that may become part of the objective; the actual data is then put into the solver
by subsequent calls of [`add_var_free!`](@ref) and the whole transaction is completed by [`add_var_free_finalize!`](@ref).
The return value of this function is passed on as `eqstate` to [`add_var_free!`](@ref).
The default implementation does nothing.
"""
add_var_free_prepare!(_, _) = nothing

"""
    add_var_free!(state, eqstate, indvals::Indvals, obj::V) where {T,V<:Real}

Add a free variable to the solver and put its value into the linear constraints (rows in the linear constraint matrix), indexed
according to `indvals`.
The variable should also be put into the objective with coefficient `obj` (which is likely to be zero).
The parameter `eqstate` is, upon first call, the value returned by [`add_var_free_prepare!`](@ref); and on all further calls,
it will be the return value of the previous call.

See also [`Indvals`](@ref).
"""
function add_var_free! end

"""
    add_var_free_finalize!(state, eqstate)

Finishes the addition of free variables to `state`; the value of `eqstate` is the return value of the last call to
[`add_var_free!`](@ref).
The default implementation does nothing.
"""
add_var_free_finalize!(_, _) = nothing

"""
    fix_constraints!(state, indvals::Indvals)

Ensures that all constraints in the optimization problem are fixed to the values according to `indvals`.
This function will be called exactly once by [`sos_setup!`](@ref) after all variables and constraints have been set up.

See also [`Indvals`](@ref).
"""
function fix_constraints! end

"""
    add_constr_slack!(state, num::Int)

Creates `num` linear fix-to-zero slack constraints in the problem (i.e., constraints that do not correspond to moments). The
result should be an abstract vector (typically a unit range) that contains the indices of all created slack constraints. The
indices should be of the same type as [`mindex`](@ref).
"""
function add_constr_slack! end