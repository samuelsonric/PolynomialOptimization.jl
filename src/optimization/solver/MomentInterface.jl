export add_constr_nonnegative!, add_constr_quadratic!, add_constr_psd!, add_constr_psd_complex!,
    add_constr_fix_prepare!, add_constr_fix!, add_constr_fix_finalize!, fix_objective!

function add_constr_nonnegative! end

"""
    add_constr_nonnegative!(state, indvals::Indvals)

Add a nonnegative constraint to the solver that contains the decision variables (columns in the linear constraint matrix)
indexed according to `indvals`.
Falls back to the vector-valued version if not implemented.

See also [`Indvals`](@ref).
"""
add_constr_nonnegative!(state, indvals::Indvals) =
    add_constr_nonnegative!(state, IndvalsIterator(indvals.indices, indvals.values, StackVec(length(indvals))))

"""
    add_constr_nonnegative!(state, indvals::IndvalsIterator)

Adds multiple nonnegative constraints to the solver that contain the decision variables (columns in the linear constraint
matrix) indices according to the entries in `indvals`.
Falls back to calling the scalar-valued version multiple times if not implemented.

See also [`IndvalsIterator`](@ref).
"""
function add_constr_nonnegative!(state, iv::IndvalsIterator)
    for indvals in iv
        add_constr_nonnegative!(state, indvals)
    end
    return
end

function add_constr_quadratic! end

@doc raw"""
    add_constr_quadratic!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds a (rotated) quadratic constraint to the `N = length(indvals)` linear combinations of decision variables (columns in the
linear constraint matrix) indexed according to the `indvals`. This will read (where ``X_i`` is
``\mathit{indvals}_i.\mathit{values} \cdot x_{\mathit{indvals}_i.\mathit{indices}}``) ``X_1, X_2 \geq 0``,
``2X_1 X_2 \geq \sum_{i = 3}^N X_i^2`` if the solver supports the rotated quadratic cone, or ``X_1 \geq 0``,
``X_1^2 \geq \sum_{i = 2}^N X_i^2`` if it only supports the standard quadratic cone.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. If the scaled
    diagonally dominant representation is requested, `indvals` can have any length.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) is defined not return
    [`SOLVER_QUADRATIC_NONE`](@ref SolverQuadratic) for the given state.
    If it does, a fallback to a 2x2 PSD constraint is used.
"""
add_constr_quadratic!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_constr_psd! end

"""
    add_constr_psd!(state, dim::Integer, data::PSDMatrixCartesian{T,V}) where {T,V<:Real}

Add a PSD constraint of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the return value
of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear constraint
matrix) indicated by the keys when iterating through `data`, which are of the type returned by [`mindex`](@ref), at positions
and with coefficients given by their values.
Note that if [`add_constr_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_constr_psd_complex!`](@ref) must be implemented.
"""
add_constr_psd!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Real})

"""
    add_constr_psd!(state, dim::Integer, data::IndvalsIterator{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_constr_psd_complex!`](@ref) must be implemented.
"""
add_constr_psd!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

function add_constr_psd_complex! end

"""
    add_constr_psd_complex!(state, dim::Int,
        data::PSDMatrixCartesian{T,V}) where {T,V<:Complex}

Add a Hermitian PSD constraint of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the
return value of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear
constraint matrix) indicated by the keys when iterating through `data`, which are of the type returned by [`mindex`](@ref),
at positions and with coefficients given by their values. The real part of the coefficient corresponds to the coefficient in
front of the real part of the matrix entry, the imaginary part is the coefficient for the imaginary part of the matrix entry.
Note that if [`add_constr_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_constr_psd_complex!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Complex})

"""
    add_constr_psd_complex!(state, dim::Int, data::IndvalsIterator{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_constr_psd_complex!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

"""
    add_constr_fix_prepare!(state, num::Int)

Prepares to add exactly `num` constraints that are fixed to a certain value; the actual data is then put into the solver by
subsequent calls of [`add_constr_fix!`](@ref) and the whole transaction is completed by [`add_constr_fix_finalize!`](@ref).
The return value of this function is passed on as `constrstate` to [`add_constr_fix!`](@ref).
The default implementation does nothing.
"""
add_constr_fix_prepare!(_, _) = nothing

"""
    add_constr_fix!(state, constrstate, indvals::Indvals, rhs::V) where {T,V<:Real}

Add a constraint fixed to `rhs` to the solver that is composed of all variables (columns in the linear constraint matrix)
indexed according to `indvals`.
The parameter `constrstate` is, upon first call, the value returned by [`add_constr_fix_prepare!`](@ref); and on all further
calls, it will be the return value of the previous call.
Note that `rhs` will almost always be zero, so if the right-hand side is represented by a sparse vector, it is worth checking
for this value (the compiler will be able to remove the check).

See also [`Indvals`](@ref).
"""
function add_constr_fix! end

"""
    add_constr_fix_finalize!(state, constrstate)

Finishes the addition of fixed constraints to `state`; the value of `constrstate` is the return value of the last call to
[`add_constr_fix!`](@ref).
The default implementation does nothing.
"""
add_constr_fix_finalize!(_, _) = nothing

"""
    fix_objective!(state, indvals::Indvals)

Puts the variables indexed according to `indvals` into the objective (that is to be minimized).
This function will be called exactly once by [`moment_setup!`](@ref) after all variables and constraints have been set up.

See also [`Indvals`](@ref).
"""
function fix_objective! end