export add_constr_nonnegative!, add_constr_quadratic!, add_constr_psd!, add_constr_psd_complex!,
    add_constr_fix_prepare!, add_constr_fix!, add_constr_fix_finalize!, fix_objective!

function add_constr_nonnegative! end

"""
    add_constr_nonnegative!(state, indices::AbstractVector{T},
        values::AbstractVector{V}) where {T,V<:Real}

Add a nonnegative constraint to the solver that contains the decision variables indexed by `indices` (columns in the linear
constraint matrix), whose eltype is of the type returned by [`mindex`](@ref), with coefficients `values`.
"""
add_constr_nonnegative!(::Any, ::AbstractVector{T}, ::AbstractVector{V}) where {T,V<:Real}

function add_constr_quadratic! end

@doc raw"""
    add_constr_quadratic!(state, indvals::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Adds a (rotated) quadratic constraint to the three linear combinations of decision variables indexed by the indices (columns in
the linear constraint matrix), whose eltypes are of the type returned by [`mindex`](@ref), with coefficients given by the
values. This will read (where ``X_i`` is ``\mathit{indvals}_{i, 2} \cdot x_{\mathit{indvals}_{i, 1}}``) ``X_1, X_2 \geq 0``,
``2X_1 X_2 \geq \sum_{i = 3} X_i^2``.

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. No other lengths
    will occur.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) is defined to return `true` for the given state.
    If not, a fallback to a 2x2 PSD constraint is used.
"""
add_constr_quadratic!(::Any, ::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

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
    add_constr_psd!(state, dim::Integer, data::PSDVector{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of decision variable indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their coefficients.
Note that if [`add_constr_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_constr_psd_complex!`](@ref) must be implemented.
"""
add_constr_psd!(::Any, ::Int, ::PSDVector{<:Any,<:Real})

function add_constr_psd_complex! end

"""
    add_constr_psd_complex!(state, dim::Int, data::PSDMatrixCartesian{T,V}) where {T,V<:Complex}

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
    add_constr_psd_complex!(state, dim::Int, data::PSDVector{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of decision variable indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their weights.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_constr_psd_complex!(::Any, ::Int, ::PSDVector{<:Any,<:Real})

"""
    add_constr_fix_prepare!(state, num::Int)

Prepares to add exactly `num` constraints that are fixed to a certain value; the actual data is then put into the solver by
subsequent calls of [`add_constr_fix!`](@ref) and the whole transaction is completed by [`add_constr_fix_finalize!`](@ref).
The return value of this function is passed on as `constrstate` to [`add_constr_fix!`](@ref).
The default implementation does nothing.
"""
add_constr_fix_prepare!(_, _) = nothing

"""
    add_constr_fix!(state, constrstate, indices::AbstractVector{T},
        values::AbstractVector{V}, rhs::V) where {T,V<:Real}

Add a constraint fixed to `rhs` to the solver that is composed of all variables given by indices (columns in the linear
constraint matrix) `indices`, whose eltype is of the type returned by [`mindex`](@ref), and with coefficients `values`.
The parameter `constrstate` is, upon first call, the value returned by [`add_constr_fix_prepare!`](@ref); and on all further
calls, it will be the return value of the previous call.
Note that `rhs` will almost always be zero, so if the right-hand side is represented by a sparse vector, it is worth checking
for this value (the compiler will be able to remove the check).
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
    fix_objective!(state, indices::Vector{T}, values::Vector{V}) where {T,V<:Real}

Puts the variables indexed by `indices` into the objective (that is to be minimized) with coefficients `values`.
This function will be called exactly once by [`moment_setup!`](@ref) after all variables and constraints have been set up.
"""
function fix_objective! end