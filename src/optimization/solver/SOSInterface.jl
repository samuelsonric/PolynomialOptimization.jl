export add_var_nonnegative!, add_var_quadratic!, add_var_psd!, add_var_psd_complex!,
    add_var_free_prepare!, add_var_free!, add_var_free_finalize!, fix_constraints!

function add_var_nonnegative! end

"""
    add_var_nonnegative!(state, indices::AbstractVector{T},
        values::AbstractVector{V}) where {T,V<:Real}

Add a nonnegative decision variable to the solver and put its value into the linear constraints indexed by `indices` (rows in
the linear constraint matrix), whose eltype is of the type returned by [`mindex`](@ref), with coefficients `values`.
"""
add_var_nonnegative!(::Any, ::AbstractVector{T}, ::AbstractVector{V}) where {T,V<:Real}

function add_var_quadratic! end

@doc raw"""
    add_var_quadratic!(state, indvals::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Adds decision variables in a rotated quadratic cone to the solver and put their values into the linear constraints indexed by
the indices (rows in the linear constraint matrix), whose eltypes are of the type returned by [`mindex`](@ref), with
coefficients given by the values. The variables will satisfy ``2x_1 x_2 \geq \sum_{i = 3} x_i^2``.

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. No other lengths
    will occur.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) is defined to return `true` for the given state.
    If not, a fallback to a 2x2 PSD constraint is used.
"""
add_var_quadratic!(::Any, ::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

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
    add_var_psd!(state, dim::Int, data::PSDVector{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of constraint indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their coefficients.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_var_psd_complex!`](@ref) must be implemented.
"""
add_var_psd!(::Any, ::Int, ::PSDVector{<:Any,<:Real})

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
    add_var_psd_complex!(state, dim::Int, data::PSDVector{T,V}) where {T,V<:Real}

Conceptually the same as above; but now, `data` is an iterable through the elements of the PSD variable one-by-one. The
individual entries are 2-Tuples of an `AbstractVector` of constraint indices, which are of the type returned by
[`mindex`](@ref), and an `AbstractVector` of their weights.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_var_psd_complex!(::Any, ::Int, ::PSDVector{T,V}) where {T,V<:Real}

"""
    add_var_free_prepare!(state, num::Int)

Prepares to add exactly `num` free variables that may become part of the objective; the actual data is then put into the solver
by subsequent calls of [`add_var_free!`](@ref) and the whole transaction is completed by [`add_var_free_finalize!`](@ref).
The return value of this function is passed on as `eqstate` to [`add_var_free!`](@ref).
The default implementation does nothing.
"""
add_var_free_prepare!(_, _) = nothing

"""
    add_var_free!(state, eqstate, indices::AbstractVector{T}, values::AbstractVector{V},
        obj::V) where {T,V<:Real}

Add a free variable to the solver and put its value into the linear constraints indexed by `indices` (rows in the linear
constraint matrix), whose eltype is of the type returned by [`mindex`](@ref), and with coefficients `values`.
The variable should also be put into the objective with coefficient `obj` (which is likely to be zero).
The parameter `eqstate` is, upon first call, the value returned by [`add_var_free_prepare!`](@ref); and on all further calls,
it will be the return value of the previous call.
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
    fix_constraints!(state, indices::Vector{T}, values::Vector{V}) where {T,V<:Real}

Ensures that all constraints in the optimization problem that appear in `indices` are fixed to their entries in `values`, while
all the rest is fixed to zero. The eltype of `indices` if the one returned by [`mindex`](@ref).
This function will be called exactly once by [`sos_setup!`](@ref) after all variables and constraints have been set up.
"""
function fix_constraints! end