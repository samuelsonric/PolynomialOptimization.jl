export add_var_nonnegative!, add_var_quadratic!, add_var_psd!, add_var_psd_complex!,
    add_var_free_prepare!, add_var_free!, add_var_free_finalize!, fix_constraints!

function add_var_nonnegative! end

"""
    add_var_nonnegative!(state, indices::AbstractVector{T},
        values::AbstractVector{V}) where {T,V<:Real}

Add a nonnegative decision variable to the solver and put its value into the linear constraints with indices (rows in the
linear constraint matrix) `indices`, whose eltype is of the type returned by [`mindex`](@ref), with coefficients `values`.
"""
add_var_nonnegative!(::Any, ::AbstractVector{T}, ::AbstractVector{V}) where {T,V<:Real}

function add_var_quadratic! end

@doc raw"""
    add_var_quadratic!(state, indices₁::AbstractVector{T}, values₁::AbstractVector{V},
        indices₂::AbstractVector{T}, values₂::AbstractVector{V},
        rest::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Add three decision variables satisfying ``x_1, x_2 \geq 0``, ``2x_1 \cdot x_2 \geq x_3^2``, corresponding to the
semidefinite constraint
``\left(\begin{smallmatrix} x_1 & \frac{x_3}{\sqrt2} \\ \frac{x_3}{\sqrt2} & x_2 \end{smallmatrix}\right) \succeq 0`` to the
solver and put their values into the linear constraints (rows in the linear constraint matrix) indicated by the indices, which
are of the type returned by [`mindex`](@ref), with coefficients given by the values (which are all the same unless the
constraint comes from a PSD matrix).

!!! note "rest vs. x₃"
    In the real-valued case, `rest` is a simple tuple corresponding to `index₃` and `value₃`; however, in the complex-valued
    case, it will be two tuples; the equivalent transformation is
    ``2x_1 \cdot x_2 \geq \operatorname{Re}^2(x_3) + \operatorname{Im}^2(x_3)``, where the decomposition into real and
    imaginary parts is already done - they should be created as different real variables.
    `rest` will always have either one or two elements.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) is defined to return `true` for the given state.
    If not, a fallback to a 2x2 PSD constraint is used.
"""
add_var_quadratic!(::Any, ::AbstractVector{T}, ::AbstractVector{V}, ::AbstractVector{T}, ::AbstractVector{V},
    ::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

"""
    add_var_quadratic!(state, indices₊::AbstractVector{T}, values₊::AbstractVector{V},
        rest_free::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

Special case of a quadratic constraint of side dimension 2 which has been simplified to x₊ ≥ 0, rest_free ∈ ℝ.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) is defined to return `true` for the given state.
    If not, a fallback to a 2x2 PSD constraint is used.
"""
add_var_quadratic!(::Any, ::AbstractVector{T}, ::AbstractVector{V},
    ::Tuple{AbstractVector{T},AbstractVector{V}}...) where {T,V<:Real}

function add_var_psd! end

"""
    add_var_psd!(state, dim::Int,
        data::Dict{FastKey{T},<:Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Real}

Add a PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the return value
of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear constraint
matrix) indicated by the keys of the `data` dictionary, which are of the type returned by [`mindex`](@ref), with coefficients
given by their values (maybe scaled).
Note that if [`add_var_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns an [`AbstractPSDIndextypeMatrix`](@ref).
For [`PSDIndextypeMatrixLinear`](@ref), there is one index in the value-tuple of the dictionary; for
[`PSDIndextypeMatrixCartesian`](@ref), there are two.

!!! hint "FastKey"
    The key of the dictionary actually is a `FastKey{T}`, where `T` is the type returned by [`mindex`](@ref). `FastKey` can be
    converted to `T` and circumvents Julia's long-winded hash calculations for integers.

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_complex_psd`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_var_psd_complex!`](@ref) must be implemented.
"""
add_var_psd!(::Any, ::Int, ::Dict{FastKey{T},<:Tuple{AbstractVector{I},AbstractVector{V}}}) where {T,I,V<:Real}

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
add_var_psd!(::Any, ::Int, ::PSDVector{T,V}) where {T,V<:Real}

function add_var_psd_complex! end

"""
    add_var_psd_complex!(state, dim::Int,
        data::Dict{FastKey{T},Tuple{AbstractVector{I}...,AbstractVector{V}}}) where {T,I,V<:Complex}

Add a Hermitian PSD variable of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the
return value of [`psd_indextype`](@ref)); these elements of the matrix are put into the linear constraints (rows in the linear
constraint matrix) indicated by the keys of the `data` dictionary, which are of the type returned by [`mindex`](@ref), with
coefficients given by their values. The real part of the coefficient corresponds to the coefficient in front of the real part
of the matrix entry, the imaginary part is the coefficient for the imaginary part of the matrix entry.
Note that if [`add_var_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns an [`AbstractPSDIndextypeMatrix`](@ref).
For [`PSDIndextypeMatrixLinear`](@ref), there is one index in the value-tuple of the dictionary; for
[`PSDIndextypeMatrixCartesian`](@ref), there are two.

!!! warning
    This function will only be called if [`supports_complex_psd`](@ref) is defined to return `true` for the given state.
"""
add_var_psd_complex!(::Any, ::Int, ::Dict{FastKey{T},Tuple{AbstractVector{I},AbstractVector{V}}}) where {T,I,V<:Complex}

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

Prepares to add no more than `num` free variables to `state`; the actual data is then put into the solver by subsequent calls
of [`add_var_free!`](@ref) and the whole transaction is completed by [`add_var_free_finalize!`](@ref).
The return value of this function is passed on as `eqstate` to [`add_var_Free!`](@ref).
The default implementation does nothing.
"""
add_var_free_prepare!(_, _) = nothing

"""
    add_var_free!(state, eqstate, indices::AbstractVector{T}, values::AbstractVector{V},
        obj::Bool) where {T,V<:Real}

Add a free variable to the solver and put its value into the linear constraints with indices (rows in the linear constraint
matrix) `indices`, whose eltype is of the type returned by [`mindex`](@ref), and with coefficients `values`.
If `obj` is `true`, the variable should also be put into the objective with coefficient `1`.
The parameter `eqstate` is, upon first call, the value returned by [`add_var_free_prepare!`](@ref); and on all further calls,
it will be the return value of the previous call.
"""
function add_var_free! end

"""
    add_var_free_finalize!(state, eqstate)

Finishes the addition of free variables to `state`; the value of `eqstate` is the return value of the last call to
[`add_var_free!`](@ref). Note that less variables than initially requested might have been added; in this case, the
implementation might decide to do some bookkeeping.
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