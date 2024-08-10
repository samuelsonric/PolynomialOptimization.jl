export add_constr_nonnegative!, add_constr_rotated_quadratic!, add_constr_quadratic!, add_constr_linf!,
    add_constr_linf_complex!, add_constr_psd!, add_constr_psd_complex!, add_constr_dddual!, add_constr_dddual_complex!,
    add_constr_fix_prepare!, add_constr_fix!, add_constr_fix_finalize!, fix_objective!, add_var_slack!

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

Adds a quadratic constraint to the `N = length(indvals)` linear combinations of decision variables (columns in the conic
constraint matrix) indexed according to the `indvals`. This will read (where ``X_i`` is
``\mathit{indvals}_i.\mathit{values} \cdot x_{\mathit{indvals}_i.\mathit{indices}}``) ``X_1 \geq 0``,
``X_1^2 \geq \sum_{i = 2}^N X_i^2``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. If the scaled
    diagonally dominant representation is requested, `indvals` can have any length.

!!! warning
    This function will only be called if [`supports_quadratic`](@ref) returns `true` for the given state.
    If (rotated) quadratic constraints are unsupported, a fallback to a 2x2 PSD constraint is used.
"""
add_constr_quadratic!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_constr_rotated_quadratic! end

@doc raw"""
    add_constr_rotated_quadratic!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds a rotated quadratic constraint to the `N = length(indvals)` linear combinations of decision variables (columns in the
conic constraint matrix) indexed according to the `indvals`. This will read (where ``X_i`` is
``\mathit{indvals}_i.\mathit{values} \cdot x_{\mathit{indvals}_i.\mathit{indices}}``) ``X_1, X_2 \geq 0``,
``2X_1 X_2 \geq \sum_{i = 3}^N X_i^2``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! note "Number of parameters"
    In the real-valued case, `indvals` is always of length three, in the complex case, it is of length four. If the scaled
    diagonally dominant representation is requested, `indvals` can have any length.

!!! warning
    This function will only be called if [`supports_rotated_quadratic`](@ref) returns `true` for the given state.
    If (rotated) quadratic constraints are unsupported, a fallback to a 2x2 PSD constraint is used.
"""
add_constr_rotated_quadratic!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_constr_psd! end

"""
    add_constr_psd!(state, dim::Integer, data::PSDMatrixCartesian{T,V}) where {T,V<:Real}

Add a PSD constraint of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the return value
of [`psd_indextype`](@ref)); these elements make up a linear matrix inequality with variables given by the keys when iterating
through `data`, which are of the type returned by [`mindex`](@ref).
Note that if [`add_constr_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_psd_complex`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_constr_psd_complex!`](@ref) must be implemented.
"""
add_constr_psd!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Real})

"""
    add_constr_psd!(state, dim::Integer, data::IndvalsIterator{T,V}) where {T,V<:Real}

Add a PSD constraint of side dimension `dim` ≥ 3 to the solver. `data` is an iterable through the elements of the PSD matrix
one-by-one, in the order specified by [`psd_indextype`](@ref). The individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).

!!! hint "Complex-valued PSD variables"
    Note that this function will also be called for complex-valued PSD cones if [`supports_psd_complex`](@ref) returns `false`.
    The data will have been rewritten in terms of a real-valued PSD cone, which doubles the dimension.
    If the solver natively supports complex-valued PSD cones, [`add_constr_psd_complex!`](@ref) must be implemented.
"""
add_constr_psd!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

function add_constr_psd_complex! end

"""
    add_constr_psd_complex!(state, dim::Int,
        data::PSDMatrixCartesian{T,V}) where {T,V<:Complex}

Add a Hermitian PSD constraint of side dimension `dim` ≥ 3 to the solver. Its requested triangle is indexed according to the
return value of [`psd_indextype`](@ref)); these elements make up a linear matrix inequality with variables given by the keys
when iterating through `data`, which are of the type returned by [`mindex`](@ref). The real part of any coefficient corresponds
to the coefficient in front of the real part of the matrix entry, the imaginary part is the coefficient for the imaginary part
of the matrix entry.
Note that if [`add_constr_quadratic!`](@ref) is not implemented, `dim` may also be `2`.
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeMatrixCartesian`](@ref).

!!! warning
    This function will only be called if [`supports_psd_complex`](@ref) is defined to return `true` for the given state.
"""
add_constr_psd_complex!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Complex}) = unsupported

"""
    add_constr_psd_complex!(state, dim::Int, data::IndvalsIterator{T,V}) where {T,V<:Real}

Add a Hermitian PSD constraint of side dimension `dim` ≥ 3 to the solver. `data` is an iterable through the elements of the PSD
matrix one-by-one, in the order specified by [`psd_indextype`](@ref). The individual entries are [`Indvals`](@ref).
This method is called if [`psd_indextype`](@ref) returns a [`PSDIndextypeVector`](@ref).
Regardless of the travelling order, for diagonal elements, there will be exactly one entry, which is the real part. For
off-diagonal elements, the real part will be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_psd_complex`](@ref) is defined to return `true` for the given state.
"""
add_constr_psd_complex!(::Any, ::Int, ::IndvalsIterator{<:Any,<:Real})

function add_constr_linf! end

@doc raw"""
    add_constr_linf!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Adds an ``\ell_\infty`` norm constraint to the `N = length(indvals)` linear combinations of decision variables (columns in the
conic constraint matrix) indexed according to the `indvals`. This will read (where ``X_i`` is
``\mathit{indvals}_i.\mathit{values} \cdot x_{\mathit{indvals}_i.\mathit{indices}}``)
``X_1 \geq \max_{i > 2} \lvert X_i\rvert``.

See also [`Indvals`](@ref), [`IndvalsIterator`](@ref).

!!! warning
    This function will only be called if [`supports_lnorm`](@ref) returns `true` for the given state.
    If ``\ell_\infty`` norm constraints are unsupported, a fallback to multiple linear constraints will be used.
"""
add_constr_linf!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_constr_linf_complex! end

@doc raw"""
    add_constr_linf_complex!(state, indvals::IndvalsIterator{T,V}) where {T,V<:Real}

Same as [`add_constr_linf!`](@ref), but now two successive items in `indvals` (starting from the second) are interpreted as
determining the real and imaginary part of a component of the ``\ell_\infty`` norm cone.

!!! warning
    This function will only be called if [`supports_lnorm_complex`](@ref) returns `true` for the given state.
    If complex-valued ``\ell_\infty`` norm constraints are unsupported, a fallback to multiple linear constraints and quadratic
    cones will be used. If [`supports_quadratic`](@ref) is not `true`, complex-valued DD cones cannot be used.
"""
add_constr_linf_complex!(::Any, ::IndvalsIterator{<:Any,<:Real})

function add_constr_dddual! end

@doc raw"""
    add_constr_dddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to diagonally dominant matrices to the solver. `data` is an iterator through
the (unscaled) lower triangle of the element of the matrix. A basis change is induced by `u`, with the meaning for the primal
cone that `M ∈ DD(u) ⇔ M = uᵀ Q u` with `Q ∈ DD`.

!!! warning
    This function will only be called if [`supports_dd`](@ref) returns `true` for the given state. If diagonally dominant cones
    are not supported directly, a fallback to a columnwise representation in terms of ``\ell_\infty`` norms will be used (or
    the fallbacks if this norm is not supported).
"""
function add_constr_dddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u, ::Val{complex}=Val(false)) where {T,V<:Real,complex}
    isempty(data.lens) && return
    diagu = u isa UniformScaling || u isa Diagonal # we just ignore uniform scaling completely as if it were 𝟙
    # Diagonal-dominant representation: this is a relaxation for the SOS formulation, where we replace M ∈ PSD by
    # M ∈ {U† D U, D ∈ DD}. Since this is more restrictive than PSD, the SOS maximization will only decrease, so we still have
    # a valid lower bound.
    # Vectorized version: vec(M) = vec(U† mat(d) U). In component form, this is
    # mᵢ = ∑_(diagonal j) Ū[row(j), row(i)] U[col(j), col(i)] dⱼ +
    #      ∑_(offdiag j) (Ū[col(j), row(i)] U[row(j), col(i)] + Ū[row(j), row(i)] U[col(j), col(i)]) dⱼ ⇔ m = Ũ d.
    # Note that if U is diagonal, mᵢ = U[col(i), col(i)] Ū[row(i), row(i)] dᵢ.
    # So define d ∈ vec(DD), m free, then demand 𝟙*m + (-Ũ)*d = 0. But actually, in SOS, m enters the linear constraints with
    # rows given by sosdata, so we don't even need to create those variables - d is sufficient. Therefore, the DD-SOS problem
    # looks like d ∈ vec(DD), and sosdata[i] contains the linear constraint row indices for the linear combination (Ũ*d)[i].
    # Then, we need to translate DD into a cone that is supported; let's assume for simplicity that the ℓ₁ cone is available.
    # DD = ℓ₁ × ... × ℓ₁ plus equality constraints that enforce symmetry.
    # However, here we construct the moment representation; so we now need the dual formulation of diagonal dominance. Due to
    # the equality constraints, this is more complicated:
    # For side dimension n, there are n ℓ₁ cones (we just take the columns - also taking into account the rows would be even
    # more restrictive).
    # Without the U, the columns would look like (note that the diagonal is moved to the first row)
    # data₁             data₄             data₆
    # 2data₂ - slack₁   slack₁            slack₂
    # 2data₃ - slack₂   2data₅ - slack₃   slack₃
    # i.e., we introduce a slack variable for every off-diagonal cell; on the upper triangle, we just put the slacks, on the
    # lower triangle, we put twice the data for this cell minus the slack.

    # If U is diagonal, things are not much more difficult:
    # U₁²data₁              U₂²data₄              U₃²data₆
    # 2U₂U₁data₂ - slack₁   slack₁                slack₂
    # 2U₃U₁data₃ - slack₂   2U₃U₂data₅ - slack₃   slack₃

    # If U is dense, we have for column j:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) U[j, row] U[j, col] data(row, col),
    #  slack(i, j) for i ∈ 1, ..., j -1,
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) (U[i, row] U[j, col] + U[j, row] U[i, col]) data(row, col) -
    #      slack(i, j) for i ∈ j +1, ..., dim} ∈ ℓ_∞

    # In the complex case with diagonal U, we have (note that the l.h.s. only has a real part, the rest are two entries)
    # |U₁|²data₁                                 |U₂|²data₆                                 |U₃|²data₈
    # 2Re(U₂Ū₁)data₂ - 2Im(U₂Ū₁)data₃ - slack₁   slack₁                                     slack₃
    # 2Im(U₂Ū₁)data₂ + 2Re(U₂Ū₁)data₃ - slack₂   slack₂                                     slack₄
    # 2Re(U₃Ū₁)data₄ - 2Im(U₃Ū₁)data₅ - slack₃   2Re(U₃Ū₂)data₇ - 2Im(U₃Ū₂)data₈ - slack₅   slack₅
    # 2Im(U₃Ū₁)data₄ + 2Re(U₃Ū₁)data₅ - slack₄   2Im(U₃Ū₂)data₇ + 2Re(U₃Ū₂)data₈ - slack₆   slack₆

    maxsize = maximum(data.lens, init=0) # how large is one dataᵢ at most?
    if complex && (!(Base.IteratorEltype(u) isa Base.HasEltype) || eltype(u) <: Complex)
        maxsize *= 2
    end
    if !diagu
        maxsize *= trisize(dim) # how large are all the dataᵢ that might be used in a single cell at most?
    end
    have_linf = complex ? supports_lnorm_complex(state) : supports_lnorm(state)
    complex && @assert(have_linf || supports_quadratic(state)) # this must have been checked during construction
    if have_linf
        indices = FastVec{T}(buffer=complex ? ((2dim -2) * (maxsize +1) + maxsize) : (maxsize +1) * dim -1)
                  # first col is possibly largest: full with data plus (dim -1) slacks
    elseif complex
        # This means that we use the quadratic cone to mimick the ℓ_∞ norm cone. This means Mᵢᵢ² ≥ Re² M₁ᵢ + Im² M₁ᵢ. So we
        # need to submit lots of cones, but all of them pretty small.
        indices = FastVec{T}(buffer=5maxsize + 2)
    else
        # If we don't have this cone, we must use linear constraints. To mimick the ℓ_∞ norm cone, we need to impose a number
        # of additional linear constraints: Mᵢᵢ - M₁ᵢ ≥ 0, Mᵢᵢ + M₁ᵢ ≥ 0, ... We will create all the first pairs of inequality
        # constraints in a single column, then flip the sign and do it all over again.
        indices = FastVec{T}(buffer=(2maxsize +1) * (dim -1))
    end
    values = similar(indices, V)
    lens = FastVec{Int}(buffer=complex ? 2dim -1 : dim)
    slacks = add_var_slack!(state, complex ? 2trisize(dim -1) : trisize(dim -1))
    s = 1
    unchecked_iterator = IndvalsIterator(indices, values, lens)
    @inbounds if diagu
        i = 1
        k = 1
        for col in 1:dim
            iᵤ = i
            kᵤ = k
            kdiag = k
            # First add the diagonal, which naturally is the first item in the L order. If we need the linear constraints, we
            # just prepare the diagonal data, as we need to add it for every off-diagonal term. In the complex case, even if we
            # use the quadratic cone, the first term will always stay the same, so we can already add it properly scaled.
            diaglen = Int(data.lens[i])
            diagr = k:k+diaglen-1
            k += diaglen
            i += 1
            if u isa Diagonal
                diaguval = abs2(u[col, col])
            else
                diaguval = true
            end
            if have_linf || complex
                # if we don't have the ℓ_∞ cone, the diagonal entry is only defined in each inequality
                unsafe_append!(indices, @view(data.indices[diagr]))
                unsafe_append!(values, @view(data.values[diagr]))
                isone(diaguval) || rmul!(@view(values[end-diaglen+1:end]), diaguval)
                unsafe_push!(lens, diaglen)
            end
            # Add all elements above the diagonal, which are just the slack variables
            if have_linf
                unsafe_append!(indices, @view(slacks[s:s+(complex ? 2col -3 : col -2)]))
                unsafe_append!(values, Iterators.repeated(one(V), complex ? 2col -2 : col -1))
                unsafe_append!(lens, Iterators.repeated(1, complex ? 2col -2 : col -1))
                s += complex ? 2col -2 : col -1
            elseif complex
                if !isone(col)
                    unsafe_push!(indices, slacks[s], slacks[s+1])
                    unsafe_push!(values, one(V), one(V))
                    unsafe_push!(lens, 1, 1)
                    add_constr_quadratic!(state, unchecked_iterator)
                    s += 2
                    for row in 2:col-1
                        indices[end-1] = slacks[s]
                        indices[end] = slacks[s+1]
                        add_constr_quadratic!(state, unchecked_iterator)
                        s += 2
                    end
                    Base._deleteend!(indices, 2)
                    Base._deleteend!(values, 2)
                    Base._deleteend!(lens, 2)
                end
            else
                for row in 1:col-1
                    unsafe_append!(indices, @view(data.indices[diagr]))
                    unsafe_append!(values, @view(data.values[diagr]))
                    isone(diaguval) || rmul!(@view(values[end-diaglen+1:end]), diaguval)
                    unsafe_push!(indices, slacks[s])
                    unsafe_push!(values, one(V))
                    unsafe_push!(lens, diaglen +1)
                    s += 1
                end
            end
            # Add all elements below the diagonal
            sbelow = s + (complex ? 2col - 2 : col -1)
            for row in col+1:dim
                @twice impart complex begin
                    len = Int(data.lens[i])
                    r = k:k+len-1
                    if have_linf || complex
                        if u isa Diagonal
                            uval = 2u[row, row] * conj(u[col, col])
                            if iszero(real(uval))
                                thislen = 0
                            else
                                thislen = len
                                unsafe_append!(indices, @view(data.indices[r]))
                                unsafe_append!(values, @view(data.values[r]))
                                isone(real(uval)) || rmul!(@view(values[end-len+1:end]), real(uval))
                            end
                            if complex && !iszero(imag(uval))
                                let lenalt=Int(data.lens[impart ? i-1 : i+1]), k=impart ? k - lenalt : k + len, r=k:k+lenalt-1,
                                    uval=impart ? imag(uval) : -imag(uval)
                                    searchrange = length(indices)-thislen+1:length(indices)
                                    searchview = @view(indices[searchrange])
                                    for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                                        idx = findfirst(isequal(ind), searchview)
                                        if isnothing(idx)
                                            unsafe_push!(indices, ind)
                                            unsafe_push!(values, uval * val)
                                            thislen += 1
                                        else
                                            values[first(searchrange)+idx-1] += uval * val
                                        end
                                    end
                                end
                            end
                        else
                            thislen = len
                            unsafe_append!(indices, @view(data.indices[r]))
                            unsafe_append!(values, @view(data.values[r]))
                            rmul!(@view(values[end-len+1:end]), V(2))
                        end
                        unsafe_push!(indices, slacks[sbelow])
                        unsafe_push!(values, -one(V))
                        unsafe_push!(lens, thislen +1)
                    else
                        # We need to add the diagonal element again for every single constraint.
                        unsafe_append!(indices, @view(data.indices[diagr]))
                        unsafe_append!(values, @view(data.values[diagr]))
                        isone(diaguval) || rmul!(@view(values[end-diaglen+1:end]), diaguval)
                        # We combine this index part with the diagonal; but this can lead to duplicates, which we must sum up.
                        if u isa Diagonal
                            uval = 2u[row, row] * u[col, col]
                        else
                            uval = V(2)
                        end
                        thislen = diaglen
                        searchrange = length(indices)-diaglen+1:length(indices)
                        searchview = @view(indices[searchrange])
                        for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                            idx = findfirst(isequal(ind), searchview)
                            if isnothing(idx)
                                unsafe_push!(indices, ind)
                                unsafe_push!(values, uval * val)
                                thislen += 1
                            else
                                values[first(searchrange)+idx-1] += uval * val
                            end
                        end
                        unsafe_push!(indices, slacks[sbelow])
                        unsafe_push!(values, -one(V))
                        unsafe_push!(lens, thislen +1)
                    end
                    k += len
                    i += 1
                    if complex
                        sbelow += impart ? 2row -3 : 1
                    else
                        sbelow += row -1
                    end
                end
                if complex && !have_linf
                    add_constr_quadratic!(state, unchecked_iterator)
                    Base._deleteend!(indices, length(indices) - diaglen)
                    Base._deleteend!(values, length(values) - diaglen)
                    Base._deleteend!(lens, 2)
                end
            end
            if have_linf
                (complex ? add_constr_linf_complex! : add_constr_linf!)(state, unchecked_iterator)
            elseif !complex
                add_constr_nonnegative!(state, unchecked_iterator)
                # Flip the sign of the nondiagonal parts. This is tricky: Every entry begins with the diagonal part, then has
                # the rest; so we can always simply flip the sign of the rest. However, if some indices of the rest coincide
                # with diagonal indices, we merged them in the front; flipping won't do the job.
                # Previously, we had diag + offdiag, now we want diag - offdiag, so let's do -(old - diag) + diag = -old+2diag
                j = 1
                for l in lens
                    @simd for di in 0:diaglen-1
                        values[j+di] = 2diaguval * data.values[kdiag+di] - values[j+di]
                    end
                    rmul!(@view(values[j+diaglen:j+l-1]), -one(V))
                    j += l
                end
                add_constr_nonnegative!(state, unchecked_iterator)
            end
            empty!(indices)
            empty!(values)
            empty!(lens)
        end
    else
        if !have_linf
            diagvals = Vector{V}(undef, maxsize) # let's store the diagonal entry to avoid having to compute it over and over
        end
        for ddcol in 1:dim
            # First the diagonal. We will add it in any case, even for the inequalities, as this time indices may combine.
            i = 1
            k = 1
            for datacol in 1:dim, datarow in datacol:dim
                len = Int(data.lens[i])
                r = k:k+len-1
                k += len
                i += 1
                uval = conj(u[ddcol, datarow]) * u[ddcol, datacol]
                iszero(uval) && continue
                if datarow != datacol
                    uval *= V(2)
                end
                searchview = @view(indices[:])
                for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                    idx = findfirst(isequal(ind), searchview)
                    if isnothing(idx)
                        unsafe_push!(indices, ind)
                        unsafe_push!(values, uval * val)
                    else
                        values[idx] += uval * val
                    end
                end
            end
            diaglen = length(indices)
            if have_linf
                unsafe_push!(lens, diaglen)
            else
                copyto!(diagvals, values)
            end
            # Then all slacks
            if have_linf
                unsafe_append!(indices, @view(slacks[s:s+ddcol-2]))
                unsafe_append!(values, Iterators.repeated(one(V), ddcol -1))
                unsafe_append!(lens, Iterators.repeated(1, ddcol -1))
                s += ddcol -1
            else
                # the first slack doesn't need the diagonal, as we just added it
                if !isone(ddcol)
                    unsafe_push!(indices, slacks[s])
                    unsafe_push!(values, one(V))
                    unsafe_push!(lens, diaglen +1)
                    s += 1
                end
                for ddrow in 2:ddcol-1
                    # but all others do
                    unsafe_append!(indices, @view(indices[1:diaglen]))
                    unsafe_append!(values, @view(diagvals[1:diaglen]))
                    unsafe_push!(indices, slacks[s])
                    unsafe_push!(values, one(V))
                    unsafe_push!(lens, diaglen +1)
                    s += 1
                end
            end
            # And finally all elements below the diagonal
            sbelow = s + ddcol -1
            for ddrow in ddcol+1:dim
                if ddrow == 2 && !have_linf
                    # special place: first column, nothing before, so the diagonal was already inserted
                    startidx = 1
                else
                    # duplicate the diagonal
                    startidx = length(indices) +1
                    if !have_linf
                        unsafe_append!(indices, @view(indices[1:diaglen]))
                        unsafe_append!(values, @view(diagvals[1:diaglen]))
                    end
                end
                i = 1
                k = 1
                for datacol in 1:dim, datarow in datacol:dim
                    len = Int(data.lens[i])
                    r = k:k+len-1
                    uval = conj(u[ddrow, datarow]) * u[ddcol, datacol] + conj(u[ddcol, datarow]) * u[ddrow, datacol]
                    k += len
                    i += 1
                    iszero(uval) && continue
                    if datarow != datacol
                        uval *= V(2)
                    end
                    searchview = @view(indices[startidx:end])
                    for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                        idx = findfirst(isequal(ind), searchview)
                        if isnothing(idx)
                            unsafe_push!(indices, ind)
                            unsafe_push!(values, uval * val)
                        else
                            values[startidx+idx-1] += uval * val
                        end
                    end
                end
                unsafe_push!(indices, slacks[sbelow])
                unsafe_push!(values, -one(V))
                unsafe_push!(lens, length(indices) - startidx +1)
                sbelow += ddrow -1
            end
            if have_linf
                add_constr_linf!(state, unchecked_iterator)
            else
                add_constr_nonnegative!(state, unchecked_iterator)
                j = 1
                for l in lens
                    @simd for di in 0:diaglen-1
                        values[j+di] = 2diagvals[1+di] - values[j+di]
                    end
                    rmul!(@view(values[j+diaglen:j+l-1]), -one(V))
                    j += l
                end
                add_constr_nonnegative!(state, unchecked_iterator)
            end
            empty!(indices)
            empty!(values)
            empty!(lens)
        end
    end
    return
end

function add_constr_dddual_complex! end

@doc raw"""
    add_constr_dddual_complex!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to complex-valued diagonally dominant matrices to the solver. `data` is an
iterator through the (unscaled) lower triangle of the element of the matrix. A basis change is induced by `u`, with the meaning
for the primal cone that `M ∈ DD(u) ⇔ M = u† Q u` with `Q ∈ DD`.
For diagonal elements, there will be exactly one entry, which is the real part. For off-diagonal elements, the real part will
be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_dd_complex`](@ref) returns `true` for the given state. If complex-valued
    diagonally dominant cones are not supported directly, a fallback to quadratic constraints on the complex-valued data is
    tried first (if supported), followed by a columnwise representation in terms of ``\ell_\infty`` norms or their fallback on
    the realification of the matrix data if not.
"""
add_constr_dddual_complex!(state, dim::Integer, data::IndvalsIterator{<:Any,<:Real}, u) =
    add_constr_dddual!(state, dim, data, u, Val(true))

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

"""
    add_var_slack!(state, num::Int)

Creates `num` slack variables in the problem. Slack variables must be free. The result should be an abstract vector (typically
a unit range) that contains the indices of all created slack variables. The indices should be of the same type as
[`mindex`](@ref).
"""
function add_var_slack! end