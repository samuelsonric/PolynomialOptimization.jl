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
add_constr_psd_complex!(::Any, ::Int, ::PSDMatrixCartesian{<:Any,<:Complex})

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

function add_constr_dddual! end

@doc raw"""
    add_constr_dddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to diagonally dominant matrices to the solver. `data` is an iterator through
the (unscaled) lower triangle of the matrix. A basis change is induced by `u`, with the meaning for the primal cone that
`M ∈ DD(u) ⇔ M = uᵀ Q u` with `Q ∈ DD`.

!!! warning
    This function will only be called if [`supports_dd`](@ref) returns `true` for the given state. If diagonally dominant cones
    are not supported directly, a fallback to a columnwise representation in terms of ``\ell_\infty`` norms will be used (or
    the fallbacks if this norm is not supported).
"""
function add_constr_dddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u, ::Val{complex}=Val(false)) where {T,V<:Real,complex}
    # If we end up here, this must be a fallback, i.e., the solver should not have set the supports_ functions to true!
    complex && supports_dd_complex(state) && throw(MethodError(add_constr_dddual_complex!, (state, dim, data, u)))
    !complex && supports_dd(state) && throw(MethodError(add_constr_dddual!, (state, dim, data, u)))
    isempty(data.lens) && return
    diagu = u isa UniformScaling || u isa Diagonal # we just ignore uniform scaling completely as if it were 𝟙
    # Diagonal-dominant representation: this is a relaxation for the SOS formulation, where we replace M ∈ PSD by
    # M ∈ {U† D U, D ∈ DD}. Since this is more restrictive than PSD, the SOS maximization will only decrease, so we still have
    # a valid lower bound.
    # Vectorized version: vec(M) = vec(U† mat(d) U). In component form, this is
    # mᵢ = ∑_(diagonal j) Ū[row(j), row(i)] U[col(j), col(i)] dⱼ +
    #      ∑_(offdiag j) (Ū[col(j), row(i)] U[row(j), col(i)] + Ū[row(j), row(i)] U[col(j), col(i)]) dⱼ ⇔ m = Ũ d.
    # Note that if U is diagonal, mᵢ = Ū[row(i), row(i)] U[col(i), col(i)] dᵢ.
    # So define d ∈ vec(DD), m free, then demand 𝟙*m + (-Ũ)*d = 0. But actually, in SOS, m enters the linear constraints with
    # rows given by sosdata, so we don't even need to create those variables - d is sufficient. Therefore, the DD-SOS problem
    # looks like d ∈ vec(DD), and sosdata[i] contains the linear constraint row indices for the linear combination (Ũ*d)[i].
    # Then, we need to translate DD into a cone that is supported; let's assume for simplicity that the ℓ₁ cone is available.
    # DD = ℓ₁ × ... × ℓ₁ plus equality constraints that enforce symmetry.
    # However, here we construct the moment representation; so we now need the dual formulation of diagonal dominance. Due to
    # the equality constraints, this is more complicated:
    # For side dimension n, there are n ℓ₁ cones (we just take the columns - also taking into account the rows would be even
    # more restrictive).
    # Without the U, the columns in the real-valued case would look like (note that the diagonal is moved to the first row)
    # data₁             data₄             data₆
    # 2data₂ - slack₁   slack₁            slack₂
    # 2data₃ - slack₂   2data₅ - slack₃   slack₃
    # i.e., we introduce a slack variable for every off-diagonal cell; on the upper triangle, we just put the slacks, on the
    # lower triangle, we put twice the data for this cell minus the slack.
    # Note slackᵢ(i, j) = -slackᵢ(j, i)

    # For a general U and complex-valued data, this is instead for the column j:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) (Re(U[j, col] Ū[j, row]) dataᵣ(row, col) +
    #                                                        Im(U[j, col] Ū[j, row]) dataᵢ(row, col)),
    #  slackᵣ(j, i), slackᵢ(j, i) for i ∈ 1, ..., j -1,
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col})
    #      ((Re(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) -
    #       (Im(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) dataᵢ(row, col)) - slackᵣ(i, j),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col})
    #      ((Im(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) +
    #       (Re(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) dataᵢ(row, col)) - slackᵢ(i, j)
    #  for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # Let's specialize the formula. If U is diagonal:
    # {|U[j, j]|² dataᵣ(j, j),
    #  slackᵣ(j, i), slackᵢ(j, i) for i ∈ 1, ..., j -1,
    #  2 (Re(U[i, i] Ū[j, j]) dataᵣ(i, j) - Im(U[i, i] Ū[j, j]) dataᵢ(i, j)) - slackᵣ(i, j),
    #  2 (Im(U[i, i] Ū[j, j]) dataᵣ(i, j) + Re(U[i, i] Ū[j, j]) dataᵢ(i, j)) - slackᵢ(i, j)
    #  for i in j +1, ..., dim
    # } ∈ ℓ_∞
    # Let's write this out:
    # |U₁|²data₁                                 |U₂|²data₆                                 |U₃|²data₈
    # 2Re(U₂Ū₁)data₂ - 2Im(U₂Ū₁)data₃ - slack₁   slack₁                                     slack₃
    # 2Im(U₂Ū₁)data₂ + 2Re(U₂Ū₁)data₃ - slack₂   slack₂                                     slack₄
    # 2Re(U₃Ū₁)data₄ - 2Im(U₃Ū₁)data₅ - slack₃   2Re(U₃Ū₂)data₇ - 2Im(U₃Ū₂)data₈ - slack₅   slack₅
    # 2Im(U₃Ū₁)data₄ + 2Re(U₃Ū₁)data₅ - slack₄   2Im(U₃Ū₂)data₇ + 2Re(U₃Ū₂)data₈ - slack₆   slack₆

    # If everything is instead real-valued:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) U[j, col] U[j, row] data(row, col),
    #  slack(j, i) for i ∈ 1, ..., j -1,
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) (U[i, row] U[j, col] + U[i, col] U[j, row]) data(row, col) -
    #      slack(i, j) for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # If everything is real and U is diagonal:
    # {U[j, j]² data(j, j),
    #  slack(j, i) for i ∈ 1, ..., j -1,
    #  2 U[i, i] U[j, j] data(i, j) - slack(i, j) for i in j +1, ..., dim
    # } ∈ ℓ_∞
    # Let's write this out:
    # U₁²data₁              U₂²data₄              U₃²data₆
    # 2U₂U₁data₂ - slack₁   slack₁                slack₂
    # 2U₃U₁data₃ - slack₂   2U₃U₂data₅ - slack₃   slack₃
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
        # This means that we use the quadratic cone to mimick the ℓ_∞ norm cone: x₁ ≥ ∑ᵢ (Re² xᵢ + Im² xᵢ). So we need to
        # submit lots of cones, but all of them pretty small.
        indices = FastVec{T}(buffer=5maxsize + 2)
    else
        # If we don't have this cone, we must use linear constraints. To mimick the ℓ_∞ norm cone, we need to impose a number
        # of additional linear constraints: xᵢ - x₁ ≥ 0, xᵢ + x₁ ≥ 0, ... We will create all the first pairs of inequality
        # constraints in a single column, then flip the sign and do it all over again. The diagonal entry, which we don't need
        # to add explicitly, still gets a placeholder value.
        indices = FastVec{T}(buffer=(2maxsize +1) * (dim -1) + maxsize)
    end
    values = similar(indices, V)
    lens = FastVec{Int}(buffer=complex ? 2dim -1 : dim)
    slacks = add_var_slack!(state, complex ? 2trisize(dim -1) : trisize(dim -1))
    s = 1
    if diagu
        idx = 1
        dataidx = 1
    end
    @inbounds for j in 1:dim
        #region Diagonal
        # First add the diagonal, which naturally is the first item in the L order. We will add this diagonal once for
        # sure. This is fine for the ℓ_∞ cone anyway, where it is supposed to be the first element; but also for the other
        # cones, where we need to repeat it, it is not a problem: for the quadratic cone, it will always be exactly the
        # first element in the cone; for the linear inequalities, it may combine with other elements. However, we add all
        # linear inequalities of the same j in one bunch, i.e., the first corresponds to diagonal - first slack, where
        # nothing combines. The only exception is the j = 1 case, where we don't have any slack. In this case, we'll just
        # add diagonal ≥ 0 anyway and skip it when passing the data to the cone (it is not wrong, but also not necessary).
        if diagu
            diaglen = Int(data.lens[idx])
            diagr = dataidx:dataidx+diaglen-1
            dataidx += diaglen
            idx += 1
            unsafe_append!(indices, @view(data.indices[diagr]))
            unsafe_append!(values, @view(data.values[diagr]))
            u isa Diagonal && rmul!(values, abs2(u[j, j]))
        else
            idx = 1
            dataidx = 1
            for col in 1:dim, row in col:dim
                uval = u[j, col] * conj(u[j, row])
                if row != col
                    uval *= V(2)
                end
                @twice impart (complex && row != col) begin
                    searchview = @view(indices[:])
                    len = Int(data.lens[idx])
                    r = dataidx:dataidx+len-1
                    dataidx += len
                    idx += 1
                    thisuval = impart ? imag(uval) : real(uval)
                    iszero(thisuval) || for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                        dupidx = findfirst(isequal(ind), searchview)
                        if isnothing(dupidx)
                            unsafe_push!(indices, ind)
                            unsafe_push!(values, thisuval * val)
                        else
                            values[dupidx] += thisuval * val
                        end
                    end
                end
            end
            diaglen = length(indices)
        end
        if have_linf || complex
            unsafe_push!(lens, diaglen)
        else
            diagvals = @view(values[:])
        end
        #endregion
        #region Above diagonal (slacks)
        if have_linf
            δ = complex ? 2j -2 : j -1
            unsafe_append!(indices, @view(slacks[s:s+δ-1]))
            unsafe_append!(values, Iterators.repeated(one(V), δ))
            unsafe_append!(lens, Iterators.repeated(1, δ))
            s += δ
        elseif complex
            if !isone(j)
                unsafe_push!(indices, slacks[s], slacks[s+1])
                unsafe_push!(values, one(V), one(V))
                unsafe_push!(lens, 1, 1)
                add_constr_quadratic!(state, IndvalsIterator(unsafe, indices, values, lens))
                s += 2
                for i in 2:j-1
                    indices[end-1] = slacks[s]
                    indices[end] = slacks[s+1]
                    add_constr_quadratic!(state, IndvalsIterator(unsafe, indices, values, lens))
                    s += 2
                end
                Base._deleteend!(indices, 2)
                Base._deleteend!(values, 2)
                Base._deleteend!(lens, 2)
            end
        else
            # the first slack doesn't need the diagonal any more, we already added it
            if !isone(j)
                unsafe_push!(indices, slacks[s])
                unsafe_push!(values, one(V))
                unsafe_push!(lens, diaglen +1)
                s += 1
                for i in 2:j-1
                    unsafe_append!(indices, @view(indices[1:diaglen]))
                    unsafe_append!(values, @view(values[1:diaglen]))
                    unsafe_push!(indices, slacks[s])
                    unsafe_push!(values, one(V))
                    unsafe_push!(lens, diaglen +1)
                    s += 1
                end
            end
        end
        #endregion
        #region Below diagonal
        sbelow = s + (complex ? 2j - 2 : j -1)
        for i in j+1:dim
            if diagu
                if (have_linf || complex) && u isa Diagonal
                    uval = 2u[i, i] * conj(u[j, j])
                end
                @twice impart complex begin
                    startidx = length(indices) +1
                    len = Int(data.lens[idx])
                    r = dataidx:dataidx+len-1
                    if have_linf || complex
                        if u isa Diagonal
                            if !iszero(real(uval))
                                unsafe_append!(indices, @view(data.indices[r]))
                                unsafe_append!(values, @view(data.values[r]))
                                isone(real(uval)) || rmul!(@view(values[startidx:end]), real(uval))
                            end
                            if complex && !iszero(imag(uval))
                                let lenalt=Int(data.lens[impart ? idx-1 : idx+1]),
                                    dataidx=impart ? dataidx - lenalt : dataidx + len, r=dataidx:dataidx+lenalt-1,
                                    uimval=impart ? imag(uval) : -imag(uval)
                                    searchrange = startidx:length(indices)
                                    searchview = @view(indices[searchrange])
                                    for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                                        dupidx = findfirst(isequal(ind), searchview)
                                        if isnothing(dupidx)
                                            unsafe_push!(indices, ind)
                                            unsafe_push!(values, uimval * val)
                                        else
                                            values[first(searchrange)+dupidx-1] += uimval * val
                                        end
                                    end
                                end
                            end
                        else
                            unsafe_append!(indices, @view(data.indices[r]))
                            unsafe_append!(values, @view(data.values[r]))
                            rmul!(@view(values[end-len+1:end]), V(2))
                        end
                    else
                        # We need to add the diagonal element again for every single constraint.
                        unsafe_append!(indices, @view(indices[1:diaglen]))
                        unsafe_append!(values, @view(values[1:diaglen]))
                        # We combine this index part with the diagonal; but this can lead to duplicates, which we must sum up.
                        # Note we are real-valued if we are here.
                        if u isa Diagonal
                            uval = 2u[i, i] * u[j, j]
                        else
                            uval = V(2)
                        end
                        searchrange = length(indices)-diaglen+1:length(indices)
                        searchview = @view(indices[searchrange])
                        for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                            dupidx = findfirst(isequal(ind), searchview)
                            if isnothing(dupidx)
                                unsafe_push!(indices, ind)
                                unsafe_push!(values, uval * val)
                            else
                                values[first(searchrange)+dupidx-1] += uval * val
                            end
                        end
                    end
                    unsafe_push!(indices, slacks[sbelow])
                    unsafe_push!(values, impart ? one(V) : -one(V))
                    unsafe_push!(lens, length(indices) - startidx +1)
                    dataidx += len
                    idx += 1
                    if complex
                        sbelow += impart ? 2i -3 : 1
                    else
                        sbelow += i -1
                    end
                end
            else
                @twice impart complex begin
                    startidx = length(indices) +1
                    if !have_linf && !complex
                        # We need to add the diagonal element again for every linear constraint
                        unsafe_append!(indices, @view(indices[1:diaglen]))
                        unsafe_append!(values, @view(values[1:diaglen]))
                    end
                    idx = 1
                    dataidx = 1
                    for col in 1:dim, row in col:dim
                        @twice imdata (complex && row != col) begin
                            searchview = @view(indices[startidx:end])
                            uval = u[i, row] * conj(u[j, col])
                            if imdata
                                uval -= u[i, col] * conj(u[j, row])
                                thisuval = impart ? real(uval) : -imag(uval)
                            else
                                uval += u[i, col] * conj(u[j, row])
                                thisuval = impart ? imag(uval) : real(uval)
                            end
                            len = Int(data.lens[idx])
                            r = dataidx:dataidx+len-1
                            dataidx += len
                            idx += 1
                            if !iszero(thisuval)
                                if row != col
                                    thisuval *= V(2)
                                end
                                for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                                    dupidx = findfirst(isequal(ind), searchview)
                                    if isnothing(dupidx)
                                        unsafe_push!(indices, ind)
                                        unsafe_push!(values, thisuval * val)
                                    else
                                        values[dupidx+startidx-1] += thisuval * val
                                    end
                                end
                            end
                        end
                    end
                    unsafe_push!(indices, slacks[sbelow])
                    unsafe_push!(values, impart ? one(V) : -one(V))
                    unsafe_push!(lens, length(indices) - startidx +1)
                    if complex
                        sbelow += impart ? 2i -3 : 1
                    else
                        sbelow += i -1
                    end
                end
            end
            if complex && !have_linf
                add_constr_quadratic!(state, IndvalsIterator(unsafe, indices, values, lens))
                Base._deleteend!(indices, length(indices) - diaglen)
                Base._deleteend!(values, length(values) - diaglen)
                Base._deleteend!(lens, 2)
            end
        end
        #endregion
        #region Add the whole column
        # ℓ_∞ or linear; quadratic meant lots of constraints that were already added on-the-fly
        if have_linf
            (complex ? add_constr_linf_complex! : add_constr_linf!)(state, IndvalsIterator(unsafe, indices, values, lens))
        elseif !complex
            if isone(j)
                # Here we added the diagonal entry in the beginning as a dummy so that we can easily subtract without
                # multiplication operations. But we don't need it in the cone
                indvals = @views IndvalsIterator(unsafe, indices[diaglen+1:end], values[diaglen+1:end], lens)
            else
                indvals = @views IndvalsIterator(unsafe, indices[1:end], values[1:end], lens) # just for type stability
            end
            add_constr_nonnegative!(state, indvals)
            # Flip the sign of the nondiagonal parts. This is tricky: Every entry begins with the diagonal part, then has
            # the rest; so we can always simply flip the sign of the rest. However, if some indices of the rest coincide
            # with diagonal indices, we merged them in the front; flipping won't do the job.
            # Previously, we had diag + offdiag, now we want diag - offdiag, so let's do -(old - diag) + diag = -old+2diag
            # Note: This relies on the implementation not changing the underlying indvals (which is not done
            # AbstractSparseMatrixSolver or any of the provided API solvers). If the implementation were to, e.g., sort the
            # indices, we'd generate invalid data.
            k = isone(j) ? diaglen : 0
            for l in lens
                @simd for di in 1:diaglen
                    values[k+di] = 2diagvals[di] - values[k+di]
                end
                rmul!(@view(values[k+diaglen+1:k+l]), -one(V))
                k += l
            end
            add_constr_nonnegative!(state, indvals)
        end
        #endregion
        empty!(indices)
        empty!(values)
        empty!(lens)
    end
    return
end

function add_constr_dddual_complex! end

@doc raw"""
    add_constr_dddual_complex!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to complex-valued diagonally dominant matrices to the solver. `data` is an
iterator through the (unscaled) lower triangle of the matrix. A basis change is induced by `u`, with the meaning for the primal
cone that `M ∈ DD(u) ⇔ M = u† Q u` with `Q ∈ DD`.
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

function add_constr_sdddual! end

@doc raw"""
    add_constr_sdddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to scaled diagonally dominant matrices to the solver. `data` is an iterator
through the (unscaled) lower triangle of the matrix. A basis change is induced by `u`, with the meaning for the primal cone
that `M ∈ SDD(u) ⇔ M = uᵀ Q u` with `Q ∈ SDD`.

!!! warning
    This function will only be called if [`supports_sdd`](@ref) returns `true` for the given state. If scaled diagonally
    dominant cones are not supported directly, a fallback to (rotated) quadratic cones will be used.
"""
function add_constr_sdddual!(state, dim::Integer, data::IndvalsIterator{T,V}, u, ::Val{complex}=Val(false)) where {T,V<:Real,complex}
    isempty(data.lens) && return
    diagu = u isa UniformScaling || u isa Diagonal # we just ignore uniform scaling completely as if it were 𝟙
    # See the comment in add_constr_dddual!. Here, the fallback implementation is done in terms of rotated quadratic cones due
    # to the relationship of SDD matrices with factor-width-2 matrices.
    # Note that our data is already scaled in such a way that the diagonal elements are scaled by 1/√2, so we don't have to do
    # this explicitly.

    # For a general U and complex-valued data, we have the following rotated quadratic constraints for the column j and the row
    # i > j:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (Re(U[j, col] Ū[j, row]) dataᵣ(row, col) - Im(U[j, col] Ū[j, row]) dataᵢ(row, col)),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (Re(U[i, col] Ū[i, row]) dataᵣ(row, col) - Im(U[i, col] Ū[i, row]) dataᵢ(row, col)),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim ((Re( U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) -
    #                                     (Im(-U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵢ(row, col)),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim ((Im( U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) +
    #                                     (Re(-U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵢ(row, col))
    # } ∈ ℛ𝒬₄

    # Let's specialize the formula. If U is diagonal:
    # {|U[j, j]|² dataᵣ(j, j),
    #  |U[i, i]|² dataᵣ(i, i),
    #  (Re(U[i, i] Ū[j, j]) dataᵣ(i, j) + Im(U[i, i] Ū[j, j]) dataᵢ(i, j)),
    #  (Im(U[i, i] Ū[j, j]) dataᵣ(i, j) - Re(U[i, i] Ū[j, j]) dataᵢ(i, j))
    # } ∈ ℛ𝒬₄

    # If U is instead real-valued:
    # {∑_{col = 1}^dim ∑_{row = col}^dim U[j, col] U[j, row] data(row, col),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim U[i, col] U[i, row] data(row, col),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (U[i, row] U[j, col] + U[i, col] U[j, row]) data(row, col)
    # } ∈ ℛ𝒬₃

    # If U is real-diagonal:
    # {U[j, j]² data(j, j),
    #  U[i, i]² data(i, i),
    #  U[i, i] U[j, j] data(i, j)
    # } ∈ ℛ𝒬₃
    maxsize = maximum(data.lens, init=0) # how large is one dataᵢ at most?

    if complex && (!(Base.IteratorEltype(u) isa Base.HasEltype) || eltype(u) <: Complex)
        maxsize *= 2
    end
    if !diagu
        maxsize *= trisize(dim) # how large are all the dataᵢ that might be used in a single cell at most?
    end
    have_rot = supports_rotated_quadratic(state)
    indices = FastVec{T}(buffer=4maxsize)
    values = similar(indices, V)
    lens = FastVec{Int}(buffer=complex ? 4 : 3)
    if diagu
        idx = 1
        dataidx = 1
        diagdataidxs = Vector{Int}(undef, dim)
        @inbounds diagdataidxs[1] = 1
        @inbounds for j in 1:dim-1
            x = diagdataidxs[j]
            for _ in 0:(complex ? 2(dim-j) : dim-j)
                x += data.lens[idx]
                idx += 1
            end
            diagdataidxs[j+1] = x
        end
        idx = 1
    end
    @inbounds for j in 1:dim-1
        #region First item
        if diagu
            len = Int(data.lens[idx])
            r = dataidx:dataidx+len-1
            dataidx += len
            idx += 1
            unsafe_append!(indices, @view(data.indices[r]))
            unsafe_append!(values, @view(data.values[r]))
            u isa Diagonal && rmul!(values, abs2(u[j, j]))
        else
            idx = 1
            dataidx = 1
            for col in 1:dim, row in col:dim
                uval = u[j, col] * conj(u[j, row])
                @twice impart (complex && row != col) begin
                    searchview = @view(indices[:])
                    len = Int(data.lens[idx])
                    r = dataidx:dataidx+len-1
                    dataidx += len
                    idx += 1
                    thisuval = impart ? -imag(uval) : real(uval)
                    iszero(thisuval) || for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                        dupidx = findfirst(isequal(ind), searchview)
                        if isnothing(dupidx)
                            unsafe_push!(indices, ind)
                            unsafe_push!(values, thisuval * val)
                        else
                            values[dupidx] += thisuval * val
                        end
                    end
                end
            end
        end
        firstlen = length(indices)
        unsafe_push!(lens, firstlen)
        #endregion
        if diagu
            otheridx = idx + (complex ? 2(dim - j) : dim-j)
        end
        for i in j+1:dim
            #region Second item
            if diagu
                len = Int(data.lens[otheridx])
                r = diagdataidxs[i]:diagdataidxs[i]+len-1
                otheridx += (complex ? 2(dim - i)+1 : dim-i+1)
                unsafe_append!(indices, @view(data.indices[r]))
                unsafe_append!(values, @view(data.values[r]))
                u isa Diagonal && rmul!(@view(values[end-len+1:end]), abs2(u[i, i]))
            else
                idx = 1
                dataidx = 1
                for col in 1:dim, row in col:dim
                    uval = u[i, col] * conj(u[i, row])
                    @twice impart (complex && row != col) begin
                        searchview = @view(indices[firstlen+1:end])
                        len = Int(data.lens[idx])
                        r = dataidx:dataidx+len-1
                        dataidx += len
                        idx += 1
                        thisuval = impart ? -imag(uval) : real(uval)
                        iszero(thisuval) || for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                            dupidx = findfirst(isequal(ind), searchview)
                            if isnothing(dupidx)
                                unsafe_push!(indices, ind)
                                unsafe_push!(values, thisuval * val)
                            else
                                values[firstlen+dupidx] += thisuval * val
                            end
                        end
                    end
                end
            end
            unsafe_push!(lens, length(indices) - firstlen)
            #endregion
            #region Third and fourth item
            if diagu
                if u isa Diagonal
                    uval = u[i, i] * conj(u[j, j])
                end
                @twice impart complex begin
                    startidx = length(indices) +1
                    len = Int(data.lens[idx])
                    r = dataidx:dataidx+len-1
                    if u isa Diagonal
                        if !iszero(real(uval))
                            unsafe_append!(indices, @view(data.indices[r]))
                            unsafe_append!(values, @view(data.values[r]))
                            if impart
                                isone(-real(uval)) || rmul!(@view(values[startidx:end]), -real(uval))
                            else
                                isone(real(uval)) || rmul!(@view(values[startidx:end]), real(uval))
                            end
                        end
                        if complex && !iszero(imag(uval))
                            let lenalt=Int(data.lens[impart ? idx-1 : idx+1]),
                                dataidx=impart ? dataidx - lenalt : dataidx + len, r=dataidx:dataidx+lenalt-1
                                searchrange = startidx:length(indices)
                                searchview = @view(indices[searchrange])
                                for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                                    dupidx = findfirst(isequal(ind), searchview)
                                    if isnothing(dupidx)
                                        unsafe_push!(indices, ind)
                                        unsafe_push!(values, imag(uval) * val)
                                    else
                                        values[first(searchrange)+dupidx-1] += imag(uval) * val
                                    end
                                end
                            end
                        end
                    else
                        unsafe_append!(indices, @view(data.indices[r]))
                        unsafe_append!(values, @view(data.values[r]))
                    end
                    unsafe_push!(lens, length(indices) - startidx +1)
                    dataidx += len
                    idx += 1
                end
            else
                @twice impart complex begin
                    startidx = length(indices) +1
                    idx = 1
                    dataidx = 1
                    for col in 1:dim, row in col:dim
                        searchview = @view(indices[startidx:end])
                        @twice imdata (complex && row != col) begin
                            uval = u[i, col] * conj(u[j, row])
                            if imdata
                                uval -= u[i, row] * conj(u[j, col])
                                thisuval = impart ? real(uval) : -imag(uval)
                            else
                                uval += u[i, row] * conj(u[j, col])
                                thisuval = impart ? imag(uval) : real(uval)
                            end
                            len = Int(data.lens[idx])
                            r = dataidx:dataidx+len-1
                            dataidx += len
                            idx += 1
                            if !iszero(thisuval)
                                for (ind, val) in zip(@view(data.indices[r]), @view(data.values[r]))
                                    dupidx = findfirst(isequal(ind), searchview)
                                    if isnothing(dupidx)
                                        unsafe_push!(indices, ind)
                                        unsafe_push!(values, thisuval * val)
                                    else
                                        values[dupidx+startidx-1] += thisuval * val
                                    end
                                end
                            end
                        end
                    end
                    unsafe_push!(lens, length(indices) - startidx +1)
                end
            end
            #endregion
            (have_rot ? add_constr_rotated_quadratic! : add_constr_quadratic!)(state, to_soc!(indices, values, lens, have_rot))
            # we can keep the first element
            resize!(indices, firstlen)
            resize!(values, firstlen)
            resize!(lens, 1)
        end
        empty!(indices)
        empty!(values)
        empty!(lens)
    end
    return
end

function add_constr_sdddual_complex! end

@doc raw"""
    add_constr_sdddual_complex!(state, dim::Integer, data::IndvalsIterator{T,V}, u) where {T,V<:Real}

Add a constraint for membership in the dual cone to complex-valued scaled diagonally dominant matrices to the solver. `data` is
an iterator through the (unscaled) lower triangle of the matrix. A basis change is induced by `u`, with the meaning for the
primal cone that `M ∈ SDD(u) ⇔ M = u† Q u` with `Q ∈ SDD`.
For diagonal elements, there will be exactly one entry, which is the real part. For off-diagonal elements, the real part will
be followed by the imaginary part. Therefore, the coefficients are real-valued.

!!! warning
    This function will only be called if [`supports_sdd_complex`](@ref) returns `true` for the given state. If complex-valued
    sclaed diagonally dominant cones are not supported directly, a fallback to quadratic constraints is automatically
    performed.
"""
add_constr_sdddual_complex!(state, dim::Integer, data::IndvalsIterator{<:Any,<:Real}, u) =
    add_constr_sdddual!(state, dim, data, u, Val(true))

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