export moment_add_matrix!, moment_add_equality!, moment_setup!
# This file is for the commuting case; nevertheless, we already write the monomial index/multiplication calculation with a
# possible extension to the noncommuting case in mind.

# generic moment matrix constraint with
# - only real-valued monomials involved in the grouping, and only real-valued polynomials involved in the constraint (so if it
#   contains complex coefficients/monomials, imaginary parts cancel out)
# - or complex-valued monomials involved in the grouping, but the solver supports the complex-valued PSD cone explicitly
function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::AbstractPSDIndextype{Tri},
    type::Union{Tuple{Val{true},Val},Tuple{Val{false},Val{true}}}) where {Tri}
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    if dim == 1 || (dim == 2 && supports_quadratic(state))
        matrix_indexing = false
        tri = :U # we always create the data in U format; this ensures the scaling is already taken care of
    elseif indextype isa PSDIndextypeMatrixCartesian
        Tri ∈ (:L, :U) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype, type)))
        matrix_indexing = true
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U. However, if Tri === :L, we will fill
                 # the values with conjugates.
    else
        Tri ∈ (:L, :U, :F) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            type)))
        matrix_indexing = false
        tri = Tri
    end
    complex = type isa Tuple{Val{false},Val{true}}
    maxlen = maximum(length, constraint, init=0)
    colcount = (tri === :F ? (complex ? 2dim^2 - dim : dim^2) : (complex ? dim^2 : trisize(dim)))
    if matrix_indexing
        rows = FastVec{T}(buffer=2maxlen * colcount)
        indices = similar(rows)
        values = similar(rows, complex ? Complex{V} : V)
    else
        lens = Vector{SimplePolynomials.smallest_unsigned(2maxlen)}(undef, colcount)
        indices = FastVec{T}(buffer=2maxlen * colcount)
        values = similar(indices, V)
    end
    # introduce a method barrier to fix the potentially unknown eltype of lens make sure "dynamic" constants can be folded
    moment_add_matrix_helper!(state, T, V, grouping, constraint, indextype, Val(tri), Val(matrix_indexing), Val(complex), lg,
        block_size, dim, matrix_indexing ? (rows, indices, values) : (lens, indices, values))
end

function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::AbstractPSDIndextype{Tri}, ::Val{tri}, ::Val{matrix_indexing},
    ::Val{complex}, lg, block_size, dim, data) where {Tri,tri,matrix_indexing,complex}
    if matrix_indexing
        rows, indices, values = data
        i = zero(T)
    else
        sqrt2 = (state isa SOSWrapper && !(dim == 2 && supports_quadratic(state))) ? inv(sqrt(V(2))) : sqrt(V(2))
        lens, indices, values = data
        i = 1
    end
    @inbounds for (exp2, g₂) in enumerate(grouping)
        isreal_g₂ = !complex || isreal(g₂)
        for block_j in 1:block_size
            exp1_range = (tri === :F ? (1:lg) : (tri === :U ? (1:exp2) : (exp2:lg)))
            for (exp1, g₁) in zip(exp1_range, @view(grouping[exp1_range]))
                isreal_g₁ = !complex || isreal(g₁)
                for block_i in (tri === :F ? (1:block_size) : (tri === :U ? (1:(exp1 == exp2 ? block_j : block_size)) :
                                                                            ((exp1 == exp2 ? block_j : 1):block_size)))
                    # - !complex: all polynomials are real-valued in total. Only take into account canonical monomials, but
                    #   double the prefactor if their conjugated must also occur.
                    # - block_i == block_j && exp1 == exp2: we are on the total diagonal. Even if not all polynomials were
                    #   real, these are for sure.
                    # - block_i == block_j && isreal(g₁) && isreal(g₂): we are on the inner diagonal, but since the outer basis
                    #   elements are real-valued, canonical and non-canonicals will also occur in pairs.
                    total_real = !complex || (block_i == block_j && (exp1 == exp2 || (isreal_g₁ && isreal_g₂)))
                    ondiag = exp1 == exp2 && block_i == block_j
                    @twice secondtime (!matrix_indexing && !total_real) begin
                        curlen = 0
                        for term_constr in constraint[block_i, block_j]
                            mon_constr = monomial(term_constr)
                            coeff_constr = coefficient(term_constr)
                            if tri !== :F && !matrix_indexing && !ondiag
                                coeff_constr *= sqrt2
                            end
                            recoeff = real(coeff_constr)
                            imcoeff = imag(coeff_constr)
                            repart, impart, canonical = getreim(state, g₁, mon_constr, SimpleConjMonomial(g₂))
                            if total_real
                                # Even if `complex` - so we are on the diagonal and the values should be Complex{V} - we can
                                # just do the push!, the conversion is implicitly done; as the imaginary part should be zero,
                                # we don't need to discriminate against matrix_indexing.
                                if canonical # ≡ iscanonical(mon_constr)
                                    if !iszero(recoeff)
                                        matrix_indexing && unsafe_push!(rows, i)
                                        unsafe_push!(indices, repart)
                                        unsafe_push!(values, repart == impart ? recoeff : V(2) * recoeff)
                                        curlen += 1
                                    end
                                    if repart == impart
                                        @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                        @assert(isreal(mon_constr))
                                    elseif !iszero(imcoeff)
                                        matrix_indexing && unsafe_push!(rows, i)
                                        unsafe_push!(indices, impart)
                                        unsafe_push!(values, V(-2) * imcoeff)
                                        curlen += 1
                                    end
                                end
                            else
                                # We are not on the diagonal, so we must work with every entry. Updating might become necessary
                                # as the conjugate can pop up later.
                                if matrix_indexing
                                    # Interpretation: Complex(x, y) means that the real part of the entry is given by x and
                                    #                 the imaginary part by y.
                                    let coeff=Complex(recoeff, Tri === :L ? -imcoeff : imcoeff)
                                        if !iszero(coeff)
                                            @pushorupdate!(repart, i, coeff)
                                            repart == impart ||
                                                @pushorupdate!(impart, i,
                                                    canonical ? Complex(-imcoeff, Tri === :L ? -recoeff : recoeff) :
                                                                Complex(imcoeff, Tri === :L ? recoeff : -recoeff))
                                        end
                                    end
                                elseif !secondtime
                                    iszero(recoeff) || @pushorupdate!(repart, i, recoeff)
                                    repart == impart || iszero(imcoeff) ||
                                        @pushorupdate!(impart, i, canonical ? -imcoeff : imcoeff)
                                else
                                    iszero(imcoeff) || @pushorupdate!(repart, i, imcoeff)
                                    repart == impart || iszero(recoeff) ||
                                        @pushorupdate!(impart, i, canonical ? recoeff : -recoeff)
                                end
                            end
                        end
                        if !matrix_indexing
                            lens[i] = Core.Intrinsics.trunc_int(eltype(lens), curlen) # no need to check for overflows
                        end
                        i += one(i)
                    end
                    if !matrix_indexing && complex && total_real && !ondiag
                        # We skipped the addition of an imaginary part because it is zero. But we have to tell this to the
                        # solver.
                        lens[i] = zero(eltype(lens))
                        i += one(i)
                    end
                end
            end
        end
    end
    if matrix_indexing
        return (complex ? add_constr_psd_complex! : add_constr_psd!)(
            state, dim, PSDMatrixCartesian{_get_offset(indextype)}(dim, Tri, finish!(rows), finish!(indices), finish!(values))
        )
    else
        @assert(i == lastindex(lens) +1)
        @inbounds if dim == 1
            @assert(length(lens) == 1)
            add_constr_nonnegative!(state, indices, values)
        elseif dim == 2 && supports_quadratic(state)
            @assert(length(lens) == (complex ? 4 : 3))
            # Note: indices/values represent the vectorized PSD cone (upper triangle) with off-diagonals already pre-multiplied
            # by sqrt(2).
            range1 = 1:lens[1]
            range2 = last(range1)+1:last(range1)+lens[2]
            range3 = last(range2)+1:last(range2)+lens[3]
            indices₁₁ = @view(indices[range1])
            values₁₁ = @view(values[range1])
            indicesₒᵣ = @view(indices[range2])
            valuesₒᵣ = @view(values[range2])
            fourquads = complex
            if fourquads
                indicesₒᵢ = @view(indices[range3])
                valuesₒᵢ = @view(values[range3])
                range4 = last(range3)+1:last(range3)+lens[4]
                indices₂₂ = @view(indices[range4])
                values₂₂ = @view(values[range4])
                if all(iszero, valuesₒᵢ)
                    fourquads = false
                elseif all(iszero(valuesₒᵣ))
                    indicesₒᵣ = indicesₒᵢ
                    valuesₒᵣ = valuesₒᵢ
                    fourquads = false
                end
            else
                indices₂₂ = @view(indices[range3])
                values₂₂ = @view(values[range3])
            end
            if !fourquads
                add_constr_quadratic!(state, (indices₁₁, values₁₁), (indices₂₂, values₂₂), (indicesₒᵣ, valuesₒᵣ))
            else
                add_constr_quadratic!(state, (indices₁₁, values₁₁), (indices₂₂, values₂₂), (indicesₒᵣ, valuesₒᵣ),
                    (indicesₒᵢ, valuesₒᵢ))
            end
        else
            (complex ? add_constr_psd_complex! : add_constr_psd!)(
                state, dim, PSDVector(finish!(indices), finish!(values), lens)
            )
        end
    end
    return
end

# generic moment matrix constraint with complex-valued monomials involved in the grouping, but the solver does not support the
# complex PSD cone explicitly
function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::AbstractPSDIndextype{Tri},
    ::Tuple{Val{false},Val{false}}) where {Tri}
    matrix_indexing = indextype isa PSDIndextypeMatrixCartesian
    if matrix_indexing
        Tri ∈ (:L, :U) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            (Val(false), Val(false)))))
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U.
    else
        Tri ∈ (:L, :U, :F) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            (Val(false), Val(false)))))
        tri = Tri
        sqrt2 = (state isa SOSWrapper && !(dim == 2 && supports_quadratic(state))) ? inv(sqrt(V(2))) : sqrt(V(2))
    end
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    if dim == 1 || (dim == 2 && supports_quadratic(state))
        # in these cases, we will rewrite the Hermitian PSD cone in terms of linear or quadratic constraints, so break off
        return moment_add_matrix_helper!(state, T, V, grouping, constraint, indextype, (Val(false), Val(true)))
    end
    maxlen = maximum(length, constraint, init=0)
    colcount = tri === :F ? 4dim^2 : trisize(2dim)
    rows = FastVec{T}(buffer=2maxlen * colcount)
    indices = similar(rows)
    values = similar(rows, V)
    # Here we implement the default way: Z ⪰ 0 ⟺ [Re Z  -Im Z; Im Z  Re Z] ⪰ 0.
    # This is relatively simple, as "rows" encodes the position of the element in the matrix (in linear index form of the upper
    # triangle columnwise).
    dimT = T(dim)
    items = tri === :F ? dimT^2 : trisize(dimT)
    i1 = zero(T)
    if tri === :U
        i2order = items
        i3 = items + dimT
    elseif tri === :L
        i2order = dimT
        i3 = (T(3) * dimT^2 + dimT) >> 1
    else
        i2order = dimT
        i2other = T(2) * items
        i3 = i2other + dimT
    end
    col = zero(T)
    @inbounds for (exp2, g₂) in enumerate(grouping)
        isreal_g₂ = isreal(g₂)
        for block_j in 1:block_size
            row = tri === :L ? col : zero(T)
            exp1_range = (tri === :F ? (1:lg) : (tri === :U ? (1:exp2) : (exp2:lg)))
            if tri === :U
                i2other = items + col + (row * (row + T(2) * dimT + one(T))) >> 1
            elseif tri === :L
                i2other = i2order
            end
            for (exp1, g₁) in zip(exp1_range, @view(grouping[exp1_range]))
                isreal_g₁ = isreal(g₁)
                for block_i in (tri === :F ? (1:block_size) : (tri === :U ? (1:(exp1 == exp2 ? block_j : block_size)) :
                                                                            ((exp1 == exp2 ? block_j : 1):block_size)))
                    total_real = block_i == block_j && (exp1 == exp2 || (isreal_g₁ && isreal_g₂))
                    ondiag = exp1 == exp2 && block_i == block_j
                    curlen = 0
                    for term_constr in constraint[block_i, block_j]
                        mon_constr = monomial(term_constr)
                        coeff_constr = coefficient(term_constr)
                        if tri !== :F && !matrix_indexing && !ondiag
                            coeff_constr *= sqrt2
                        end
                        recoeff = real(coeff_constr)
                        imcoeff = imag(coeff_constr)
                        repart, impart, canonical = getreim(state, g₁, mon_constr, SimpleConjMonomial(g₂))
                        if total_real
                            if canonical
                                if !iszero(recoeff)
                                    unsafe_push!(rows, i1, i3)
                                    unsafe_push!(indices, repart, repart)
                                    if repart != impart
                                        recoeff *= V(2)
                                    end
                                    unsafe_push!(values, recoeff, recoeff)
                                    curlen += 2
                                end
                                if repart == impart
                                    @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                    @assert(isreal(mon_constr))
                                elseif !iszero(imcoeff)
                                    unsafe_push!(rows, i1, i3)
                                    unsafe_push!(indices, impart, impart)
                                    imcoeff *= V(-2)
                                    unsafe_push!(values, imcoeff, imcoeff)
                                    curlen += 2
                                end
                            end
                        else
                            @assert(i2order != i2other)
                            if !iszero(recoeff)
                                @pushorupdate!(repart, i1, recoeff, i3, recoeff)
                                if repart != impart
                                    care = canonical ? recoeff : -recoeff
                                    # We don't really need the L/U distinction, as it does not matter where to place the
                                    # minus. However, to make testing simpler, we keep our promise.
                                    if tri === :L || tri === :F
                                        @pushorupdate!(impart, i2order, care, i2other, -care)
                                    else
                                        @pushorupdate!(impart, i2other, care, i2order, -care)
                                    end
                                end
                            end
                            if !iszero(imcoeff)
                                if tri === :L || tri === :F
                                    @pushorupdate!(repart, i2order, imcoeff, i2other, -imcoeff)
                                else
                                    @pushorupdate!(repart, i2other, imcoeff, i2order, -imcoeff)
                                end
                                if repart != impart
                                    care = canonical ? -imcoeff : imcoeff
                                    @pushorupdate!(impart, i1, care, i3, care)
                                end
                            end
                        end
                    end
                    row += one(T)
                    i1 += one(T)
                    if tri === :U
                        i2order += one(T)
                        i2other += dimT + row
                    elseif tri === :L
                        i2order += one(T)
                        i2other += T(2) * dimT - row
                    else
                        i2order += one(T)
                        i2other += one(T)
                    end
                    i3 += one(T)
                end
            end
            if tri === :U
                i2order += dimT
                i3 += dimT
            elseif tri === :L
                i1 += dimT
                i2order += dimT
            else
                i1 += dimT
                i2order += dimT
                i2other += dimT
                i3 += dimT
            end
            col += one(T)
        end
    end
    if matrix_indexing
        return add_constr_psd!(state, 2dim, PSDMatrixCartesian{_get_offset(indextype)}(2dim, Tri, finish!(rows),
            finish!(indices), finish!(values)))
    else
        # We constructed everything in matrix form, as it was easier to assign arbitrary positions in this way; but now we need
        # to convert it to the iterative form. We cannot re-use rows for this, unfortunately. It may happen that some rows
        # (i.e., entries for the PSD matrix) are zero and don't arise in rows, although they will have to be present in lens.
        # So to avoid potentially overwriting data, we need a whole new vector.
        @inbounds let rows=finish!(rows), indices=finish!(indices), values=finish!(values)
            lens = Vector{SimplePolynomials.smallest_unsigned(2maxlen)}(undef, (Tri === :F ? 4dim^2 : trisize(2dim)))
            sort_along!(rows, indices, values)
            row = zero(T)
            i = 1
            j = 1
            remaining = length(rows)
            while !iszero(remaining)
                currow = rows[j]
                while currow > row
                    lens[i] = zero(eltype(lens))
                    i += 1
                    row += one(T)
                end
                curlen = one(eltype(lens))
                j += 1
                remaining -= 1
                while !iszero(remaining) && rows[j] == currow
                    curlen += one(eltype(lens))
                    j += 1
                    remaining -= 1
                end
                lens[i] = curlen
                i += 1
                row += one(T)
            end
            fill!(@view(lens[i:end]), zero(eltype(lens)))

            add_constr_psd!(state, 2dim, PSDVector(indices, values, lens))
        end
    end
end

"""
    moment_add_matrix!(state, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}})

Parses a constraint in the moment hierarchy with a basis given in `grouping` (this might also be a partial basis due to
sparsity), premultiplied by `constraint` (which may be the unit polynomial for the moment matrix) and calls the appropriate
solver functions to set up the problem structure.

To make this function work for a solver, implement the following low-level primitives:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_equality!`](@ref).
"""
moment_add_matrix!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::Union{P,<:AbstractMatrix{P}}) where {P<:SimplePolynomial} =
    moment_add_matrix_helper!(
        state,
        Base.promote_op(mindex, typeof(state), monomial_type(P)),
        realtype(coefficient_type(P)),
        grouping,
        constraint isa AbstractMatrix ? constraint : ScalarMatrix(constraint),
        psd_indextype(state),
        (Val((length(grouping) == 1 || isreal(grouping)) &&
             (constraint isa SimplePolynomial || isreal(constraint))), Val(supports_complex_psd(state)))
    )

"""
    moment_add_equality!(state, grouping::SimpleMonomialVector, constraint::SimplePolynomial)

Parses a polynomial equality constraint for moments and calls the appropriate solver functions to set up the problem structure.
`grouping` contains the basis that will be squared in the process to generate the prefactor.

To make this function work for a solver, implement the following low-level primitives:
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_finalize_fix!`](@ref) (optional)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_matrix!`](@ref).
"""
function moment_add_equality!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::P) where {Nr,Nc,I<:Integer,P<:SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}}}
    # We need to traverse all unique elements in groupings * groupings†. For purely complex-valued groupings, this is the full
    # list; as soon as we have a real variable present, it is smaller.
    # To avoid rehashings, get an overestimator of the total grouping size first.
    unique_groupings = sizehint!(Set{FastKey{I}}(), iszero(Nc) ? trisize(length(grouping)) : length(grouping)^2)
    complex_groupings = 0
    for (i, g₁) in enumerate(grouping)
        if !iszero(Nc)
            complex_groupings += !isreal(g₁)
        end
        # In the real case, we can skip the first i-1 entries as they would lead to duplicates.
        # In the complex case, we can also skip the first i-1 entries, as they would lead to exact conjugates, which in the
        # end give rise to the same conditions.
        for g₂ in Iterators.drop(grouping, i -1)
            # We don't use mindex, as this can have unintended side-effects on the solver state (such as creating a
            # representation for this monomial, although we don't even know whether we need it - if constraint does not contain
            # a constant term, this function must not automatically add all the squared groupings as monomials, even if they
            # will probably appear at some place).
            push!(unique_groupings, FastKey(monomial_index(g₁, SimpleConjMonomial(g₂))))
        end
    end

    # Assume we have a grouping g = (gᵣ + im*gᵢ) and a polynomial p = pᵣ + im*pᵢ, where the individual parts are real-valued.
    # Then, add_equality! means that g*p = 0 and ḡ*p = 0. Of course we can also conjugate everything. We must split each
    # constraint into its real and imaginary parts:
    # (I)   Re(g*p) = gᵣ*pᵣ - gᵢ*pᵢ
    # (II)  Im(g*p) = gᵣ*pᵢ + gᵢ*pᵣ
    # (III) Re(ḡ*p) = gᵣ*pᵣ + gᵢ*pᵢ
    # (IV)  Im(ḡ*p) = gᵣ*pᵢ - gᵢ*pᵣ
    # To analyze this (which would be easier if we added and subtracted the equalities, but in the SimplePolynomials setup, the
    # given form is most easy to handle), let's consider linear dependencies.
    # - If the constraint is real-valued, (III) is equal to (I) and (IV) is -(II), so we only take (I) and (II).
    # - If the grouping is real-valued, (III) is equal to (I) and (IV) is equal to (II), so we only take (I) and (II).
    # - If both are real-valued, (III) is equal to (I) while (II) and (IV) are zero, so we only take (I).
    # - If both are complex-valued, all constraints are linearly independent.
    # Rearranging this, we always take (I); if at least one is complex-valued, we also take (II); if both are, we take all.
    # Note that we don't have to consider the conjugates of the groupings separately, as they only yield a global sign in the
    # zero-equality.
    real_groupings = length(grouping) - complex_groupings
    real_constr = isreal(constraint)

    # now we can
    # - combine real_grouping with real_grouping, getting trisize(real_grouping) purely real groupings
    # - combine real_grouping with complex_grouping, getting real_grouping*complex_grouping complex-valued groupings up to
    #   conjugation
    # - combine complex_grouping with itself conjugated, getting complex_grouping purely real groupings
    # - combine complex_grouping with all other entries conjugates, getting trisize(complex_groupings-1) complex-valued
    #   grouping up to conjugation
    totalsize = let groupings_re=trisize(real_groupings) + complex_groupings,
        groupings_cp=real_groupings*complex_groupings + trisize(complex_groupings -1)
        groupings_re + groupings_cp + # we always get (I)
        groupings_cp + # complex-valued groupings give (II) for all constraints
        (real_constr ? 0 :
                       groupings_re + groupings_cp - # complex-valued constraints give (II) for all groupings
                       groupings_cp +                # subtract the intersection that counted (II) twice
                       groupings_cp + groupings_cp   # complex-valued constraints and groupings give (III) and (IV)
        )
    end

    constrstate = @inline add_constr_fix_prepare!(state, totalsize)
    V = realtype(coefficient_type(P))
    indices₁ = FastVec{Base.promote_op(mindex, typeof(state), monomial_type(P))}(buffer=2length(constraint))
    values₁ = similar(indices₁, V)
    # While we could conditionally define those variables only if the requirements are satisfied, the compiler might not be
    # able to infer that we only use them later on if the same conditions (potentially stricter) are met. So define them
    # always, but not using any memory.
    indices₂ = similar(indices₁, 0, buffer=iszero(complex_groupings) && real_constr ? 0 : 2length(constraint))
    values₂ = similar(indices₂, V)
    indices₃ = similar(indices₁, 0, buffer=iszero(complex_groupings) || real_constr ? 0 : 2length(constraint))
    values₃ = similar(indices₃, V)
    indices₄ = similar(indices₃)
    values₄ = similar(values₃)

    e = ExponentsAll{Nr+2Nc,I}()

    for grouping_idx in unique_groupings
        grouping = SimpleMonomial{Nr,Nc}(unsafe, e, convert(I, grouping_idx))
        real_grouping = isreal(grouping)
        skip₂ = real_constr && real_grouping
        for term_constr in constraint
            mon_constr = monomial(term_constr)
            coeff_constr = coefficient(term_constr)
            recoeff = real(coeff_constr)
            imcoeff = imag(coeff_constr)
            repart, impart, canonical = getreim(state, grouping, mon_constr)
            if !iszero(recoeff)
                pushorupdate!(indices₁, repart, values₁, recoeff)
                skip₂ || repart == impart || pushorupdate!(indices₂, impart, values₂, canonical ? recoeff : -recoeff)
            end
            if !iszero(imcoeff)
                repart == impart || pushorupdate!(indices₁, impart, values₁, canonical ? -imcoeff : imcoeff)
                skip₂ || pushorupdate!(indices₂, repart, values₂, imcoeff)
            end
            if !real_grouping && !real_constr
                repart₂, impart₂, canonical₂ = getreim(state, SimpleConjMonomial(grouping), mon_constr)
                if !iszero(recoeff)
                    pushorupdate!(indices₃, repart₂, values₃, recoeff)
                    repart₂ == impart₂ || pushorupdate!(indices₄, impart₂, values₄, canonical₂ ? recoeff : -recoeff)
                end
                if !iszero(imcoeff)
                    repart₂ == impart₂ || pushorupdate!(indices₃, impart₂, values₃, canonical₂ ? -imcoeff : imcoeff)
                    pushorupdate!(indices₄, repart₂, values₄, imcoeff)
                end
            end
        end
        constrstate = @inline add_constr_fix!(state, constrstate, indices₁, values₁, zero(V))
        empty!(indices₁); empty!(values₁)
        if !skip₂
            if !isempty(indices₂)
                constrstate = @inline add_constr_fix!(state, constrstate, indices₂, values₂, zero(V))
                empty!(indices₂); empty!(values₂)
            end
            if !real_grouping && !real_constr
                if !isempty(indices₃)
                    constrstate = @inline add_constr_fix!(state, constrstate, indices₃, values₃, zero(V))
                    empty!(indices₃); empty!(values₃)
                end
                if !isempty(indices₄)
                    constrstate = @inline add_constr_fix!(state, constrstate, indices₄, values₄, zero(V))
                    empty!(indices₄); empty!(values₄)
                end
            end
        end
    end
    @inline add_constr_fix_finalize!(state, constrstate)
    return
end

"""
    moment_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings)

Sets up all the necessary moment matrices, variables, constraints, and objective of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_fix_finalize!`](@ref) (optional)
- [`fix_objective!`](@ref)

!!! warning "Indices"
    The variable indices used in all solver functions directly correspond to the indices given back by [`mindex`](@ref).
    However, in a sparse problem there may be far fewer indices present; therefore, when the problem is finally given to the
    solver, care must be taken to eliminate all unused indices.

!!! info "Order"
    This function is guaranteed to set up the fixed constraints first, then followed by all the others. However, the order of
    nonnegative, quadratic, and PSD constraints is undefined (depends on the problem).

See also [`sos_setup!`](@ref), [`moment_add_matrix!`](@ref), [`moment_add_equality!`](@ref).
"""
function moment_setup!(state, relaxation::AbstractRelaxation{<:Problem{P}}, groupings::RelaxationGroupings) where {P}
    problem = poly_problem(relaxation)
    T = Base.promote_op(mindex, typeof(state), monomial_type(P))
    V = realtype(coefficient_type(problem.objective))

    # fixed items
    # fix constant term to 1
    if isone(problem.prefactor)
        @inline add_constr_fix_finalize!(
            state,
            add_constr_fix!(
                state,
                add_constr_fix_prepare!(state, 1),
                StackVec(mindex(state, constant_monomial(P))),
                StackVec(one(V)),
                one(V)
            )
        )
    else
        let buffer=length(problem.prefactor), indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
            for t in problem.prefactor
                # We know that if imaginary parts pop up somewhere, they will cancel out in the end, as the conjugates also
                # appear. So use the same strategy as always and only add canonical ones, but doubled.
                mon = monomial(t)
                coeff = coefficient(t)
                recoeff = real(coeff)
                imcoeff = imag(coeff)
                repart, impart, canonical = getreim(state, mon)
                if repart == impart
                    @assert(iszero(imcoeff)) # else the objective would not be real-valued
                    unsafe_push!(indices, repart)
                    unsafe_push!(values, recoeff)
                elseif canonical
                    if !iszero(recoeff)
                        unsafe_push!(indices, repart)
                        unsafe_push!(values, recoeff + recoeff)
                    end
                    if !iszero(imcoeff)
                        unsafe_push!(indices, impart)
                        unsafe_push!(values, -(imcoeff + imcoeff))
                    end
                end
            end
            @inline add_constr_fix_finalize!(
                state,
                add_constr_fix!(state, add_constr_fix_prepare!(state, 1), indices, values, one(V))
            )
        end
    end
    for (groupingsᵢ, constrᵢ) in zip(groupings.zeros, problem.constr_zero)
        for grouping in groupingsᵢ
            moment_add_equality!(state, collect_grouping(grouping), constrᵢ)
        end
    end

    # SOS term for objective
    constantP = SimplePolynomial(constant_monomial(P), coefficient_type(problem.objective))
    for grouping in groupings.obj
        moment_add_matrix!(state, collect_grouping(grouping), constantP)
    end
    # localizing matrices
    for (groupingsᵢ, constrᵢ) in zip(groupings.nonnegs, problem.constr_nonneg)
        for grouping in groupingsᵢ
            moment_add_matrix!(state, collect_grouping(grouping), constrᵢ)
        end
    end
    for (groupingsᵢ, constrᵢ) in zip(groupings.psds, problem.constr_psd)
        for grouping in groupingsᵢ
            moment_add_matrix!(state, collect_grouping(grouping), constrᵢ)
        end
    end

    # Riesz functional in the objective
    let buffer=length(problem.objective), indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
        for t in problem.objective
            mon = monomial(t)
            coeff = coefficient(t)
            recoeff = real(coeff)
            imcoeff = imag(coeff)
            repart, impart, canonical = getreim(state, mon)
            # repart is the variable that is associated with the real part of the monomial mon;
            # impart is the variable associated with the imaginary part of the canonicalized monomial mon.
            if repart == impart
                @assert(iszero(imcoeff)) # else the objective would not be real-valued
                unsafe_push!(indices, repart)
                unsafe_push!(values, recoeff)
            elseif canonical # skip over noncanonicals - to have a real-valued objective, we know how they must look like
                if !iszero(recoeff)
                    unsafe_push!(indices, repart)
                    unsafe_push!(values, recoeff + recoeff)
                end
                if !iszero(imcoeff)
                    unsafe_push!(indices, impart)
                    unsafe_push!(values, -(imcoeff + imcoeff))
                end
            end
        end
        fix_objective!(state, finish!(indices), finish!(values))
    end

    return
end