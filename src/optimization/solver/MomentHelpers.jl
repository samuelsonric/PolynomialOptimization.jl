export moment_add_matrix!, moment_add_equality!, moment_setup!
# This file is for the commuting case; nevertheless, we already write the monomial index/multiplication calculation with a
# possible extension to the noncommuting case in mind.

function to_soc!(indices, values, lens, supports_rotated)
    @assert(length(lens) ≥ 2)
    @inbounds if supports_rotated
        for j in length(lens):-1:3
            iszero(lens[j]) && deleteat!(lens, j)
        end
    else
        range₁ = 1:lens[1]
        range₂ = lens[1]+1:lens[1]+lens[2]
        prevlen = last(range₂)
        inds = @views count_uniques(indices[range₁], indices[range₂])
        total = 2inds
        if total > prevlen
            Base._growat!(indices, prevlen +1, total - prevlen)
            Base._growat!(values, prevlen +1, total - prevlen)
        end
        # we must make a copy to avoid potential overwrites, doesn't matter in which direction we work
        oldinds = indices[1:prevlen]
        oldvals = values[1:prevlen]
        count_uniques(@view(oldinds[range₁]), @view(oldinds[range₂]), let oldinds=oldinds, oldvals=oldvals
            (o₁, i₁, i₂) -> @inbounds begin
                if ismissing(i₁)
                    indices[o₁+inds] = indices[o₁] = oldinds[lens[1]+i₂]
                    values[o₁+inds] = -(values[o₁] = oldvals[lens[1]+i₂])
                elseif ismissing(i₂)
                    indices[o₁+inds] = indices[o₁] = oldinds[i₁]
                    values[o₁+inds] = values[o₁] = oldvals[i₁]
                else
                    indices[o₁+inds] = indices[o₁] = oldinds[i₁]
                    values[o₁] = oldvals[i₁] + oldvals[lens[1]+i₂]
                    values[o₁+inds] = oldvals[i₁] - oldvals[lens[1]+i₂]
                end
            end
        end)
        lens[1] = inds
        lens[2] = inds
        for j in length(lens):-1:2
            iszero(lens[j]) && deleteat!(lens, j)
        end
    end
    return IndvalsIterator(unsafe, indices, values, lens)
end

# generic moment matrix constraint with
# - only real-valued monomials involved in the grouping, and only real-valued polynomials involved in the constraint (so if it
#   contains complex coefficients/monomials, imaginary parts cancel out)
# - or complex-valued monomials involved in the grouping, but the solver supports the complex-valued PSD cone explicitly
# - or DD/SDD representations in the real case and in the complex case if the quadratic cone is requested
function moment_add_matrix_helper!(state::AnySolver{T,V}, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri},
    type::Union{Tuple{Val{true},Val},Tuple{Val{false},Val{true}}}, representation::RepresentationMethod) where {T,V,Tri}
    # Note: We rely on a lot of checks having already been done, so that the combination of arguments is always correct. This
    # function should not be called from the outside.

    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    matrix_indexing = indextype isa PSDIndextypeMatrixCartesian
    dim = lg * block_size
    complex = type isa Tuple{Val{false},Val{true}}
    if matrix_indexing
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U. However, if Tri === :L, we will fill
                 # the values with conjugates.
    else
        tri = Tri
    end
    maxlen = maximum(length, constraint, init=0)
    colcount = (tri === :F ? (complex ? 2dim^2 - dim : dim^2) : (complex ? dim^2 : trisize(dim)))
    indlen = 2maxlen * colcount
    if matrix_indexing
        rows = FastVec{T}(buffer=indlen)
        indices = similar(rows)
        values = similar(rows, complex ? Complex{V} : V)
    else
        lens = FastVec{SimplePolynomials.smallest_unsigned(2maxlen)}(buffer=colcount)
        indices = FastVec{T}(buffer=indlen)
        values = similar(indices, V)
    end

    # introduce a method barrier to fix the potentially unknown eltype of lens due to the dynamic maxlen
    moment_add_matrix_helper!(state, grouping, constraint, indextype, Val(tri), Val(matrix_indexing), Val(complex), lg,
        block_size, dim, matrix_indexing ? (rows, indices, values) : (lens, indices, values), representation)
end

function moment_add_matrix_helper!(state::AnySolver{T,V}, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri}, ::Val{tri}, ::Val{matrix_indexing},
    ::Val{complex}, lg, block_size, dim, data, representation::RepresentationMethod) where {T,V,Tri,tri,matrix_indexing,complex}
    if matrix_indexing
        rows, indices, values = data
        row = zero(T)
        scaleoffdiags = false
    else
        lens, indices, values = data
        if representation isa RepresentationPSD
            # Off-diagonals are multiplied by √2 in order to put variables into the vectorized PSD cone. Even if state isa
            # SOSWrapper (then, the variables directly correspond to a vectorized PSD cone), the actual values in the PSD
            # matrix are still multiplied by 1/√2, so we must indeed always multiply the coefficients by √2 to undo this.
            # We also have to account for the unwanted factor of 2 in the rotated quadratic cone.
            # In the case of a (normal) quadratic cone, we canonically take the rotated cone and transform it by multiplying
            # the left-hand side by 1/√2, giving (x₁/√2)² ≥ (x₂/√2)² + ∑ᵢ (√2 xᵢ)² ⇔ x₁² ≥ x₂² + ∑ᵢ (2 xᵢ)².
            scaleoffdiags = tri !== :F
            if scaleoffdiags
                if dim == 2
                    rquad = supports_rotated_quadratic(state)
                    quad = supports_quadratic(state)
                    if !rquad && quad
                        scaling = V(2)
                    else
                        scaling = sqrt(V(2))
                    end
                else
                    scaling = sqrt(V(2))
                end
            end
        else
            rquad = quad = false
            scaleoffdiags = true
            scaling = (representation isa RepresentationDD && ((complex && supports_dd_complex(state)) ||
                                                               (!complex && supports_dd(state)))) ||
                      (representation isa RepresentationSDD && ((complex && supports_sdd_complex(state)) ||
                                                                (!complex && supports_sdd(state)))) ? sqrt(V(2)) : V(2)
        end
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
                            if scaleoffdiags && !ondiag
                                coeff_constr *= scaling
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
                                        matrix_indexing && unsafe_push!(rows, row)
                                        unsafe_push!(indices, repart)
                                        unsafe_push!(values, repart == impart ? recoeff : V(2) * recoeff)
                                        curlen += 1
                                    end
                                    if repart == impart
                                        @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                        @assert(isreal(mon_constr))
                                    elseif !iszero(imcoeff)
                                        matrix_indexing && unsafe_push!(rows, row)
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
                                            @pushorupdate!(repart, row, coeff)
                                            repart == impart ||
                                                @pushorupdate!(impart, row,
                                                    canonical ? Complex(-imcoeff, Tri === :L ? -recoeff : recoeff) :
                                                                Complex(imcoeff, Tri === :L ? recoeff : -recoeff))
                                        end
                                    end
                                elseif !secondtime
                                    iszero(recoeff) || @pushorupdate!(repart, row, recoeff)
                                    repart == impart || iszero(imcoeff) ||
                                        @pushorupdate!(impart, row, canonical ? -imcoeff : imcoeff)
                                else
                                    iszero(imcoeff) || @pushorupdate!(repart, row, imcoeff)
                                    repart == impart || iszero(recoeff) ||
                                        @pushorupdate!(impart, row, canonical ? recoeff : -recoeff)
                                end
                            end
                        end
                        if matrix_indexing
                            row += one(row)
                        else
                            unsafe_push!(lens, Core.Intrinsics.trunc_int(eltype(lens), curlen)) # no check for overflows
                        end
                    end
                    if !matrix_indexing && complex && total_real && !ondiag
                        # We skipped the addition of an imaginary part because it is zero. But we have to tell this to the
                        # solver.
                        unsafe_push!(lens, zero(eltype(lens)))
                    end
                end
            end
        end
    end
    if matrix_indexing # implies dim ≥ 3
        mc = PSDMatrixCartesian{_get_offset(indextype)}(dim, Tri, finish!(rows), finish!(indices), finish!(values))
        if representation isa RepresentationPSD
            return (complex ? add_constr_psd_complex! : add_constr_psd!)(state, dim, mc)
        elseif representation isa RepresentationDD
            return (complex ? add_constr_dddual_complex! : add_constr_dddual!)(state, dim, mc)
        else
            return (complex ? add_constr_sdddual_complex! : add_constr_sdddual!)(state, dim, mc)
        end
    else
        let indices=finish!(indices), values=finish!(values)
            @inbounds if dim == 1
                @assert(length(lens) == 1)
                add_constr_nonnegative!(state, Indvals(indices, values))
            elseif dim == 2 && (rquad || quad)
                @assert(length(lens) == (complex ? 4 : 3))
                # Note: indices/values represent the vectorized PSD cone (lower triangle) with off-diagonals already
                # pre-multiplied by sqrt(2). To make this exactly equivalent to a rotated quadratic cone, we must bring the
                # last element to the second position.
                @views @unroll for x in (indices, values)
                    reverse!(x[lens[1]+1:end-lens[end]])
                    reverse!(x[end-lens[end]+1:end])
                    reverse!(x[lens[1]+1:end])
                end
                if complex
                    lens[2], lens[3], lens[4] = lens[4], lens[2], lens[3]
                else
                    lens[2], lens[3] = lens[3], lens[2]
                end
                (rquad ? add_constr_rotated_quadratic! : add_constr_quadratic!)(state, to_soc!(indices, values, lens, rquad))
            else # implies dim ≥ 3
                ii = IndvalsIterator(indices, values, lens)
                if representation isa RepresentationPSD
                    return (complex ? add_constr_psd_complex! : add_constr_psd!)(state, dim, ii)
                elseif representation isa RepresentationDD
                    if complex
                        if supports_dd_complex(state)
                            return add_constr_dddual_complex!(state, dim, ii, representation.u)
                        else
                            return moment_add_dddual_transform!(state, dim, ii, representation.u, Val(true))
                        end
                    else
                        if supports_dd(state)
                            return add_constr_dddual!(state, dim, ii, representation.u)
                        else
                            return moment_add_dddual_transform!(state, dim, ii, representation.u, Val(false))
                        end
                    end
                else
                    if complex
                        if supports_sdd_complex(state)
                            return add_constr_sdddual_complex!(state, dim, ii, representation.u)
                        else
                            return moment_add_sdddual_transform!(state, dim, ii, representation.u, Val(true))
                        end
                    else
                        if supports_sdd(state)
                            return add_constr_sdddual!(state, dim, ii, representation.u)
                        else
                            return moment_add_sdddual_transform!(state, dim, ii, representation.u, Val(false))
                        end
                    end
                end
            end
        end
    end
    return
end

# generic moment matrix constraint with complex-valued monomials involved in the grouping, but the solver does not support the
# complex PSD cone explicitly
function moment_add_matrix_helper!(state::AnySolver{T,V}, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri},
    ::Tuple{Val{false},Val{false}}, representation::Union{<:RepresentationDD,<:RepresentationSDD,RepresentationPSD}) where {T,V,Tri}
    # Note: We rely on a lot of checks having already been done, so that the combination of arguments is always correct. This
    # function should not be called from the outside.

    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    matrix_indexing = indextype isa PSDIndextypeMatrixCartesian
    dim = lg * block_size
    dim2 = 2dim
    if matrix_indexing
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U.
        scaleoffdiags = false
    else
        tri = Tri
        if representation isa RepresentationPSD
            scaleoffdiags = tri !== :F
            if scaleoffdiags
                scaling = sqrt(V(2))
            end
        else
            scaleoffdiags = true
            scaling = (representation isa RepresentationDD ? supports_dd(state) : supports_sdd(state)) ? sqrt(V(2)) : V(2)
        end
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
                        if scaleoffdiags && !ondiag
                            coeff_constr *= scaling
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
        return add_constr_psd!(state, dim2, PSDMatrixCartesian{_get_offset(indextype)}(dim2, Tri, finish!(rows),
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

            ii = IndvalsIterator(indices, values, lens)
            if representation isa RepresentationPSD
                return add_constr_psd!(state, 2dim, ii)
            elseif representation isa RepresentationDD
                if supports_dd(state)
                    return add_constr_dddual!(state, 2dim, ii, representation.u)
                else
                    return moment_add_dddual_transform!(state, 2dim, ii, representation.u, Val(false))
                end
            else
                if supports_sdd(state)
                    return add_constr_sdddual!(state, 2dim, ii, representation.u)
                else
                    return moment_add_sdddual_transform!(state, 2dim, ii, representation.u, Val(false))
                end
            end
        end
    end
end

function moment_add_dddual_transform!(state::AnySolver{T,V}, dim::Integer, data::IndvalsIterator{T,V}, u, ::Val{complex}) where {T,V,complex}
    !complex && (Base.IteratorEltype(u) isa Base.HasEltype) && eltype(u) <: Complex &&
        throw(MethodError(moment_add_dddual_transform!, (state, dim, data, u)))
    @assert(dim > 1)
    diagu = u isa UniformScaling || u isa Diagonal # we just ignore uniform scaling completely as if it were 𝟙
    # Diagonal-dominant representation: this is a relaxation for the SOS formulation, where we replace M ∈ PSD by
    # M ∈ {U† D U, D ∈ DD}. Since this is more restrictive than PSD, the SOS maximization will only decrease, so we still have
    # a valid lower bound.
    # Vectorized version: vec(M) = vec(U† mat(d) U). In component form, this is
    # mᵢ = ∑_(diagonal j) Ū[row(j), row(i)] U[col(j), col(i)] dⱼ +
    #      ∑_(offdiag j) (Ū[col(j), row(i)] U[row(j), col(i)] + Ū[row(j), row(i)] U[col(j), col(i)]) dⱼ ⇔ m = Ũ d.
    # Note that if U is diagonal, mᵢ = Ū[row(i), row(i)] U[col(i), col(i)] dᵢ.
    # So define d ∈ vec(DD), m free, then demand 𝟙*m + (-Ũ)*d = 0. While we could eliminate m fully, it is actually
    # advantageous to keep them: in this way, we have direct access to the rotated DD cone in the solution returned by the
    # solver. Additionally, using m instead of -Ũ*d in the constraints can give a higher sparsity. Of course, if U is the
    # identity, we would be better off eliminating m.
    # Then, we need to translate DD into a cone that is supported; let's assume for simplicity that the ℓ₁ cone is available.
    # DD = ℓ₁ × ... × ℓ₁ plus equality constraints that enforce symmetry.
    # However, here we construct the moment representation; so we now need the dual formulation of diagonal dominance. Due to
    # the equality constraints, this is more complicated:
    # For side dimension n, there are n ℓ₁ cones (we just take the columns - also taking into account the rows would be even
    # more restrictive). For every data entry, we need a new slack variable which we fix to be equal to this data entry (a
    # variable for each the real and the imaginary part). We need additional slacks for every entry in the strict upper
    # triangle (twice in the complex-valued case).
    # Without the U, all this would look as follows (for a 3x3 real matrix):
    # data₁ = slack₁, data₂ = slack₂, data₃ = slack₃, data₄ = slack₄, data₅ = slack₅, data₆ = slack₆
    # {slack₁, slack₂ - slack₇, slack₃ - slack₈} ∈ ℓ_∞
    # {slack₄, slack₇,          slack₅ - slack₉} ∈ ℓ_∞
    # {slack₆, slack₈,          slack₉}          ∈ ℓ_∞
    # effectively corresponding to the DDDual matrix
    # data₁            slack₇           slack₈
    # data₂ - slack₇   data₄            slack₉
    # data₃ - slack₈   data₅ - slack₉   data₆

    # The assignment of slack variables to the data will always be the same and unrelated to U. Therefore, in the following,
    # for better clarity, we will omit the equality constraints and simply write slack(row, col), where it is understood that
    # for row ≥ col (lower triangle), this corresponds to the equality-constrained slack variable that is (twice) the data, and
    # for row < col (strict upper triangle), this corresponds to a free slack variable required due to symmetry.

    # For a general U and complex-valued data, we then have for the column j:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (Re(U[j, col] Ū[j, row]) slackᵣ(row, col) +
    #                                     Im(U[j, col] Ū[j, row]) slackᵢ(row, col)),
    #  slackᵣ(i, j), slackᵢ(i, j) for i ∈ 1, ..., j -1,
    #  ∑_{col = 1}^dim ∑_{row = col}^dim ((Re(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) slackᵣ(row, col) -
    #                                     (Im(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) slackᵢ(row, col)) - slackᵣ(j, i),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim ((Im(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) slackᵣ(row, col) +
    #                                     (Re(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) slackᵢ(row, col)) + slackᵢ(j, i)
    #  for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # Let's specialize the formula. If U is diagonal:
    # {|U[j, j]|² slackᵣ(j, j),
    #  slackᵣ(i, j), slackᵢ(i, j) for i ∈ 1, ..., j -1,
    #  (Re(U[i, i] Ū[j, j]) slackᵣ(i, j) - Im(U[i, i] Ū[j, j]) slackᵢ(i, j)) - slackᵣ(j, i),
    #  (Im(U[i, i] Ū[j, j]) slackᵣ(i, j) + Re(U[i, i] Ū[j, j]) slackᵢ(i, j)) + slackᵢ(j, i)
    #  for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # If everything is instead real-valued:
    # {∑_{col = 1}^dim ∑_{row = col}^dim U[j, col] U[j, row] slack(row, col),
    #  slack(i, j) for i ∈ 1, ..., j -1,
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (U[i, row] U[j, col] + U[i, col] U[j, row]) slack(row, col) - slack(j, i)
    #  for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # If everything is real and U is diagonal:
    # {U[j, j]² slack(j, j),
    #  slack(i, j) for i ∈ 1, ..., j -1,
    #  U[i, i] U[j, j] slack(i, j) - slack(j, i) for i in j +1, ..., dim
    # } ∈ ℓ_∞

    # If the ℓ_∞ cone is not available, we can instead use the nonnegative cone if we are not complex. The most efficient way
    # to do this is in fact to directly rewrite the ℓ_∞ cone formulation in terms of two nonnegative constraints for all but
    # the diagonal entry.
    # If we are complex, then in the SOS formulation, we have to split the complex ℓ₁-norm cone. For this, we use quadratic
    # cones to obtain an upper bound on the absolute value from the real and imaginary parts. The diagonal entries are then the
    # sum of all absolute values plus a nonnegative variable. As a consequence, we no longer need to maintain the other
    # triangle, it is now implicit. Translated into the moment domain, we get a dim-dimensional nonnegative cone and
    # trisize(dim -1) norm cones. The first entry in the norm cone is the sum of the nonnegative variables corresponding to the
    # current row and column.

    maxsize = maximum(data.lens, init=0) # how large is one dataᵢ at most?

    have_linf = complex ? supports_lnorm_complex(state) : supports_lnorm(state)
    complex && @assert(have_linf || supports_quadratic(state)) # this must have been checked during construction
    ts = trisize(dim)
    dsq = dim^2
    if have_linf
        indices = FastVec{T}(buffer=max(1 + maxsize,
            if diagu
                if complex
                    6dim -5 # 1 + 6(dim -1)
                else
                    2dim -1 # 1 + 2(dim -1)
                end
            elseif complex
                2 * ((ts +1) * (2dim -1) - dim) # 2trisize(dim) + (2trisize(dim) +1) * 2(dim -1)
            else
                (ts +1) * dim -1 # trisize(dim) + (trisize(dim) +1) * (dim -1)
            end) # either the first col or the longest data assignment are the largest
        )
        lens = FastVec{Int}(buffer=complex ? 2dim -1 : dim)
        slacks = add_var_slack!(state, complex ? dsq + 2trisize(dim -1) : dsq)
    elseif complex
        # This means that we use the quadratic cone to mimick the ℓ_∞ norm cone: x₁ ≥ ∑ᵢ (Re² xᵢ + Im² xᵢ). So we need to
        # submit lots of cones, but all of them pretty small.
        indices = FastVec{T}(buffer=max(1 + maxsize, 6, 3dsq))
        slacks = add_var_slack!(state, dsq)
    else
        # If we don't have this cone, we must use linear constraints. While we could do the whole matrix in a single large
        # nonnegative constraint vector, we'll do it columnwise as well.
        indices = FastVec{T}(buffer=max(1 + maxsize, diagu ? 2 * 3(dim -1) : 2 * (ts +1) * (dim -1)))
        if diagu
            lens = FastVec{Int}(buffer=2dim -2)
        end
        slacks = add_var_slack!(state, dsq)
    end
    values = similar(indices, V)

    # Add the equality constraints first; this is independent of U or the method to model the cone
    @inbounds let eqstate=@inline add_constr_fix_prepare!(state, complex ? dsq : ts)
        # When iterating through data, we need to add our slack element; it does not matter whether at the beginning or the
        # end. So under the assertion that data contains more than just a single Indvals (which is always true, else this
        # function would not be called), we append the slack at the end, temporarily overwriting what is there for the first
        # item; and we prepend it for all others (no need to restore the value afterwards), avoiding copying around.
        firstindvals, restindvals = Iterators.peel(data)
        dataindices, datavalues = firstindvals.indices, firstindvals.values
        # while we could simply do @inbounds indices[end+1], this won't work in the interpreter; and we need the index multiple
        # times anyway, so let's just compute it under the assumption that everything is as it should be.
        @assert(dataindices isa Base.FastContiguousSubArray)
        pos = dataindices.offset1 + length(dataindices) +1
        storedind = dataindices.parent[pos]
        storedval = datavalues.parent[pos]
        dataindices.parent[pos] = slacks[1]
        datavalues.parent[pos] = -one(V)
        eqstate = @inline add_constr_fix!(state, eqstate, Indvals(view(dataindices.parent, dataindices.offset1+1:pos),
                                                                  view(datavalues.parent, dataindices.offset1+1:pos)), zero(V))
        dataindices.parent[pos] = storedind
        datavalues.parent[pos] = storedval
        offdiags = dim -1
        col = 1
        slack = 2
        let indvalsiter=iterate(restindvals)
            while !isnothing(indvalsiter)
                indvals = indvalsiter[1]
                dataindices, datavalues = indvals.indices, indvals.values
                pos = dataindices.offset1
                dataindices.parent[pos] = slacks[slack]
                datavalues.parent[pos] = -one(V)
                if iszero(offdiags)
                    col += 1
                    offdiags = dim - col
                else
                    if complex
                        slack += 1
                        eqstate = @inline add_constr_fix!(state, eqstate,
                                                            Indvals(view(dataindices.parent,
                                                                        pos:dataindices.offset1+length(dataindices)),
                                                                    view(datavalues.parent,
                                                                        pos:dataindices.offset1+length(dataindices))),
                                                            zero(V))
                        indvalsiter = iterate(restindvals, indvalsiter[2])::Tuple
                        indvals = indvalsiter[1]
                        dataindices, datavalues = indvals.indices, indvals.values
                        pos = dataindices.offset1
                        dataindices.parent[pos] = slacks[slack]
                        datavalues.parent[pos] = -one(V)
                    end
                    offdiags -= 1
                end
                slack += 1
                eqstate = @inline add_constr_fix!(state, eqstate,
                                                    Indvals(view(dataindices.parent,
                                                                pos:dataindices.offset1+length(dataindices)),
                                                            view(datavalues.parent,
                                                                pos:dataindices.offset1+length(dataindices))), zero(V))
                indvalsiter = iterate(restindvals, indvalsiter[2])
            end
        end
        @inline add_constr_fix_finalize!(state, eqstate)
    end

    # Note: We always add the items and don't check whether the value is zero, potentially reducing the sparsity pattern (only
    # for diagonal u, we respect it). This is so that the solver data is always at the same position and re-optimizations with
    # a different u will not change the structure of the solver problem (of course, we'd have to write support for re-using the
    # problem in the first place...).
    if have_linf
        if diagu
            upperslack = (complex ? dsq : ts) +1
            lowerslack = 1
            @inbounds for j in 1:dim
                #region Diagonal (naturally is the first item in the L order, and must be the first in the ℓ_∞ cone)
                unsafe_push!(indices, slacks[lowerslack])
                unsafe_push!(values, u isa Diagonal ? abs2(u[j, j]) : one(V))
                unsafe_push!(lens, 1)
                lowerslack += 1
                #endregion
                #region Above diagonal (slacks)
                δ = complex ? 2j -2 : j -1
                unsafe_append!(indices, @view(slacks[upperslack:upperslack+δ-1]))
                unsafe_append!(values, Iterators.repeated(one(V), δ))
                unsafe_append!(lens, Iterators.repeated(1, δ))
                upperslack += δ
                #endregion
                #region Below diagonal
                sbelow = upperslack + (complex ? 2j - 2 : j -1)
                for i in j+1:dim
                    uval = u isa Diagonal ? u[i, i] * conj(u[j, j]) : one(V)
                    if complex
                        unsafe_push!(indices, slacks[lowerslack], slacks[lowerslack+1], slacks[sbelow])
                        unsafe_push!(values, real(uval), -imag(uval), -one(V))
                        unsafe_push!(lens, 3)
                    else
                        unsafe_push!(indices, slacks[lowerslack], slacks[sbelow])
                        unsafe_push!(values, real(uval), -one(V)) # should be real anyway
                        unsafe_push!(lens, 2)
                    end
                    if complex
                        unsafe_push!(indices, slacks[lowerslack], slacks[lowerslack+1], slacks[sbelow+1])
                        unsafe_push!(values, imag(uval), real(uval), one(V))
                        unsafe_push!(lens, 3)
                        sbelow += 2i -2
                        lowerslack += 2
                    else
                        sbelow += i -1
                        lowerslack += 1
                    end
                end
                #endregion
                #region Add the whole column
                (complex ? add_constr_linf_complex! : add_constr_linf!)(state, IndvalsIterator(unsafe, indices, values, lens))
                #endregion
                empty!(indices)
                empty!(values)
                empty!(lens)
            end
        else
            upperslack = (complex ? dsq : ts) +1
            @inbounds for j in 1:dim
                #region Diagonal (naturally is the first item in the L order, and must be the first in the ℓ_∞ cone)
                lowerslack = 1
                for col in 1:dim
                    unsafe_push!(indices, slacks[lowerslack])
                    unsafe_push!(values, abs2(u[j, col]))
                    lowerslack += 1
                    for row in col+1:dim
                        uval = u[j, col] * conj(u[j, row])
                        @twice imdata complex begin
                            unsafe_push!(indices, slacks[lowerslack+imdata])
                            unsafe_push!(values, imdata ? imag(uval) : real(uval))
                        end
                        lowerslack += (complex ? 2 : 1)
                    end
                end
                unsafe_push!(lens, length(indices))
                #endregion
                #region Above diagonal (slacks)
                δ = complex ? 2j -2 : j -1
                unsafe_append!(indices, @view(slacks[upperslack:upperslack+δ-1]))
                unsafe_append!(values, Iterators.repeated(one(V), δ))
                unsafe_append!(lens, Iterators.repeated(1, δ))
                upperslack += δ
                #endregion
                #region Below diagonal
                sbelow = upperslack + (complex ? 2j - 2 : j -1)
                for i in j+1:dim
                    @twice impart complex begin
                        beforeidx = length(indices)
                        slack = 1
                        for col in 1:dim
                            uval = 2u[i, col] * conj(u[j, col])
                            unsafe_push!(indices, slacks[slack])
                            unsafe_push!(values, impart ? imag(uval) : real(uval))
                            slack += 1
                            for row in col+1:dim
                                @twice imdata complex begin
                                    uval = u[i, row] * conj(u[j, col])
                                    if imdata
                                        uval -= u[i, col] * conj(u[j, row])
                                        thisuval = impart ? real(uval) : -imag(uval)
                                    else
                                        uval += u[i, col] * conj(u[j, row])
                                        thisuval = impart ? imag(uval) : real(uval)
                                    end
                                    unsafe_push!(indices, slacks[slack])
                                    unsafe_push!(values, thisuval)
                                    slack += 1
                                end
                            end
                        end
                        unsafe_push!(indices, slacks[sbelow])
                        unsafe_push!(values, impart ? one(V) : -one(V))
                        unsafe_push!(lens, length(indices) - beforeidx)
                        if complex
                            sbelow += impart ? 2i -3 : 1
                        else
                            sbelow += i -1
                        end
                    end
                end
                #endregion
                #region Add the whole column
                (complex ? add_constr_linf_complex! : add_constr_linf!)(state, IndvalsIterator(unsafe, indices, values, lens))
                #endregion
                empty!(indices)
                empty!(values)
                empty!(lens)
            end
        end
    elseif complex
        if diagu
            rowdiagslack = 1
            lowerslack = 1
            @inbounds for j in 1:dim
                #region Diagonal (naturally is the first item in the L order, and we must put it separately in the nonneg cone)
                unsafe_push!(indices, slacks[lowerslack])
                unsafe_push!(values, u isa Diagonal ? abs2(u[j, j]) : one(V))
                add_constr_nonnegative!(state, Indvals(indices, values))
                lowerslack += 1
                #endregion
                #region Below diagonal
                # We keep the variable from the column diagonal, but we also need the variable from the row diagonal
                resize!(indices, 6)
                resize!(values, 6)
                rowdiagslack += 2(dim - j) +1
                srowdiag = rowdiagslack
                for i in j+1:dim
                    uval = u isa Diagonal ? u[i, i] * conj(u[j, j]) : one(V)

                    indices[2] = slacks[srowdiag]
                    values[2] = u isa Diagonal ? abs2(u[i, i]) : one(V)
                    indices[5] = indices[3] = slacks[lowerslack]
                    indices[6] = indices[4] = slacks[lowerslack+1]
                    values[3] = real(uval)
                    values[4] = -imag(uval)
                    values[5] = imag(uval)
                    values[6] = real(uval)
                    srowdiag += 2(dim - i) +1
                    lowerslack += 2
                    @inline add_constr_quadratic!(state, IndvalsIterator(unsafe, indices, values, 2))
                end
                #endregion
                empty!(indices)
                empty!(values)
            end
        else
            unsafe_append!(indices, slacks)
            unsafe_append!(indices, slacks)
            unsafe_append!(indices, slacks)
            @inbounds for j in 1:dim
                #region Diagonal (naturally is the first item in the L order, and we must put it separately in the nonneg cone)
                @inbounds for col in 1:dim
                    unsafe_push!(values, abs2(u[j, col]))
                    for row in col+1:dim
                        uval = u[j, col] * conj(u[j, row])
                        unsafe_push!(values, real(uval), imag(uval))
                    end
                end
                add_constr_nonnegative!(state, Indvals(slacks, values))
                empty!(values)
                #endregion
                #region Below diagonal
                for i in j+1:dim
                    # first the diagonal, which is our col diagonal plus the row diagonal that we must recompute.
                    slack = 1
                    for col in 1:dim
                        unsafe_push!(values, abs2(u[j, col]) + abs2(u[i, col]))
                        slack += 1
                        for row in col+1:dim
                            uval = u[j, col] * conj(u[j, row]) + u[i, col] * conj(u[i, row])
                            unsafe_push!(values, real(uval), imag(uval))
                        end
                    end
                    # then every off-diagonal
                    @twice impart true begin
                        slack = 1
                        for col in 1:dim
                            uval = 2u[i, col] * conj(u[j, col])
                            unsafe_push!(values, impart ? imag(uval) : real(uval))
                            slack += 1
                            for row in col+1:dim
                                @twice imdata true begin
                                    uval = u[i, row] * conj(u[j, col])
                                    if imdata
                                        uval -= u[i, col] * conj(u[j, row])
                                        thisuval = impart ? real(uval) : -imag(uval)
                                    else
                                        uval += u[i, col] * conj(u[j, row])
                                        thisuval = impart ? imag(uval) : real(uval)
                                    end
                                    unsafe_push!(values, thisuval)
                                    slack += 1
                                end
                            end
                        end
                    end
                    @inline add_constr_quadratic!(state, IndvalsIterator(unsafe, indices, values, dsq))
                    empty!(values)
                end
                #endregion
            end
        end
    else
        # this case is real-valued
        upperslack = ts +1
        if diagu
            lowerslack = 1
            @inbounds for j in 1:dim
                #region Diagonal
                udiagval = u isa Diagonal ? u[j, j]^2 : one(V)
                diagslack = lowerslack
                lowerslack += 1
                #endregion
                #region Above diagonal (slacks)
                for i in 0:j-2
                    unsafe_push!(indices, slacks[diagslack], slacks[upperslack+i], slacks[diagslack], slacks[upperslack+i])
                    unsafe_push!(values, udiagval, one(V), udiagval, -one(V))
                    unsafe_push!(lens, 2, 2)
                end
                #endregion
                #region Below diagonal
                upperslack += j -1
                sbelow = upperslack + j -1
                for i in j+1:dim
                    if u isa Diagonal
                        uval = u[i, i] * u[j, j]
                    else
                        uval = one(V)
                    end
                    unsafe_push!(indices, slacks[diagslack], slacks[lowerslack], slacks[sbelow],
                                            slacks[diagslack], slacks[lowerslack], slacks[sbelow])
                    unsafe_push!(values, udiagval, uval, -one(V), udiagval, -uval, one(V))
                    unsafe_push!(lens, 3, 3)
                    sbelow += i -1
                    lowerslack += 1
                end
                #endregion
                #region Add the whole column
                add_constr_nonnegative!(state, IndvalsIterator(unsafe, indices, values, lens))
                #endregion
                empty!(indices)
                empty!(values)
                empty!(lens)
            end
        else
            @inbounds for j in 1:dim
                #region Above diagonal (slacks)
                lowerslack = 1
                for col in 1:dim
                    unsafe_push!(indices, slacks[lowerslack])
                    unsafe_push!(values, u[j, col]^2)
                    lowerslack += 1
                    for row in col+1:dim
                        unsafe_push!(indices, slacks[lowerslack])
                        unsafe_push!(values, u[j, col] * u[j, row])
                        lowerslack += 1
                    end
                end
                diagvalues = @view(values[1:lowerslack-1]) # these are the isolated values for the diagonal u, we can use them
                                                            # to save some re-computations. But beware: if j = 1, there is no
                                                            # slack, so we should not even have inserted the values in this way
                                                            # here! We need to account for this later.
                                                            # [Note 1:lowerslack-1 = begin:end]
                diagslacks = @view(slacks[1:lowerslack-1])
                unsafe_push!(indices, slacks[upperslack]) # for j = 1, these are also invalid
                unsafe_push!(values, -one(V)) # this should be 1 on the upper triangle, but we need it to be -1 for the lower
                unsafe_append!(indices, diagslacks)
                unsafe_append!(values, diagvalues)
                unsafe_push!(indices, slacks[upperslack])
                unsafe_push!(values, one(V)) # and this should be -1
                for i in 1:j-2
                    unsafe_append!(indices, diagslacks)
                    unsafe_append!(values, diagvalues)
                    unsafe_push!(indices, slacks[upperslack+i])
                    unsafe_push!(values, one(V))
                    unsafe_append!(indices, diagslacks)
                    unsafe_append!(values, diagvalues)
                    unsafe_push!(indices, slacks[upperslack+i])
                    unsafe_push!(values, -one(V))
                end
                #endregion
                #region Below diagonal
                # Here we now have to be careful. If j = 1, then we already have the coefficients for the slacks due to the
                # diagonal part written into values. This is wrong, we need to update them and add the part for the (2, 1)
                # entry. However, it is better if we actually start to insert the starting from (3, 1), then at the end
                # re-visit the (2, 1) part and update it only now; in this way, we can use our knowledge of the diagonal part,
                # saving some small re-computation.
                upperslack += j -1
                sbelow = upperslack + j -1
                if isone(j)
                    sbelow += j
                end
                for i in (isone(j) ? 3 : j +1):dim
                    @twice negative true begin
                        slack = 1
                        for col in 1:dim
                            unsafe_push!(indices, slacks[slack])
                            unsafe_push!(values, diagvalues[slack] + (negative ? -2 : 2) * u[i, col] * u[j, col])
                            slack += 1
                            for row in col+1:dim
                                uval = u[i, row] * u[j, col] + u[i, col] * u[j, row]
                                unsafe_push!(indices, slacks[slack])
                                unsafe_push!(values, diagvalues[slack] + (negative ? -uval : uval))
                                slack += 1
                            end
                        end
                        unsafe_push!(indices, slacks[sbelow])
                        unsafe_push!(values, negative ? one(V) : -one(V))
                    end
                    sbelow += i -1
                end
                # Now let's take care of (2, 1). indices and lens are already set up properly, only the values have to be
                # adjusted.
                if isone(j)
                    @twice negative true begin
                        slack = 1
                        for col in 1:dim
                            diagvalues[slack] += (negative ? -2 : 2) * u[2, col] * u[1, col]
                            slack += 1
                            for row in col+1:dim
                                uval = u[2, row] * u[1, col] + u[2, col] * u[1, row]
                                diagvalues[slack] += (negative ? -uval : uval)
                                slack += 1
                            end
                        end
                        if !negative
                            # now map to the next value range
                            diagvalues = @view(values[diagvalues.offset1+length(diagvalues)+2:diagvalues.offset1+2length(diagvalues)+1])
                        end
                    end
                else
                    # Finally correct the signs (actually, it does not matter at all if our upper triangle is represented by
                    # dummy₁ - dummy₂ or dummy₁ + dummy₂; let's just do this for consistency).
                    values[lowerslack] = one(V)
                    values[2lowerslack] = -one(V)
                end
                #endregion
                #region Add the whole column
                add_constr_nonnegative!(state, IndvalsIterator(unsafe, indices, values, lowerslack))
                #endregion
                empty!(indices)
                empty!(values)
            end
        end
    end
    return
end

function moment_add_sdddual_transform!(state::AnySolver{T,V}, dim::Integer, data::IndvalsIterator{T,V}, u, ::Val{complex}) where {T,V,complex}
    diagu = u isa UniformScaling || u isa Diagonal # we just ignore uniform scaling completely as if it were 𝟙
    # See the comment in add_constr_dddual!. Here, the fallback implementation is done in terms of rotated quadratic cones due
    # to the relationship of SDD matrices with factor-width-2 matrices.
    # We must take care of scaling the off-diagonal data; as we didn't know about the rotation, this could not have been done
    # before. The rotated quadratic cone is 2x₁ x₂ ≥ ∑ᵢ xᵢ², so we'll scale the x₃ by √2 (i.e. multiply all the coefficients
    # that use x₃ by 1/√2) to make this equivalent to [x₁ x₃; x₃ x₂] ⪰ 0. However, since only one triangle is considered, we
    # also need to scale the coefficients by 2, so in total we end up with √2.

    # For a general U and complex-valued data, we have the following rotated quadratic constraints for the column j and the row
    # i > j:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) (Re(U[j, col] Ū[j, row]) dataᵣ(row, col) +
    #                                                        Im(U[j, col] Ū[j, row]) dataᵢ(row, col)),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) (Re(U[i, col] Ū[i, row]) dataᵣ(row, col) +
    #                                                        Im(U[i, col] Ū[i, row]) dataᵢ(row, col)),
    #  √2 ∑_{col = 1}^dim ∑_{row = col}^dim ((Re(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) -
    #                                        (Im(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) dataᵢ(row, col)),
    #  √2 ∑_{col = 1}^dim ∑_{row = col}^dim ((Im(U[i, row] Ū[j, col] + U[i, col] Ū[j, row]) dataᵣ(row, col) +
    #                                        (Re(U[i, row] Ū[j, col] - U[i, col] Ū[j, row]) dataᵢ(row, col))
    # } ∈ ℛ𝒬₄

    # Let's specialize the formula. If U is diagonal:
    # {|U[j, j]|² dataᵣ(j, j),
    #  |U[i, i]|² dataᵣ(i, i),
    #  √2 (Re(U[i, i] Ū[j, j]) dataᵣ(i, j) - Im(U[i, i] Ū[j, j]) dataᵢ(i, j)),
    #  √2 (Im(U[i, i] Ū[j, j]) dataᵣ(i, j) + Re(U[i, i] Ū[j, j]) dataᵢ(i, j))
    # } ∈ ℛ𝒬₄

    # If everything is instead real-valued:
    # {∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) U[j, col] U[j, row] data(row, col),
    #  ∑_{col = 1}^dim ∑_{row = col}^dim (2 - δ_{row, col}) U[i, col] U[i, row] data(row, col),
    #  √2 ∑_{col = 1}^dim ∑_{row = col}^dim (U[i, row] U[j, col] + U[i, col] U[j, row]) data(row, col)
    # } ∈ ℛ𝒬₃

    # If everything is real and U is diagonal:
    # {U[j, j]² data(j, j),
    #  U[i, i]² data(i, i),
    #  √2 U[i, i] U[j, j] data(i, j)
    # } ∈ ℛ𝒬₃
    maxsize = maximum(data.lens, init=0) # how large is one dataᵢ at most?
    scaling = sqrt(V(2))

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
                    if row != col
                        uval *= V(2)
                    end
                    @twice impart (complex && row != col) begin
                        searchview = @view(indices[firstlen+1:end])
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
                    uval = u[i, i] * conj(u[j, j]) * scaling
                end
                @twice impart complex begin
                    startidx = length(indices) +1
                    len = Int(data.lens[idx])
                    r = dataidx:dataidx+len-1
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
                        rmul!(@view(values[end-len+1:end]), scaling)
                    end
                    iszero(length(indices) - startidx +1) || unsafe_push!(lens, length(indices) - startidx +1)
                    dataidx += len
                    idx += 1
                end
            else
                @twice impart complex begin
                    startidx = length(indices) +1
                    idx = 1
                    dataidx = 1
                    for col in 1:dim, row in col:dim
                        @twice imdata (complex && row != col) begin
                            uval = u[i, row] * conj(u[j, col])
                            searchview = @view(indices[startidx:end])
                            if imdata
                                uval -= u[i, col] * conj(u[j, row])
                                thisuval = impart ? real(uval) : -imag(uval)
                            else
                                if row != col
                                    uval += u[i, col] * conj(u[j, row])
                                end
                                thisuval = impart ? imag(uval) : real(uval)
                            end
                            thisuval *= scaling
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
                    iszero(length(indices) - startidx +1) || unsafe_push!(lens, length(indices) - startidx +1)
                end
            end
            #endregion
            # Some possible reductions to simpler cones:
            rg₁ = 1:lens[1]
            rg₂ = lens[1]+1:lens[1]+lens[2]
            @views if length(lens) == 2 || all(iszero, values[last(rg₂)+1:end])
                @label lastzero
                # 2x₁x₂ ≥ 0, x₁, x₂ ≥ 0
                has₁ = !iszero(lens[1]) && any(!iszero, values[rg₁])
                has₂ = !iszero(lens[2]) && any(!iszero, values[rg₂])
                if has₁ || has₂
                    if has₁ && has₂
                        # x₁, x₂ ≥ 0
                        add_constr_nonnegative!(state, IndvalsIterator(unsafe, indices, values, lens))
                    else
                        # x₁ ≥ 0 or x₂ ≥ 0
                        add_constr_nonnegative!(state, Indvals(indices, values))
                    end
                # else 0 ≥ 0
                end
            else
                zero₁ = iszero(lens[1]) && all(iszero, values[rg₁])
                zero₂ = iszero(lens[2]) && all(iszero, values[rg₂])
                if zero₁ || zero₂
                    # 0 ≥ x₃² + x₄², x₁, x₂ ≥ 0
                    if zero₁
                        # x₂ ≥ 0
                        add_constr_nonnegative!(state, Indvals(indices[rg₂], values[rg₂]))
                    else
                        # x₁ ≥ 0
                        add_constr_nonnegative!(state, Indvals(indices[rg₁], values[rg₁]))
                    end
                    # x₃ = x₄ = 0
                    rg₃ = last(rg₂)+1:last(rg₂)+lens[3]
                    zero₃ = all(iszero, values[rg₃])
                    zero₄ = length(lens) < 4 || all(iszero, values[last(rg₃)+1:end])
                    zero₃ && zero₄ && @goto lastzero
                    prep = add_constr_fix_prepare!(state, !zero₃ + !zero₄)
                    zero₃ ||
                        (prep = @views add_constr_fix!(state, prep, Indvals(indices[rg₃], values[rg₃]), zero(V)))
                    zero₄ ||
                        (prep = @views add_constr_fix!(state, prep, Indvals(indices[last(rg₃)+1:end], values[last(rg₃)+1:end]),
                            zero(V)))
                    add_constr_fix_finalize!(state, prep)
                else
                    # full case
                    (have_rot ? add_constr_rotated_quadratic! : add_constr_quadratic!)(state, to_soc!(indices, values, lens, have_rot))
                end
            end
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

"""
    moment_add_matrix!(state::AbstractSolver, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}},
        representation::RepresentationMethod=RepresentationPSD())

Parses a constraint in the moment hierarchy with a basis given in `grouping` (this might also be a partial basis due to
sparsity), premultiplied by `constraint` (which may be the unit polynomial for the moment matrix) and calls the appropriate
solver functions to set up the problem structure according to `representation`.

To make this function work for a solver, implement the following low-level primitives:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_psd_complex`](@ref) to `true`)
- [`add_constr_dddual!`](@ref) (optional, then set [`supports_dd`](@ref) to `true`)
- [`add_constr_dddual_complex!`](@ref) (optional, then set [`supports_dd_complex`](@ref) to `true`)
- [`add_constr_linf!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_constr_linf_complex!`](@ref) (optional, then set [`supports_lnorm_complex`](@ref) to `true`)
- [`add_constr_sdddual!`](@ref) (optional, then set [`supports_sdd`](@ref) to `true`)
- [`add_constr_sdddual_complex!`](@ref) (optional, then set [`supports_sdd_complex`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_var_slack!`](@ref)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_equality!`](@ref), [`RepresentationMethod`](@ref).
"""
function moment_add_matrix!(state::AnySolver, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::Union{P,<:AbstractMatrix{P}}, representation::RepresentationMethod=RepresentationPSD()) where {P<:SimplePolynomial}
    dim = length(grouping) * (constraint isa AbstractMatrix ? LinearAlgebra.checksquare(constraint) : 1)
    if (dim == 1 || (dim == 2 && (supports_rotated_quadratic(state) || supports_quadratic(state))))
        if representation isa RepresentationPSD
            indextype = PSDIndextypeVector(:U)
        else
            return moment_add_matrix!(state, grouping, constraint, RepresentationPSD())
        end
    else
        indextype = psd_indextype(state)
    end

    real_valued = (length(grouping) == 1 || isreal(grouping)) && (!(constraint isa AbstractMatrix) || isreal(constraint))
    complex_cone = !real_valued &&
        (representation isa RepresentationDD{<:Any,true} &&
            (supports_dd_complex(state) || supports_lnorm_complex(state) || supports_quadratic(state))) ||
        # TODO (unlikely): we could still do it if the solver supports only the rotated, but not the standard cone. But
        # are there any solvers in this category?
        representation isa RepresentationSDD{<:Any,true} ||
        (representation isa RepresentationPSD &&
            (dim == 1 || (dim == 2 && (supports_rotated_quadratic(state) || supports_quadratic(state))) ||
             supports_psd_complex(state)))

    representation isa RepresentationSDD &&
        !supports_rotated_quadratic(state) && !supports_quadratic(state) &&
        ((complex_cone && !supports_sdd_complex(state)) || (!complex_cone && !supports_sdd(state))) &&
        error("The solver does not support the required scaled diagonally dominant cone or the fallback (rotated) quadratic cones, so a representation via scaled diagonally-dominant matrices is not possible")

    if representation isa RepresentationDD || representation isa RepresentationSDD
        try
            sizecheck = real_valued || complex_cone ? dim : 2dim
            representation.u[sizecheck, sizecheck] # just for bounds checking - we cannot access size, as u might be anything
                                                   # (e.g., UniformScaling)
        catch e
            e isa BoundsError &&
                throw(ArgumentError("The given matrix for rotating the DD cone was not large enough (required dimension: $dim)"))
            rethrow()
        end
    end
    if (representation isa RepresentationDD && !(complex_cone ? supports_dd_complex(state) : supports_dd(state))) ||
        (representation isa RepresentationSDD && !(complex_cone ? supports_sdd_complex(state) : supports_sdd(state)))
        indextype = PSDIndextypeVector(:L)
    end

    return moment_add_matrix_helper!(
        state,
        grouping,
        constraint isa AbstractMatrix ? constraint : ScalarMatrix(constraint),
        indextype,
        (Val(real_valued), Val(complex_cone)),
        representation
    )
end

"""
    moment_add_equality!(state::AbstractSolver, grouping::SimpleMonomialVector, constraint::SimplePolynomial)

Parses a polynomial equality constraint for moments and calls the appropriate solver functions to set up the problem structure.
`grouping` contains the basis that will be squared in the process to generate the prefactor.

To make this function work for a solver, implement the following low-level primitives:
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_fix_finalize!`](@ref) (optional)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_matrix!`](@ref).
"""
function moment_add_equality!(state::AnySolver{T}, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::P) where {T,Nr,Nc,I<:Integer,P<:SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}}}
    # We need to traverse all unique elements in groupings * groupings†. For purely complex-valued groupings, this is the full
    # list; as soon as we have a real variable present, it is smaller.
    # To avoid rehashings, get an overestimator of the total grouping size first.
    # TODO (maybe): In the first loop to populate unique_groupings, we determine whether the grouping is real-valued. So we
    # could instead populate two sets, saving isreal and a lot of conditionals in the second loop.
    unique_groupings = sizehint!(Set{FastKey{I}}(), iszero(Nc) ? trisize(length(grouping)) : length(grouping)^2)
    real_grouping = true
    totalsize = 0
    for (i, g₁) in enumerate(grouping)
        if !iszero(Nc)
            g₁real = !iszero(Nr) && isreal(g₁)
            # Consider the g₂ = ḡ₁ case separately in the complex case. Explanations below.
            let g₂=g₁
                prodidx = FastKey(monomial_index(g₁, SimpleConjMonomial(g₂)))
                indexug, sh = Base.ht_keyindex2_shorthash!(unique_groupings.dict, prodidx)
                if indexug ≤ 0
                    @inbounds Base._setindex!(unique_groupings.dict, nothing, prodidx, -indexug, sh)
                    totalsize += 1
                end
            end
        end
        # In the real case, we can skip the entries behind i as they would lead to duplicates.
        # In the complex case, we can also skip them, as they would lead to exact conjugates, which in the end give rise to the
        # same conditions (but note that i is already handled above).
        for g₂ in Iterators.take(grouping, iszero(Nc) ? i : i -1)
            # We don't use mindex, as this can have unintended side-effects on the solver state (such as creating a
            # representation for this monomial, although we don't even know whether we need it - if constraint does not contain
            # a constant term, this function must not automatically add all the squared groupings as monomials, even if they
            # will probably appear at some place).
            prodidx = FastKey(monomial_index(g₁, SimpleConjMonomial(g₂)))
            # We need to add the product to the set if it does not exists; we also need to count the number of conditions that
            # we get out of it.
            indexug, sh = Base.ht_keyindex2_shorthash!(unique_groupings.dict, prodidx)
            if indexug ≤ 0
                # It does not exist.
                @inbounds Base._setindex!(unique_groupings.dict, nothing, prodidx, -indexug, sh)
                # Assume we have a grouping g = (gᵣ + im*gᵢ) and a polynomial p = pᵣ + im*pᵢ, where the individual parts are
                # real-valued. Then, add_equality! means that g*p = 0 and ḡ*p = 0. Of course we can also conjugate everything.
                # We must split each constraint into its real and imaginary parts:
                # (I)   Re(g*p) = gᵣ*pᵣ - gᵢ*pᵢ
                # (II)  Im(g*p) = gᵣ*pᵢ + gᵢ*pᵣ
                # (III) Re(ḡ*p) = gᵣ*pᵣ + gᵢ*pᵢ
                # (IV)  Im(ḡ*p) = gᵣ*pᵢ - gᵢ*pᵣ
                # To analyze this (which would be easier if we added and subtracted the equalities, but in the
                # SimplePolynomials setup, the given form is most easy to handle), let's consider linear dependencies.
                # - If the constraint is real-valued, (III) is equal to (I) and (IV) is -(II), so we only take (I) and (II).
                # - If the grouping is real-valued, (III) is equal to (I) and (IV) is equal to (II); we only take (I) and (II).
                # - If both are real-valued, (III) is equal to (I) while (II) and (IV) are zero, so we only take (I).
                # - If both are complex-valued, all constraints are linearly independent.
                # Rearranging this, we always take (I); if at least one is complex-valued, we also take (II); if both are, we
                # take all. Note that we don't have to consider the conjugates of the groupings separately, as they only yield
                # a global sign in the zero-equality.
                # For this loop, this means that we will only check whether g₁*ḡ₂ belongs to a real-valued monomial, in which
                # case we add 1; or to a complex-valued monomial, in which case we add 2. After the loop, we multiply by 2 if
                # the constraint was also complex-valued.
                if iszero(Nc) || (!iszero(Nr) && g₁real && isreal(g₂)) # note that g₁ ≠ ḡ₂
                    totalsize += 1
                else
                    totalsize += 2
                    real_grouping = false
                end
            end
        end
    end

    real_constr = isreal(constraint)
    if !real_constr
        totalsize *= 2
    end

    constrstate = @inline add_constr_fix_prepare!(state, totalsize)
    V = real(coefficient_type(P))
    indices₁ = FastVec{T}(buffer=2length(constraint))
    values₁ = similar(indices₁, V)
    # While we could conditionally define those variables only if the requirements are satisfied, the compiler might not be
    # able to infer that we only use them later on if the same conditions (potentially stricter) are met. So define them
    # always, but not using any memory.
    indices₂ = similar(indices₁, 0, buffer=real_grouping && real_constr ? 0 : 2length(constraint))
    values₂ = similar(indices₂, V)
    indices₃ = similar(indices₁, 0, buffer=real_grouping || real_constr ? 0 : 2length(constraint))
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
        constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₁, values₁), zero(V))
        empty!(indices₁); empty!(values₁)
        if !skip₂
            if !isempty(indices₂)
                constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₂, values₂), zero(V))
                empty!(indices₂); empty!(values₂)
            end
            if !real_grouping && !real_constr
                if !isempty(indices₃)
                    constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₃, values₃), zero(V))
                    empty!(indices₃); empty!(values₃)
                end
                if !isempty(indices₄)
                    constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₄, values₄), zero(V))
                    empty!(indices₄); empty!(values₄)
                end
            end
        end
    end
    @inline add_constr_fix_finalize!(state, constrstate)
    return
end

"""
    moment_setup!(state::AbstractSolver, relaxation::AbstractRelaxation, groupings::RelaxationGroupings[; representation])

Sets up all the necessary moment matrices, variables, constraints, and objective of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_psd_complex`](@ref) to `true`)
- [`add_constr_dddual!`](@ref) (optional, then set [`supports_dd`](@ref) to `true`)
- [`add_constr_dddual_complex!`](@ref) (optional, then set [`supports_dd_complex`](@ref) to `true`)
- [`add_constr_linf!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_constr_linf_complex!`](@ref) (optional, then set [`supports_lnorm_complex`](@ref) to `true`)
- [`add_constr_sdddual!`](@ref) (optional, then set [`supports_sdd`](@ref) to `true`)
- [`add_constr_sdddual_complex!`](@ref) (optional, then set [`supports_sdd_complex`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_fix_finalize!`](@ref) (optional)
- [`fix_objective!`](@ref)
- [`add_var_slack!`](@ref)

!!! warning "Indices"
    The variable indices used in all solver functions directly correspond to the indices given back by [`mindex`](@ref).
    However, in a sparse problem there may be far fewer indices present; therefore, when the problem is finally given to the
    solver, care must be taken to eliminate all unused indices. The functionality provided by [`AbstractAPISolver`](@ref) and
    [`AbstractSparseMatrixSolver`](@ref) already takes care of this.

!!! info "Order"
    The individual constraint types can be added in any order (including interleaved).

!!! info "Representation"
    This function may also be used to describe simplified cones such as the (scaled) diagonally dominant one. The
    `representation` parameter can be used to define a representation that is employed for the individual groupings. This may
    either be an instance of a [`RepresentationMethod`](@ref) - which requires the method to be independent of the dimension of
    the grouping - or a callable. In the latter case, it will be passed as a first parameter an identifier[^3] of the current
    conic variable, and as a second parameter the side dimension of its matrix. The method must then return a
    [`RepresentationMethod`](@ref) instance.

    [^3]: This identifier will be a tuple, where the first element is a symbol - either `:objective`, `:nonneg`, or `:psd` - to
          indicate the general reason why the variable is there. The second element is an `Int` denoting the index of the
          constraint (and will be undefined for the objective, but still present to avoid extra compilataion). The last element
          is an `Int` denoting the index of the grouping within the constraint/objective.

See also [`sos_setup!`](@ref), [`moment_add_matrix!`](@ref), [`moment_add_equality!`](@ref),
[`RepresentationMethod`](@ref).
"""
function moment_setup!(state::AnySolver{T,V}, relaxation::AbstractRelaxation{<:Problem{P}}, groupings::RelaxationGroupings;
    representation::Union{<:RepresentationMethod,<:Base.Callable}=RepresentationPSD()) where {T,V,P}
    problem = poly_problem(relaxation)
    (real(coefficient_type(problem.objective)) <: V) ||
        @warn("Expected value type for the solver $V might not be compatible with polynomial coefficient type $(real(coefficient_type(problem.objective)))")

    # fixed items
    # fix constant term to 1
    if isone(problem.prefactor)
        @inline add_constr_fix_finalize!(
            state,
            add_constr_fix!(
                state,
                add_constr_fix_prepare!(state, 1),
                Indvals(StackVec(mindex(state, constant_monomial(P))), StackVec(one(V))),
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
                add_constr_fix!(state, add_constr_fix_prepare!(state, 1), Indvals(indices, values), one(V))
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
    for (i, grouping) in enumerate(groupings.obj)
        g = collect_grouping(grouping)
        moment_add_matrix!(state, g, constantP,
            representation isa RepresentationMethod ? representation : representation((:objective, 0, i), length(g)))
    end
    # localizing matrices
    for (i, (groupingsᵢ, constrᵢ)) in enumerate(zip(groupings.nonnegs, problem.constr_nonneg))
        for (j, grouping) in enumerate(groupingsᵢ)
            g = collect_grouping(grouping)
            moment_add_matrix!(state, g, constrᵢ,
                representation isa RepresentationMethod ? representation : representation((:nonneg, i, j), length(g)))
        end
    end
    for (i, (groupingsᵢ, constrᵢ)) in enumerate(zip(groupings.psds, problem.constr_psd))
        for (j, grouping) in enumerate(groupingsᵢ)
            g = collect_grouping(grouping)
            moment_add_matrix!(state, g, constrᵢ,
                representation isa RepresentationMethod ? representation :
                                                          representation((:psd, i, j), length(g) * size(constrᵢ, 1)))
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
        fix_objective!(state, Indvals(finish!(indices), finish!(values)))
    end

    return
end