# This file is for the commuting case; nevertheless, we already write the monomial index/multiplication calculation with a
# possible extension to the noncommuting case in mind.

trisize(n) = (n * (n +1)) >> 1
realtype(::Type{<:Union{R,Complex{R}}}) where {R<:Real} = R

# For complex-valued problems, there are m and conj(m); all solvers require real-valued inputs.
# Therefore, we first define the canonicalized monomial m̃ as the one of m, conj(m) that has the smaller index.
iscanonical(::SimplePolynomials.SimpleRealMonomial) = true
Base.@assume_effects :consistent iscanonical(m::SimpleMonomial) = m.exponents_complex ≤ m.exponents_conj
canonicalize(m::SimpleMonomial) = iscanonical(m) ? m : conj(m)
# Next, the real part of m̃ is stored in the index associated with m; the imaginary part of m̃ in the index associated with
# conj(m), unless those two are equal (then we don't store the imaginary part).
# Note that Julia should be able to infer that the indices in the case of real-valued problems are the same and optimize a lot
# away, as we marked monomial_index and conj to be :consistent.

# This is for the quadratic case, everything is preallocated
@inline function pushorupdate!(idxvec::FastVec, index, valuevec::FastVec, value)
    i = findfirst(isequal(index), idxvec)
    if isnothing(i)
        unsafe_push!(idxvec, index)
        unsafe_push!(valuevec, value)
    else
        @inbounds valuevec[i] += value
    end
end

# In the PSD case, we might need to grow the list
@inline function pushorupdate!((idxvec, valuevec)::NTuple{2,FastVec}, index, value)
    i = findfirst(isequal(index), idxvec)
    if isnothing(i)
        push!(idxvec, index)
        push!(valuevec, value)
    else
        @inbounds valuevec[i] += value
    end
end

@inline function pushorupdate!((idxrvec, idxcvec, valuevec)::NTuple{3,FastVec}, idxr, idxc, value)
    found = false
    i = 1
    for outer i in 1:length(idxrvec)
        @inbounds if idxrvec[i] == idxr && idxcvec[i] == idxc
            found = true
            break
        end
    end
    if !found
        push!(idxrvec, idxr)
        push!(idxcvec, idxc)
        push!(valuevec, value)
    else
        @inbounds valuevec[i] += value
    end
end

@inline function getreonly(state, args::SimpleMonomial{Nr,Nc,P}...) where {Nr,Nc,P<:Unsigned}
    idx₁ = sos_solver_mindex(state, args...)
    @assert(idx₁ == sos_solver_mindex(state, conj.(reverse(args))...)) # reverse is rather unnecessary for the commuting case
    return idx₁
end

@inline function getreim(state, args::SimpleMonomial{Nr,Nc,P}...) where {Nr,Nc,P<:Unsigned}
    idx₁ = sos_solver_mindex(state, args...)
    idx₂ = sos_solver_mindex(state, conj.(reverse(args))...) # reverse is rather unnecessary for the commuting case
    if idx₁ ≤ idx₂
        return idx₁, idx₂, true # make sure to test for equality whenever the imaginary part is involved
    else
        return idx₂, idx₁, false
    end
end

# g * constraint * conj(g) ≥ 0 - linear constraint
function sos_add_matrix_helper!(state, T, V, g::SimpleMonomial, constraint::SimplePolynomial)
    buffer = length(constraint)
    if isone(buffer)
        # we just have a single monomial in the constraint. This is only possible if it is of the form m * conj(m) - i.e., the
        # monomial has no imaginary part. But then, the coefficient must be real-valued as well.
        let term_constr=first(constraint), mon_constr=monomial(term_constr), coeff_constr=coefficient(term_constr)
            @assert(iszero(imag(coeff_constr)))
            sos_solver_add_scalar!(state, getreonly(state, g, mon_constr, conj(g)), real(coeff_constr))
        end
    else
        # There are multiple terms in the constraint. Those may even be complex-valued: we know that in the construction of the
        # problem, the validity was checked. So we know for sure that for every complex valued term α m, there will later be a
        # corresponding term conj(α m). Instead of trying to handle those "duplicate" terms, we will simply add double the
        # coefficient and ignore the later (or earlier) version.
        let indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
            for term_constr in constraint
                mon_constr = monomial(term_constr)
                coeff_constr = coefficient(term_constr)
                recoeff = real(coeff_constr)
                imcoeff = imag(coeff_constr)
                if isreal(mon_constr)
                    @assert(iszero(imcoeff))
                    unsafe_push!(indices, getreonly(state, g, mon_constr, conj(g)))
                    unsafe_push!(values, recoeff)
                elseif iscanonical(mon_constr)
                    repart, impart, canonical = getreim(state, g, mon_constr, conj(g))
                    @assert(canonical)
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
            sos_solver_add_scalar!(state, indices, values)
        end
    end
    return
end

# [g₁; g₂] * constraint * [conj(g₁) conj(g₂)] ⪰ 0 - quadratic constraint
function sos_add_matrix_helper!(state, T, V, g₁::M, g₂::M, constraint::SimplePolynomial) where {M<:SimpleMonomial}
    buffer = length(constraint)
    real_valued = isreal(g₁) && isreal(g₂)
    if isone(buffer)
        # we just have a single monomial in the constraint. This is only possible if it is of the form m * conj(m) - i.e., the
        # monomial has no imaginary part. But then, the coefficient must be real-valued as well.
        let term_constr=first(constraint), mon_constr=monomial(term_constr), coeff_constr=coefficient(term_constr)
            @assert(iszero(imag(coeff_constr)))
            recoeff = real(coeff_constr)
            if real_valued
                sos_solver_add_quadratic!(
                    state,
                    getreonly(state, g₁, mon_constr, g₁), recoeff,
                    getreonly(state, g₂, mon_constr, g₂), recoeff,
                    (getreonly(state, g₂, mon_constr, g₁), recoeff)
                )
            else
                offrepart, offimpart, offcanonical = getreim(state, g₁, mon_constr, conj(g₂))
                sos_solver_add_quadratic!(
                    state,
                    getreonly(state, g₁, mon_constr, conj(g₁)), recoeff,
                    getreonly(state, g₂, mon_constr, conj(g₂)), recoeff,
                    (offrepart, recoeff),
                    (offimpart, offcanonical ? recoeff : -recoeff)
                    # We don't really need this here as the sign is gobbled by the cone, but for consistency let's put it.
                )
            end
        end
    else
        # There are multiple terms in the constraint. These may be of the form (m * conj(m)), but also α*m + conj(α)*conj(m)
        # with α ∈ ℂ is allowed or α*(m - conj(m)) with α ∈ iℝ. We know that in the construction of the problem, the validity
        # was checked, so for sure there will not be constraints with dangling imaginary parts.
        @inbounds let indices₁=FastVec{T}(; buffer), indices₂=similar(indices₁),
                      indices₃=FastVec{T}(buffer=real_valued ? buffer : 2buffer),
                      values₁=FastVec{V}(; buffer), values₂=similar(values₁), values₃=similar(indices₃, V)
            # For a real-valued basis:
            # - every real term will get at most one entry in any of the indices
            # - every other complex-valued term will be skipped
            # - the remaining complex-valued terms will get up to two entries in the diagonal indices and up to four in the
            #   off-diagonals
            if !real_valued
                indices₄ = similar(indices₃)
                values₄ = similar(values₃)
            end
            for term_constr in constraint
                mon_constr = monomial(term_constr)
                coeff_constr = coefficient(term_constr)
                recoeff = real(coeff_constr)
                imcoeff = imag(coeff_constr)
                if isreal(mon_constr)
                    @assert(iszero(imcoeff)) # else the polynomial would not be real-valued
                    unsafe_push!(indices₁, getreonly(state, g₁, mon_constr, conj(g₁)))
                    unsafe_push!(values₁, recoeff)
                    unsafe_push!(indices₂, getreonly(state, g₂, mon_constr, conj(g₂)))
                    unsafe_push!(values₂, recoeff)
                    offrepart, offimpart, offcanonical = getreim(state, g₁, mon_constr, conj(g₂))
                    unsafe_push!(indices₃, offrepart)
                    unsafe_push!(values₃, recoeff)
                    if offrepart ≠ offimpart
                        unsafe_push!(indices₄, offimpart)
                        unsafe_push!(values₄, offcanonical ? recoeff : -recoeff)
                    end
                elseif iscanonical(mon_constr)
                    # In principle, we have to go through all the terms and successively add their coefficients. Note that we
                    # require the monomials in constraint to be unique, and also the gs are all distinct and nonconjugated.
                    # As constraint is real-valued in total, every term will occur in its canonical form and once again
                    # conjugated. We simplify matters here and skip all the non-canonical forms, while for the canonical ones,
                    # we add the contribution for both terms.
                    # On the diagonal, and also on the off-diagonal if `real_valued`, this means doubling the coefficients,
                    # as the exact same term will appear once again and lead to a real-valued contribution only.
                    # On the off-diagonal with `!real_valued`, we won't ever get an exact duplicate.
                    repart₁, impart₁, canonical₁ = getreim(state, g₁, mon_constr, conj(g₁))
                    repart₂, impart₂, canonical₂ = getreim(state, g₂, mon_constr, conj(g₂))
                    @assert(canonical₁ && canonical₂)
                    @assert(repart₁ ≠ impart₁ && repart₂ ≠ impart₂)
                    if !iszero(recoeff)
                        doublere = recoeff + recoeff
                        unsafe_push!(indices₁, repart₁)
                        unsafe_push!(values₁, doublere)
                        unsafe_push!(indices₂, repart₂)
                        unsafe_push!(values₂, doublere)
                    end
                    if !iszero(imcoeff)
                        doubleim = imcoeff + imcoeff
                        unsafe_push!(indices₁, impart₁)
                        unsafe_push!(values₁, -doubleim)
                        unsafe_push!(indices₂, impart₂)
                        unsafe_push!(values₂, -doubleim)
                    end
                    # We must be more careful on the off-diagonal, as now iscanonical(mon_constr) does not necessarily
                    # translate into iscanonical of the product.
                    offrepart₁, offimpart₁, offcanonical₁ = getreim(state, g₁, mon_constr, conj(g₂))
                    if real_valued
                        if !iszero(recoeff)
                            unsafe_push!(indices₃, offrepart₁)
                            unsafe_push!(values₃, doublere)
                        end
                        if !iszero(imcoeff) && offrepart₁ ≠ offimpart₁ # maybe mon_constr = conj(g₁) * g₂?
                            unsafe_push!(indices₃, offimpart₁)
                            unsafe_push!(values₃, offcanonical₁ ? -doubleim : doubleim)
                        end
                    else
                        offrepart₂, offimpart₂, offcanonical₂ = getreim(state, g₁, conj(mon_constr), conj(g₂))
                        @assert(offrepart₂ ≠ offrepart₁ && offimpart₂ ≠ offimpart₁)
                        if !iszero(recoeff)
                            unsafe_push!(indices₃, offrepart₁, offrepart₂)
                            unsafe_push!(values₃, recoeff, recoeff)
                            if offrepart₁ ≠ offimpart₁
                                unsafe_push!(indices₄, offimpart₁)
                                unsafe_push!(values₄, offcanonical₁ ? recoeff : -recoeff)
                            end
                            if offrepart₂ ≠ offimpart₂
                                unsafe_push!(indices₄, offimpart₂)
                                unsafe_push!(values₄, offcanonical₂ ? recoeff : -recoeff)
                            end
                        end
                        if !iszero(imcoeff)
                            if offrepart₁ ≠ offimpart₁
                                unsafe_push!(indices₃, offimpart₁)
                                unsafe_push!(values₃, offcanonical₁ ? -imcoeff : imcoeff)
                            end
                            if offrepart₂ ≠ offimpart₂
                                unsafe_push!(indices₃, offimpart₂)
                                unsafe_push!(values₃, offcanonical₂ ? imcoeff : -imcoeff)
                            end
                            unsafe_push!(indices₄, offrepart₁, offrepart₂)
                            unsafe_push!(values₄, imcoeff, -imcoeff)
                        end
                    end
                end
            end
            if real_valued || isempty(values₄)
                sos_solver_add_quadratic!(state, indices₁, values₁, indices₂, values₂, (indices₃, values₃))
            else
                sos_solver_add_quadratic!(state, indices₁, values₁, indices₂, values₂, (indices₃, values₃),
                    (indices₄, values₄))
            end
        end
    end
    return
end

# g * [c₁₁ c₁₂; conj(c₁₂) c₂₂] * conj(g) ⪰ 0 - quadratic constraint
function sos_add_matrix_helper!(state, T, V, g::SimpleMonomial, c₁₁::P, c₁₂::P, c₂₂::P) where {P<:SimplePolynomial}
    # c₁₁ and c₂₂ must be real-valued anyway, but c₁₂ might not be so.
    real_valued = isreal(c₁₂)
    if isone(length(c₁₁)) && isone(length(c₁₂)) && isone(length(c₂₂))
        # All matrix entries have just a single monomial. Therefore, the diagonal monomials cannot have an imaginary part and
        # their coefficients are real-valued. But the off-diagonal term may be complex-valued.
        let t₁₁=first(c₁₁), t₂₁=first(c₁₂), t₂₂=first(c₂₂)
            @assert(isreal(monomial(t₁₁)) && isreal(monomial(t₂₂)))
            @assert(iszero(imag(coefficient(t₁₁))) && iszero(imag(coefficient(t₂₂))))
            offrepart, offimpart, offcanonical = getreim(state, g, monomial(t₂₁), conj(g))
            offrecoeff = real(coefficient(t₂₁))
            offimcoeff = imag(coefficient(t₂₁))
            if real_valued
                @assert(offrepart == offimpart && iszero(offimcoeff))
                sos_solver_add_quadratic!(
                    Val(:check),
                    state,
                    getreonly(state, g, monomial(t₁₁), conj(g)), real(coefficient(t₁₁)),
                    getreonly(state, g, monomial(t₂₂), conj(g)), real(coefficient(t₂₂)),
                    (offrepart, offrecoeff)
                )
            elseif offrepart == offimpart
                # The off-diagonal monomial is real-valued, but the coefficient might still be complex.
                if iszero(offimcoeff)
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        getreonly(state, g, monomial(t₁₁), conj(g)), real(coefficient(t₁₁)),
                        getreonly(state, g, monomial(t₂₂), conj(g)), real(coefficient(t₂₂)),
                        (offrepart, offrecoeff)
                    )
                elseif iszero(offrecoeff)
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        getreonly(state, g, monomial(t₁₁), conj(g)), real(coefficient(t₁₁)),
                        getreonly(state, g, monomial(t₂₂), conj(g)), real(coefficient(t₂₂)),
                        (offrepart, offimcoeff)
                    )
                else
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        StackVec(getreonly(state, g, monomial(t₁₁), conj(g))), StackVec(real(coefficient(t₁₁))),
                        StackVec(getreonly(state, g, monomial(t₂₂), conj(g))), StackVec(real(coefficient(t₂₂))),
                        (offrepart, offrecoeff),
                        (offrepart, offimcoeff)
                    )
                end
            else
                if iszero(offimcoeff)
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        getreonly(state, g, monomial(t₁₁), conj(g)), real(coefficient(t₁₁)),
                        getreonly(state, g, monomial(t₂₂), conj(g)), real(coefficient(t₂₂)),
                        (offrepart, offrecoeff),
                        (offimpart, offcanonical ? offrecoeff : -offrecoeff)
                    )
                elseif iszero(offrecoeff)
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        getreonly(state, g, monomial(t₁₁), conj(g)), real(coefficient(t₁₁)),
                        getreonly(state, g, monomial(t₂₂), conj(g)), real(coefficient(t₂₂)),
                        (offimpart, offcanonical ? -offimcoeff : offimcoeff),
                        (offrepart, offimcoeff)
                    )
                else
                    sos_solver_add_quadratic!(
                        Val(:check),
                        state,
                        StackVec(getreonly(state, g, monomial(t₁₁), conj(g))), StackVec(real(coefficient(t₁₁))),
                        StackVec(getreonly(state, g, monomial(t₂₂), conj(g))), StackVec(real(coefficient(t₂₂))),
                        (StackVec(offrepart, offimpart), StackVec(offrecoeff, offcanonical ? -offimcoeff : offimcoeff)),
                        (StackVec(offrepart, offimpart), StackVec(offimcoeff, offcanonical ? offrecoeff : -offrecoeff))
                    )
                end
            end
        end
    else
        # The matrix entries may have multiple terms, also complex-valued ones whose imaginary parts cancel at a later time.
        # Howver, the off-diagonal term may be entirely complex-valued.
        let indices₁=FastVec{T}(buffer=length(c₁₁)), values₁=similar(indices₁, V),
            indices₂=FastVec{T}(buffer=length(c₂₂)), values₂=similar(indices₂, V),
            indices₃=FastVec{T}(buffer=2length(c₁₂)), values₃=similar(indices₃, V)
            # - every real term will get up to one entry in the diagonal indices
            # - every other complex-valued term will be skipped in the diagonal
            # - the remaining complex-valued terms will get two entries in the diagonal
            # - every term will get up to two entries in the off-diagonal, but merging might reduce this a lot
            for (constr, ind, val) in (real_valued ? ((c₁₁, indices₁, values₁), (c₂₂, indices₂, values₂),
                                                      (c₁₂, indices₃, values₃)) :
                                                     ((c₁₁, indices₁, values₁), (c₂₂, indices₂, values₂)))
                for term_constr in constr
                    mon_constr = monomial(term_constr)
                    coeff_constr = coefficient(term_constr)
                    recoeff = real(coeff_constr)
                    imcoeff = imag(coeff_constr)
                    if isreal(mon_constr)
                        @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                        unsafe_push!(ind, getreonly(state, g, mon_constr, conj(g)))
                        unsafe_push!(val, recoeff)
                    elseif iscanonical(mon_constr)
                        repart, impart, canonical = getreim(state, g, mon_constr, conj(g))
                        @assert(canonical)
                        @assert(repart ≠ impart)
                        if !iszero(recoeff)
                            unsafe_push!(ind, repart)
                            unsafe_push!(val, recoeff + recoeff)
                        end
                        if !iszero(imcoeff)
                            unsafe_push!(ind, impart)
                            unsafe_push!(val, -(imcoeff + imcoeff))
                        end
                    end
                end
            end
            if real_valued
                sos_solver_add_quadratic!(Val(:check), state, indices₁, values₁, indices₂, values₂, (indices₃, values₃))
            else
                indices₄ = similar(indices₃)
                values₄ = similar(values₃)
                # Now for c₂₁ we don't know anything any more. We really must work with every single term in the polynomial,
                # but it may still happen that the conjugate is the current monomial was already inserted before, maybe leading
                # to cancellation, maybe not. At least some imaginary parts will not cancel.
                for term_constr in c₁₂
                    mon_constr = monomial(term_constr)
                    coeff_constr = coefficient(term_constr)
                    recoeff = real(coeff_constr)
                    imcoeff = imag(coeff_constr)
                    repart, impart, canonical = getreim(state, g, mon_constr, conj(g))
                    if !iszero(recoeff)
                        pushorupdate!(indices₃, repart, values₃, recoeff)
                        impart != repart && pushorupdate!(indices₄, impart, values₄, canonical ? recoeff : -recoeff)
                        # The off-diagonals need not be real-valued. But if they are, we don't have an extra index for the
                        # imaginary part (as it is always zero). Beware!
                    end
                    if !iszero(imcoeff)
                        impart != repart && pushorupdate!(indices₃, impart, values₃, canonical ? -imcoeff : imcoeff)
                        pushorupdate!(indices₄, repart, values₄, imcoeff)
                    end
                end
                sos_solver_add_quadratic!(Val(:check), state, indices₁, values₁, indices₂, values₂, (indices₃, values₃),
                    (indices₄, values₄))
            end
        end
    end
end

# For the PSD case, we don't want to implement everything more often than necessary. We already need a lot of implementations:
# (1) only real-valued monomials
# (2) complex-valued monomials, solver supports complex PSD
# (3) complex-valued monomials, solver doesn't support complex PSD
# in every combination with
# (a) solver provides (i, j)-indexed PSD variable (upper/lower triangle or full): X ⪰ 0, now work with X[i, j]
# (b) solver provides linearly indexed PSD variable (upper/lower triangle or full): X ⪰ 0, now work with X[k]
# (c) solver provides PSD cone membership (upper/lower triangle scaled/unscaled or full): M x ⪰ 0, now work with x
# This is all different logic (though (a) and (b) share a lot of code and are therefore implemented in on method that relies
# heavily on constant folding). We don't want to additionally implement everything explicitly for scalar and matrix
# constraints, so we'll define a 1x1 matrix so that Julia can do the work by itself.
struct ScalarMatrix{X} <: AbstractMatrix{X} # we could make it mutable and fully implement the AbstractArray interface, but we
                                            # don't need it
    x::X
end

Base.size(::ScalarMatrix) = (1, 1)
Base.length(::ScalarMatrix) = 1
@inline function Base.getindex(m::ScalarMatrix, args...)
    @boundscheck checkbounds(m, args...)
    return m.x
end
Base.iterate(m::ScalarMatrix) = m.x, nothing
Base.iterate(::ScalarMatrix, ::Nothing) = nothing

# generic SOS constraint with
# - only real-valued monomials involved in the grouping, and only real-valued polynomials involved in the constraint (so if it
#   contains complex coefficients/monomials, imaginary parts cancel out)
# - or complex-valued monomials involved in the grouping, but the solver supports the complex-valued PSD cone explicitly
# Interface for getting values from a PSD variable (e.g., Mosek barvar, COPT PSDVar; none for the complex case).
function sos_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial}, (Tidx, tri, offset)::Tuple{Union{DataType,<:Tuple},Symbol,Integer},
    type::Union{Tuple{Val{true},Val},Tuple{Val{false},Val{true}}})
    @assert(Tidx <: Integer || Tidx <: Tuple{Integer,Integer})
    @assert(tri ∈ (:L, :U))
    complex = type isa Tuple{Val{false},Val{true}}
    Vtype = complex ? Complex{V} : V
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    items = trisize(dim)
    linear_indexing = Tidx <: Integer
    if linear_indexing
        Tl = Tidx
    else
        T1 = fieldtype(Tidx, 1)
        T2 = fieldtype(Tidx, 2)
    end
    # we use a sizehint to make avoid growing and re-hashing the dictionary. We don't want to compute the exact
    # number of distinct monomials that can occur by multiplying `grouping` with itself and `constraint`, but
    # we also don't want to overestimate too much (although it's not extremely costly).
    # Certainly, we cannot get more items than monomial_count(2maxdegree(grouping) + maxdegree(constraint)) -
    #                                          monomial_count(2mindegree(grouping) + mindegree(constraint)).
    # However, a better bound for more incomplete groupings might be to assume that every monomial in
    # `grouping` can be multiplied with every monomial in `grouping` at the same or a later position (but not
    # before, this would certainly lead to duplicates). This will still be an overestimate, but gives `items`
    # monomials in the square grouping. We then crudely multiply this by the number of terms in the constraints
    # and get another overestimate.
    # For the complex case, this is even more crude. `grouping` will only contain the non-conjugated monomials, but
    # `constraint` will also contain conjugated ones. TODO: adjust effective_nvariable so that it ignores the conj part.
    # And the 2maxdegree argument also doesn't hold, instead it would be 2monomial_count(maxdegree, ...). But when mixing
    # real and complex variables, we need to double the real ones only. This makes the guesstimate ever larger than it has to
    # be.
    data = let nvars=effective_nvariables(grouping, monomials.(constraint)), constrex=extdegree(constraint),
        groupingex=extdegree(grouping)
        sizehint!(
            Dict{FastKey{T},linear_indexing ? Tuple{FastVec{Tl},FastVec{Vtype}} :
                                              Tuple{FastVec{T1},FastVec{T2},FastVec{Vtype}}}(),
            (complex ? 2 : 1) *
                min(monomial_count(2groupingex[2] + constrex[2], nvars) - monomial_count(2groupingex[1] + constrex[1], nvars),
                    items * length(constraint))
        )
    end
    if linear_indexing
        i = Tl(offset)
        gen_new = @capture(() -> (FastVec{$Tl}(), FastVec{$Vtype}()))
    else
        col = T2(offset)
        gen_new = @capture(() -> (FastVec{$T1}(), FastVec{$T2}(), FastVec{$Vtype}()))
    end
    grouping₂ = lazy_unalias(grouping)
    @inbounds for (exp2, g₂) in enumerate(grouping)
        for block_j in 1:block_size
            if !linear_indexing
                row = tri === :U ? T1(offset) : col
            end
            exp1_range = (tri === :U ? (1:exp2) : (exp2:lg))
            for (exp1, g₁) in zip(exp1_range, @view(grouping₂[exp1_range]))
                for block_i in (tri === :U ? (1:(exp1 == exp2 ? block_j : block_size)) :
                                             ((exp1 == exp2 ? block_j : 1):block_size))
                    for term_constr in constraint[block_i, block_j]
                        mon_constr = monomial(term_constr)
                        coeff_constr = coefficient(term_constr)
                        recoeff = real(coeff_constr)
                        imcoeff = imag(coeff_constr)
                        repart, impart, canonical = getreim(state, g₁, mon_constr, conj(g₂))
                        # Interpretation: repart is the constraint that defines the real part of the coefficient in front of
                        # g₁*mon_constr*ḡ₂, impart is the constraint that defines the imaginary part of the coefficient.
                        if !complex || (block_i == block_j && (exp1 == exp2 || (isreal(g₁) && isreal(g₂))))
                            # - !complex: all polynomials are real-valued in total. Same strategy as before, only take into
                            #   account canonical monomials. The doubling of will be done automatically by the solver on the
                            #   total off-diagonal.
                            # - block_i == block_j && exp1 == exp2: we are on the total diagonal. Even if not all polynomials
                            #   were real, those are for sure.
                            # - block_i == block_j && isreal(g₁) && isreal(g₂): we are on the inner diagonal, but since the
                            #   outer basis elements are real-valued, canonical and non-canonicals will also occur in pairs.
                            # Even if `complex` - so we are on the diagonal and the values should be Complex{V} - we can just
                            # do the push!, the conversion is implicitly done.
                            if canonical # ≡ iscanonical(mon_constr)
                                if !iszero(recoeff)
                                    svre = get!(gen_new, data, FastKey(repart))
                                    if linear_indexing
                                        push!(svre[1], i)
                                        push!(svre[2], recoeff)
                                    else
                                        push!(svre[1], row)
                                        push!(svre[2], col)
                                        push!(svre[3], recoeff)
                                    end
                                end
                                if repart == impart
                                    @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                    @assert(isreal(mon_constr))
                                elseif !iszero(imcoeff)
                                    svim = get!(gen_new, data, FastKey(impart))
                                    if linear_indexing
                                        push!(svim[1], i)
                                        push!(svim[2], imcoeff)
                                    else
                                        push!(svim[1], row)
                                        push!(svim[2], col)
                                        push!(svim[3], imcoeff)
                                    end
                                end
                            end
                        else
                            # We are not on the diagonal, so we must work with every entry. Updating might become necessary as
                            # the conjugate can pop up later.
                            if repart != impart
                                recoeff *= 1//2 # The solver will implicitly double; reverse
                                imcoeff *= 1//2 # While this won't necessarily work for V<:Integer, we don't really care about
                                                # it. No solver has integer data types.
                                # If repart == impart, the whole multiplied package is a real-valued monomial. Therefore, it
                                # will occur exactly as it is again in the transpose of the full matrix, so we need the
                                # implicit doubling by the solver.
                                svim = get!(gen_new, data, FastKey(impart))
                            end
                            # Interpretation: Complex(x, y) means to extract x*real(matrix entry) + y*imag(matrix entry).
                            svre = get!(gen_new, data, FastKey(repart))
                            if linear_indexing
                                pushorupdate!(svre, i, Complex(recoeff, -imcoeff))
                                if repart != impart
                                    pushorupdate!(svim, i, canonical ? Complex(imcoeff, recoeff) :
                                                                       Complex(-imcoeff, -recoeff))
                                end
                            else
                                pushorupdate!(svre, row, col, Complex(recoeff, -imcoeff))
                                if repart != impart
                                    pushorupdate!(svim, row, col, canonical ? Complex(imcoeff, recoeff) :
                                                                              Complex(-imcoeff, -recoeff))
                                end
                            end
                        end
                    end
                    if linear_indexing
                        i += one(Tl)
                    else
                        row += one(T1)
                    end
                end
            end
            if !linear_indexing
                col += one(T1)
            end
        end
    end
    (complex ? sos_solver_add_psd_complex! : sos_solver_add_psd!)(state, dim, data)
    return
end

# generic SOS constraint with complex-valued monomials involved in the grouping; the solver does not support complex-valued PSD
# cones, so we'll have to rewrite everything in terms of a real PSD cone
# Interface for getting values from a PSD variable (e.g., Mosek barvar, COPT PSDVar).
function sos_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial}, (Tidx, tri, offset)::Tuple{Union{DataType,<:Tuple},Symbol,Integer},
    ::Tuple{Val{false},Val{false}})
    @assert(Tidx <: Integer || Tidx <: Tuple{Integer,Integer})
    @assert(tri ∈ (:L, :U))
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    items = trisize(dim) # we still keep it at the trisize, although we now have dim^2 distinct elements
    linear_indexing = Tidx <: Integer
    if linear_indexing
        Tl = Tidx
    else
        T1 = fieldtype(Tidx, 1)
        T2 = fieldtype(Tidx, 2)
    end
    # TODO: see previous methods
    data = let nvars=effective_nvariables(grouping, monomials.(constraint)), constrex=extdegree(constraint),
        groupingex=extdegree(grouping)
        sizehint!(
            Dict{FastKey{T},linear_indexing ? Tuple{FastVec{Tl},FastVec{V}} : Tuple{FastVec{T1},FastVec{T2},FastVec{V}}}(),
            2min(monomial_count(2groupingex[2] + constrex[2], nvars) - monomial_count(2groupingex[1] + constrex[1], nvars),
                items * length(constraint))
        )
    end
    # There are two ways, and we need to implement them based on the different solver styles.
    # Default: Z ⪰ 0 ⟺ [Re Z  -Im Z; Im Z  Re Z] ⪰ 0.
    #   This is good for solvers of the type G x ∈ PSD, as we simply put the variables in their appropriate positions one after
    #   the other
    # Wang-style: rewrite the whole SDP
    #    max ⟨C, H⟩
    #    s.t. A(H) = b
    #         H ⪰ 0
    # ⟺ max ⟨Cᵣ, X₁ + X₂⟩ - ⟨Cᵢ, X₃ - X₃ᵀ⟩
    #    s.t. Aᵣ(X₁ + X₂) - Aᵢ(X₃ - X₃ᵀ) = bᵣ
    #         Aᵣ(X₃ - X₃ᵀ) + Aᵢ(X₁ + X₂) = bᵢ
    #         [X₁ X₃; X₃ᵀ X₂] ⪰ 0
    #   This is good for solvers of the type X ∈ PSD, do something with X, as they would have to enforce the block-equality of
    #   the diagonals manually. Conversion back works by H = Complex(X₁ + X₂, X₃ - X₃ᵗ)
    # In this method, we are in the second case.
    if linear_indexing
        # For linear indexing, we must keep track of the index in X₁, X₂, and X₃. Note that we always travel a triangle!
        # In the comment, we assume offset = 1, in the code, we deal with it appropriately.
        # X₁ starts from 1; in :U order, it just increases linearly. In :L order, after one column, we need to skip over dim
        # entries.
        i1 = Tl(offset)
        # X₂ is always a dense matrix, but we still travel one triangle. We therefore need an upper and a lower index.
        # In :U order, the index starts at items+1; and after one column, we need to skip over dim entries for the upper index.
        # The lower index is given by items+rowidx+(colidx-1)*(colidx+2dim)÷2.
        # In :L order, we have access to X₃ᵀ. The index starts at dim+1 and we need to skip over dim entries after one column
        # for the lower index. The upper index is given by rowidx+colidx*(4dim+1-colidx)÷2-dim.
        i2order = Tl((tri === :U ? items : dim) + offset)
        # X₃ starts at items+dim+1 in :U order and after one column, we need to skip dim entries. In :L order, it starts at
        # dim(3dim +1)÷2+1 and just increases linearly.
        i3 = Tl((tri === :U ? items + dim : ((dim * (3dim +1)) >> 1)) + offset)
        gen_new = @capture(() -> (FastVec{$Tl}(), FastVec{$V}()))
    else
        gen_new = @capture(() -> (FastVec{$T1}(), FastVec{$T2}(), FastVec{$V}()))
    end
    δ = offset -1
    col = 1
    grouping₂ = lazy_unalias(grouping)
    # There's a lot of constant folding going on here; in particular, conditioning on linear_indexing or tri will be done at
    # compile time.
    @inbounds for (exp2, g₂) in enumerate(grouping)
        for block_j in 1:block_size
            row = tri === :U ? 1 : col
            exp1_range = (tri === :U ? (1:exp2) : (exp2:lg))
            for (exp1, g₁) in zip(exp1_range, @view(grouping₂[exp1_range]))
                for block_i in (tri === :U ? (1:(exp1 == exp2 ? block_j : block_size)) :
                                             ((exp1 == exp2 ? block_j : 1):block_size))
                    if linear_indexing
                        i2other = Tl(tri === :U ? δ + items + col + ((row -1) * (row + 2dim)) >> 1 :
                                                  offset + col + dim - row + ((row -1) * (4dim + 2 - row)) >> 1)
                    end
                    for term_constr in constraint[block_i, block_j]
                        # This method is a bit different from the others. To exploit the advantages of Wang's reformulation,
                        # we have to keep the monomials complex-valued, not multiplying out the re/im decomposition.
                        # Aᵣ then corresponds to the real part of the coefficients and Aᵢ to their imaginary parts. In turn,
                        # bᵣ (still) is the index associated with the real part of the monomial and bᵢ with its imaginary part.
                        mon_constr = monomial(term_constr)
                        coeff_constr = coefficient(term_constr)
                        recoeff = real(coeff_constr)
                        imcoeff = imag(coeff_constr)
                        repart, impart, canonical = getreim(state, g₁, mon_constr, conj(g₂))
                        if block_i == block_j && (exp1 == exp2 || (isreal(g₁) && isreal(g₂)))
                            if canonical
                                # this is bᵣ + i bᵢ = recoeff * real(H(row, col)) + i * imcoeff * real(H(row, col))
                                if !iszero(recoeff)
                                    svre = get!(gen_new, data, FastKey(repart))
                                    # part 1a: Aᵣ(X₁ + X₂)
                                    if linear_indexing
                                        push!(svre[1], i1, i3)
                                        push!(svre[2], recoeff, recoeff)
                                    else
                                        push!(svre[1], T1(row + δ), T1(row + dim + δ))
                                        push!(svre[2], T2(col + δ), T2(col + dim + δ))
                                        push!(svre[3], recoeff, recoeff)
                                    end
                                end
                                if repart == impart
                                    @assert(iszero(imcoeff))
                                    @assert(isreal(mon_constr))
                                elseif !iszero(imcoeff)
                                    svim = get!(gen_new, data, FastKey(impart))
                                    # part 2b: Aᵢ(X₁ + X₂)
                                    if linear_indexing
                                        push!(svim[1], i1, i3)
                                        push!(svim[2], imcoeff, imcoeff)
                                    else
                                        push!(svim[1], T1(row + δ), T1(row + dim + δ))
                                        push!(svim[2], T2(col + δ), T2(col + dim + δ))
                                        push!(svim[3], imcoeff, imcoeff)
                                    end
                                end
                            end
                        else
                            if repart != impart
                                recoeff *= 1//2
                                imcoeff *= 1//2
                                svim = get!(gen_new, data, FastKey(impart))
                            end
                            # this is bᵣ + i bᵢ = recoeff * real(H(row, col)) - imcoeff * imag(H(row, col)) +
                            #                     i * imcoeff * real(H(row, col)) + i * recoeff * imag(H(row, col))
                            svre = get!(gen_new, data, FastKey(repart))
                            # Now we need both the real and imaginary part at position (row, col).
                            if !iszero(recoeff)
                                # part 1a: Aᵣ(X₁ + X₂)
                                if linear_indexing
                                    pushorupdate!(svre, i1, recoeff)
                                    pushorupdate!(svre, i3, recoeff)
                                else
                                    pushorupdate!(svre, T1(row + δ), T2(col + δ), recoeff)
                                    pushorupdate!(svre, T1(row + dim + δ), T2(col + dim + δ), recoeff)
                                end
                                # part 2a: Aᵣ(X₃ - X₃ᵀ)
                                if repart != impart
                                    if linear_indexing
                                        if i2order != i2other
                                            pushorupdate!(svim, tri === :U ? i2order : i2other, canonical ? recoeff : -recoeff)
                                            pushorupdate!(svim, tri === :U ? i2other : i2order, canonical ? -recoeff : recoeff)
                                        end
                                    elseif row != col
                                        if tri === :U
                                            pushorupdate!(svim, T1(row + δ), T2(col + dim + δ), canonical ? recoeff : -recoeff)
                                            pushorupdate!(svim, T1(col + δ), T2(row + dim + δ), canonical ? -recoeff : recoeff)
                                        else
                                            pushorupdate!(svim, T1(col + dim + δ), T2(row + δ), canonical ? recoeff : -recoeff)
                                            pushorupdate!(svim, T1(row + dim + δ), T2(col + δ), canonical ? -recoeff : recoeff)
                                        end
                                    end
                                end
                            end
                            if !iszero(imcoeff)
                                # part 1b: -Aᵢ(X₃ - X₃ᵀ)
                                if linear_indexing
                                    if i2order != i2other
                                        pushorupdate!(svre, tri === :U ? i2order : i2other, -imcoeff)
                                        pushorupdate!(svre, tri === :U ? i2other : i2order, imcoeff)
                                    end
                                elseif row != col
                                    if tri === :U
                                        pushorupdate!(svre, T1(row + δ), T2(col + dim + δ), -imcoeff)
                                        pushorupdate!(svre, T1(col + δ), T2(row + dim + δ), imcoeff)
                                    else
                                        pushorupdate!(svre, T1(col + dim + δ), T2(row + δ), -imcoeff)
                                        pushorupdate!(svre, T1(row + dim + δ), T2(col + δ), imcoeff)
                                    end
                                end
                                # part 2b: Aᵢ(X₁ + X₂)
                                if repart != impart
                                    if linear_indexing
                                        pushorupdate!(svim, i1, canonical ? imcoeff : -imcoeff)
                                        pushorupdate!(svim, i3, canonical ? imcoeff : -imcoeff)
                                    else
                                        pushorupdate!(svim, T1(row + δ), T2(col + δ),
                                                      canonical ? imcoeff : -imcoeff)
                                        pushorupdate!(svim, T1(row + dim + δ), T2(col + dim + δ),
                                                      canonical ? imcoeff : -imcoeff)
                                    end
                                end
                            end
                        end
                    end
                    if linear_indexing
                        i1 += one(Tl)
                        i2order += one(Tl)
                        i3 += one(Tl)
                    end
                    row += 1
                end
            end
            if linear_indexing
                if tri === :U
                    i3 += Tl(dim)
                else
                    i1 += Tl(dim)
                end
                i2order += Tl(dim)
            end
            col += 1
        end
    end
    sos_solver_add_psd!(state, 2dim, data)
    return
end


"""
    sos_add_matrix!(state, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}})

Parses a SOS constraint with a basis given in `grouping` (this might also be a partial basis due to sparsity), premultiplied by
`constraint` (which may be the unit polynomial for the moment matrix) and calls the appropriate solver functions to set up the
problem structure.

To make this function work for a solver, implement the following low-level primitives:
- [`sos_solver_mindex`](@ref)
- [`sos_solver_add_scalar!`](@ref)
- [`sos_solver_add_quadratic!`](@ref) (optional)
- [`sos_solver_add_psd!`](@ref)
- [`sos_solver_psd_indextype`](@ref)

Usually, this function does not have to be called explicitly; use [`sos_setup!`](@ref) instead.

See also [`sos_add_equality!`](@ref).
"""
function sos_add_matrix!(state, grouping::AbstractVector{M} where M<:SimpleMonomial, constraint::SimplePolynomial)
    lg = length(grouping)
    T = Base.promote_op(sos_solver_mindex, typeof(state), monomial_type(constraint))
    V = realtype(coefficient_type(constraint))
    if lg == 1
        @inbounds sos_add_matrix_helper!(state, T, V, grouping[1], constraint)
    elseif lg == 2 && sos_solver_supports_quadratic(state)
        @inbounds sos_add_matrix_helper!(state, T, V, grouping[1], lazy_unalias(grouping)[2], constraint)
    else
        sos_add_matrix_helper!(state, T, V, grouping, ScalarMatrix(constraint), sos_solver_psd_indextype(state),
            (Val(isreal(grouping) && isreal(constraint)), Val(sos_solver_supports_complex_psd(state))))
    end
    return
end

function sos_add_matrix!(state, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial})
    block_size = LinearAlgebra.checksquare(constraint)
    isone(block_size) && @inbounds return sos_add_matrix!(state, grouping, constraint[1, 1])
    lg = length(grouping)
    T = Base.promote_op(sos_solver_mindex, typeof(state), monomial_type(eltype(constraint)))
    V = realtype(coefficient_type(eltype(constraint)))
    if lg == 1 && block_size == 2 && sos_solver_supports_quadratic(state)
        @inbounds sos_add_matrix_helper!(state, T, V, grouping[1], constraint[1, 1], constraint[1, 2], constraint[2, 2])
    else
        sos_add_matrix_helper!(state, T, V, grouping, constraint, sos_solver_psd_indextype(state),
            (Val(isreal(grouping) && isreal(constraint)), Val(sos_solver_supports_complex_psd(state))))
    end
    return
end

"""
    sos_add_equality!(state, groupings::AbstractVector{M} where M<:SimpleMonomialVector, constraint::SimplePolynomial)

Parses a polynomial equality constraint and calls the appropriate solver functions to set up the problem structure.
`groupings` contains all the individual bases that will be squared in the process to generate the prefactor.

To make this function work for a solver, implement the following low-level primitives:
- [`sos_solver_add_free_prepare!`](@ref) (optional)
- [`sos_solver_add_free!`](@ref) (required)
- [`sos_solver_add_free_finalize!`](@ref) (optional)

Usually, this function does not have to be called explicitly; use [`sos_setup!`](@ref) instead.

See also [`sos_add_matrix!`](@ref).
"""
function sos_add_equality!(state, groupings::AbstractVector{MV} where {M<:SimpleMonomial,MV<:AbstractVector{M}}, constraint::SimplePolynomial)
    real_constr = isreal(constraint)
    real_basis = true
    buffer = length(constraint)
    # First we need an overestimator on the number of variables that might come from all our multiplications. Some solvers will
    # require adding the variables first.
    maxbasisitems = 0
    for grouping in groupings
        real_grouping = 0
        complex_grouping = 0
        for g in grouping
            if isreal(g)
                real_grouping += 1
            else
                complex_grouping += 1
                real_basis = false
            end
        end
        maxbasisitems += trisize(real_grouping) + real_grouping * 2complex_grouping + complex_grouping^2
        # only reals             mix a real with a complex or conj   only complexes
    end
    # Now the prefactor is an arbitrary polynomial, and also constraint can be anything - there is no need that they be
    # real-valued any more. However, as we transfer everything to the objective, we then need to consider real and imaginary
    # part separately.
    # We will simply separate real and imaginary part in the constraint to give two separate constraints. Our prefactors will
    # in total always be real-valued.
    eqstate = @inline sos_solver_add_free_prepare!(state, real_constr ? maxbasisitems : 2maxbasisitems)
    # Construct the basis for the zero constraints by considering the squares of all the elements in a given basis
    # If we were to multiply every item in grouping with every item in conj(grouping), for real-valued problems, this would
    # give rise to a large number of redundant constraints. Passing these linear dependencies to the optimizer might lead to
    # severe numerical issues, even if a linear dependency checker is employed by the solver.
    # Example: Problem (6.1) from the correlative sparsity paper with
    # (nx = 2, ny = 4, μ = 0, M = 18). Mosek will have tons of issues.
    # - Using the default parameters, it reports 0.053532 as optimal.
    # - Switching off the presolver or PRESOLVE_LINDEP_USE gives the correct answer (which is lower!), 0.051707.
    # - Setting PRESOLVE_ELIMINATOR_MAX_NUM_TRIES to 0, 1, 2 gives 0.215861, 0.052381, 0.053533.
    # But in general, we want the presolver to work, so we'll have to take care that we don't generate all those duplicates.
    # Unfortunately, this requires keeping track of everything we already did (unless we are in the specia case
    # isone(length(groupings)) and first(groupings) is a LazyMonomials... Should we treat this separately? But then, our whole
    # two-loop must be transformed into a one-loop.)
    zero_basis = sizehint!(Set{FastKey{Int}}(), maxbasisitems)
    for grouping in groupings
        grouping₂ = lazy_unalias(grouping)
    @inbounds let T=Base.promote_op(sos_solver_mindex, typeof(state), monomial_type(constraint)),
        V=realtype(coefficient_type(constraint)),
        indices=FastVec{T}(buffer=real_constr && real_basis ? buffer : (real_constr || real_basis ? 2buffer : 4buffer)),
        values=similar(indices, V)
        if !real_constr || !real_basis
            indices₂ = similar(indices)
            values₂ = similar(values)
            if !real_constr && !real_basis
                indices₃ = similar(indices)
                values₃ = similar(values)
                indices₄ = similar(indices)
                values₄ = similar(values)
            end
        end
        grouping₂ = lazy_unalias(grouping)
            for (exp2, g₂) in enumerate(grouping)
                for g₁ in Iterators.drop(grouping₂, isreal(g₂) ? exp2 -1 : 0)
                real_grouping = real_basis || g₁.exponents_complex == g₂.exponents_complex
                # only canonical products g₁ * conj(g₂)
                real_grouping || g₁.exponents_complex < g₂.exponents_complex || continue
                    # no duplicates
                    let grouping_idx=FastKey(monomial_index(g₁, conj(g₂)))
                        grouping_idx ∈ zero_basis ? continue : push!(zero_basis, grouping_idx)
                    end
                for term_constr in constraint
                    mon_constr = monomial(term_constr)
                    coeff_constr = coefficient(term_constr)
                    recoeff = real(coeff_constr)
                    imcoeff = imag(coeff_constr)
                    repart, impart, canonical = getreim(state, g₁, mon_constr, conj(g₂))
                    if real_constr
                        # Even here, it may happen that we get duplicates: this is if real_grouping is false, for then we also
                        # might add something with the conjugate grouping; and a later (or earlier) term may, with the
                        # canonical grouping, give rise to the same monomial.
                        if isreal(mon_constr)
                            @assert(iszero(imcoeff) && canonical)
                            if !iszero(recoeff)
                                pushorupdate!(indices, repart, values, recoeff)
                                if repart != impart
                                    @assert(!real_basis)
                                    pushorupdate!(indices₂, impart, values₂, recoeff)
                                end
                            end
                        elseif iscanonical(mon_constr)
                            @assert(canonical)
                            if real_grouping
                                @assert(repart != impart)
                                iszero(recoeff) || pushorupdate!(indices, repart, values, recoeff)
                                iszero(imcoeff) || pushorupdate!(indices, impart, values, imcoeff)
                            else
                                repart₂, impart₂, canonical₂ = getreim(state, g₂, mon_constr, conj(g₁))
                                @assert(repart != repart₂ && impart != impart₂)
                                if !iszero(recoeff)
                                    if repart == impart
                                        @assert(repart₂ != impart₂)
                                        # complex monomial, complex grouping, but real product
                                        # -> will be very different for g₁*conj(mon_constr)*conj(g₂), but then, the
                                        # conjugate grouping will have this.
                                        pushorupdate!(indices, repart, values, recoeff + recoeff)
                                        pushorupdate!(indices, repart₂, values, recoeff)
                                        pushorupdate!(indices₂, impart₂, values₂, canonical₂ ? -recoeff : recoeff)
                                    elseif repart₂ == impart₂
                                        @assert(repart != impart)
                                        pushorupdate!(indices, repart, values, recoeff)
                                        pushorupdate!(indices, repart₂, values, recoeff + recoeff)
                                        pushorupdate!(indices₂, impart, values₂, recoeff)
                                    else
                                        pushorupdate!(indices, repart, values, recoeff)
                                        pushorupdate!(indices, repart₂, values, recoeff)
                                        pushorupdate!(indices₂, impart, values₂, recoeff)
                                        pushorupdate!(indices₂, impart₂, values₂, canonical₂ ? -recoeff : recoeff)
                                    end
                                end
                                if !iszero(imcoeff)
                                    if repart == impart
                                        @assert(repart₂ != impart₂)
                                        pushorupdate!(indices, impart₂, values, canonical₂ ? imcoeff : -imcoeff)
                                        pushorupdate!(indices₂, repart, values₂, -(imcoeff + imcoeff))
                                        pushorupdate!(indices₂, repart₂, values₂, imcoeff)
                                    elseif repart₂ == impart₂
                                        @assert(repart != impart)
                                        pushorupdate!(indices, impart, values, imcoeff)
                                        pushorupdate!(indices₂, repart, values₂, -imcoeff)
                                        pushorupdate!(indices₂, repart₂, values₂, imcoeff + imcoeff)
                                    else
                                        pushorupdate!(indices, impart, values, imcoeff)
                                        pushorupdate!(indices, impart₂, values, canonical₂ ? imcoeff : -imcoeff)
                                        pushorupdate!(indices₂, repart, values₂, -imcoeff)
                                        pushorupdate!(indices₂, repart₂, values₂, imcoeff)
                                    end
                                end
                            end
                        end
                    elseif isreal(mon_constr)
                        @assert(canonical)
                        if !iszero(recoeff)
                            pushorupdate!(indices, repart, values, recoeff)
                            repart == impart || pushorupdate!(indices₃, impart, values₃, recoeff)
                        end
                        if !iszero(imcoeff)
                            pushorupdate!(indices₂, repart, values₂, imcoeff)
                            repart == impart || pushorupdate!(indices₄, impart, values₄, imcoeff)
                        end
                    else
                        if repart != impart || !real_grouping
                            recoeff *= 1//2
                            imcoeff *= 1//2
                        end
                        if real_grouping
                            if !iszero(recoeff)
                                pushorupdate!(indices, repart, values, recoeff)
                                repart != impart &&
                                    pushorupdate!(indices₂, impart, values₂, canonical ? -recoeff : recoeff)
                            end
                            if !iszero(imcoeff)
                                repart != impart && pushorupdate!(indices, impart, values, canonical ? imcoeff : -imcoeff)
                                pushorupdate!(indices₂, repart, values₂, imcoeff)
                            end
                        else
                            repart₂, impart₂, canonical₂ = getreim(state, g₂, mon_constr, conj(g₁))
                            @assert(repart != repart₂ && impart != impart₂)
                            if !iszero(recoeff)
                                if repart == impart
                                    # special: the scaling didn't happen, but it should
                                    @assert(repart₂ != impart₂)
                                    pushorupdate!(indices, repart, values, recoeff + recoeff)
                                    pushorupdate!(indices₄, repart, values₄, recoeff + recoeff)
                                    pushorupdate!(indices, repart₂, values, recoeff)
                                    pushorupdate!(indices₄, repart₂, values₄, -recoeff)
                                    pushorupdate!(indices₂, impart₂, values₂, canonical₂ ? -recoeff : recoeff)
                                    pushorupdate!(indices₃, impart₂, values₃, canonical₂ ? -recoeff : recoeff)
                                elseif repart₂ == impart₂
                                    @assert(repart != impart)
                                    pushorupdate!(indices, repart, values, recoeff)
                                    pushorupdate!(indices₄, repart, values₄, recoeff)
                                    pushorupdate!(indices, repart₂, values, recoeff + recoeff)
                                    pushorupdate!(indices₄, repart₂, values₄, -(recoeff + recoeff))
                                    pushorupdate!(indices₂, impart, values₂, canonical ? -recoeff : recoeff)
                                    pushorupdate!(indices₃, impart, values₃, canonical ? recoeff : -recoeff)
                                else
                                    pushorupdate!(indices, repart, values, recoeff)
                                    pushorupdate!(indices₄, repart, values₄, recoeff)
                                    pushorupdate!(indices, repart₂, values, recoeff)
                                    pushorupdate!(indices₄, repart₂, values₄, -recoeff)
                                    pushorupdate!(indices₂, impart, values₂, canonical ? -recoeff : recoeff)
                                    pushorupdate!(indices₃, impart, values₃, canonical ? recoeff : -recoeff)
                                    pushorupdate!(indices₂, impart₂, values₂, canonical₂ ? -recoeff : recoeff)
                                    pushorupdate!(indices₃, impart₂, values₃, canonical₂ ? -recoeff : recoeff)
                                end
                            end
                            if !iszero(imcoeff)
                                if repart == impart
                                    pushorupdate!(indices₂, repart, values₂, imcoeff + imcoeff)
                                    pushorupdate!(indices₃, repart, values₃, -(imcoeff + imcoeff))
                                    pushorupdate!(indices₂, repart₂, values₂, imcoeff)
                                    pushorupdate!(indices₃, repart₂, values₃, imcoeff)
                                    pushorupdate!(indices, impart₂, values, canonical₂ ? imcoeff : -imcoeff)
                                    pushorupdate!(indices₄, impart₂, values₄, canonical₂ ? -imcoeff : imcoeff)
                                elseif repart₂ == impart₂
                                    pushorupdate!(indices₂, repart, values₂, imcoeff)
                                    pushorupdate!(indices₃, repart, values₃, -imcoeff)
                                    pushorupdate!(indices₂, repart₂, values₂, imcoeff + imcoeff)
                                    pushorupdate!(indices₃, repart₂, values₃, imcoeff + imcoeff)
                                    pushorupdate!(indices, impart, values, canonical ? imcoeff : -imcoeff)
                                    pushorupdate!(indices₄, impart, values₄, canonical ? imcoeff : -imcoeff)
                                else
                                    pushorupdate!(indices₂, repart, values₂, imcoeff)
                                    pushorupdate!(indices₃, repart, values₃, -imcoeff)
                                    pushorupdate!(indices₂, repart₂, values₂, imcoeff)
                                    pushorupdate!(indices₃, repart₂, values₃, imcoeff)
                                    pushorupdate!(indices, impart, values, canonical ? imcoeff : -imcoeff)
                                    pushorupdate!(indices₄, impart, values₄, canonical ? imcoeff : -imcoeff)
                                    pushorupdate!(indices, impart₂, values, canonical₂ ? imcoeff : -imcoeff)
                                    pushorupdate!(indices₄, impart₂, values₄, canonical₂ ? -imcoeff : imcoeff)
                                end
                            end
                        end
                    end
                end
                eqstate = @inline sos_solver_add_free!(state, eqstate, indices, values, false)
                empty!(indices)
                empty!(values)
                if !real_constr || !real_basis
                    if !isempty(indices₂)
                        eqstate = @inline sos_solver_add_free!(state, eqstate, indices₂, values₂, false)
                        empty!(indices₂)
                        empty!(values₂)
                    end
                    if !real_constr && !real_basis
                        if !isempty(indices₃)
                            eqstate = @inline sos_solver_add_free!(state, eqstate, indices₃, values₃, false)
                            empty!(indices₃)
                            empty!(values₃)
                        end
                        if !isempty(indices₄)
                            eqstate = @inline sos_solver_add_free!(state, eqstate, indices₄, values₄, false)
                            empty!(indices₄)
                            empty!(values₄)
                        end
                    end
                end
            end
        end
    end
    end
    @inline sos_solver_add_free_finalize!(state, eqstate)
    return
end

collect_grouping(g::AbstractVector{M} where M<:SimpleMonomial) = g
collect_grouping(g) = collect(g)

"""
    sos_setup!(state, relaxation::AbstractPORelaxation, groupings::RelaxationGroupings)

Sets up all the necessary SOS matrices, free variables, objective, and constraints of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`sos_solver_mindex`](@ref)
- [`sos_solver_add_scalar!`](@ref)
- [`sos_solver_add_quadratic!`](@ref) (optional)
- [`sos_solver_add_psd!`](@ref)
- [`sos_solver_psd_indextype`](@ref)
- [`sos_solver_add_free_prepare!`](@ref) (optional)
- [`sos_solver_add_free!`](@ref)
- [`sos_solver_add_free_finalize!`](@ref) (optional)
- [`sos_solver_fix_constraints!`](@ref)

!!! warning "Indices"
    The constraint indices used in all solver functions directly correspond to the indices given back by
    [`sos_solver_mindex`](@ref). However, in a sparse problem there may be far fewer indices present; therefore, when the
    problem is finally given to the solver, care must be taken to eliminate all unused indices.

See also [`sos_add_matrix!`](@ref), [`sos_add_equality!`](@ref).
"""
function sos_setup!(state, relaxation::AbstractPORelaxation{<:POProblem{P}}, groupings::RelaxationGroupings) where {P}
    problem = poly_problem(relaxation)
    # SOS term for objective
    for grouping in groupings.obj
        sos_add_matrix!(state, collect_grouping(grouping),
            SimplePolynomial(constant_monomial(P), coefficient_type(problem.objective)))
    end
    # free items
    for (groupingsᵢ, constrᵢ) in zip(groupings.zeros, problem.constr_zero)
        sos_add_equality!(state, collect_grouping.(groupingsᵢ), constrᵢ)
    end
    # localizing matrices
    for (groupingsᵢ, constrᵢ) in zip(groupings.nonnegs, problem.constr_nonneg)
        for grouping in groupingsᵢ
            sos_add_matrix!(state, collect_grouping(grouping), constrᵢ)
        end
    end
    for (groupingsᵢ, constrᵢ) in zip(groupings.psds, problem.constr_psd)
        for grouping in groupingsᵢ
            sos_add_matrix!(state, collect_grouping(grouping), constrᵢ)
        end
    end

    T = Base.promote_op(sos_solver_mindex, typeof(state), monomial_type(P))
    V = realtype(coefficient_type(problem.objective))
    # add lower bound
    if isone(problem.prefactor)
        @inline sos_solver_add_free_finalize!(
            state,
            sos_solver_add_free!(
                state,
                sos_solver_add_free_prepare!(state, 1),
                StackVec(sos_solver_mindex(state, constant_monomial(P))),
                StackVec(one(V)),
                true
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
            @inline sos_solver_add_free_finalize!(
                state,
                sos_solver_add_free!(state, sos_solver_add_free_prepare!(state, 1), indices, values, true)
            )
        end
    end

    # fix constraints to objective
    let buffer=length(problem.objective), indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
        for t in problem.objective
            mon = monomial(t)
            coeff = coefficient(t)
            recoeff = real(coeff)
            imcoeff = imag(coeff)
            repart, impart, canonical = getreim(state, mon)
            # repart is the constraint that is associated with the real part of the monomial mon;
            # impart is the constraint associated with the imaginary part of the canonicalized monomial mon.
            if repart == impart
                @assert(iszero(imcoeff)) # else the objective would not be real-valued
                unsafe_push!(indices, repart)
                unsafe_push!(values, recoeff)
            elseif canonical
                if !iszero(recoeff)
                    unsafe_push!(indices, repart)
                    unsafe_push!(values, recoeff)
                end
                if !iszero(imcoeff)
                    unsafe_push!(indices, impart)
                    unsafe_push!(values, imcoeff)
                end
            end
        end
        sos_solver_fix_constraints!(state, finish!(indices), finish!(values))
    end

    return
end