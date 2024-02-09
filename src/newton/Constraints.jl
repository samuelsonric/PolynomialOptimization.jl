function merge_constraints_postproc(T, nv, mons_idx_set, degree, dense::Bool, verbose::Bool)
    mons_idx = sort!(collect(mons_idx_set))
    next_col = 1
    max_col = length(mons_idx)
    iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), T(degree), zeros(T, nv), fill(T(degree), nv), true)
    if verbose
        iterlen = length(iter)
        @printf("└ checking %d monomials (%.2f)%%", iterlen, 100maxcol / iterlen)
    end
    if dense
        coeffs = resizable_array(T, nv, max_col)
        @inbounds for (idx, powers) in enumerate(iter)
            if idx == mons_idx[next_col]
                copyto!(@view(coeffs[:, next_col]), powers)
                next_col += 1
                next_col > max_col && break
            end
        end
        @assert(next_col == max_col +1)
        return coeffs
    else
        colptr = resizable_array(UInt, max_col +1)
        rowval = FastVec{UInt}()
        nzval = FastVec{T}()
        @inbounds for (idx, powers) in enumerate(iter)
            if idx == mons_idx[next_col]
                colptr[next_col] = length(rowval) +1
                for (row, val) in enumerate(powers)
                    if !iszero(val)
                        push!(rowval, row)
                        push!(nzval, val)
                    end
                end
                next_col += 1
                next_col > max_col && break
            end
        end
        @assert(next_col == max_col +1)
        colptr[next_col] = length(rowval) +1
        return SparseMatrixCSC{T,UInt}(nv, max_col, colptr, finish!(rowval), finish!(nzval))
    end
end

function merge_constraints(degree, indextype, objective::SimpleRealPolynomial, zero, nonneg, psd, dense::Bool, verbose::Bool)
    @verbose_info("Incorporating constraints into set of exponents")
    nv = nvariables(objective)
    T = SimplePolynomials.smallest_unsigned(2degree)
    mons_idx_set = sizehint!(Set{FastKey{indextype}}(), length(objective))
    # we start by storing the indices of the monomials only, which is the most efficient way for eliminating duplicates
    # afterwards
    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(indextype(monomial_index(mon))))
    end
    # If there are constraints present, things are not so simple. We assume a Putinar certificate:
    # f ≥ 0 on {zero == 0, nonneg ≥ 0, psd ⪰ 0} ⇐ f = σ₀ + ∑ᵢ nonnegᵢ σᵢ + ∑ⱼ ⟨psdⱼ, Mⱼ⟩ + ∑ₖ zeroₖ pₖ
    #                                              where σ₀, σᵢ ∈ SOS, Mⱼ ∈ SOSmatrix, pₖ ∈ poly
    # This can simply be reformulated into f - ∑ᵢ nonnegᵢ σᵢ - ∑ⱼ ⟨psdⱼ, Mⱼ⟩ - ∑ₖ zeroₖ pₖ ∈ SOS, i.e., we can now apply
    # Newton methods to the polynomial with subtracted constraint certifiers. The variable degree influences how large
    # the σᵢ, Mⱼ, and pₖ will maximally be.
    # Note that since the coefficients of the σᵢ, Mⱼ, and pₖ are unknowns, we don't need to ask ourselves whether some
    # cancellation may occur - we don't know. So every monomial that is present in any of the constraints, multiplied
    # by any monomial of allowed degree for the multiplier, will give rise to an additional entry in the coeffs array.
    # This can quickly become disastrous if `degree` is high but the degree of the constraints is low (as then, the
    # prefactors have lots of entries), but it is not so harmful in the other regime.
    minmultideg = zeros(T, nv)
    maxmultideg = similar(minmultideg)
    powers₁ = similar(minmultideg)
    powers₂ = similar(minmultideg)
    monomial₁ = SimpleMonomial{nv,0,T,typeof(powers₁)}(powers₁, SimplePolynomials.absent, SimplePolynomials.absent)
    monomial₂ = SimpleMonomial{nv,0,T,typeof(powers₂)}(powers₂, SimplePolynomials.absent, SimplePolynomials.absent)

    # 1. zero constraints
    isempty(zero) || @verbose_info("├ zero constraints")
    for zeroₖ in zero
        maxdeg = T(2(degree - maxhalfdegree(zeroₖ)))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        sizehint!(mons_idx_set, length(mons_idx_set) + length(iter) * length(zeroₖ))
        for t in zeroₖ
            monₜ = monomial(t)
            for _ in iter
                push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, monₜ))))
            end
        end
    end

    # 2. nonneg constraints
    # Note that we can still exploit that the σⱼ must be sums of squares: despite being polynomials of degree deg(σⱼ)
    # with unknown coefficients, _some_ of these must for sure be zero, namely those that cannot be reached by
    # combining any two of the possible coefficients that are in the valid multidegree range of σⱼ.
    isempty(nonneg) || @verbose_info("├ nonnegative constraints")
    for nonnegᵢ in nonneg
        maxdeg = T(degree - maxhalfdegree(nonnegᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        len = length(iter)
        sizehint!(mons_idx_set, length(mons_idx_set) + (len * (len +1) ÷ 2) * length(nonnegᵢ))
        for t in nonnegᵢ
            monₜ = monomial(t)
            for _ in iter
                # no need to run over duplicates
                push!(mons_idx_set, monomial_index(monomial₁, conj(monomial₁), monₜ))
                @inbounds copyto!(powers₂, powers₁)
                for _ in InitialStateIterator(MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg,
                                                                                 maxmultideg, powers₂),
                                              moniter_state(powers₂))
                    push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, monomial₂, monₜ))))
                end
            end
        end
    end

    # 3. psd constraints
    # Those are modeled in terms of SOS matrices. Given the basis u of `degree`, an m×m-matrix M is a SOS matrix iff
    # M(x) = (u ⊗ 1ₘ)ᵀ Z (u ⊗ 1ₘ) with Z ⪰ 0. Still, there is no duplication of the Z-coefficients (apart from
    # symmetry), so there is no way in which these unknown coefficients could potentially cancel: every entry in Z will
    # appear in exactly one cell in a triangle of M. So basically, all that we have to do is to apply the SOS
    # decomposition cell-wise.
    isempty(psd) || @verbose_info("├ PSD constraints")
    for psdᵢ in psd
        dim = LinearAlgebra.checksquare(psdᵢ)
        maxdeg = T(degree - maxhalfdegree(psdᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        len = length(iter)
        sizehint!(mons_idx_set, length(mons_idx_set) + (len * (len +1) ÷ 2) *
                                sum(@capture(length($psdᵢ[i, j]) for j in 1:dim for i in 1:j), init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for _ in iter
                    # no need to run over duplicates
                    push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, conj(monomial₁), monₜ))))
                    @inbounds copyto!(powers₂, powers₁)
                    for _ in InitialStateIterator(MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg,
                                                                                     maxmultideg, powers₂),
                                                  moniter_state(powers₂))
                        push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, monomial₂, monₜ))))
                    end
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    @verbose_info("├ Total number of coefficients: ", length(mons_idx_set),
        "\nConverting back from intermediate to exponent representation")
    return merge_constraints_postproc(T, nv, mons_idx_set, 2degree, dense, verbose)
end

_realify(m::SimpleComplexMonomial{Nc,P,V}) where {Nc,P<:Unsigned,V<:AbstractVector{P}} =
    SimpleMonomial{Nc,0,P,V}(m.exponents_complex, SimplePolynomials.absent, SimplePolynomials.absent)

function merge_constraints(degree, indextype, objective::SimpleComplexPolynomial, zero, nonneg, psd,
    dense::Bool, verbose::Bool)
    @verbose_info("Incorporating constraints into set of exponents")
    # Note that in the complex-valued case there's no mixing - i.e., no real variables. And every monomial appears once in its
    # original form, once in its conjugate. We are only interested in the "original" (whichever it is), so we effectively treat
    # every monomial as real, discarding the conjugate part. (It would be possible to do it differently, but then in the end
    # when we reduce everything to the complex part dropping the conjugates, deleting duplicates would be necessary - in this
    # way, we don't even generate duplicates.)

    nv = nvariables(objective) ÷ 2 # don't double Nc
    T = SimplePolynomials.smallest_unsigned(2degree)
    mons_idx_set = sizehint!(Set{FastKey{indextype}}(), length(objective))

    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(indextype(monomial_index(_realify(mon)))))
    end

    minmultideg = zeros(T, nv)
    maxmultideg = similar(minmultideg)
    powers₁ = similar(minmultideg)
    monomial₁ = SimpleMonomial{nv,0,T,typeof(powers₁)}(powers₁, SimplePolynomials.absent, SimplePolynomials.absent)

    # 1. zero constraints
    # 2. nonneg constraints
    for constraints in (zero, nonneg)
        isempty(constraints) || @verbose_info("├ ", constraints === zero ? "zero" : "nonnegative", " constraints")
        for constrₖ in constraints
            maxdeg = T(degree - maxhalfdegree(constrₖ))
            fill!(maxmultideg, maxdeg)
            iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
            sizehint!(mons_idx_set, length(mons_idx_set) + 2length(iter) * length(constrₖ))
            for t in constrₖ
                monₜ = monomial(t)
                for _ in iter
                    push!(mons_idx_set,
                        monomial_index(monomial₁, FastKey(indextype(_realify(monₜ)))),
                        monomial_index(monomial₁, FastKey(indextype(_realify(conj(monₜ)))))
                    )
                end
            end
        end
    end

    # 3. psd constraints
    isempty(psd) || @verbose_info("├ PSD constraints")
    for psdᵢ in psd
        dim = LinearAlgebra.checksquare(psdᵢ)
        maxdeg = T(degree - maxhalfdegree(psdᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        sizehint!(mons_idx_set, length(mons_idx_set) + 2length(iter) * sum(length, psdᵢ, init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for _ in iter
                    push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, _realify(monₜ)))))
                    i == j || push!(mons_idx_set, FastKey(indextype(monomial_index(monomial₁, _realify(conj(monₜ))))))
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    @verbose_info("├ Total number of coefficients: ", length(mons_idx_set),
        "\nConverting back from intermediate to exponent representation")
    return merge_constraints_postproc(T, nv, mons_idx_set, degree, dense, verbose)
end