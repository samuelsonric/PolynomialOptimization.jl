function merge_constraints(objective::SimpleRealPolynomial{<:Any,Nr,P}, zero, nonneg, psd, groupings::RelaxationGroupings,
    dense::Val, verbose::Bool) where {Nr,P<:Unsigned}
    @verbose_info("Incorporating constraints into set of exponents")
    mons_idx_set = sizehint!(Set{FastKey{Int}}(), length(objective))
    # we start by storing the indices of the monomials only, which is the most efficient way for eliminating duplicates
    # afterwards
    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(monomial_index(mon)))
    end
    # If there are constraints present, things are not so simple. We assume a Putinar certificate:
    # f ≥ 0 on {zero == 0, nonneg ≥ 0, psd ⪰ 0} ⇐ f = σ₀ + ∑ᵢ nonnegᵢ σᵢ + ∑ⱼ ⟨psdⱼ, Mⱼ⟩ + ∑ₖ zeroₖ pₖ
    #                                              where σ₀, σᵢ ∈ SOS, Mⱼ ∈ SOSmatrix, pₖ ∈ poly
    # This can simply be reformulated into f - ∑ᵢ nonnegᵢ σᵢ - ∑ⱼ ⟨psdⱼ, Mⱼ⟩ - ∑ₖ zeroₖ pₖ ∈ SOS, i.e., we can now apply Newton
    # methods to the polynomial with subtracted constraint certifiers. We get our exact form for the multipliers from
    # `groupings` - if sparsity methods have already been applied, we can exploit them.
    # Note that since the coefficients of the σᵢ, Mⱼ, and pₖ are unknowns, we don't need to ask ourselves whether some
    # cancellation may occur - we don't know. So every monomial that is present in any of the constraints, multiplied
    # by any monomial of allowed degree for the multiplier, will give rise to an additional entry in the coeffs array.
    # This can quickly become disastrous if `degree` is high but the degree of the constraints is low (as then, the
    # prefactors have lots of entries), but it is not so harmful in the other regime.

    # 1./2. zero/nonneg constraints
    # Note that we can still exploit that the σⱼ must be made up of products of elements in the grouping: despite being
    # polynomials of degree deg(σⱼ) with unknown coefficients, _some_ of these must for sure be zero, namely those that cannot
    # be reached by combining any two of the possible coefficients that are in the valid multidegree range of σⱼ.
    for (s, constr_groupings, constrs) in (("zero", groupings.zeros, zero), ("nonnegative", groupings.nonnegs, nonneg))
        isempty(constrs) || @verbose_info("├ ", s, " constraints")
        for (groupings, constrᵢ) in zip(constr_groupings, constrs)
            sizehint!(mons_idx_set, length(mons_idx_set) + length(constrᵢ) * sum(g -> length(g) * (length(g) +1) ÷ 2,
                                                                                 groupings))
            for t in constrᵢ
                monₜ = monomial(t)
                for grouping in groupings
                    grouping₂ = lazy_unalias(grouping)
                    for (i, g₁) in enumerate(grouping)
                        for g₂ in @view(grouping₂[i:end])
                            push!(mons_idx_set,
                                FastKey(monomial_index(g₁, monₜ, conj(g₂))),
                                FastKey(monomial_index(g₂, monₜ, conj(g₁)))
                            )
                        end
                    end
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
    for (groupings, psdᵢ) in zip(groupings.psds, psd)
        dim = LinearAlgebra.checksquare(psdᵢ)
        sizehint!(mons_idx_set, length(mons_idx_set) + sum(g -> length(g) * (length(g) +1) ÷ 2, groupings.psds) *
                                sum(@capture(length($psdᵢ[i, j]) for j in 1:dim for i in 1:j), init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for grouping in groupings
                    grouping₂ = lazy_unalias(grouping)
                    for (i, g₁) in enumerate(grouping)
                        for g₂ in @view(grouping₂[i:end])
                            push!(mons_idx_set,
                                FastKey(monomial_index(g₁, monₜ, conj(g₂))),
                                FastKey(monomial_index(g₂, monₜ, conj(g₁)))
                            )
                        end
                    end
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    @verbose_info("├ Total number of coefficients: ", length(mons_idx_set),
        "\nConverting back from intermediate to exponent representation")
    # Taking indices is about twice as slow as iterating through monomials. So let's check how many monomials we have and which
    # method is more efficient.
    return exponents_from_indices(P, Nr, mons_idx_set, dense)
end

_realify(m::SimpleComplexMonomial{Nc,P,V}) where {Nc,P<:Unsigned,V<:AbstractVector{P}} =
    SimpleMonomial{Nc,0,P,V}(m.exponents_complex, SimplePolynomials.absent, SimplePolynomials.absent)

function merge_constraints(objective::SimpleComplexPolynomial{<:Any,Nc,P}, zero, nonneg, psd, groupings::RelaxationGroupings,
    dense::Val, verbose::Bool) where {Nc,P<:Unsigned}
    @verbose_info("Incorporating constraints into set of exponents")
    # Note that in the complex-valued case there's no mixing - i.e., no real variables. And every monomial appears once in its
    # original form, once in its conjugate. We are only interested in the "original" (whichever it is), so we effectively treat
    # every monomial as real, discarding the conjugate part. (It would be possible to do it differently, but then in the end
    # when we reduce everything to the complex part dropping the conjugates, deleting duplicates would be necessary - in this
    # way, we don't even generate duplicates.)
    mons_idx_set = sizehint!(Set{FastKey{Int}}(), length(objective))
    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(monomial_index(_realify(mon))))
    end

    # 1. zero constraints
    isempty(zero) || @verbose_info("├ zero constraints")
    for (grouping, zeroₖ) in zip(groupings.zeros, zero)
        sizehint!(mons_idx_set, length(mons_idx_set) + 4length(zeroₖ) * length(grouping))
        for t in zeroₖ
            monₜ = monomial(t)
            for g in grouping
                push!(mons_idx_set,
                    FastKey(monomial_index(_realify(g), _realify(monₜ))),
                    FastKey(monomial_index(_realify(g), _realify(conj(monₜ)))),
                    FastKey(monomial_index(_realify(conj(g)), _realify(monₜ))),
                    FastKey(monomial_index(_realify(conj(g)), _realify(conj(monₜ))))
                )
            end
        end
    end

    # 2. nonneg constraints
    isempty(nonneg) || @verbose_info("├ nonnegative constraints")
    for (groupings, nonnegᵢ) in zip(groupings.nonnegs, nonneg)
        sizehint!(mons_idx_set, length(mons_idx_set) + length(nonnegᵢ) * sum(length, groupings))
        for t in nonnegᵢ
            monₜ = monomial(t)
            for grouping in groupings, g in grouping
                push!(mons_idx_set,
                    FastKey(monomial_index(_realify(g), _realify(monₜ)))
                    # the monomial will appear conjugated anyway
                )
            end
        end
    end

    # 3. psd constraints
    isempty(psd) || @verbose_info("├ PSD constraints")
    for (groupings, psdᵢ) in zip(groupings.psds, psd)
        dim = LinearAlgebra.checksquare(psdᵢ)
        sizehint!(mons_idx_set, length(mons_idx_set) + sum(length, groupings.psds, init=0) * sum(length, psdᵢ, init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for grouping in groupings, g in grouping
                    push!(mons_idx_set, FastKey(monomial_index(_realify(g), _realify(monₜ))))
                    i != j && push!(mons_idx_set, FastKey(monomial_index(_realify(g), _realify(conj(monₜ)))))
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    @verbose_info("├ Total number of coefficients: ", length(mons_idx_set),
        "\nConverting back from intermediate to exponent representation")
    return exponents_from_indices(P, Nc, mons_idx_set, dense)
end