# We only accept ExponentsAll; else, we cannot guarantee that all the resulting exponents will in fact be possible. In this
# way, we don't have to check for equality of the exponents sets (or form the largest cover), which is also beneficial.
# And converting polynomials from the MP interface will yield ExponentsAll anyway.
function merge_constraints(objective::SimplePolynomial{<:Any,Nr,0}, zero::AbstractVector{<:SimplePolynomial{<:Any,Nr,0}},
    nonneg::AbstractVector{<:SimplePolynomial{<:Any,Nr,0}},
    psd::AbstractVector{<:AbstractMatrix{<:SimplePolynomial{<:Any,Nr,0}}}, groupings::RelaxationGroupings{Nr,0},
    verbose::Bool, need_copy::Bool) where {Nr}
    @verbose_info("Incorporating constraints into set of exponents")
    e = ExponentsAll{Nr,UInt}()

    if isempty(zero) && isempty(nonneg) && isempty(psd)
        # short path: just directly add the stuff from the objective. While we can get duplicates (due to the elimination of
        # conjugates), we have a clear and manageable upper bound to the size, so no need to determine duplicates on-the-fly.
        @verbose_info("No constraints found, adding all objective monomials")
        mons = monomials(objective)
        if monomials(objective).e == e
            @verbose_info("Aliasing monomial indices")
            return need_copy ? copy(mons) : mons
        else
            @verbose_info("Converting monomial indices")
            return SimpleMonomialVector{Nr,0}(e, monomial_index.((e,), mons))
        end
    end

    mons_idx_set = sizehint!(Set{FastKey{UInt}}(), length(objective))
    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(monomial_index(e, mon)))
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
                    for (i, g₁) in enumerate(grouping)
                        for g₂ in @view(grouping[i:end])
                            push!(mons_idx_set,
                                FastKey(monomial_index(e, g₁, monₜ, g₂)),
                                FastKey(monomial_index(e, g₂, monₜ, g₁))
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
        dim = size(psdᵢ, 1)
        sizehint!(mons_idx_set, length(mons_idx_set) + sum(g -> length(g) * (length(g) +1) ÷ 2, groupings.psds) *
                                sum(@capture(length($psdᵢ[i, j]) for j in 1:dim for i in 1:j), init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for grouping in groupings
                    for (i, g₁) in enumerate(grouping)
                        for g₂ in @view(grouping₂[i:end])
                            push!(mons_idx_set,
                                FastKey(monomial_index(e, g₁, monₜ, g₂)),
                                FastKey(monomial_index(e, g₂, monₜ, g₁))
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
    return exponents_from_indices(P, Nr, mons_idx_set)
end

function merge_constraints(objective::SimplePolynomial{<:Any,0,Nc}, zero::AbstractVector{<:SimplePolynomial{<:Any,0,Nc}},
    nonneg::AbstractVector{<:SimplePolynomial{<:Any,0,Nc}},
    psd::AbstractVector{<:AbstractMatrix{<:SimplePolynomial{<:Any,0,Nc}}}, groupings::RelaxationGroupings{0,Nc},
    verbose::Bool) where {Nc}
    @verbose_info("Incorporating constraints into set of exponents")
    # Note that in the complex-valued case there's no mixing - i.e., no real variables. And every monomial appears once in its
    # original form, once in its conjugate. We are only interested in the "original" (whichever it is), so we effectively treat
    # every monomial as real, discarding the conjugate part. (It would be possible to do it differently, but then in the end
    # when we reduce everything to the complex part dropping the conjugates, deleting duplicates would be necessary - in this
    # way, we don't even generate duplicates.)
    e = ExponentsAll{2Nc,UInt}()

    if isempty(zero) && isempty(nonneg) && isempty(psd)
        # short path: just directly add the stuff from the objective. While we can get duplicates (due to the elimination of
        # conjugates), we have a clear and manageable upper bound to the size, so no need to determine duplicates on-the-fly.
        @verbose_info("No constraints found, adding all objective monomials")
        mons = monomials(objective)
        candidates = FastVec{UInt}(buffer=length(mons))
        for mon in mons
            @inbounds unsafe_push!(candidates, exponents_to_index(e, KillConjugates(exponents(mon))))
        end
        @verbose_info("Sorting and removing duplicates")
        return SimpleMonomialVector{0,Nc}(e, Base._groupedunique(sort!(finish!(candidates))))
    end

    mons_idx_set = sizehint!(Set{FastKey{UInt}}(), length(objective))
    @verbose_info("├ objective")
    for mon in monomials(objective)
        push!(mons_idx_set, FastKey(exponents_to_index(ea, KillConjugates(exponents(mon)))))
    end

    # 1. zero constraints
    isempty(zero) || @verbose_info("├ zero constraints")
    for (grouping, zeroₖ) in zip(groupings.zeros, zero)
        sizehint!(mons_idx_set, length(mons_idx_set) + 4length(zeroₖ) * length(grouping))
        for t in zeroₖ
            monₜ = monomial(t)
            monₜe = KillConjugates(exponents(monₜ))
            monₜec = KillConjugates(exponents(SimpleConjMonomial(monₜ)))
            for g in grouping
                ge = KillConjugates(exponents(g))
                gec = KillConjugates(exponents(SimpleConjMonomial(g)))
                push!(mons_idx_set,
                    FastKey(exponents_sum(ea, ge, monₜe)),
                    FastKey(exponents_sum(ea, ge, monₜec)),
                    FastKey(exponents_sum(ea, gec, monₜe)),
                    FastKey(exponents_sum(ea, gec, monₜec))
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
                    FastKey(exponents_sum(ea, KillConjugates(exponents(g)), KillConjugates(exponents(monₜ))))
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
                    push!(mons_idx_set, FastKey(exponents_sum(ea, KillConjugates(exponents(g)), KillConjugates(exponents(monₜ)))))
                    i != j &&
                        push!(mons_idx_set, FastKey(exponents_sum(ea, KillConjugates(exponents(g)),
                                                                      KillConjugates(exponents(SimpleConjMonomial(monₜ))))))
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    @verbose_info("├ Total number of coefficients: ", length(mons_idx_set),
        "\nConverting back from intermediate to exponent representation")
    return SimpleMonomialVector{0,Nc}(e, sort!(convert.(UInt, mons_idx_set)))
end