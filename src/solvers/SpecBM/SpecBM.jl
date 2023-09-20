# This is an implementation of the SpecPM primal solver, https://arxiv.org/abs/2307.07651v1 with a reference implementation on
# https://github.com/soc-ucsd/SpecBM.git, tightly integrated with the PolynomialOptimization framework
"""
    specbm_primal(A, b, c; free=missing, psd::Vector{<:Integer}, ϵ=1e-4, β=0.1, α=1., αfree=α, maxiter=500, ml=0.001,
        mu=min(1.5β, 1), αmin=1e-5, αmax=1000., verbose=true, offset=0, rescale=true, max_cols, ρ, evec_past, evec_current,
        At=transpose(A), AAt=A*At, adaptive=true, step=1)
"""
function specbm_primal(A::AbstractMatrix{R}, b::AbstractVector{R}, c::AbstractVector{R};
    free::Union{Missing,Integer}=missing, psd::AbstractVector{<:Integer},
    ρ::Real, r_past::Union{<:AbstractVector{<:Integer},<:Integer}, r_current::Union{<:AbstractVector{<:Integer},<:Integer},
    ϵ::Real=1e-4, β::Real=0.1, maxiter::Integer=500,
    α::Real=1., adaptive::Bool=true, αmin::Real=1e-5, αmax::Real=1000.,
    ml::Real=0.001, mr::Real=min(1.5β, 1), Nmin::Integer=10,
    verbose::Bool=true, step::Integer=20, offset::R=zero(R),
    At::Union{Missing,AbstractMatrix{R}}=missing, AAt::Union{Missing,AbstractMatrix{R}}=missing,
    subsolver::Symbol=:Mosek) where {R}
    #region Input validation
    subsolver === :Mosek || error("Unsupported subsolver ", subsolver)
    # Problem data A₁, ..., Aₘ, C ∈ 𝕊ⁿ, b ∈ ℝⁿ. Here, we also allow for free variables, as in the reference implementation.
    # We do not store the matrices A directly, but instead interpret all PSD variables by their vectorized upper triangle
    # (contrary to the reference implementation, which uses vectorized full storage). Therefore, A contains the stacked
    # vectorized matrices and C is also a vector. All free variables come before the PSD variables.
    num_conds, num_vars = size(A)
    (num_conds == length(b) && num_vars == length(c)) || error("Incompatible dimensions")
    all(j -> j > 0, psd) || error("PSD dimensions must be positive")
    if ismissing(free)
        free = num_vars - sum(packedsize, psd, init=0)
        free ≥ 0 || error("Incompatible dimensions")
    elseif free < 0
        error("Number of free variables must be nonnegative")
    elseif sum(packedsize, psd, init=0) + free != num_vars
        error("Incompatible dimensions")
    end
    num_psd = length(psd)
    if isa(r_current, Integer)
        r_current ≥ 0 || error("r_current must be positive")
        r_current = min.(r_current, psd)
    elseif length(r_current) != num_psd
        error("Number of r_current must be the same as number of psd constraints")
    else
        all(x -> x ≥ 1, r_current) || error("r_current must be positive")
        all(splat(≤), zip(r_current, psd)) || error("No r_current must not exceed its associated dimension")
    end
    if isa(r_past, Integer)
        r_past ≥ 0 || error("r_past must be nonnegative")
        r_past = min.(fill(r_past, num_psd), psd .- r_current) # which is guaranteed to be nonnegative
    elseif length(r_past) != num_psd
        error("Number of r_past must be the same as number of psd constraints")
    else
        all(x -> x ≥ 0, r_past) || error("r_past must be nonnegative")
        all((r_currentⱼ, r_pastⱼ, dimⱼ) -> r_currentⱼ + r_pastⱼ ≤ dimⱼ) ||
            error("r_past + r_current must not exceed the associated dimension")
    end
    # Parameters rₚ ≥ 0, r_c ≥ 1, α > 0, β ∈ (0, 1), ϵ ≥ 0, tₘₐₓ ≥ 1
    α > 0 || error("α must be positive")
    0 < β < 1 || error("β must be in (0, 1)")
    ϵ ≥ 0 || error("ϵ must be nonnegative")
    maxiter > 1 || error("maxiter must be larger than 1")
    # Adaptive parameters mᵣ > β, 0 < mₗ < β
    if adaptive
        mr > β || error("mr must be larger than β")
        0 < ml < β || error("ml must be in (0, β)")
        0 < Nmin || error("Nmin must be positive")
        α = inv(R(2))
    end
    if ismissing(At)
        At = copy(transpose(A))
        # A has off-diagonal elements scaled by a factor of 2 (required for scalar product between packed matrices), but we
        # don't need this for At (returns packed matrix)
        i = free +1
        for nⱼ in psd
            curcol = 2
            i += 1
            while curcol ≤ nⱼ
                lmul!(inv(R(2)), @view(At[i:i+curcol-2, :]))
                i += curcol
                curcol += 1
            end
        end
    end
    if ismissing(AAt)
        AAt = A * At
    end
    #endregion

    @verbose_info("SpecBM Primal Solver with parameters ρ = $ρ, r_past = $r_past, r_current = $r_current, ϵ = $ϵ, β = $β, $α ",
        adaptive ? "∈ [$αmin, $αmax], ml = $ml, mr = $mr" : "= $α")
    @verbose_info("Iteration | Primal objective | Primal infeas | Dual infeas | Duality gap | Rel. accuracy | Rel. primal inf. | Rel. dual inf. |    Rel. gap | Descent step | Consecutive null steps")

    invnormbplus1 = inv(norm(b) + one(R))
    invnormcplus1 = inv(norm(c) + one(R))

    #region Allocations
    # An initial point Ω₀ ∈ 𝕊ⁿ.  As in the reference implementation, we take zero for the free variables and the vectorized
    # identity for the PSD variables.
    Ω = zeros(R, num_vars)
    Ω_psds = Vector{PackedMatrix{R,typeof(@view(Ω[begin:end]))}}(undef, num_psd)
    # 1: Initialization. Let r = rₚ + r_c
    r = r_past .+ r_current
    rdims = packedsize.(r)
    Σr = sum(rdims, init=0)
    # Initialize W̄₀ ∈ 𝕊₊ⁿ with tr(W̄₀) = 1. As in the reference implementation, we take the (1,1) elementary matrix. Note that
    # the reference implementation only allows for a single block; we map this to multiple semidefinite constraints not merely
    # by mimicking a block-diagonal matrix, but taking the constraints into account individually!
    Ws = Vector{PackedMatrix{R,Vector{R}}}(undef, num_psd)
    # Compute P₀ ∈ ℝⁿˣʳ with columns being the top r orthonormal eigenvectors of -Ω₀. As Ω₀ is the identity, we can do this
    # explicitly.
    Ps = Vector{Matrix{R}}(undef, num_psd)
    Pkrons = Vector{Matrix{R}}(undef, num_psd)
    # We also need some temporaries to avoid allocations.
    A_psds = Vector{typeof(@view(A[:, begin:end]))}(undef, num_psd)
    At_psds = Vector{typeof(@view(At[begin:end, :]))}(undef, num_psd)
    c_free = @view(c[1:free])
    C_psds = Vector{PackedMatrix{R,Vector{R}}}(undef, num_psd)
    bigbuf = Vector{R}(undef, max(num_conds * max(num_psd, Σr), maximum(r)^2, maximum(psd)^2))
    γstars = Vector{R}(undef, num_psd)
    Wstars = similar(Ws)
    Sstar = @view(bigbuf[1:Σr])
    Sstars = Vector{PackedMatrix{R,typeof(@view(Sstar[begin:end]))}}(undef, num_psd)
    Xstar = similar(Ω)
    Xstar_free = @view(Xstar[1:free])
    Xstar_psds = similar(Ω_psds)
    ystar = Vector{R}(undef, num_conds)
    condtmp = similar(ystar)
    i = free +1
    @inbounds for (j, (nⱼ, rⱼ, rdimⱼ)) in enumerate(zip(psd, r, rdims))
        dimⱼ = packedsize(nⱼ)
        Ω_psds[j] = Ωⱼ = PackedMatrix(nⱼ, @view(Ω[i:i+dimⱼ-1]))
        for k in 1:nⱼ
            Ωⱼ[k, k] = one(R)
        end
        Ws[j] = Wⱼ = PackedMatrix(nⱼ, zeros(R, dimⱼ))
        Wⱼ[1, 1] = one(R)
        Ps[j] = Pⱼ = zeros(R, nⱼ, rⱼ)
        for k in 1:rⱼ
            Pⱼ[k, k] = one(R)
        end
        Pkrons[j] = Matrix{R}(undef, rdimⱼ, dimⱼ)
        A_psds[j] = @view(A[:, i:i+dimⱼ-1])
        At_psds[j] = @view(At[i:i+dimⱼ-1, :])
        C_psds[j] = Cⱼ = PackedMatrix{R}(undef, nⱼ) # @view(c[i:i+dimⱼ-1])
        # Cⱼ will be a copy of c[i:i+dimⱼ-1], but with off-diagonals as they are (whereas in c, they have to be doubled)
        let i=i, k=1, nextdiag=0, Cⱼ=vec(Cⱼ)
            while k < length(Cⱼ)
                Cⱼ[k:k+nextdiag-1] .= inv(R(2)) .* @view(c[i:i+nextdiag-1])
                Cⱼ[k+nextdiag] = c[i+nextdiag]
                nextdiag += 1
                i += nextdiag
                k += nextdiag
            end
        end
        Wstars[j] = PackedMatrix{R}(undef, nⱼ)
        # Sstars is initialized later, as we need a different index
        Xstar_psds[j] = PackedMatrix(nⱼ, @view(Xstar[i:i+dimⱼ-1]))
        i += dimⱼ
    end
    # To solve the main problem, several precomputations can be done, and a couple of preallocations will be useful
    cache = let m₁ = Vector{R}(undef, num_psd),
        m₂ = Vector{R}(undef, Σr),
        M = Symmetric(Matrix{R}(undef, num_psd + Σr, num_psd + Σr), :L),
        M₁₁ = @view(parent(M)[1:num_psd, 1:num_psd]),
        M₂₂ = @view(parent(M)[num_psd+1:end, num_psd+1:end]),
        M₂₁ = @view(parent(M)[num_psd+1:end, 1:num_psd]),
        # q₁ = m₁
        q₂s = Vector{typeof(@view(m₂[begin:end]))}(undef, num_psd),
        q₃ = Vector{R}(undef, num_conds),
        Q₁₁ = @view(M₁₁[begin:num_psd+1:end]), # this is a view to the diagonal of M₁₁
        Q₃₃inv = try EfficientCholmod(ldlt(AAt)) catch; qr(AAt) end,
        Q₂₁s = Vector{typeof(@view(M₂₁[begin:end, begin]))}(undef, num_psd), # this is really a block-diagonal matrix
        Q₃₁ = Matrix{R}(undef, num_conds, num_psd),
        Q₃₂ = Matrix{R}(undef, num_conds, Σr),
        Q₃₂s = Vector{typeof(@view(Q₃₂[:, begin:end]))}(undef, num_psd),
        minus2Ac = R(-2) * (A * c),
        tmpm1 = reshape(@view(bigbuf[1:num_conds*num_psd]), (num_conds, num_psd)),
        tmpm2 = reshape(@view(bigbuf[1:num_conds*Σr]), (num_conds, Σr)),
        i = 1
        @inbounds for (j, (rⱼ, dimⱼ)) in enumerate(zip(r, rdims))
            q₂s[j] = @view(m₂[i:i+dimⱼ-1])
            Q₂₁s[j] = @view(M₂₁[i:i+dimⱼ-1, j])
            Q₃₂s[j] = @view(Q₃₂[:, i:i+dimⱼ-1])
            Sstars[j] = PackedMatrix(rⱼ, @view(Sstar[i:i+dimⱼ-1]))
            i += dimⱼ
        end
        (m₁, m₂, M, M₁₁, M₂₂, M₂₁, q₂s, q₃, Q₁₁, Q₃₃inv, Q₂₁s, Q₃₁, Q₃₂, Q₃₂s, minus2Ac, Pkrons, A, A_psds, At, At_psds, b, c,
            c_free, C_psds, tmpm1, tmpm2, psd, r, Σr, ϵ)
    end
    subsolver_data = specbm_setup_primal_subsolver(Val(subsolver), num_psd, r, rdims, Σr, ρ)
    #endregion

    # We need some additional variables for the adaptive strategy, following the naming in the reference implementation
    # (in the paper, the number of consecutive null steps N_c is used instead).
    null_count = 0
    has_descended = true

    # 2: for t = 0, ..., tₘₐₓ do [we fix this to 1:maxiter]
    local FΩ, relative_pfeasi, quality
    for t in 1:maxiter
        # 3: solve (24) to obtain Xₜ₊₁*, γₜ*, Sₜ*
        # combined with
        # 4: form the iterate Wₜ* in (28) and dual iterate yₜ* in (29)
        dfeasi, dfeasi_psd, dfeasi_free, gap = direction_qp_primal_free!(γstars, ystar, Wstars, Sstar, Sstars, Xstar,
            Xstar_free, Xstar_psds, Ω, Ω_psds, Ws, Ps, !isone(t), α, cache, subsolver_data)
        # We also calculate some quality criteria here
        dual_feasi = max(dfeasi_free, dfeasi_psd)
        relative_dfeasi = sqrt(dfeasi * invnormcplus1)
        if has_descended
            copyto!(condtmp, b) # we don't need y any more, so we can use it as a temporary
            mul!(condtmp, A, Ω, true, -one(R))
            relative_pfeasi = norm(condtmp) * invnormbplus1
            # else we no not need to recompute this, the value from the last iteration is still valid
        end
        # 5: if t = 0 and A(Ωₜ) ≠ b then
        if isone(t) && relative_pfeasi > ϵ # note: reference implementation does not check A(Ωₜ) ≠ b
            copyto!(Ω, Xstar)
        # 7: else
        else
            # 8: if (25) holds then
            # (25): β( F(Ωₜ) - ̂F_{Wₜ, Pₜ}(Xₜ₊₁*)) ≤ F(Ωₜ) - F(Xₜ₊₁*)
            # where (20): F(X) := ⟨C, X⟩ - ρ min(λₘᵢₙ(X), 0)
            if has_descended
                FΩ = dot(c, Ω) - ρ * sum(min(eigmin(Ωⱼ), zero(R)) for Ωⱼ in Ω_psds; init=zero(R))
                # else we do not need to recalculate this, it did not change from the previous iteration
            end
            cXstar = dot(c, Xstar)
            Fmodel = cXstar - sum(dot(Wstarⱼ, Xstarⱼ) for (Wstarⱼ, Xstarⱼ) in zip(Wstars, Xstar_psds); init=zero(R))
            FXstar = cXstar - ρ * sum(min(eigmin(Xstarⱼ), zero(R)) for Xstarⱼ in Xstar_psds; init=zero(R))
            estimated_drop = FΩ - Fmodel
            cost_drop = FΩ - FXstar
            if (has_descended = (β * estimated_drop ≤ cost_drop))
                # 9: set primal iterate Ωₜ₊₁ = Xₜ₊₁*
                copyto!(Ω, Xstar)
                # 6.1.1. Adaptive strategy (can only be lower case due to mₗ < β < mᵣ)
                if adaptive
                    if mr * estimated_drop ≤ cost_drop
                        α = max(α / 2, αmin)
                    end
                    null_count = 0
                end
            # 10: else
            else
                # 11: set primal iterate Ωₜ₊₁ = Ωₜ (no-op)
                # 6.1.1. Adaptive strategy (can only be upper case)
                if adaptive
                    null_count += 1
                    if null_count ≥ Nmin && ml * estimated_drop ≥ cost_drop
                        α = min(2α, αmax)
                        null_count = 0
                    end
                end
            # 12: end if
            end
            relative_accuracy = estimated_drop / (abs(FΩ) + one(R))
        # 13: end if
        end
        relative_gap = gap / (one(R) + abs(dot(c, Ω)) + abs(dot(b, ystar))) # now Ω is corrected
        # 14: compute Pₜ₊₁ as (26), and ̄Wₜ₊₁ as (27)
        # (26): Pₜ₊₁ = orth([Vₜ; Pₜ Q₁])
        # where Vₜ: top r_c ≥ 1 eigenvectors of -Xₜ₊₁*
        # and S* = [Q₁ Q₂] * Diagonal(Σ₁, Σ₂) * [Q₁; Q₂] with division in (rₚ, r - rₚ)
        # (27): ̄Wₜ₊₁ = 1/(γ* + tr(Σ₂)) * (γ* ̄Wₜ + Pₜ Q₂ Σ₂ Q₂ᵀ Pₜᵀ)
        primal_feasi = zero(R)
        @inbounds for (rⱼ, r_currentⱼ, r_pastⱼ, γstarⱼ, Wstarⱼ, Sstarⱼ, Xstarⱼ, Wⱼ, Pⱼ, tmpⱼ) in zip(r, r_current, r_past,
                                                                            γstars, Wstars, Sstars, Xstar_psds, Ws, Ps, Pkrons)
            # note: we adjusted r such that it cannot exceed the side dimension of Xstar_psd, but we cannot do the same with
            # r_current and r_past, as only their sum has an upper bound.
            nⱼ = size(Xstarⱼ, 1)
            @assert(size(Sstarⱼ) == (rⱼ, rⱼ))
            V = r_currentⱼ < nⱼ ? eigen!(Xstarⱼ, 1:r_currentⱼ) : eigen!(Xstarⱼ)
            primal_feasi = min(primal_feasi, first(V.values))
            r_pastⱼ = min(r_pastⱼ, rⱼ)
            if iszero(r_pastⱼ)
                copyto!(Wⱼ, Wstarⱼ)
                rmul!(Wⱼ, inv(tr(Wⱼ)))
                copyto!(Pⱼ, V.vectors)
            else
                γstarⱼ = max(γstarⱼ, zero(R)) # prevent numerical issues
                Sstareig = eigen!(Sstarⱼ)
                Q₁ = @view(Sstareig.vectors[:, end-r_pastⱼ+1:end]) # sorted in ascending order; we need the largest rₚ, but
                                                                   # the order doesn't really matter
                Q₂ = @view(Sstareig.vectors[:, 1:end-r_pastⱼ])
                Σ₂ = @view(Sstareig.values[1:end-r_pastⱼ])
                tmpmⱼ_large = reshape(@view(tmpⱼ[1:nⱼ*rⱼ]), (nⱼ, rⱼ))
                # Wⱼ = (γstar * Wⱼ + Pⱼ * Q₂ * Diagonal(Σ₂) * Q₂' * Pⱼ') / (γstar + tr(Σ₂))
                den = γstarⱼ + sum(v -> max(v, zero(R)), Σ₂) # also prevent numerical issues here
                if den > 1e-8
                    tmpmⱼ_small = reshape(@view(bigbuf[1:rⱼ^2]), (rⱼ, rⱼ))
                    tmpmⱼ_small2 = reshape(@view(tmpⱼ[1:length(Q₂)]), size(Q₂))
                    mul!(tmpmⱼ_small2, Q₂, Diagonal(Σ₂))
                    mul!(tmpmⱼ_small, tmpmⱼ_small2, transpose(Q₂))
                    mul!(tmpmⱼ_large, Pⱼ, tmpmⱼ_small)
                    tmpmⱼ_verylarge = reshape(@view(bigbuf[1:nⱼ^2]), (nⱼ, nⱼ))
                    gemmt!('U', 'N', 'T', true, tmpmⱼ_large, Pⱼ, false, tmpmⱼ_verylarge)
                    trttp!('U', tmpmⱼ_verylarge, vec(Wstarⱼ)) # Wstarⱼ is just another temporary now
                    den = inv(den)
                    axpby!(den, vec(Wstarⱼ), γstarⱼ * den, vec(Wⱼ))
                end # else no update of W
                # Pⱼ = orth([V.vectors Pⱼ*Q₁])
                # for orthogonalization, we use QR to be numerically stable; unfortunately, this doesn't produce Q directly, so
                # we need another temporary. For consistency with the reference implementation, we put Pⱼ*Q₁ first.
                mul!(@view(tmpmⱼ_large[:, 1:r_pastⱼ]), Pⱼ, Q₁)
                copyto!(@view(tmpmⱼ_large[:, r_pastⱼ+1:end]), V.vectors)
                copyto!(Pⱼ, qr!(tmpmⱼ_large).Q)
            end
        end
        # 15: if stopping criterion then
        #     16: quit
        isone(t) && continue
        # Iteration | Primal objective | Primal infeas | Dual infeas | Duality gap | Rel. accuracy | Rel. primal inf. | Rel. dual inf. | Rel. gap | Descent step | Consecutive null steps
        iszero(t % step) && @verbose_info(@sprintf("%9d | %16g | %13g | %11g | %11g | %13g | %16g | %14g | %11g | %12s | %22d",
            t, FΩ + offset, primal_feasi, dual_feasi, gap, relative_accuracy, relative_pfeasi, relative_dfeasi, relative_gap,
            has_descended, null_count))
        quality = max(relative_accuracy, relative_pfeasi, relative_dfeasi, relative_gap, -primal_feasi)
        quality < ϵ && break
        # 17: end if
    # 18: end for
    end

    specbm_finalize_primal_subsolver(subsolver_data)

    return FΩ + offset, Ω, ystar, quality
end

function specbm_setup_primal_subsolver end
function specbm_finalize_primal_subsolver end
function specbm_primal_subsolve end

if isdefined(Mosek, :appendafes)
    if VersionNumber(Mosek.getversion()) ≥ v"10.1.11"
        include("SpecBMMosek.jl")
    else
        @warn "The SpecBM method Mosek is not available: upgrade your Mosek distribution to at least version 10.1.11."
    end
end

@inline function direction_qp_primal_free!(γstars::AbstractVector{R}, ystar, Wstars, Sstar, Sstars, Xstar, Xstar_free,
    Xstar_psds, Ω, Ω_psds, Ws, Ps, feasible, α, cache, subsolver) where {R}
    m₁, m₂, M, M₁₁, M₂₂, M₂₁, q₂s, q₃, Q₁₁, Q₃₃inv, Q₂₁s, Q₃₁, Q₃₂, Q₃₂s, minus2Ac, Pkrons, A, A_psds, At, At_psds, b, c,
        c_free, C_psds, tmpm1, tmpm2, psd, r, Σr, ϵ = cache
    invα = inv(α)
    # We need to (34): minimize dot(v, M, v) + dot(m, v) + c
    #                      s.t. v = [γ; vec(S)]
    #                           γ ≥ 0, S ∈ 𝕊₊ʳ, γ + tr(S) ≤ ρ
    # Note that as we have multiple PSD blocks which we all treat separately (and not just as a single block-diagonal
    # constraint, we actually get multiple γ and multiple S matrices), though there is just one ρ.
    # Creating the data from the given parameters is detailed in C.1
    # We create a matrix Pkron (symmetrized Kronecked product) such that vec(Pᵀ W P) = Pkron*w, if w is the packed vector of W
    # Pkronᵢ is packedsize(rᵢ) × packedsize(nᵢ)
    @inbounds @fastmath for (Pⱼ, Pkronⱼ) in zip(Ps, Pkrons)
        cols, rows = size(Pⱼ) # Pᵀ is to the left
        colidx = 1
        for l in 1:cols
            for k in 1:l-1
                rowidx = 1
                for p in 1:rows
                    @simd for q in 1:p
                        Pkronⱼ[rowidx, colidx] = Pⱼ[k, q] * Pⱼ[l, p] + Pⱼ[l, q] * Pⱼ[k, p]
                        rowidx += 1
                    end
                end
                colidx += 1
            end
            rowidx = 1
            for p in 1:rows
                @simd for q in 1:p
                    Pkronⱼ[rowidx, colidx] = Pⱼ[l, q] * Pⱼ[l, p]
                    rowidx += 1
                end
            end
            colidx += 1
        end
    end
    # m₁ = q₁ - Q₁₃ Q₃₃⁻¹ q₃
    # q₁ = 2⟨W̄ⱼ, α Ωⱼ - Cⱼ⟩
    # Q₃₁ = hcat(Aⱼ(W̄ⱼ))
    # q₃ = -2α(b - A(Ω)) - 2A(C)
    # We can use Xstar_psd as temporaries for 2(α Ωⱼ - Cⱼ)
    twoαΩminusC = Xstar_psds
    for (twoαΩminusCⱼ, Ωⱼ, Cⱼ) in zip(twoαΩminusC, Ω_psds, C_psds)
        twoαΩminusCⱼ .= R(2) .* (α .* Ωⱼ .- Cⱼ)
    end
    m₁ .= dot.(Ws, twoαΩminusC) # q₁ ≡ m₁
    mul!.(eachcol(Q₃₁), A_psds, Ws)
    if feasible
        copyto!(q₃, minus2Ac)
    else
        copyto!(q₃, b)
        mul!(q₃, A, Ω, 2α, R(-2) * α)
        q₃ .+= minus2Ac
    end
    copyto!(ystar, q₃) # we'll construct ystar successively, let's save q₃ for the moment
    ldiv!(Q₃₃inv, q₃) # now q₃ ← Q₃₃⁻¹ q₃
    mul!(m₁, transpose(Q₃₁), q₃, -one(R), true)

    # m₂ = q₂ - Q₂₃ Q₃₃⁻¹ q₃
    # q₂ = (2vec(Pⱼᵀ (α Ωⱼ - Cⱼ) Pⱼ))
    mul!.(q₂s, Pkrons, twoαΩminusC) # note that q₂s aliases m₂, so we already set the first part!
    # Q₃₂ = [hcat(vec(Pⱼᵀ Aᵢ Pⱼ)ᵀ for j in 1:num_psd) for i in 1:num_conds]
    mul!.(Q₃₂s, transpose.(At_psds), transpose.(Pkrons)) # transpose(At_psds) ≠ A_psds: the former contains the unscaled matrix
    # correct the scaling
    @inbounds for (q₂ⱼ, Q₃₂ⱼ, rⱼ) in zip(q₂s, Q₃₂s, r)
        let i=2, nextdiag=1
            for col in 2:rⱼ
                @view(q₂ⱼ[i:i+nextdiag-1]) .*= R(2)
                @view(Q₃₂ⱼ[:, i:i+nextdiag-1]) .*= R(2)
                nextdiag += 1
                i += nextdiag
            end
        end
    end
    # multiply each Q₃₂s by diag([1, 2, 1, 2, 2, 1, 2, 2, 2, ...])
    mul!(m₂, transpose(Q₃₂), q₃, -one(R), true) # q₃ already contains the inverse part

    # M₁₁ = Q₁₁ - Q₁₃ Q₃₃⁻¹ Q₁₃ᵀ
    # Q₁₁ = Diag(⟨W̄ⱼ, W̄ⱼ⟩)
    ldiv!(tmpm1, Q₃₃inv, Q₃₁)
    mul!(M₁₁, transpose(Q₃₁), tmpm1, -one(R), false) # note: tmpm1, tmpm2, and Sstar share memory
    Q₁₁ .+= norm.(Ws) .^ 2 # Q₁₁ is a diagonal view into M₁₁

    # M₁₂ = Q₁₂ - Q₁₃ Q₃₃⁻¹ Q₂₃ᵀ ⇔ M₂₁ = Q₂₁ - Q₂₃ Q₃₃⁻¹ Q₃₁
    # Q₂₁ = Diag(vec(Pⱼᵀ W̄ⱼ Pⱼ)) - but this is a block diagonal for which there is no native support, so we use Vector{Vector}
    fill!(M₂₁, zero(R))
    mul!.(Q₂₁s, Pkrons, Ws) # note that Q₂₁ aliases M₂₁, so we already set the first part!
    @inbounds for (Q₂₁ⱼ, rⱼ) in zip(Q₂₁s, r)
        let i=2, nextdiag=1
            for col in 2:rⱼ
                @view(Q₂₁ⱼ[i:i+nextdiag-1]) .*= R(2)
                nextdiag += 1
                i += nextdiag
            end
        end
    end
    mul!(M₂₁, transpose(Q₃₂), tmpm1, -one(R), true) # tmpm already contains the inverse part

    # M₂₂ = Q₂₂ - Q₂₃ Q₃₃⁻¹ Q₂₃ᵀ
    # Q₂₂ = id_{Σr}
    ldiv!(tmpm2, Q₃₃inv, Q₃₂)
    mul!(M₂₂, transpose(Q₃₂), tmpm2, -one(R), false)
    # This is adding the identity - but vectorized off-diagonals actually need the factor 2
    let i = 1, Δ = size(M₂₂, 1) +1 # next item on diagonal - this is a view, so we don't use stride
        @inbounds for rⱼ in r
            δ = -Δ
            for col in 1:rⱼ
                M₂₂[i:Δ:i+δ] .= @view(M₂₂[i:Δ:i+δ]) .+ R(2)
                δ += Δ
                M₂₂[i+δ] += one(R)
                i += δ + Δ
            end
        end
    end

    # Now we have the matrix M and can in principle directly invoke Mosek using putqobj. However, this employs a sparse
    # Cholesky factorization for large matrices. In our case, the matrix M is dense and not very large, so we are better of
    # calculating the dense factorization by ourselves and then using the conic formulation. This also makes it easier to use
    # other solvers which have a similar syntax.
    Mfact = cholesky!(M, RowMaximum(), tol=ϵ^2, check=false)
    specbm_primal_subsolve(subsolver, Mfact, m₁, q₂s, Σr, γstars, Sstars, m₂) # we no longer need m₂, so it's scratch space now

    # Reconstruct y = Q₃₃⁻¹(-q₃/2 - Q₁₃ᵀ γ - Q₂₃ᵀ vec(S))
    # Note that at this stage, y = q₃
    mul!(ystar, Q₃₁, γstars, -one(R), inv(R(-2)))
    mul!(ystar, Q₃₂, Sstar, -one(R), true)
    ldiv!(Q₃₃inv, ystar)
    # Reconstruct Wⱼ = γⱼ W̄ⱼ + Pⱼ Sⱼ Pⱼᵀ and Xⱼ = Ωⱼ + (W - C + A*(y))/α
    Xstar_free .= .-c_free
    for (Wstarⱼ, γstarⱼ, Wⱼ, Pkronⱼ, Sstarⱼ, Xstarⱼ, Cⱼ, nⱼ, rⱼ) in zip(Wstars, γstars, Ws, Pkrons, Sstars, Xstar_psds,
                                                                        C_psds, psd, r)
        copyto!(Wstarⱼ, Wⱼ)
        # transpose(Pkronⱼ)*vec(Sstarⱼ) ≠ vec(Pⱼ Sⱼ Pⱼᵀ): the off-diagonal scaling is very different. It seems to be most
        # reasonable to first overwrite Pkronⱼ with the proper rescaling (we don't need it any more) and then just do the
        # multiplication.
        Pkronᵀ = transpose(Pkronⱼ)
        i = 2
        for col in 2:rⱼ
            @inbounds rmul!(@view(Pkronᵀ[:, i:i+col-2]), R(2))
            i += col
        end
        i = 2
        for row in 2:nⱼ
            @inbounds rmul!(@view(Pkronᵀ[i:i+row-2, :]), inv(R(2)))
            i += row
        end
        mul!(vec(Wstarⱼ), Pkronᵀ, Sstarⱼ, true, γstarⱼ)
        Xstarⱼ .= Wstarⱼ .- Cⱼ
    end
    mul!(Xstar, At, ystar, invα, invα)
    # before we complete the Ω reconstruction, calculate some feasibility quantifiers
    dfeasible_psd = (α * sum(norm, Xstar_psds, init=zero(R)))^2
    dfeasible_free = (α * norm(Xstar_free))^2
    dfeasible = dfeasible_free + dfeasible_psd
    Xstar .+= Ω

    gap = abs(dot(b, ystar) - dot(c, Xstar))
    return dfeasible, dfeasible_free, dfeasible_psd, gap
end


#=function direction_qp_primal_free(ω_free, ω_psd, Wt, Pt, feasible, A_free, A_sdp, c_sdp, α, A, c,)
    # Wt is a fixed atom
    # Pt is the transformation matrix
    # feasible means b - A ω = 0.
    # Consider free variables as well
    # The two changes are Q₃₃ and q₃
    # But Q₃₃ is precomputed
    kronPtPt = kron(Pt, Pt)
    Q11 = dot(Wt, Wt)
    Q12 = Wt * kronPtPt
    Q31 = A_sdp * Wt
    # Original strategy
    Q32 = A_sdp * kronPtPt
    Q13 = Q31'
    Q23 = Q32'
    temp = 2(-c_sdp .+ α .* ω_psd)
    q1 = dot(Wt, temp)
    q2T = temp' * kronPtPt
    q2 = q2T'

    if feasible
        q3 = -2(A * c)
    else
        q3 = -2(α * (b .- A_sdp * ω_psd .- A_free * ω_free) .+ A * c)
    end

    M11 = Q11 - dot(Q13, AAtinv \ Q31)
    M22 = Ir2 - Q23 * (AAtinv \ Q32)
    M12 = Q12 - Q13 * (AAtinv \ Q32)
    m1 = q1 - dot(Q13, AAtinv \ q3)
    m2 = q2 - dot(Q23, AAtinv \ q3)
    M = Hermitian([M11 M12; M12' M22])
    if iszero(evec_past) && isone(evec_current) && feasible
        # closed-form solution
        m = [m1; m2]
        rmul!(m, inv(eltype(m)(2)))
        v = -qr(M) \ m
        if v[1] < 0 || v[2] < 0 || v[1] + v[2] > ρ
            denominator = (2M11 + 2M22 - 4M12)
            c1 = let guess=iszero(denominator) ? denominator : (2M22*ρ - 2M12*ρ - M1 + M2) / denominator,
                tmp=max(zero(guess), min(ρ, guess))
                (tmp, ρ - tmp)
            end
            c2 = let guess=-m2 / (2M22), tmp=max(zero(guess), min(ρ, guess))
                (zero(tmp), tmp)
            end
            c3 = let guess=-m1 / (2M11), tmp=max(zero(guess), min(ρ, guess))
                (tmp, zero(tmp))
            end
            f1 = dot(c1, M, c1) + dot(m, c1) # TODO: tuples, won't work
            f2 = dot(c2, M, c2) + dot(m, c2)
            f3 = dot(c3, M, c3) + dot(m, c3)
            c = f1 ≤ f2 ? (f1 ≤ f3 ? c1 : c3) : (f2 ≤ f3 ? c2 : c3)
            Gammastar, Sstar = c
        else
            Gammastar, Sstar = v
        end
    else
        eigs = eigen!(M)
        eigs.values .= max.(eigs.values, zero(eltype(eigs.values)))
        M05 = Hermitian(eigen.vectors * Diagonal(sqrt.(eigs.values)) * eigen.vectors')
end=#