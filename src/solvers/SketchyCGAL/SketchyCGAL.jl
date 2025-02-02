# We implement not only the translation of the polynomial problem to the SketchCGAL solver, but also this solver directly.
# While there already is a Julia implementation, we need to change a few things.
# The algorithm comes from "Scalable Semidefinite Programming" by Yurtsever et. al, https://doi.org/10.1137/19M1305045
export Status, sketchy_cgal

"""
    Status{R}

This struct contains the current information for the Sketchy CGAL solver. Per-iteration callbacks will receive this structure
to gather current information.
Note that if `ϵ` is zero, the fields `infeasibility` and `duality_gap` are not available.

See also [`sketchy_cgal`](@ref).
"""
mutable struct Status{R}
    # static information
    max_iter::UInt
    time_limit::UInt
    ϵ::R
    # dynamic information
    iteration::UInt
    time::UInt
    objective::R
    infeasibility::R
    infeasibility_stop::R
    suboptimality::R
    suboptimality_stop::R

    Status(max_iter::Integer, time_limit::Integer, ϵ::R) where {R} = new{R}(UInt(max_iter), UInt(time_limit), ϵ)
end

struct MatrixFreeOperator{T}
    size::Tuple{Int,Int}
    primitive!
end

Base.@propagate_inbounds LinearAlgebra.mul!(C, AorB::MatrixFreeOperator, X, α, β) = AorB.primitive!(C, X, α, β)
Base.size(A::MatrixFreeOperator, i::Int) = A.size[i]
Base.eltype(::MatrixFreeOperator{T}) where {T} = T

@doc raw"""
    sketchy_cgal(; A, b, C, α=(0, 1), rank, ϵ=1e-4, max_iter=0, time_limit=0, verbose=false,
        β₀=1, K=∞, method=:auto, callback=(_) -> ())

Implementation of the [SketchyCGAL algorithm](https://doi.org/10.1137/19M1305045). This solves the following problem:
```math
    \min \bigl\{
        \operatorname{tr}(C X) : \operatorname{tr}(A_i X) = b_i \ \forall i,
                                 \alpha_1 \leq \operatorname{tr}(X) \leq \alpha_2,
                                 X \succeq 0
    \bigr\}
```
The function returns the optimization status, objective value, and the optimal matrix ``X`` (in the form of an SVD
factorization object).

## Parameters
- `rank` controls the rank that is used to approximate the end result.
- The solution accuracy can be controlled by the parameter `ϵ`; however, no more than `max_iter` iterations are carried out,
  and no more iterations will be performed if `time_limit` was exceeded (in seconds), regardless of `ϵ`. Set any of those three
  parameters to zero to disable the check.
- The `callback` may be called after each iteration and will receive a [`Status`](@ref) as parameter. If the
  callback returns `false`, the iteration will be the last one.
- The parameters `β₀` and `K` allow to tune the optimization. `β₀` is a smoothing, `K` limits the dual vector to a generalized
  sphere of radius `K` around the origin.
- The `method` determines the way in which the smallest eigenvalue and its eigenvector are calculated during each iteration.
  Possible values are
  - `:lanczos_space` (uses the space-efficient implementation described in the SketchyCGAL paper, memory is linear in the
    problem size)
  - `:lanczos_time` (uses the same principle, but can save about half of the operations by using more memory: quadratic in the
    problem size)
  - `:lobpcg_fast` (uses the LOBPCG solver from the [`IterativeSolvers`](https://iterativesolvers.julialinearalgebra.org/dev/)
    package, bounding the number of iterations with the same heuristic as for the Lanczos methods)
  - `:lobpcg_accurate` (bounds the error instead to `ϵ/100`)
  - `:auto` chooses `:lobpcg_accurate` for problem sizes smaller than 10, `:lanczos_time` for problem sizes less than 11500
    (where roughly 1 GiB is required for the speedup), and `lanczos_space` for all larger problems.

## Optimization status values
- `:optimal`: the desired accuracy `ϵ` was reached in both the relative suboptimality gap as well as the relative infeasibility
- `:max_iter`: the maximum number of iterations was reached
- `:timeout`: the maximum computation time was hit
- `:canceled`: the callback returned `false`
- `:unknown`: an internal error has happened

See also [`Status`](@ref).



    sketchy_cgal(primitive1!, primitive2!, primitive3!, n, b, α; rank, ϵ=1e-8, max_iter=10_000, verbose=false,
        rescale_C=1, rescale_A=[1, ...], rescale_X=1, β₀=1, K=∞, method=:auto, callback=(_) -> ())

This is the black-box version that allows for matrix-free operations. `n` is the side dimension of the semidefinite variable;
`b` is the right-hand side of the constraint vector (of length `d`); and the primitives effectively calculate
- `primitive1!(v, u, α, β) = (v = α * C * u + β * v)`, ``\mathbb R^n \to \mathbb R^n``
- `primitive2!(v, u, z, α, β) = (v = α * adjoint(A)(z) * u + β * v)`, ``\mathbb R^n \times \mathbb R^d \to \mathbb R^n``
- `primitive3!(v, u) = (v = A(u * u'))`, ``\mathbb R^n \to \mathbb R^d``
where `A(X)` is the linear map `[⟨A[1], X⟩, ..., ⟨A[d], X⟩]` and `adjoint(A)(z) = sum(z[i] * A[i])`. All of them must also
return their outputs.

If you are able to calculate these oracles faster or more memory-efficient than the straightforward implementation (which is
based on `mul!`), use the blackbox method.
*The (Frobenius) norm of all the `A[i]` and `C` must be one for the blackbox method.*
If you had to rescale those matrices in order to achieve this normalization condition, you may pass the corresponding rescaling
factors in `rescale_C` and `rescale_A`. Additionally, the upper bound in `α` should be one for a better performance, for which
`rescale_X` can be used. These are posterior factors that indicate the multiplication that has been done before calling the
function in order to enforce compliance. They will be taken into account when calculating the termination criteria (such that
`ϵ` then corresponds to the original problem and not the rescaled one), filling the status structure or verbose output, and
the final values of primal objective and optimal ``X``.
"""
function sketchy_cgal(primitive1!, primitive2!, primitive3!, n, b, α::Tuple{R,R}; rank::Integer, ϵ::R=1e-4,
    max_iter::Integer=10_000, time_limit::Integer=0, rescale_C::R=one(R), rescale_A::Vector{R}=fill(one(R), length(b)),
    rescale_X::R=one(R), β₀::R=one(R), K::R=R(Inf), verbose::Bool=false, callback::Union{Nothing,Function}=nothing,
    method::Symbol=:auto) where {R<:Real}
    @assert(ϵ ≥ 0 && max_iter ≥ 0 && time_limit ≥ 0 && rank ≥ 1 && all(α .≥ zero(R)) && any(α .> zero(R)))
    @assert(ϵ > 0 || max_iter > 0 || time_limit > 0)
    if method === :auto
        if n < 10
            method = :lobpcg_accurate
        elseif n < 11_500
            method = :lanczos_time
        else
            method = :lanczos_space
        end
    else
        @assert(method ∈ (:lanczos_space, :lanczos_time, :lobpcg_fast, :lobpcg_accurate))
    end
    starting_time = time_ns()
    info = Status(max_iter, time_limit, ϵ)
    if iszero(ϵ)
        info.infeasibility = R(NaN)
        info.suboptimality = R(NaN)
        detail_calc = false
    else
        detail_calc = verbose || !isnothing(callback)
    end
    if !iszero(time_limit)
        time_limit = starting_time + 1_000_000_000 * time_limit
    end
    @verbose_info("SketchyCGAL solver, implemented by PolynomialOptimization, using eigensolver method ", method, "\n",
        "Iteration | Primal objective | Suboptimality | Infeasibility | rel. subopt. | rel. infeas. | Time")

    d = length(b)
    α₁ = min(α...)
    α₂ = max(α...)
    sqrtα₁ = sqrt(α₁)
    sqrtα₂ = sqrt(α₂)
    status = :unknown
    # Scale problem data (or better: problem data is scaled, but adjust termination criteria appropriately)
    obj_rescale = 1 / (rescale_C * rescale_X)
    infeas_rescale = (1 / rescale_X) ./ rescale_A
    ϵ_infeas = ϵ * max(sqrt(sum(x -> prod(x)^2, zip(b, infeas_rescale))), 1)
    info.infeasibility_stop = ϵ_infeas
    # β₀ ← 1 and K ← ∞ (we expose these parameters)
    # NystromSketch.Init(n, R)
    sketch = NystromSketch{R}(n, rank)
    # z ← 0_d and y ← 0_d
    z = zeros(R, d)
    y = zeros(R, d)
    # we also define some temporaries to avoid allocations and speed up calculations in the loop
    zminusb = -b
    tmpd = Vector{R}(undef, d)
    tmpv = Vector{R}(undef, n)
    eig_tmp = setup_approx_min_evec(Val(method), n, ϵ, tmpd, primitive1!, primitive2!)
    p = zero(R)
    trace = zero(R)
    # for t ← 1, 2, 3, ..., T do
    t = 1
    @inbounds while true
        # β ← β₀ * √(t+1) and η ← 2/(t+1)
        β = β₀ * sqrt(t +1)
        η = 2 / (t +1)
        # [ξ, v] ← ApproxMinEvec(C + A*(y + β(z - b)); q) with q = t^(1/4) log n
        tmpd .= y .+ β .* zminusb
        ξ, v = approx_min_evec(t, eig_tmp)
        # depending on the sign of ξ, we need to pick either the upper or lower trace bound.
        # Note that here, we already multiply v by sqrtα. We need to do this anyway in the end for the sketch update, so we can
        # just do it here and avoid some α multiplications afterwards.
        lmul!(ξ > 0 ? sqrtα₁ : sqrtα₂, v)
        # check the stopping criterion (for the current point, where we need ξ)
        finish = false
        if !iszero(ϵ)
            # the infeasibility is just z - b
            info.infeasibility = sqrt(sum(x -> prod(x)^2, zip(zminusb, infeas_rescale), init=zero(R)))
            stop_feasible = info.infeasibility ≤ ϵ_infeas
            if stop_feasible || detail_calc
                # we can bound the suboptimality via
                # g - ⟨y, z - b⟩ - 1/2 β ‖ z - b ‖^2
                # where g = p + ⟨y + β (z - b), z⟩ - λₘᵢₙ(D)
                # and λₘᵢₙ(D) ≈ ξ
                info.suboptimality = (p + dot(tmpd, z) - ξ - dot(y, zminusb) - β * norm(zminusb)^2/2) * obj_rescale
                info.suboptimality_stop = ϵ * max(abs(p * obj_rescale), 1)
                finish = stop_feasible && info.suboptimality ≤ info.suboptimality_stop
            end
        end
        if !isnothing(callback) || verbose
            info.iteration = t
            info.objective = p * obj_rescale
            info.time = (time_ns() - starting_time) ÷ 1_000_000_000
            if verbose && (finish || isone(t) || iszero(t % 100))
                @verbose_info(@sprintf("%9d | %16g | %13g | %13g | %12g | %12g | %02d:%02d",
                    t, # iteration
                    info.objective, # primal objective
                    info.suboptimality, # actual upper bound to suboptimality
                    info.infeasibility, # actual infeasibility
                    ϵ * info.suboptimality / info.suboptimality_stop, # relative value of suboptimality, relevant for stopping
                    ϵ * info.infeasibility / info.infeasibility_stop, # relative value of infeasibility, relevant for stopping
                    info.time ÷ 60, info.time % 60))
            end
            if !isnothing(callback) && callback(info) === false
                status = :canceled
                break
            end
        end
        if finish
            status = :optimal
            break
        end
        if !iszero(max_iter) && t == max_iter
            status = :max_iter
            break
        end
        if !iszero(time_limit) && time_ns() ≥ time_limit
            status = :timeout
            break
        end

        # update pₜ₊₁ = (1 - η) pₜ + η * α * vₜ*(C vₜ) - strictly speaking, we don't need this for zero ϵ, but we want to keep
        # track of the objective anyway (and even for no output, we output it in the end)
        p = (1 - η) * p + η * dot(v, primitive1!(tmpv, v, true, false))
        # z = (1 - η) z + η A(α v v*)
        z .= (1 - η) .* z .+ η .* primitive3!(tmpd, v)
        zminusb .= z .- b
        trace = (1 - η) * trace + η * (ξ > 0 ? α₁ : α₂)
        # γ is the largest solution to γ ‖ zₜ₊₁ - b ‖² ≤ 4α²β₀ / (t +1)^(3/2) ‖ A ‖^2 and 0 ≤ γ ≤ β₀,
        # which can also be written as γ ‖ zₜ₊₁ - b ‖² ≤ β η²α² ‖ A ‖² and 0 ≤ γ ≤ β₀
        # If needed, set γ = 0 to prevent ‖ yₜ₊₁ ‖ > K (which "in practice is not necessary").
        γ = min(β * (η * α₂)^2 / norm(zminusb)^2, β₀)
        # Note that in MATLAB, this strangely uses β₊ = β₀ * sqrt(t +2) (so the β for the next iteration), but still η from
        # this iteration...
        # yₜ₊₁ = yₜ + γ (zₜ₊₁ - b)
        tmpd .= y .+ γ .* zminusb
        if norm(tmpd) ≤ K
            y .= tmpd
        end
        # NystromSketch.RankOneUpdate(√α v, η)
        rank_one_update!(sketch, v, η)

        t += 1
    end
    # [U, Λ] ← NystromSketch.Reconstruct()
    U, Λ = reconstruct(sketch)
    # Λ ← Λ + (α - tr(Λ)) I / R
    # Here, we do not use α as the trace reference, since α might actually define an interval. Instead, we kept track of what
    # the trace was supposed to be (assuming an ideal storage). We then compute the actual trace based on the sketch
    # reconstruction and perform the correction step based on this.
    Λ .= (Λ .+ ((trace .- Λ) ./ rank)) ./ rescale_X
    # X = U Λ U*. Instead of returning the full matrix that arises in this way, we give back an appropriate factorization.
    return status, p * obj_rescale, SVD(U, Λ, U')
end

function sketchy_cgal(; A::AbstractVector{<:AbstractMatrix{R}}, b::AbstractVector{R}, C::AbstractMatrix{R}, kwargs...) where {R<:Real}
    @assert(length(A) == length(b) && all(size.(A) .== (size(C),)) && length(A) ≥ 1)
    n = LinearAlgebra.checksquare(C)
    d = length(b)
    rescale_C = 1 / norm(C)
    rescale_A = 1 ./ norm.(A)
    kwargs = Dict(kwargs) # immutable
    alpha = pop!(kwargs, :α, (zero(R), one(R)))::Tuple{R,R}
    rescale_X = 1 / max(alpha...)
    status, obj, X = @inbounds sketchy_cgal(
        (v, u, α, β) -> mul!(v, C, u, α * rescale_C, β),
        (v, u, z, α, β) -> begin
            mul!(v, A[1], u, α * z[1] * rescale_A[1], β)
            for i in 2:d
                mul!(v, A[i], u, α * z[i] * rescale_A[i], one(R))
            end
            v
        end,
        (v, u) -> begin
            for i in 1:d
                v[i] = rescale_A[i] * dot(u, A[i], u)
            end
            v
        end,
        n, b .* rescale_A .* rescale_X, alpha .* rescale_X;
        kwargs...,
        rescale_C=rescale_C,
        rescale_A=rescale_A
    )
    return status, obj, X
end

struct LanczosTimeTmp{R}
    n
    logn
    vmat::Matrix{R}
    ρ::Vector{R}
    ω::Vector{R}
    v::Vector{R}
    tmpd::Vector{R}
    primitive1!
    primitive2!
end

function setup_approx_min_evec(::Val{:lanczos_time}, n, ϵ, tmpd::Vector{R}, primitive1!, primitive2!) where {R}
    return LanczosTimeTmp{R}(n, log(n), Matrix{R}(undef, n, n), Vector{R}(undef, n -1), Vector{R}(undef, n -1),
        Vector{R}(undef, n), tmpd, primitive1!, primitive2!)
end

function approx_min_evec(t, tmp::LanczosTimeTmp{R}) where {R}
    vmat, ρ, ω, v, tmpd, primitive1!, primitive2! = tmp.vmat, tmp.ρ, tmp.ω, tmp.v, tmp.tmpd, tmp.primitive1!, tmp.primitive2!
    @views @inbounds begin
        # v₁ ← randn(n, 1)
        # v₁ ← v₁ / ‖ v₁ ‖
        normalize!(Random.randn!(vmat[:, 1]))
        local i
        # for i ← 1, 2, 3, ..., min{q, n -1} do
        for outer i in 1:min(ceil(Int, t^R(1//4) * tmp.logn), tmp.n -1)
            vᵢ = vmat[:, i]
            vᵢ₊₁ = vmat[:, i+1]
            # ωᵢ ← Re(vᵢ* (M vᵢ)). So we first need M vᵢ, which is C + A*(y + β(z - b))
            primitive1!(vᵢ₊₁, vᵢ, true, false)
            primitive2!(vᵢ₊₁, vᵢ, tmpd, true, true)
            ω[i] = real(dot(vᵢ, vᵢ₊₁))
            # vᵢ₊₁ ← M vᵢ - ωᵢ vᵢ - ρᵢ₋₁ vᵢ₋₁ where ρ₀ v₀ = 0
            if i == 1
                vᵢ₊₁ .-= ω[i] .* vᵢ
            else
                vᵢ₊₁ .-= ω[i] .* vᵢ .+ ρ[i-1] .* vmat[:, i-1]
            end
            # ρᵢ ← ‖ vᵢ ‖ (should be vᵢ₊₁)
            ρ[i] = norm(vᵢ₊₁)
            # if ρᵢ = 0 then break
            ρ[i] < 1e-8 && break
            vᵢ₊₁ ./= ρ[i]
        end
        # T ← tridiag(ρ[1:i-1], ω[1:i], ρ[1:i-1])
        # [ξ, u] ← MinEvec(T)
        w, Z = LAPACK.stegr!('V', 'I', ω[1:i], ρ[1:i-1], 0.0, 0.0, 1, 1) # TODO: call LAPACK ourselves, so we avoid allocations
        ξ = w[1]
        # v ← ∑ⱼⁱ uⱼ vⱼ
        mul!(v, vmat[:, 1:i], Z[:, 1])
    end
    nrm = norm(v)
    return ξ * nrm, lmul!(1/nrm, v)
end

struct LanczosSpaceTmp{R}
    n
    logn
    v₁::Vector{R}
    v₂::Vector{R}
    v₃::Vector{R}
    ρ::Vector{R}
    ω::Vector{R}
    v::Vector{R}
    tmpd::Vector{R}
    primitive1!
    primitive2!
end

function setup_approx_min_evec(::Val{:lanczos_space}, n, ϵ, tmpd::Vector{R}, primitive1!, primitive2!) where {R}
    return LanczosSpaceTmp{R}(n, log(n), Vector{R}(undef, n), Vector{R}(undef, n), Vector{R}(undef, n), Vector{R}(undef, n -1),
        Vector{R}(undef, n -1), Vector{R}(undef, n), tmpd, primitive1!, primitive2!)
end

function approx_min_evec(t, tmp::LanczosSpaceTmp{R}) where {R}
    vᵢ₋₁, vᵢ, vᵢ₊₁, ρ, ω, v, tmpd, primitive1!, primitive2! = tmp.v₁, tmp.v₂, tmp.v₃, tmp.ρ, tmp.ω, tmp.v, tmp.tmpd,
        tmp.primitive1!, tmp.primitive2!
    @views @inbounds begin
        # v₁ ← randn(n, 1)
        # v₁ ← v₁ / ‖ v₁ ‖
        normalize!(Random.randn!(v))
        copyto!(vᵢ, v)
        local i
        # for i ← 1, 2, 3, ..., min{q, n -1} do
        for outer i in 1:min(ceil(Int, t^R(1//4) * tmp.logn), tmp.n -1)
            # ωᵢ ← Re(vᵢ* (M vᵢ)). So we first need M vᵢ, which is C + A*(y + β(z - b))
            primitive1!(vᵢ₊₁, vᵢ, true, false)
            primitive2!(vᵢ₊₁, vᵢ, tmpd, true, true)
            ω[i] = real(dot(vᵢ, vᵢ₊₁))
            # vᵢ₊₁ ← M vᵢ - ωᵢ vᵢ - ρᵢ₋₁ vᵢ₋₁ where ρ₀ v₀ = 0
            if i == 1
                vᵢ₊₁ .-= ω[i] .* vᵢ
            else
                vᵢ₊₁ .-= ω[i] .* vᵢ .+ ρ[i-1] .* vᵢ₋₁
            end
            # ρᵢ ← ‖ vᵢ ‖ (should be vᵢ₊₁)
            ρ[i] = norm(vᵢ₊₁)
            # if ρᵢ = 0 then break
            ρ[i] < 1e-8 && break
            vᵢ₊₁ ./= ρ[i]
            vᵢ₋₁, vᵢ, vᵢ₊₁ = vᵢ, vᵢ₊₁, vᵢ₋₁ # just reference cross-swapping
        end
        # T ← tridiag(ρ[1:i-1], ω[1:i], ρ[1:i-1])
        # [ξ, u] ← MinEvec(T)
        w, Z = LAPACK.stegr!('V', 'I', ω[1:i], ρ[1:i-1], 0.0, 0.0, 1, 1) # TODO: call LAPACK ourselves, so we avoid allocations
        ξ = w[1]
        # as we don't have the vectors available, we must redo all of the computation. We stored our initial random vector in
        # v, which we now flip back to vᵢ. Note that LAPACK overwrites both ω and ρ, so we also have to re-compute those.
        vᵢ, v = v, vᵢ
        local ωᵢ, ρᵢ
        for i in 1:i
            primitive1!(vᵢ₊₁, vᵢ, true, false)
            primitive2!(vᵢ₊₁, vᵢ, tmpd, true, true)
            ωᵢ = real(dot(vᵢ, vᵢ₊₁)) # no need to do some index calculations, just a simple variable
            if i == 1
                vᵢ₊₁ .-= ωᵢ .* vᵢ
                v .= Z[1, 1] * vᵢ
            else
                vᵢ₊₁ .-= ωᵢ .* vᵢ .+ ρᵢ .* vᵢ₋₁ # indeed, this is ρᵢ₋₁ now, but we don't introduce an alias
                v .+= Z[i, 1] * vᵢ
            end
            ρᵢ = norm(vᵢ₊₁)
            ρᵢ < 1e-8 && break
            vᵢ₊₁ ./= ρᵢ
            vᵢ₋₁, vᵢ, vᵢ₊₁ = vᵢ, vᵢ₊₁, vᵢ₋₁ # just reference cross-swapping
        end
    end
    nrm = norm(v)
    return ξ * nrm, lmul!(1/nrm, v)
end

struct LOBPCGFastTmp{R}
    n
    logn
    iterator
    v::Vector{R}
    tmpd::Vector{R}
end

function setup_approx_min_evec(::Val{:lobpcg_fast}, n, ϵ, tmpd::Vector{R}, primitive1!, primitive2!) where {R}
    v = Vector{R}(undef, n)
    return LOBPCGFastTmp{R}(
        n,
        log(n),
        IterativeSolvers.LOBPCGIterator(
            MatrixFreeOperator{R}((n, n), (out, in, alpha, beta) -> begin
                primitive1!(out, in, alpha, beta)
                primitive2!(out, in, tmpd, alpha, true)
            end), false, reshape(v, :, 1)
        ),
        v,
        tmpd
    )
end

function approx_min_evec(t, tmp::LOBPCGFastTmp{R}) where R
    normalize!(Random.randn!(tmp.v))
    ξ = IterativeSolvers.lobpcg!(tmp.iterator, tol=1e-8, maxiter=min(ceil(Int, t^R(1//4) * tmp.logn), tmp.n -1)).λ[1]
    return ξ, tmp.v
end

struct LOBPCGAccurateTmp{R}
    ϵ::R
    iterator
    v::Vector{R}
    tmpd::Vector{R}
end

function setup_approx_min_evec(::Val{:lobpcg_accurate}, n, ϵ, tmpd::Vector{R}, primitive1!, primitive2!) where {R}
    v = Vector{R}(undef, n)
    return LOBPCGAccurateTmp{R}(
        iszero(ϵ) ? IterativeSolvers.default_tolerance(R) : ϵ/100,
        IterativeSolvers.LOBPCGIterator(
            MatrixFreeOperator{R}((n, n), (out, in, alpha, beta) -> begin
                primitive1!(out, in, alpha, beta)
                primitive2!(out, in, tmpd, alpha, true)
            end), false, reshape(v, :, 1)
        ),
        v,
        tmpd
    )
end

function approx_min_evec(t, tmp::LOBPCGAccurateTmp)
    normalize!(Random.randn!(tmp.v))
    ξ = IterativeSolvers.lobpcg!(tmp.iterator, tol=tmp.ϵ).λ[1]
    return ξ, tmp.v
end

struct NystromSketch{T}
    Ω::Matrix{T}
    S::Matrix{T}
    tmp::Vector{T}

    NystromSketch{T}(n::Integer, R::Integer) where {T} = new(randn(T, n, R), zeros(T, n, R), Vector{T}(undef, R))
end

rank_one_update!(sketch::NystromSketch{T}, v::AbstractVector{T}, η::T) where {T} =
    # S ← (1 - η) S + η v (v* Ω)
    mul!(sketch.S, v, transpose(mul!(sketch.tmp, transpose(sketch.Ω), conj(v))), η, 1 - η)
    # TODO: benchmark whether scaling + rank-1 update is better than level 3 function.

function reconstruct(sketch::NystromSketch{T}) where {T}
    n = size(sketch.Ω, 1)
    # σ ← √n eps(norm(S))
    σ = sqrt(n) * eps(T) * maximum(norm, eachcol(sketch.S); init=zero(T))
    # Sσ ← S + σ Ω
    Sₛ = sketch.S + σ * sketch.Ω
    # L ← chol(Ω* Sσ)
    L = cholesky!(Hermitian(sketch.Ω' * Sₛ))
    # [U, Σ, ~] ≤ svd(Sσ / L)
    U, Σ, _ = svd!(Sₛ / L.U)
    # Λ ← max{0, Σ² - σ I}
    Λ = max.(0, Σ .^ 2 .- σ)

    return U, Λ
end