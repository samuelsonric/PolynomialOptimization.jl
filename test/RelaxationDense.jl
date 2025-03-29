include("./shared.jl")

@testset "Some real-valued examples (scalar)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-(x[1] - 1)^2 - (x[1] - x[2])^2 - (x[2] - 3)^2,
                        nonneg=[1 - (x[1] - 1)^2, 1 - (x[1] - x[2])^2, 1 - (x[2] - 3)^2])
    bad_quality = (:SCSMoment, :SpecBMSOS)
    for solver in (:ClarabelMoment, :COPTMoment, :HypatiaMoment, :LoRADSMoment, :LoraineMoment, :MosekMoment, :MosekSOS,
                   :ProxSDPMoment, :SCSMoment, :SpecBMSOS)
        skipsolver(solver) && continue
        @testset let solver=solver
            res = poly_optimize(solver, prob, 1, precision=1e-7)
            @test res.method === solver
            @test res.objective ≈ -3 atol=5e-6
            @test issuccess(res) skip=solver===:SpecBMSOS
            @test optimality_certificate(res) === :Unknown
        end
        @testset let solver=solver
            res = poly_optimize(solver, prob, 2, precision=solver === :LoraineMoment ? 4e-7 : 1e-7)
            @test res.method === solver
            @test res.objective ≈ -2 atol=(solver ∈ bad_quality ? 1e-2 : 1e-5)
            solver ∈ bad_quality && continue
            @test issuccess(res)
            @test optimality_certificate(res) === :Optimal
            Random.seed!(123) # mvhankel is randomized; if we are unlucky, it won't be successful, so fix a seed that works
            sol = poly_all_solutions(res)
            sol₁ = sol₂ = sol₃ = false
            for (solᵢ, badness) in sol
                if isapprox(solᵢ[1], 1, atol=1e-5) && isapprox(solᵢ[2], 2, atol=1e-5)
                    @test !sol₁
                    @test isapprox(badness, 0, atol=2e-5)
                    sol₁ = true
                    continue
                elseif isapprox(solᵢ[1], 2, atol=1e-5)
                    if isapprox(solᵢ[2], 3, atol=1e-5)
                        @test !sol₂
                        @test isapprox(badness, 0, atol=2e-5)
                        sol₂ = true
                        continue
                    elseif isapprox(solᵢ[2], 2, atol=1e-5)
                        @test !sol₃
                        @test isapprox(badness, 0, atol=2e-5)
                        sol₃ = true
                        continue
                    end
                end
                @test false
            end
            @test sol₁ && sol₂ & sol₃
        end
    end
    @test poly_optimize(:SketchyCGAL, prob, 1, α=(8., 9.), rank=2) broken=true # size 1 not supported
    @test poly_optimize(:SketchyCGAL, prob, 2, α=(47., 48.), rank=3).objective ≈ -2 atol=1e-2 broken=true # super bad
end

@testset "Some real-valued examples (matrix)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-x[1]^2 - x[2]^2, psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]], perturbation=1e-5)
    for solver in (:ClarabelMoment, :COPTMoment, :HypatiaMoment, :LoRADSMoment, :LoraineMoment, :MosekMoment, :MosekSOS,
                   :ProxSDPMoment, :SCSMoment, :SpecBMSOS)
        skipsolver(solver) && continue
        @testset let solver=solver
            @test poly_optimize(solver, prob, 1, precision=1e-7).objective ≈ -4. atol=1e-3

            res = poly_optimize(solver, prob, 2, precision=1e-7)
            @test res.objective ≈ -4. atol = 1e-3
            @test issuccess(res) broken=solver===:SpecBMSOS
            sols = poly_all_solutions(:heuristic, res)
            for sol in sols
                @test abs.(sol[1]) ≈ [0., 2.] atol=(solver === :SpecBMSOS ? 1e-1 : 1e-2)
                @test sol[2] ≈ 0 atol=2e-2
            end
        end
    end
    # SketchyCGAL at level 1 doesn't work at all and at level 2 is the epitome of slow tail convergence (never checked whether
    # it actually converges)
end

# Now we have verified all solvers. In the sequel, we'll only use Clarabel and, if exotic cones are used, Hypatia.

@testset "Some real-valued examples (matrix and equality)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-x[1]^2 - x[2]^2, zero=[x[1] + x[2] - 1], psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]])
    @test poly_optimize(:Clarabel, prob, 1).objective ≈ -4. atol=1e-7

    res = poly_optimize(:Clarabel, prob, 2)
    @test issuccess(res)
    @test res.objective ≈ -3.904891578336841 atol=1e-7
    Random.seed!(123)
    for extraction in (:mvhankel, :heuristic)
        pas = poly_all_solutions(extraction, res)
        @test isone(length(pas))
        @test pas[1][1] ≈ [-0.8047780612688199, 1.804778061268820] atol=1e-6
    end
end

@testset "Some complex-valued examples (equality)" begin
    DynamicPolynomials.@complex_polyvar z
    prob = poly_problem(z + conj(z), zero=[z * conj(z) - 1])
    @test poly_optimize(:Clarabel, prob, 1).objective ≈ -2. atol=1e-7
    @test poly_optimize(:Hypatia, prob, 1).objective ≈ -2. atol=1e-7
end

@testset "Some complex-valued examples (inequalities)" begin
    DynamicPolynomials.@complex_polyvar z[1:2]
    prob = poly_problem(3 - z[1] * conj(z[1]) - 0.5im * z[1] * conj(z[2])^2 + 0.5im * z[2]^2 * conj(z[1]),
                        zero=[z[1] * conj(z[1]) - 0.25z[1]^2 - 0.25conj(z[1])^2 - 1,
                              z[1] * conj(z[1]) + z[2] * conj(z[2]) - 3,
                              1im * z[2] - 1im * conj(z[2])], nonneg=[z[2] + conj(z[2])])
    @test poly_optimize(:Clarabel, prob, 3).objective ≈ 0.42817465 atol=1e-6
    res = poly_optimize(:Hypatia, prob, 3, preprocess=true)
    @test issuccess(res)
    @test res.objective ≈ 0.4281746445 atol=1e-5
    Random.seed!(123)
    for extraction in (:mvhankel, :heuristic)
        pas = poly_all_solutions(extraction, res)
        @test isone(length(pas))
        @test pas[1][1] ≈ [-sqrt(2/3)*im, sqrt(7/3)] atol=1e-5
    end
end

@testset "Some complex-valued examples (matrix)" begin
    DynamicPolynomials.@complex_polyvar x[1:2]
    prob = poly_problem(-x[1] * conj(x[1]) - x[2] * conj(x[2]),
                        psd=[[1-2*(x[1]*x[2]+conj(x[1] * x[2])) x[1]; conj(x[1]) 4-x[1]*conj(x[1])-x[2]*conj(x[2])]])
    @test poly_optimize(:Clarabel, prob, 2).objective ≈ -4. atol = 1e-7
    @test poly_optimize(:Clarabel, prob, 3).objective ≈ -4. atol = 1e-7
end