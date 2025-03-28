using Test
using PolynomialOptimization, GALAHAD
using PolynomialOptimization.Solvers.LANCELOT: LANCELOT_simple
using MultivariatePolynomials
import DynamicPolynomials

@testset "LANCELOT documentation" begin
    obj(x) = 100*(x[2] - x[1]^2)^2 + (1 - x[1])^2
    result = LANCELOT_simple(2, [-1.2, 1.0], obj, print_level=0)
    @test ≈(result[1], 0, atol=1e-10) && result[2] == 60 && result[3] == 0

    function MY_GRAD(g, x)
        g[1] = -400*(x[2]-x[1]^2)*x[1]-2*(1-x[1])
        g[2] = 200*(x[2]-x[1]^2)
    end
    result = LANCELOT_simple(2, [-1.2, 1.0], obj; print_level=0, MY_GRAD)
    @test ≈(result[1], 0, atol=1e-14) && result[2] == 60 && result[3] == 0

    function MY_HESS(h, x)
        h[1] = -400*(x[2]-3x[1]^2) + 2
        h[2] = -400*x[1]
        h[3] = 200
    end
    result = LANCELOT_simple(2, [-1.2, 1.0], obj; print_level=0, MY_GRAD, MY_HESS)
    @test ≈(result[1], 0, atol=1e-14) && result[2] == 23 && result[3] == 0

    function obj(x, i)
        i == 1 && return x[1] + 3x[2] - 3
        i == 2 && return x[1]^2 + x[2]^2 - 4
        return NaN
    end
    function MY_GRAD(g, x, i)
        i == 1 && (g[1] = 1; g[2] = 3; return)
        i == 2 && (g[1] = 2x[1]; g[2] = 2x[2]; return)
        fill!(g, NaN)
        return
    end
    function MY_HESS(h, x, i)
        i == 1 && (fill!(h, 0.); return)
        i == 2 && (h[1] = 2; h[2] = 0; h[3] = 2; return)
        fill!(h, NaN)
        return
    end
    result = LANCELOT_simple(2, [-1.2, 1.0], obj; print_level=0, MY_GRAD, MY_HESS, BL=[0., -1.], neq=1, nin=1)
    @test ≈(result[1], 0.023313439474602547, atol=1e-6) && result[2] == 8 && result[3] == 0
end

const startval = [0.7210476969933126, 0.6380699455835934, 0.8532153291949746, 0.3952826714480566, 0.3064291214270869,
                  0.9778462338230722]

@testset "Some real-valued examples (scalar)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-(x[1] - 1)^2 - (x[1] - x[2])^2 - (x[2] - 3)^2,
                        nonneg=[1 - (x[1] - 1)^2, 1 - (x[1] - x[2])^2, 1 - (x[2] - 3)^2])
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:2])
    @test optval ≈ -2 atol = 2e-5
    @test optpt ≈ [1., 2.] atol = 2e-5
end

@testset "Some real-valued examples (matrix)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-x[1]^2 - x[2]^2, psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]])
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:2])
    @test optval ≈ -4 atol = 2e-5
    @test optpt ≈ [0., 2.] atol = 2e-5
end

@testset "Some real-valued examples (matrix and equality)" begin
    DynamicPolynomials.@polyvar x[1:2]
    prob = poly_problem(-x[1]^2 - x[2]^2, zero=[x[1] + x[2] - 1], psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]])
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:2])
    # global would be -3.90489157833684, [-0.804778061268820., 1.804778061268820]
    @test optval ≈ -3.50442134489527 atol = 2e-5
    @test optpt ≈ [1.725646827649273, -0.7256475529616774] atol = 2e-5
end

@testset "Example 3.1 from correlative term sparsity paper" begin
    DynamicPolynomials.@polyvar x[1:3]
    prob = poly_problem(1 + x[1]^2 + x[2]^2 + x[3]^2 + x[1] * x[2] + x[2] * x[3] + x[3])
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:3])
    @test optval ≈ 0.625 atol = 2e-5
    @test optpt ≈ [-0.25, 0.5, -0.75] atol = 2e-5
end

@testset "Example 3.4 from correlative term sparsity paper" begin
    DynamicPolynomials.@polyvar x[1:6]
    prob = poly_problem(1 + sum(x[i]^4 for i in 1:6) + x[1] * x[2] * x[3] + x[3] * x[4] * x[5] + x[3] * x[4] * x[6] +
                      x[3] * x[5] * x[6] + x[4] * x[5] * x[6])
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:6])
    @test optval ≈ 0.9412824077069458 atol = 2e-5
    @test optpt ≈ [0.33346423231291855, -0.3334642984669165, 0.4447934153234278, 0.3948134188295012, -0.5023550515807035,
                   0.3948134185292575] atol = 2e-5
end

@testset "Example 4.1 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:50]
    obj = sum((x[i-1] + x[i] + x[i+1])^4 for i in 2:49)
    prob = poly_problem(obj)
    optval, optpt = poly_optimize(:LANCELOT, prob)(Float64.(1:50))
    @test optval ≈ 0 atol = 1e-7
    @test obj(optpt) ≈ 0 atol = 1e-7
end

@testset "Example 4.4 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:4]
    prob = poly_problem(2 + x[1]^2 * x[4]^2 * (x[1]^2 * x[4]^2 - 1) - x[1]^2 + x[1]^4 +
                        sum((x[i]^2 * x[i-1]^2 * (x[i]^2 * x[i-1]^2 - 1) - x[i]^2 + x[i]^4 for i in 2:4)))
    optval, optpt = poly_optimize(:LANCELOT, prob)(startval[1:4])
    @test optval ≈ 0.110118425157690 atol = 2e-5
    @test optpt ≈ [0.793700525984100, 0.793700525984100, 0.793700525984100, 0.793700525984100] atol = 2e-5
end