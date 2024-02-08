include("./shared.jl")

@testset "Some real-valued examples (scalar)" begin
    DynamicPolynomials.@polyvar x[1:2]
    probs = [poly_problem(-(x[1] - 1)^2 - (x[1] - x[2])^2 - (x[2] - 3)^2, i,
                          nonneg=[1 - (x[1] - 1)^2, 1 - (x[1] - x[2])^2, 1 - (x[2] - 3)^2], perturbation=1e-3)
             for i = 1:2]
    if optimize
        for solver in solvers
            @test getproperty.(poly_optimize.(solver, probs), :objective) ≈ [-3, -2] atol = 2e-2
        end
    end
end

@testset "Some real-valued examples (matrix)" begin
    DynamicPolynomials.@polyvar x[1:2]
    probs = [poly_problem(-x[1]^2 - x[2]^2, i,
                          psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]], perturbation=1e-4) for i in 1:2]
    if optimize
        for solver in solvers
            @test getproperty.(poly_optimize.(solver, probs), :objective) ≈ [-4, -4] atol = 1e-3
        end
    end
end

@testset "Some real-valued examples (matrix and equality)" begin
    DynamicPolynomials.@polyvar x[1:2]
    probs = [poly_problem(-x[1]^2 - x[2]^2, i, zero=[x[1] + x[2] - 1],
                          psd=[[1-4x[1]*x[2] x[1]; x[1] 4-x[1]^2-x[2]^2]]) for i in 1:2]
    if optimize
        for solver in solvers
            @test getproperty.(poly_optimize.(solver, probs), :objective) ≈ [-4, -3.904891578336841] atol = 5e-3
        end
    end
end

@testset "Some complex-valued examples (equality)" begin
    DynamicPolynomials.@polycvar z
    prob = poly_problem(z + conj(z), 1, zero=[z * conj(z) - 1])
    if optimize
        for solver in solvers
            @test poly_optimize(solver, prob).objective ≈ -2 atol = 1e-8
        end
    end
end

@testset "Some complex-valued examples (inequalities)" begin
    DynamicPolynomials.@polycvar z[1:2]
    prob = poly_problem(3 - z[1] * conj(z[1]) - 0.5im * z[1] * conj(z[2])^2 + 0.5im * z[2]^2 * conj(z[1]), 3,
                        zero=[z[1] * conj(z[1]) - 0.25z[1]^2 - 0.25conj(z[1])^2 - 1,
                              z[1] * conj(z[1]) + z[2] * conj(z[2]) - 3,
                              1im * z[2] - 1im * conj(z[2])], nonneg=[z[2] + conj(z[2])])
    if optimize
        for solver in solvers
            @test poly_optimize(solver, prob).objective ≈ 0.42817465 atol = 1e-5
        end
    end
end

@testset "Some complex-valued examples (matrix)" begin
    DynamicPolynomials.@polycvar x[1:2]
    probs = [poly_problem(-x[1] * conj(x[1]) - x[2] * conj(x[2]), i,
                          psd=[[1-2*(x[1]*x[2]+conj(x[1] * x[2])) x[1]; conj(x[1]) 4-x[1]*conj(x[1])-x[2]*conj(x[2])]],
                          perturbation=1e-4) for i in 2:3]
    if optimize
        for solver in solvers
            @test getproperty.(poly_optimize.(solver, probs), :objective) ≈ [-4, -4] atol = 5e-3
        end
    end
end