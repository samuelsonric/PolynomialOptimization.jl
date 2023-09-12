using Test
using PolynomialOptimization
using MultivariatePolynomials
import DynamicPolynomials

all_solvers = [:MosekMoment, :MosekSOS, :COSMOMoment, :HypatiaMoment];
complex_solvers = [:MosekMoment, :HypatiaMoment];

function strRep(x)
    io = IOBuffer()
    show(IOContext(io, :limit => true, :displaysize => (10, 10)), "text/plain", x)
    return String(take!(io))
end;

@testset "Example 3.1 from correlative term sparsity paper" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityCorrelative(poly_problem(1 + x[1]^2 + x[2]^2 + x[3]^2 + x[1] * x[2] + x[2] * x[3] + x[3], 2))
    @test strRep(sp) == "SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2]
  x[2], x[3]
Block sizes:
  [6 => 2]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.625 atol = 1e-6
        end
    end
end

@testset "Example 3.4 from correlative term sparsity paper" begin
    DynamicPolynomials.@polyvar x[1:6]
    sp = SparsityCorrelative(
        poly_problem(1 + sum(x[i]^4 for i in 1:6) + x[1] * x[2] * x[3] + x[3] * x[4] * x[5] + x[3] * x[4] * x[6] +
                     x[3] * x[5] * x[6] + x[4] * x[5] * x[6], 2, perturbation=1e-4)
    )
    @test strRep(sp) == "SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
  x[3], x[4], x[5], x[6]
Block sizes:
  [15 => 1, 10 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5042 atol = 1e-3
        end
    end
end

@testset "Example 4.1 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:50]
    sp = SparsityCorrelative(
        poly_problem(sum((x[i-1] + x[i] + x[i+1])^4 for i in 2:49), 2)
    )
    @test strRep(sp) == "SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
  x[2], x[3], x[4]
  x[3], x[4], x[5]
  x[4], x[5], x[6]
  x[5], x[6], x[7]
  x[6], x[7], x[8]
  x[7], x[8], x[9]
  x[8], x[9], x[10]
  x[9], x[10], x[11]
  x[10], x[11], x[12]
  x[11], x[12], x[13]
  x[12], x[13], x[14]
  x[13], x[14], x[15]
  x[14], x[15], x[16]
  x[15], x[16], x[17]
  x[16], x[17], x[18]
  x[17], x[18], x[19]
  x[18], x[19], x[20]
  x[19], x[20], x[21]
  x[20], x[21], x[22]
  x[21], x[22], x[23]
  x[22], x[23], x[24]
  x[23], x[24], x[25]
  x[24], x[25], x[26]
  x[25], x[26], x[27]
  x[26], x[27], x[28]
  x[27], x[28], x[29]
  x[28], x[29], x[30]
  x[29], x[30], x[31]
  x[30], x[31], x[32]
  x[31], x[32], x[33]
  x[32], x[33], x[34]
  x[33], x[34], x[35]
  x[34], x[35], x[36]
  x[35], x[36], x[37]
  x[36], x[37], x[38]
  x[37], x[38], x[39]
  x[38], x[39], x[40]
  x[39], x[40], x[41]
  x[40], x[41], x[42]
  x[41], x[42], x[43]
  x[42], x[43], x[44]
  x[43], x[44], x[45]
  x[44], x[45], x[46]
  x[45], x[46], x[47]
  x[46], x[47], x[48]
  x[47], x[48], x[49]
  x[48], x[49], x[50]
Block sizes:
  [10 => 48]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0 atol = 2e-6
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0 atol = 2e-6
        @test sparse_optimize(:COSMOMoment, sp, eps_abs=1e-8, eps_rel=1e-8)[2] ≈ 0 atol = 1e-4
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 0 atol = 1e-5
    end
end

@testset "Example 4.4 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:4]
    prob = poly_problem(2 + x[1]^2 * x[4]^2 * (x[1]^2 * x[4]^2 - 1) - x[1]^2 + x[1]^4 +
                        sum((x[i]^2 * x[i-1]^2 * (x[i]^2 * x[i-1]^2 - 1) - x[i]^2 + x[i]^4 for i in 2:4)), 4,
        perturbation=1e-4)
    sp = SparsityCorrelative(prob)
    @test strRep(sp) == "SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[4]
  x[2], x[3], x[4]
Block sizes:
  [35 => 2]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.11008 atol = 1e-3
        end
    end

    sp = SparsityCorrelative(prob, chordal_completion=false)
    @test strRep(sp) == "SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2]
  x[1], x[4]
  x[2], x[3]
  x[3], x[4]
Block sizes:
  [15 => 4]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.11008 atol = 1e-3
        end
    end

    @test isnothing(sparse_iterate!(sp))
end