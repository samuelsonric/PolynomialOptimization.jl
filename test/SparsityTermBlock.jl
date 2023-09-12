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

@testset "Example 4.3" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityTermBlock(poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1] * x[2] * x[3] + x[2], 2))
    @test strRep(sp) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [6 => 1, 2 => 2]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.4752747 atol = 1e-6
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [6 => 1, 4 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.4752747 atol = 1e-6
        end
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Example 4.5 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityTermBlock(poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1]^2 * x[2]^2 + x[1]^2 * x[3]^2 +
                                        x[2]^2 * x[3]^2 + x[2] * x[3], 2))
    @test strRep(sp) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [5 => 1, 2 => 1, 1 => 3]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.91666667 atol = 1e-6
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [5 => 1, 2 => 2, 1 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.91666667 atol = 1e-6
        end
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Example 7.2" begin
    DynamicPolynomials.@polyvar x[1:3] y[1:3]
    sp = SparsityTermBlock(poly_problem(27 - ((x[1] - x[2])^2 + (y[1] - y[2])^2) * ((x[1] - x[3])^2 + (y[1] - y[3])^2) *
                                        ((x[2] - x[3])^2 + (y[2] - y[3])^2), 3,
                                        zero=[x[1]^2 + y[1]^2 + x[2]^2 + y[2]^2 + x[3]^2 + y[3]^2 - 3]))
    @test strRep(sp) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], y[1], y[2], y[3]
Block sizes:
  [31 => 2, 7 => 1, 1 => 15]
  [13 => 1, 9 => 1, 1 => 6]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0 atol = 1e-9
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0 atol = 1e-7
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 0 atol = 1e-6
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 0 atol = 2e-6
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], y[1], y[2], y[3]
Block sizes:
  [31 => 2, 13 => 1, 9 => 1]
  [13 => 1, 9 => 1, 3 => 2]"

    @test isnothing(sparse_iterate!(sp))

    sp = SparsityTermBlock(poly_problem(27 - ((x[1] - x[2])^2 + (y[1] - y[2])^2) * ((x[1] - x[3])^2 + (y[1] - y[3])^2) *
                                        ((x[2] - x[3])^2 + (y[2] - y[3])^2), 3,
                                        zero=[x[1]^2 + y[1]^2 + x[2]^2 + y[2]^2 + x[3]^2 + y[3]^2 - 3],
                                        equality_method=emCalculateGröbner))
    @test strRep(sp) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], y[1], y[2], y[3]
Block sizes:
  [28 => 2, 12 => 1, 9 => 1]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0 atol = 1e-8
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0 atol = 1e-8
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 0 atol = 1e-4
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 0 atol = 1e-6
    end

    @test isnothing(sparse_iterate!(sp))

    sp = SparsityTermBlock(poly_problem(27 - ((x[1] - x[2])^2 + (y[1] - y[2])^2) * ((x[1] - x[3])^2 + (y[1] - y[3])^2) *
                                        ((x[2] - x[3])^2 + (y[2] - y[3])^2), 4,
                                        zero=[x[1]^2 + y[1]^2 + x[2]^2 + y[2]^2 + x[3]^2 + y[3]^2 - 3],
                                        equality_method=emInequalities))
    @test strRep(sp) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], y[1], y[2], y[3]
Block sizes:
  [79 => 1, 69 => 1, 31 => 2]
  [31 => 2, 13 => 1, 9 => 1]"
    @test isnothing(sparse_iterate!(sp))
end

@testset "Something with matrices" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityTermBlock(poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1] * x[2] * x[3] + x[2], 2,
                                        psd=[[x[1] x[2]; x[2] x[1]]]))
    @test strRep(sp) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [8 => 1, 2 => 1]
  [4 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5355788 atol = 1e-6
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [10 => 1]
  [4 => 1]"
    @test isnothing(sparse_iterate!(sp))
end

@testset "Other example" begin
    n = 7
    DynamicPolynomials.@polyvar x[1:n]
    sp = SparsityTermBlock(poly_problem(
        sum((x[i] * (2 + 5x[i]^2) + 1 -
             sum((j != i) && (max(1, i - 5) ≤ j ≤ min(n, i + 1)) ? (1 + x[j]) * x[j] : 0
                 for j = 1:n))^2 for i = 1:n),
        3
    ))
    @test strRep(sp) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [85 => 1, 1 => 35]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0 atol = 2e-10
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0 atol = 1e-8
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 0 atol = 1e-4
        @test sparse_optimize(:HypatiaMoment, sp, dense=true)[2] ≈ 0 atol = 1e-7
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [120 => 1]"
end

@testset "Complex-valued" begin
    DynamicPolynomials.@polycvar z[1:2]
    sp = SparsityTermBlock(poly_problem(z[1] + conj(z[1]), 2, nonneg=[1 - z[1] * conj(z[1]) - z[2] * conj(z[2])]))
    @test strRep(sp) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  z[1], z[2]
Block sizes:
  [2 => 1, 1 => 4]
  [2 => 1, 1 => 1]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ -2 atol = 1e-8
        @test sparse_optimize(:HypatiaMoment, sp, dense=true)[2] ≈ -2 atol = 1e-8
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityTermBlock with 1 constraint(s)
Variable cliques:
  z[1], z[2]
Block sizes:
  [3 => 1, 2 => 1, 1 => 1]
  [2 => 1, 1 => 1]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ -2 atol = 1e-7
        @test sparse_optimize(:HypatiaMoment, sp, dense=true)[2] ≈ -2 atol = 1e-7
    end

    @test isnothing(sparse_iterate!(sp))
end