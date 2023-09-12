using Test
using PolynomialOptimization
using MultivariatePolynomials
import DynamicPolynomials

all_solvers = [:MosekMoment, :MosekSOS, :COSMOMoment, :HypatiaMoment];
complex_solvers = [:MosekMoment, :HypatiaMoment];

function strRep(x)
    io = IOBuffer()
    show(io, "text/plain", x)
    return String(take!(io))
end;

@testset "Example 3.1" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityCorrelativeTerm(poly_problem(1 + x[1]^2 + x[2]^2 + x[3]^2 + x[1] * x[2] + x[2] * x[3] + x[3], 1))
    @test strRep(sp) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2]
Block sizes:
  [2 => 1, 1 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[2], x[3]
Block sizes:
  [3 => 1]"
    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0.625 atol = 1e-7
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0.625 atol = 1e-7
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 0.625 atol = 1e-5
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 0.625 atol = 1e-7
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2]
Block sizes:
  [3 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[2], x[3]
Block sizes:
  [3 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.625 atol = 1e-6
        end
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Example 3.4" begin
    DynamicPolynomials.@polyvar x[1:6]
    sp = SparsityCorrelativeTerm(
        poly_problem(1 + sum(x[i]^4 for i in 1:6) + x[1] * x[2] * x[3] + x[3] * x[4] * x[5] + x[3] * x[4] * x[6] +
                     x[3] * x[5] * x[6] + x[4] * x[5] * x[6], 2))
    @test strRep(sp) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [4 => 1, 2 => 3]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6]
Block sizes:
  [10 => 1, 5 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5042 atol = 1e-3
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [4 => 1, 2 => 3]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6]
Block sizes:
  [15 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5042 atol = 1e-3
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [6 => 1, 2 => 2]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6]
Block sizes:
  [15 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5042 atol = 1e-3
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [6 => 1, 4 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6]
Block sizes:
  [15 => 1]"
    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.5042 atol = 1e-3
        end
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Broyden Banded Function" begin
    n = 20
    DynamicPolynomials.@polyvar x[1:n]
    prob = poly_problem(sum((x[i] * (2 + 5x[i]^2) + 1 -
                             sum(j == i ? 0 : (1 + x[j]) * x[j] for j = max(1, i - 5):min(n, i + 1)))^2 for i = 1:n), 3)
    sps = [SparsityCorrelativeTerm(prob, clique_chordal_completion=false, term_mode=tm_cliques),
        SparsityCorrelativeTerm(prob, clique_chordal_completion=false, term_mode=tm_chordal_cliques),
        SparsityCorrelativeTerm(prob, term_mode=tm_block)]
    @test strRep(sps[1]) == "SparsityCorrelativeTerm with 14 cliques(s)
=========
Clique #1
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4], x[5], x[6], x[7], x[8]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #3
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6], x[7], x[8], x[9]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #4
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[4], x[5], x[6], x[7], x[8], x[9], x[10]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #5
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[5], x[6], x[7], x[8], x[9], x[10], x[11]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #6
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[6], x[7], x[8], x[9], x[10], x[11], x[12]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #7
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[7], x[8], x[9], x[10], x[11], x[12], x[13]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #8
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[8], x[9], x[10], x[11], x[12], x[13], x[14]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #9
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[9], x[10], x[11], x[12], x[13], x[14], x[15]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #10
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[10], x[11], x[12], x[13], x[14], x[15], x[16]
Block sizes:
  [10 => 14, 9 => 20, 8 => 15, 7 => 10, 6 => 2, 5 => 24, 4 => 64, 3 => 9, 1 => 35]
=========
Clique #11
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[11], x[12], x[13], x[14], x[15], x[16], x[17]
Block sizes:
  [10 => 16, 9 => 20, 8 => 16, 7 => 10, 6 => 5, 5 => 26, 4 => 54, 3 => 9, 1 => 35]
=========
Clique #12
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[12], x[13], x[14], x[15], x[16], x[17], x[18]
Block sizes:
  [10 => 8, 9 => 23, 8 => 15, 7 => 15, 6 => 6, 5 => 26, 4 => 47, 3 => 11, 1 => 35]
=========
Clique #13
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[13], x[14], x[15], x[16], x[17], x[18], x[19]
Block sizes:
  [10 => 7, 9 => 16, 8 => 13, 7 => 17, 6 => 12, 5 => 28, 4 => 45, 3 => 16, 1 => 35]
=========
Clique #14
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [11 => 4, 10 => 5, 9 => 21, 8 => 11, 7 => 11, 6 => 11, 5 => 37, 4 => 38, 3 => 11, 1 => 35]"
    @test strRep(sps[2]) == "SparsityCorrelativeTerm with 14 cliques(s)
=========
Clique #1
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4], x[5], x[6], x[7], x[8]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #3
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6], x[7], x[8], x[9]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #4
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[4], x[5], x[6], x[7], x[8], x[9], x[10]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #5
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[5], x[6], x[7], x[8], x[9], x[10], x[11]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #6
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[6], x[7], x[8], x[9], x[10], x[11], x[12]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #7
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[7], x[8], x[9], x[10], x[11], x[12], x[13]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #8
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[8], x[9], x[10], x[11], x[12], x[13], x[14]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #9
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[9], x[10], x[11], x[12], x[13], x[14], x[15]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #10
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[10], x[11], x[12], x[13], x[14], x[15], x[16]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #11
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[11], x[12], x[13], x[14], x[15], x[16], x[17]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 1, 7 => 6, 6 => 11, 4 => 4, 1 => 35]
=========
Clique #12
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[12], x[13], x[14], x[15], x[16], x[17], x[18]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 9, 8 => 2, 7 => 4, 6 => 12, 4 => 4, 1 => 35]
=========
Clique #13
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[13], x[14], x[15], x[16], x[17], x[18], x[19]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 3, 11 => 6, 10 => 5, 9 => 7, 8 => 2, 7 => 5, 6 => 10, 4 => 6, 1 => 35]
=========
Clique #14
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [18 => 3, 17 => 4, 15 => 2, 14 => 3, 13 => 3, 12 => 4, 11 => 5, 10 => 4, 9 => 8, 8 => 3, 7 => 4, 6 => 11, 4 => 5, 1 => 35]"
    @test strRep(sps[3]) == "SparsityCorrelativeTerm with 14 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4], x[5], x[6], x[7], x[8]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #3
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6], x[7], x[8], x[9]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #4
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[4], x[5], x[6], x[7], x[8], x[9], x[10]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #5
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[5], x[6], x[7], x[8], x[9], x[10], x[11]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #6
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[6], x[7], x[8], x[9], x[10], x[11], x[12]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #7
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[7], x[8], x[9], x[10], x[11], x[12], x[13]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #8
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[8], x[9], x[10], x[11], x[12], x[13], x[14]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #9
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[9], x[10], x[11], x[12], x[13], x[14], x[15]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #10
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[10], x[11], x[12], x[13], x[14], x[15], x[16]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #11
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[11], x[12], x[13], x[14], x[15], x[16], x[17]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #12
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[12], x[13], x[14], x[15], x[16], x[17], x[18]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #13
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[13], x[14], x[15], x[16], x[17], x[18], x[19]
Block sizes:
  [85 => 1, 1 => 35]
=========
Clique #14
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [85 => 1, 1 => 35]"

    # we test some of the optimziations, but some just take too long
    if optimize
        @test sparse_optimize(:MosekMoment, sps[1])[2] ≈ 0 atol = 4e-6
        @test sparse_optimize(:MosekSOS, sps[1])[2] ≈ 0 atol = 1e-6
        @test sparse_optimize(:COSMOMoment, sps[1])[2] ≈ 0 atol = 2e-3
        @test sparse_optimize(:HypatiaMoment, sps[1])[2] ≈ 0 atol = 1e-3 # numerical failure

        @test sparse_optimize(:MosekMoment, sps[2])[2] ≈ 0 atol = 1e-6
        @test sparse_optimize(:MosekSOS, sps[2])[2] ≈ 0 atol = 2e-7
        #@test sparse_optimize(:COSMOMoment, sps[2])[2] ≈ 0 atol=1e-3
        #@test sparse_optimize(:HypatiaMoment, sps[2])[2] ≈ 0 atol=1e-3

        #@test sparse_optimize(:MosekMoment, sps[3])[2] ≈ 0 atol = 1e-6
        @test sparse_optimize(:MosekSOS, sps[3])[2] ≈ 0 atol = 2e-7
        #@test sparse_optimize(:COSMOMoment, sps[3])[2] ≈ 0 atol=1e-3
        #@test sparse_optimize(:HypatiaMoment, sps[3])[2] ≈ 0 atol=1e-3
    end

    @test isnothing(sparse_iterate!(sps[1]))
    @test strRep(sparse_iterate!(sps[2])) == "SparsityCorrelativeTerm with 14 cliques(s)
=========
Clique #1
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [36 => 1, 35 => 3, 34 => 1, 33 => 3, 32 => 3, 28 => 1, 27 => 2, 26 => 2, 25 => 1, 24 => 2, 23 => 2, 22 => 2, 21 => 2, 20 => 2, 19 => 4, 18 => 2, 17 => 1, 16 => 1, 15 => 1, 14 => 2, 13 => 2, 12 => 2, 11 => 1, 10 => 3, 8 => 3, 7 => 1, 6 => 11, 5 => 4, 4 => 4, 3 => 7, 1 => 4]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4], x[5], x[6], x[7], x[8]
Block sizes:
  [39 => 1, 38 => 1, 36 => 2, 35 => 1, 33 => 1, 31 => 1, 30 => 1, 29 => 2, 28 => 1, 26 => 2, 25 => 2, 24 => 1, 23 => 2, 22 => 2, 21 => 4, 20 => 2, 19 => 1, 18 => 3, 17 => 1, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 4, 8 => 4, 6 => 11, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #3
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6], x[7], x[8], x[9]
Block sizes:
  [40 => 1, 38 => 2, 37 => 3, 35 => 4, 32 => 1, 31 => 1, 29 => 1, 28 => 1, 26 => 1, 25 => 1, 24 => 1, 23 => 5, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 17 => 1, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 5, 8 => 4, 6 => 10, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #4
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[4], x[5], x[6], x[7], x[8], x[9], x[10]
Block sizes:
  [42 => 1, 40 => 2, 39 => 3, 37 => 1, 35 => 2, 33 => 2, 31 => 1, 28 => 1, 27 => 1, 26 => 3, 24 => 1, 23 => 4, 22 => 2, 21 => 2, 20 => 1, 19 => 2, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #5
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[5], x[6], x[7], x[8], x[9], x[10], x[11]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #6
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[6], x[7], x[8], x[9], x[10], x[11], x[12]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #7
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[7], x[8], x[9], x[10], x[11], x[12], x[13]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #8
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[8], x[9], x[10], x[11], x[12], x[13], x[14]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #9
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[9], x[10], x[11], x[12], x[13], x[14], x[15]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #10
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[10], x[11], x[12], x[13], x[14], x[15], x[16]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #11
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[11], x[12], x[13], x[14], x[15], x[16], x[17]
Block sizes:
  [41 => 1, 40 => 2, 39 => 1, 38 => 2, 37 => 1, 36 => 2, 35 => 2, 32 => 1, 31 => 1, 28 => 1, 27 => 2, 26 => 1, 24 => 2, 23 => 3, 22 => 2, 21 => 2, 20 => 2, 19 => 1, 18 => 2, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 11 => 1, 10 => 6, 8 => 4, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #12
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[12], x[13], x[14], x[15], x[16], x[17], x[18]
Block sizes:
  [41 => 1, 40 => 2, 39 => 2, 37 => 1, 36 => 3, 35 => 1, 32 => 1, 29 => 1, 28 => 1, 27 => 3, 26 => 1, 25 => 1, 24 => 2, 23 => 3, 22 => 2, 20 => 2, 19 => 3, 18 => 1, 17 => 1, 16 => 1, 15 => 2, 14 => 1, 13 => 1, 12 => 1, 11 => 1, 10 => 4, 9 => 1, 8 => 3, 7 => 1, 6 => 9, 5 => 4, 4 => 6, 3 => 4, 1 => 4]
=========
Clique #13
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[13], x[14], x[15], x[16], x[17], x[18], x[19]
Block sizes:
  [40 => 2, 39 => 2, 38 => 1, 37 => 1, 36 => 1, 35 => 1, 34 => 1, 33 => 2, 31 => 1, 29 => 1, 28 => 2, 26 => 2, 25 => 1, 24 => 3, 23 => 2, 22 => 3, 20 => 2, 19 => 3, 18 => 2, 16 => 1, 15 => 2, 14 => 2, 13 => 1, 12 => 1, 11 => 1, 10 => 3, 9 => 1, 8 => 3, 7 => 2, 6 => 7, 5 => 4, 4 => 6, 3 => 6, 1 => 4]
=========
Clique #14
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [37 => 4, 36 => 1, 35 => 3, 34 => 1, 33 => 3, 31 => 1, 29 => 1, 28 => 1, 27 => 1, 25 => 2, 24 => 3, 23 => 3, 22 => 1, 21 => 3, 19 => 2, 18 => 3, 17 => 1, 16 => 2, 15 => 2, 14 => 2, 13 => 1, 11 => 1, 10 => 3, 9 => 1, 8 => 2, 7 => 4, 6 => 3, 5 => 6, 4 => 3, 3 => 11, 1 => 4]"
    @test strRep(sparse_iterate!(sps[3])) == "SparsityCorrelativeTerm with 14 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7]
Block sizes:
  [120 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4], x[5], x[6], x[7], x[8]
Block sizes:
  [120 => 1]
=========
Clique #3
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3], x[4], x[5], x[6], x[7], x[8], x[9]
Block sizes:
  [120 => 1]
=========
Clique #4
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[4], x[5], x[6], x[7], x[8], x[9], x[10]
Block sizes:
  [120 => 1]
=========
Clique #5
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[5], x[6], x[7], x[8], x[9], x[10], x[11]
Block sizes:
  [120 => 1]
=========
Clique #6
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[6], x[7], x[8], x[9], x[10], x[11], x[12]
Block sizes:
  [120 => 1]
=========
Clique #7
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[7], x[8], x[9], x[10], x[11], x[12], x[13]
Block sizes:
  [120 => 1]
=========
Clique #8
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[8], x[9], x[10], x[11], x[12], x[13], x[14]
Block sizes:
  [120 => 1]
=========
Clique #9
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[9], x[10], x[11], x[12], x[13], x[14], x[15]
Block sizes:
  [120 => 1]
=========
Clique #10
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[10], x[11], x[12], x[13], x[14], x[15], x[16]
Block sizes:
  [120 => 1]
=========
Clique #11
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[11], x[12], x[13], x[14], x[15], x[16], x[17]
Block sizes:
  [120 => 1]
=========
Clique #12
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[12], x[13], x[14], x[15], x[16], x[17], x[18]
Block sizes:
  [120 => 1]
=========
Clique #13
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[13], x[14], x[15], x[16], x[17], x[18], x[19]
Block sizes:
  [120 => 1]
=========
Clique #14
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [120 => 1]"
end

@testset "Generalized Rosenbrock" begin
    n = 40
    DynamicPolynomials.@polyvar x[1:n]
    sp = SparsityCorrelativeTerm(poly_problem(1 + sum(100 * (x[i] - x[i-1]^2)^2 + (1 - x[i])^2 for i in 2:n), 2,
                                              nonneg=[1 - sum(x[i]^2 for i in (20j-19):(20j)) for j in 1:Int(n / 20)]),
                                 term_mode=tm_chordal_cliques)
    @test strRep(sp) == "SparsityCorrelativeTerm with 3 cliques(s)
=========
Clique #1
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [21 => 1, 3 => 19, 2 => 19, 1 => 171]
  [2 => 19, 1 => 1]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[20], x[21]
Block sizes:
  [3 => 2, 2 => 2]
=========
Clique #3
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40]
Block sizes:
  [21 => 1, 3 => 19, 2 => 20, 1 => 171]
  [2 => 20]"

    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 38.0494 atol = 1e-4
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 38.0494 atol = 1e-4
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 38.0494 atol = 2e-1
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 38.0494 atol = 1e-4
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 3 cliques(s)
=========
Clique #1
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [40 => 1, 3 => 171, 2 => 19]
  [2 => 19, 1 => 1]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[20], x[21]
Block sizes:
  [4 => 1, 3 => 1, 2 => 1]
=========
Clique #3
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40]
Block sizes:
  [41 => 1, 3 => 190]
  [2 => 20]"

    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 38.0514 atol = 1e-4
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 38.0514 atol = 1e-4
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 38.0514 atol = 2e-1
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 38.0514 atol = 1e-4
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 3 cliques(s)
=========
Clique #1
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [40 => 1, 4 => 171, 2 => 19]
  [20 => 1, 1 => 1]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[20], x[21]
Block sizes:
  [4 => 1, 3 => 1, 2 => 1]
=========
Clique #3
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40]
Block sizes:
  [41 => 1, 4 => 190]
  [21 => 1]"

    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 38.0514 atol = 1e-4
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 38.0514 atol = 1e-4
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 38.0514 atol = 2e-1
        @test sparse_optimize(:HypatiaMoment, sp)[2] ≈ 38.0514 atol = 1e-4
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 3 cliques(s)
=========
Clique #1
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [149 => 1, 133 => 2, 89 => 5, 58 => 10, 20 => 1]
  [20 => 1, 1 => 1]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[20], x[21]
Block sizes:
  [4 => 1, 3 => 1, 2 => 1]
=========
Clique #3
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40]
Block sizes:
  [165 => 1, 141 => 2, 107 => 1, 93 => 4, 77 => 1, 60 => 10]
  [21 => 1]"

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 3 cliques(s)
=========
Clique #1
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20]
Block sizes:
  [211 => 1, 20 => 1]
  [20 => 1, 1 => 1]
=========
Clique #2
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[20], x[21]
Block sizes:
  [4 => 1, 3 => 1, 2 => 1]
=========
Clique #3
SparsityTermCliques with 1 constraint(s)
Variable cliques:
  x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40]
Block sizes:
  [231 => 1]
  [21 => 1]"

    @test isnothing(sparse_iterate!(sp))
end

@testset "Example 4.4 from Zhen, Fantuzzi, Papachristodoulou review" begin
    DynamicPolynomials.@polyvar x[1:4]
    sp = SparsityCorrelativeTerm(
        poly_problem(2 + x[1]^2 * x[4]^2 * (x[1]^2 * x[4]^2 - 1) - x[1]^2 + x[1]^4 +
                     sum((x[i]^2 * x[i-1]^2 * (x[i]^2 * x[i-1]^2 - 1) - x[i]^2 + x[i]^4 for i in 2:4)), 4)
    )
    @test strRep(sp) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[4]
Block sizes:
  [10 => 1, 4 => 6, 1 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[2], x[3], x[4]
Block sizes:
  [10 => 1, 4 => 6, 1 => 1]"

    if optimize
        for solver in all_solvers
            @test sparse_optimize(solver, sp)[2] ≈ 0.110118 atol = 1e-3
        end
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Something with matrices" begin
    DynamicPolynomials.@polyvar x[1:3]
    sp = SparsityCorrelativeTerm(poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1] * x[2] + x[2], 2,
                                              psd=[[x[1] x[2]; x[2] x[1]]]))
    @test strRep(sp) == "SparsityCorrelativeTerm with 2 cliques(s)
=========
Clique #1
SparsityTermBlock with 1 constraint(s)
Variable cliques:
  x[1], x[2]
Block sizes:
  [6 => 1]
  [3 => 1]
=========
Clique #2
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[3]
Block sizes:
  [2 => 1, 1 => 1]"

    if optimize
        @test sparse_optimize(:MosekMoment, sp)[2] ≈ 0.283871 atol = 1e-6
        @test sparse_optimize(:MosekSOS, sp)[2] ≈ 0.283871 atol = 1e-6
        @test sparse_optimize(:COSMOMoment, sp)[2] ≈ 0.283871 atol = 1e-6
        @test sparse_optimize(:HypatiaMoment, sp, dense=true)[2] ≈ 0.283871 atol = 1e-6
    end

    @test isnothing(sparse_iterate!(sp))
end

@testset "Complex" begin
    DynamicPolynomials.@polycvar z[1:2]
    sp = SparsityCorrelativeTerm(poly_problem(z[1] + conj(z[1]), 2,
                                              nonneg=[1 - z[1] * conj(z[1]) - z[2] * conj(z[2])]))
    @test strRep(sp) == "SparsityCorrelativeTerm with 1 cliques(s)
=========
Clique #1
SparsityTermBlock with 1 constraint(s)
Variable cliques:
  z[1], z[2]
Block sizes:
  [2 => 1, 1 => 4]
  [2 => 1, 1 => 1]"

    if optimize
        for solver in complex_solvers
            @test sparse_optimize(solver, sp)[2] ≈ -2 atol = 1e-7
        end
    end

    @test strRep(sparse_iterate!(sp)) == "SparsityCorrelativeTerm with 1 cliques(s)
=========
Clique #1
SparsityTermBlock with 1 constraint(s)
Variable cliques:
  z[1], z[2]
Block sizes:
  [3 => 1, 2 => 1, 1 => 1]
  [2 => 1, 1 => 1]"

    if optimize
        for solver in complex_solvers
            @test sparse_optimize(solver, sp)[2] ≈ -2 atol = 1e-7
        end
    end

    @test isnothing(sparse_iterate!(sp))
end