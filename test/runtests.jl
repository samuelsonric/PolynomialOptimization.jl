include("./shared.jl")
using Documenter: doctest;

@testset "PolynomialOptimization" begin
    @testset "Documentation" begin
        doctest(PolynomialOptimization)
    end
    @testset "no sparsity" begin
        include("./SparsityNone.jl")
    end
    @testset "correlative sparsity" begin
        include("./SparsityCorrelative.jl")
    end
    @testset "term sparsity" begin
        include("./SparsityTermBlock.jl")
    end
    @testset "term sparsity with chordal extension" begin
        include("./SparsityTermCliques.jl")
    end
    @testset "correlative and term sparsity" begin
        include("./SparsityCorrelativeTerm.jl")
    end
    if :MosekSOS ∈ all_solvers
        # Tightening requires Mosek at the moment
        @testset "tightening using Nie's method" begin
            include("./Tightening.jl")
        end
    end
    @testset "multiplication by prefactor using Mai et al.'s method" begin
        include("./Noncompact.jl")
    end
end