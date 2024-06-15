include("./shared.jl")
using Documenter: doctest;

@testset "PolynomialOptimization" begin
    @testset "Documentation" begin
        doctest(PolynomialOptimization)
    end
    @testset "SimplePolynomials" begin
        include("SimplePolynomials.jl")
    end
    @testset "Moment solver interface" begin
        include("./MomentHelpers.jl")
    end
    @testset "no sparsity" begin
        include("./RelaxationDense.jl")
    end
    @testset "correlative sparsity" begin
        include("./RelaxationSparsityCorrelative.jl")
    end
    @testset "term sparsity" begin
        include("./RelaxationSparsityTermBlock.jl")
    end
    @testset "term sparsity with chordal extension" begin
        include("./RelaxationSparsityTermCliques.jl")
    end
    if :MosekSOS ∈ all_solvers
        # Tightening requires Mosek at the moment
        @testset "tightening using Nie's method" begin
            include("./Tightening.jl")
        end
        # Mosek is the only linear solver implemented at the moment
        @testset "Newton" begin
            include("./Newton.jl")
        end
    end
    @testset "multiplication by prefactor using Mai et al.'s method" begin
        include("./Noncompact.jl")
    end
end