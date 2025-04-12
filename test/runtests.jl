include("./shared.jl")
using Documenter: doctest;

if isone(Threads.nthreads())
    @testset "PolynomialOptimization" begin
        @testset "Documentation" begin
            doctest(PolynomialOptimization)
        end
        @testset "FastVector" begin
            include("./FastVector.jl")
        end
        @testset "IntPolynomials" begin
            include("./IntPolynomials.jl")
        end
        @testset "Moment solver interface" begin
            include("./MomentHelpers.jl")
        end
        @testset "no sparsity" begin
            include("./RelaxationDense.jl")
        end
        @testset "SOSCertificate" begin
            include("./SOSCertificate.jl")
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
        @testset "correlative and term sparsity" begin
            include("./RelaxationSparsityCorrelativeTerm.jl")
        end
        @testset "tightening using Nie's method" begin
            include("./Tightening.jl")
        end
        @testset "Newton" begin
            include("./Newton.jl")
        end
        @testset "multiplication by prefactor using Mai et al.'s method" begin
            include("./Noncompact.jl")
        end
        @testset "Lancelot" begin
            include("./Lancelot.jl")
        end
        @testset "SketchyCGAL solver based" begin
            include("./SketchyCGAL.jl")
        end
    end
else
    @testset "PolynomialOptimization (multi-threaded)" begin
        # we don't want to test everything again - the things that actually have a different behavior when multi-threaded
        @testset "Newton" begin
            include("./Newton.jl")
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
        @testset "correlative and term sparsity" begin
            include("./RelaxationSparsityCorrelativeTerm.jl")
        end
    end
end