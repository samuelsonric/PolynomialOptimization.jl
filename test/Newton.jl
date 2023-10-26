include("./shared.jl")
using LinearAlgebra

@testset "Monomial iterator" begin
    DynamicPolynomials.@polyvar x[1:3]
    multiit = Iterators.product(0:4, 0:4, 0:4)
    for mindeg in 0:4, maxdeg in 0:4, minmultideg in multiit, maxmultideg in multiit
        minm, maxm = collect(minmultideg), collect(maxmultideg)
        if mindeg > maxdeg || any(minmultideg .> maxmultideg)
            @test_throws ErrorException PolynomialOptimization.MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minm, maxm)
        else
            mi = PolynomialOptimization.MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minm, maxm)
            exp = exponents.(monomials(x, mindeg:maxdeg, m -> all(minm .≤ exponents(m) .≤ maxm)))
            @test collect(mi) == exp
            @test length(mi) == length(exp)
        end
    end
end

readlines_killr(io::IO) = last.(rsplit.(readlines(io), ("\r",), limit=2))

# Adapter from https://discourse.julialang.org/t/a-minimal-example-with-base-redirect-stdout/64245/8
function capture_stdout(f::Function, @nospecialize(readfn=readlines_killr))
    pipe = Pipe()
    started = Base.Event()
    writer = @async redirect_stdout($pipe) do
        try
            notify($started)
            return $f()
        finally
            close(Base.pipe_writer($pipe))
        end
    end
    wait(started)
    result = readfn(pipe)
    return fetch(writer), result
end

# BPT: Blekhermann, Parrilo, Thomas - Semidefinite Optimization and Convex Algebraic Geometry
# Actually, there only the full Newton polytope is given, so for the examples, the half polytope was calculated manually.
@testset "Newton polytope (BPT Example 3.92)" begin
    # First check the quick algo
    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar x y
        @test newton_halfpolytope(5 - x*y - x^2*y^2 + 3y^2 + x^4, preprocess_quick=true, verbose=true) ==
            monomial_vector([1, y, x, x*y, x^2])
    end
    @test output[2] == "Removing redundancies from the convex hull - quick heuristic, 5 initial candidates"
    @test startswith(output[3], "Found 4 potential extremal points of the convex hull in")
    # Then the same for the fine algo
    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar x y
        @test newton_halfpolytope(5 - x*y - x^2*y^2 + 3y^2 + x^4, preprocess_fine=true, verbose=true) ==
            monomial_vector([1, y, x, x*y, x^2])
    end
    @test output[2] == "Removing redundancies from the convex hull - fine, 5 initial candidates"
    @test startswith(output[3], "Found 4 extremal points of the convex hull in")
end

@testset "Newton polytope (BPT Example 3.93)" begin
    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar x y
        @test newton_halfpolytope(1 - x^2 + x*y + 4y^4, preprocess_quick=true, verbose=true) == monomial_vector([1, y, x, y^2])
    end
    @test output[2] == "Removing redundancies from the convex hull - quick heuristic, 4 initial candidates"
    @test startswith(output[3], "Found 3 potential extremal points of the convex hull")

    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar x y
        @test newton_halfpolytope(1 - x^2 + x*y + 4y^4, preprocess_fine=true, verbose=true) == monomial_vector([1, y, x, y^2])
    end
    @test output[2] == "Removing redundancies from the convex hull - fine, 4 initial candidates"
    @test startswith(output[3], "Found 3 extremal points of the convex hull")
end

@testset "Newton polytope (BPT, Example 3.95)" begin
    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar w x y z
        @test newton_halfpolytope((w^4 + 1) * (x^4 + 1) * (y^4 + 1) * (z^4 + 1) + 2w + 3x + 4y + 5z,
            preprocess_quick=true, verbose=true) == PolynomialOptimization.makemonovec([w, x, y, z],
                collect(PolynomialOptimization.MonomialIterator{Graded{LexOrder}}(0, 8, [0, 0, 0, 0], [2, 2, 2, 2])))
    end
    @test output[2] == "Removing redundancies from the convex hull - quick heuristic, 20 initial candidates"
    @test startswith(output[3], "Found 16 potential extremal points of the convex hull in")

    _, output = capture_stdout() do
        DynamicPolynomials.@polyvar w x y z
        @test newton_halfpolytope((w^4 + 1) * (x^4 + 1) * (y^4 + 1) * (z^4 + 1) + 2w + 3x + 4y + 5z,
            preprocess_fine=true, verbose=true) == PolynomialOptimization.makemonovec([w, x, y, z],
                collect(PolynomialOptimization.MonomialIterator{Graded{LexOrder}}(0, 8, [0, 0, 0, 0], [2, 2, 2, 2])))
    end
    @test output[2] == "Removing redundancies from the convex hull - fine, 20 initial candidates"
    @test startswith(output[3], "Found 16 extremal points of the convex hull in")
end

# Motzkin (BPT Exercise 3.97) is already checked as part of the documentation

@testset "Newton polytope (BPT Exercise 4.5)" begin
    DynamicPolynomials.@polyvar x[1:3]
    @test newton_halfpolytope(x[1]*x[2]^2 + x[2]^2 + prod(x)) == monomial_vector([x[2]])
end

@testset "Newton polytope (mutually unbiased bases)" begin
    function findmubs(d, n)
        DynamicPolynomials.@polycvar x[1:d, 1:d, 1:n]
        obj = zero(polynomialtype(x[1,1,1], Float64))
        # make it an ONB
        @views for i in 1:n
            obj += sum(z -> real(z)^2 + imag(z)^2, x[:, :, i] * x[:, :, i]' - I)
        end
        fac = inv(Float64(d))
        for i in 1:n-1
            for j in 1:i-1
                obj += sum(z -> (real(z)^2 + imag(z)^2 - fac)^2, x[:, :, i] * x[:, :, j]')
            end
        end
        obj
    end

    _, output = capture_stdout() do
        obj = findmubs(2, 3)
        @test length(newton_halfpolytope(obj, preprocess=true, verbose=true)) == 1453
    end
    @test output[2] == "Removing redundancies from the convex hull - quick heuristic, 689 initial candidates"
    @test startswith(output[3], "Found 517 potential extremal points of the convex hull in")
    @test output[4] == "Removing redundancies from the convex hull - randomized, 517 initial candidates"
    m = match(r"^Found (\d+) extremal points of the convex hull via randomization in", output[5])
    @test !isnothing(m)
    pts = parse(Int, m[1])
    @test 57 ≤ pts ≤ 517
    @test output[6] == "Removing redundancies from the convex hull - fine, $pts initial candidates"
    @test startswith(output[7], "Found 57 extremal points of the convex hull in")
end