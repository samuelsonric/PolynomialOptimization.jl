# This testsuite uses some of MP's tests (and some more), but does so explicitly, since SimplePolynomials on purpose does not
# support a lot of the things that you would want to do with polynomials.
using Test
using LinearAlgebra, SparseArrays

using MultivariatePolynomials
const MP = MultivariatePolynomials
import DynamicPolynomials

using PolynomialOptimization.SimplePolynomials
import PolynomialOptimization.SimplePolynomials: Absent

testdir = dirname(pathof(MP)) * "/../test"
include("$testdir/utils.jl")

@testset "Variable" begin
    # No polyvar macro index set: we don't provide a macro for variable construction.
    # You shouldn't build up SimplePolynomials from variables anyway!

    @test_throws DomainError SimpleRealVariable{1,0}(-1)
    @test_throws DomainError SimpleRealVariable{1,0}(0)
    @test_throws DomainError SimpleRealVariable{1,5}(2)
    @test_throws DomainError SimpleComplexVariable{0,1}(-1)
    @test_throws DomainError SimpleComplexVariable{0,1}(0)
    @test_throws DomainError SimpleComplexVariable{5,1}(2)
    for (t, i) in ((UInt8, Int8(1)), (UInt16, Int16(1)), (UInt32, Int32(1)), (UInt64, Int64(1)))
        x = SimpleRealVariable{typemax(t),0}(i)
        @test x.index === t(i)
        @test convert(typeof(x), x) === x # these are a few of the only allowed converts
        alloc_test(() -> convert(typeof(x), x), 0)
        @test convert(variable_union_type(x), x) === x
        alloc_test(() -> convert(variable_union_type(x), x), 0)
        z = SimpleComplexVariable{0,typemax(t)}(i)
        @test z.index === t(i)
        @test convert(typeof(z), z) === z
        alloc_test(() -> convert(typeof(z), z), 0)
        @test convert(variable_union_type(z), z) === z
        alloc_test(() -> convert(variable_union_type(z), z), 0)
    end

    x = SimpleRealVariable{1,1}(0x1)
    z = SimpleComplexVariable{1,1}(0x1)

    @test isreal(x)
    @test !isconj(x)
    @test ordinary_variable(x) === x

    @test !isreal(z)
    @test !isconj(z)
    @test ordinary_variable(z) === z

    zc = SimpleComplexVariable{1,1}(0x1, true)
    @test !isreal(zc)
    @test isconj(zc)
    @test ordinary_variable(zc) === z

    @test zc != z

    @test 1 != x
    @test x != 0
    @test nvariables(x) == 3 # different from MP - our ring is fixed
    @test !isapproxzero(x)
    @test !iszero(x)
    # no tests that involve zero(x): this would involve a conversion

    typetests(x)
    # no tests with polynomial(x): don't construct them in this way

    @test nterms(x) == 1
    # no tests for terms(x): we cannot convert, as we don't know whether to a dense or sparse monomial and we also don't know
    # the exponent type.

    @test degree(x, x) == 1
    @test degree(x, z) == 0
    @test length(exponents(x)) == 3
    @test first(exponents(x)) == 1
    @test isconstant(x) == false

    # no Issue #82, we cannot convert

    @testset "Effective variables" begin
        @test [x] == @inferred effective_variables(x)
        @test [z] == @inferred effective_variables(z)
    end

    # no creation of similar variables, this reeks of bottom-up construction
end
@testset "Monomial (real)" begin
    # lots of tests to skip, we don't construct monomials by multiplying variables

    @test_throws ArgumentError SimpleMonomial{2,0}(UInt8[1])
    m = SimpleMonomial{7,0}(UInt8[1, 0, 1, 0, 1, 0, 1])
    alloc_test(let v=UInt8[1, 0, 1, 0, 1, 0, 1]; () -> SimpleMonomial{7,0}(v) end, 0)
    @test m == SimpleMonomial{7,0}(Int8[1, 0, 1, 0, 1, 0, 1])
    alloc_test(() -> convert(typeof(m), m), 0)
    @test_throws ArgumentError SimpleMonomial{1,0}([1, 2])

    @test nvariables(SimpleMonomialVector{4,0}([2 0; 0 1; 0 1; 0 1])) == 4

    @test nterms(SimpleMonomial{1,0}([2])) == 1

    @test degree(SimpleMonomial{3,0}([1, 0, 2]), SimpleRealVariable{3,0}(1)) == UInt(1)
    @test degree(SimpleMonomial{3,0}([1, 0, 2]), SimpleRealVariable{3,0}(2)) == UInt(0)
    @test degree(SimpleMonomial{3,0}([1, 0, 2]), SimpleRealVariable{3,0}(3)) == UInt(2)

    @test_throws InexactError variable(SimpleMonomial{1,0}([2]))
    @test_throws InexactError variable(SimpleMonomial{2,0}([1, 1]))
    @test_throws InexactError variable(constant_monomial(typeof(SimpleRealVariable{1,0}(1))))

    x = SimpleRealVariable{2,0}(1)
    @test x != constant_monomial(typeof(x))
    @test constant_monomial(typeof(x)) != x

    # no tests for squaring monomials

    typetests(m)

    xmon = SimpleMonomial{2,0}([1, 0])
    @test variable(xmon) === x
    # no arithmetics
    @test variable(Term(1.0, xmon)) === x
    @test_throws InexactError variable(Term(3, xmon)) === x

    @test transpose(x) === x
    @test adjoint(x) === x
    @test transpose(m) === m
    @test adjoint(m) === m

    @testset "Effective variables" begin
        x = SimpleRealVariable{8,0}(1)
        y = ntuple(i -> SimpleRealVariable{8,0}(i+1), Val(7))
        T = variable_union_type(x)
        @test x isa T
        @test y[2] isa T
        @test T[x, y[2]] == @inferred effective_variables(SimpleMonomial{8,0}([1, 0, 1, 0, 0, 0, 0, 0]))
        @test T[x] == @inferred effective_variables(      SimpleMonomial{8,0}([1, 0, 0, 0, 0, 0, 0, 0]))
        @test T[y[2]] == @inferred effective_variables(   SimpleMonomial{8,0}([0, 0, 1, 0, 0, 0, 0, 0]))
    end

    # no mapexponents
end
@testset "Monomial (complex)" begin
    # lots of tests to skip, we don't construct monomials by multiplying variables

    @test_throws ArgumentError SimpleMonomial{0,1}([1, 0], [0])
    @test_throws ArgumentError SimpleMonomial{0,2}([1, 0], [0])
    m = SimpleMonomial{0,7}(UInt8[1, 0, 0, 0, 1, 0, 1], UInt8[0, 0, 1, 0, 0, 0, 0])
    alloc_test(let v1=UInt8[1, 0, 0, 0, 1, 0, 1], v2=UInt8[0, 0, 1, 0, 0, 0, 0]; () -> SimpleMonomial{0,7}(v1, v2) end, 0)
    @test m == SimpleMonomial{0,7}(Int8[1, 0, 0, 0, 1, 0, 1], Int8[0, 0, 1, 0, 0, 0, 0])
    alloc_test(() -> convert(typeof(m), m), 0)
    @test_throws ArgumentError SimpleMonomial{0,1}([1, 2], [0, 0])
    @test_throws ArgumentError SimpleMonomial{0,2}([1, 2], [0])
    @test_throws ArgumentError SimpleMonomial{0,2}([0], [1, 2])

    @test nvariables(SimpleMonomialVector{0,4}([2 0; 0 1; 0 0; 0 1], [0 0; 0 0; 0 1; 0 0])) == 8

    @test nterms(SimpleMonomial{0,1}([2], [0])) == 1

    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(1)) === UInt(1)
    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(1, true)) === UInt(0)
    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(2)) === UInt(0)
    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(2, true)) === UInt(1)
    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(3)) === UInt(2)
    @test degree(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(3, true)) === UInt(3)
    @test degree_complex(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(1)) === UInt(1)
    @test degree_complex(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(2)) === UInt(1)
    @test degree_complex(SimpleMonomial{0,3}([1, 0, 2], [0, 1, 3]), SimpleComplexVariable{0,3}(3)) === UInt(3)

    @test_throws InexactError variable(SimpleMonomial{0,1}([2], [0]))
    @test_throws InexactError variable(SimpleMonomial{0,2}([1, 0], [0, 1]))
    @test_throws InexactError variable(constant_monomial(typeof(SimpleComplexVariable{0,1}(1))))

    z = SimpleComplexVariable{0,2}(1)
    @test z != constant_monomial(typeof(z))
    @test constant_monomial(typeof(z)) != z

    # no tests for squaring monomials

    typetests(m)

    zmon = SimpleMonomial{0,2}([1, 0], [0, 0])
    @test variable(zmon) === z
    # no arithmetics
    @test variable(Term(1.0, zmon)) === z
    @test_throws InexactError variable(Term(3, zmon)) === z

    @test transpose(z) === z
    @test adjoint(z) === conj(z)
    @test transpose(m) === m
    @test adjoint(m) === conj(m) == SimpleMonomial{0,7}(UInt8[0, 0, 1, 0, 0, 0, 0], UInt8[1, 0, 0, 0, 1, 0, 1])

    @testset "Effective variables" begin
        y = ntuple(i -> SimpleComplexVariable{0,8}(i, true), Val(7))
        z = SimpleComplexVariable{0,8}(8)
        T = variable_union_type(z)
        @test z isa T
        @test y[2] isa T
        @test T[z, y[2]] == @inferred effective_variables(SimpleMonomial{0,8}([0, 0, 0, 0, 0, 0, 0, 1],
                                                                              [0, 1, 0, 0, 0, 0, 0, 0]))
        @test T[z] == @inferred effective_variables(      SimpleMonomial{0,8}([0, 0, 0, 0, 0, 0, 0, 1],
                                                                              [0, 0, 0, 0, 0, 0, 0, 0]))
        @test T[y[2]] == @inferred effective_variables(   SimpleMonomial{0,8}([0, 0, 0, 0, 0, 0, 0, 0],
                                                                              [0, 1, 0, 0, 0, 0, 0, 0]))
    end

    # no mapexponents
end
@testset "Monomial (mixed)" begin
    # lots of tests to skip, we don't construct monomials by multiplying variables

    @test_throws ArgumentError SimpleMonomial{2,2}([1], [1, 2], [0, 1])
    @test_throws ArgumentError SimpleMonomial{2,2}([1, 0], [1], [0, 1])
    @test_throws ArgumentError SimpleMonomial{2,2}([1, 0], [1, 0], [0])
    @test_throws ArgumentError SimpleMonomial{2,2}([1, 0], [1], [0])
    m = SimpleMonomial{2,7}(UInt8[2, 3], UInt8[1, 0, 0, 0, 1, 0, 1], UInt8[0, 0, 1, 0, 0, 0, 0])
    alloc_test(let v1=UInt8[2, 3], v2=UInt8[1, 0, 0, 0, 1, 0, 1], v3=UInt8[0, 0, 1, 0, 0, 0, 0]
        () -> SimpleMonomial{2,7}(v1, v2, v3)
    end, 0)
    @test m == SimpleMonomial{2,7}(Int8[2, 3], Int8[1, 0, 0, 0, 1, 0, 1], Int8[0, 0, 1, 0, 0, 0, 0])
    alloc_test(() -> convert(typeof(m), m), 0)
    @test_throws ArgumentError SimpleMonomial{1,1}([1, 2], [1], [1])
    @test_throws ArgumentError SimpleMonomial{1,1}([1], [1, 2], [0, 0])
    @test_throws ArgumentError SimpleMonomial{1,2}([1], [1, 2], [0])
    @test_throws ArgumentError SimpleMonomial{1,2}([1], [0], [1, 2])

    @test nvariables(SimpleMonomialVector{3,4}([2 0; 0 0; 1 1], [2 0; 0 1; 0 0; 0 1], [0 0; 0 0; 0 1; 0 0])) == 11

    @test nterms(SimpleMonomial{1,1}([3], [2], [0])) == 1

    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleRealVariable{2,3}(1)) === UInt(2)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleRealVariable{2,3}(2)) === UInt(1)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(1)) === UInt(1)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(1, true)) === UInt(0)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(2)) === UInt(0)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(2, true)) === UInt(1)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(3)) === UInt(2)
    @test degree(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(3, true)) === UInt(3)
    @test degree_complex(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleRealVariable{2,3}(1)) === UInt(2)
    @test degree_complex(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleRealVariable{2,3}(2)) === UInt(1)
    @test degree_complex(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(1)) === UInt(1)
    @test degree_complex(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(2)) === UInt(1)
    @test degree_complex(SimpleMonomial{2,3}([2, 1], [1, 0, 2], [0, 1, 3]), SimpleComplexVariable{2,3}(3)) === UInt(3)

    @test_throws InexactError variable(SimpleMonomial{1,1}([3], [2], [0]))
    @test_throws InexactError variable(SimpleMonomial{1,2}([1], [1, 0], [0, 1]))
    @test_throws InexactError variable(constant_monomial(typeof(SimpleComplexVariable{1,1}(1))))

    # no constant monomial tests -> these belong to SimpleSparseMonomial, which is the default for monomial construction
    x = SimpleRealVariable{3,2}(1)
    @test x != constant_monomial(typeof(x))
    @test constant_monomial(typeof(x)) != x
    z = SimpleComplexVariable{3,2}(1)
    @test z != constant_monomial(typeof(z))
    @test constant_monomial(typeof(z)) != z
    @test x != z

    # no tests for squaring monomials

    typetests(m)

    xmon = SimpleMonomial{3,2}([1, 0, 0], [0, 0], [0, 0])
    zmon = SimpleMonomial{3,2}([0, 0, 0], [1, 0], [0, 0])
    @test variable(xmon) === x
    @test variable(zmon) === z
    # no arithmetics
    @test variable(Term(1.0, xmon)) === x
    @test variable(Term(1.0, zmon)) === z
    @test_throws InexactError variable(Term(3, xmon)) === z
    @test_throws InexactError variable(Term(3, zmon)) === z

    @test transpose(x) === x
    @test transpose(z) === z
    @test adjoint(x) === x
    @test adjoint(z) === conj(z)
    @test transpose(m) === m
    @test adjoint(m) === conj(m) == SimpleMonomial{2,7}(UInt8[2, 3], UInt8[0, 0, 1, 0, 0, 0, 0], UInt8[1, 0, 0, 0, 1, 0, 1])

    @testset "Effective variables" begin
        x = SimpleRealVariable{3,5}(2)
        y = SimpleComplexVariable{3,5}(1, true)
        z = SimpleComplexVariable{3,5}(3)
        T = SimpleVariable{3,5}
        @test z isa T
        @test y isa T
        @test T[x, z, y] == @inferred effective_variables(SimpleMonomial{3,5}([0, 2, 0], [0, 0, 1, 0, 0],
                                                                                         [1, 0, 0, 0, 0]))
        @test T[x] == @inferred effective_variables(      SimpleMonomial{3,5}([0, 2, 0], [0, 0, 0, 0, 0],
                                                                                         [0, 0, 0, 0, 0]))
        @test T[z] == @inferred effective_variables(      SimpleMonomial{3,5}([0, 0, 0], [0, 0, 1, 0, 0],
                                                                                         [0, 0, 0, 0, 0]))
        @test T[y] == @inferred effective_variables(      SimpleMonomial{3,5}([0, 0, 0], [0, 0, 0, 0, 0],
                                                                                         [1, 0, 0, 0, 0]))
    end

    # no mapexponents
end
@testset "Term" begin
    # as our implementation uses the default Term, there's not much to test
    t = Term(3, SimpleMonomial{2,0}([2, 4]))
    alloc_test(() -> convert(typeof(t), t), 0)
    typetests(t)
    typetests([t, Term(2, SimpleMonomial{2,0}([2, 0]))])
end
@testset "Monomial vector" begin
    # by default, our monomial vector does not impose any order
    @test_throws ArgumentError monomials(2, 0, 1:0)
    X = [SimpleMonomial{2,0}(UInt8[0, 2]),
         SimpleMonomial{2,0}(UInt8[1, 1]),
         SimpleMonomial{2,0}(UInt8[2, 0])]
    monos = monomials(2, 0, 2:2)
    @test length(monos) == length(X)
    for (x, m) in zip(X, monos)
        @test m == x
    end
    monos = monomials(2, 0, 2:2, representation=:sparse)
    @test length(monos) == length(X)
    X = [SimpleMonomial{2,0}(sparsevec(UInt8[2], UInt8[2], 2)),
         SimpleMonomial{2,0}(sparsevec(UInt8[1, 2], UInt8[1, 1], 2)),
         SimpleMonomial{2,0}(sparsevec(UInt8[1], UInt8[2], 2))]
    for (x, m) in zip(X, monos)
        @test m == x
    end
    # we don't provide monomial_vector_type

    X = SimpleMonomialVector{2,0}(UInt8[0 1 1; 0 0 1]) # directly sorted
    @test X == collect(X)
    @test nvariables(X) == 2
    @test variables(X)[1] == SimpleRealVariable{2,0}(1)
    @test variables(X)[2] == SimpleRealVariable{2,0}(2)
    @test X[2:3][1] == SimpleMonomial{2,0}([0x1, 0x0])
    @test X[2:3][2] == SimpleMonomial{2,0}([0x1, 0x1])

    # no sort, no merge

    @test monomials(1, 0, 1:3) == SimpleMonomialVector{1,0}(UInt8[1 2 3])

    @testset "monomials" begin
        @test monomials(3, 0, 0:3) == SimpleMonomialVector{3,0}(
            copy(transpose(UInt8[
                0 0 0
                0 0 1
                0 1 0
                1 0 0
                0 0 2
                0 1 1
                0 2 0
                1 0 1
                1 1 0
                2 0 0
                0 0 3
                0 1 2
                0 2 1
                0 3 0
                1 0 2
                1 1 1
                1 2 0
                2 0 1
                2 1 0
                3 0 0
            ]))
        )
    end

    @test_throws ArgumentError monomials(1, 0, -1:0)

    # new tests
    @test effective_nvariables(monomials(2, 3, 0:2)) == 8
    @test effective_nvariables(monomials(2, 3, 0:1), monomials(2, 3, 0:2, representation=:sparse)) == 8
    @test effective_nvariables(monomials(2, 3, 0:0)) == 0
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 0; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 2; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 0; 1 0])) == 2
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 0; 0 0]), SimpleMonomialVector{2,0}([1 0; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 0; 0 0]), SimpleMonomialVector{2,0}([1 0; 1 0])) == 2
end
@testset "Polynomial" begin
    # polynomial is just a very thin wrapper, and its main importance is in converting other MP polynomials to SimplePolynomial
    # (which is also mainly a feature of SimpleMonomialVector, but we'll test it here)
    DynamicPolynomials.@polyvar x

    @test terms(SimplePolynomial(polynomial([1, x^2, x, 2x^2]))) ==
        [Term(1, SimpleMonomial{1,0}([0x0])),
         Term(1, SimpleMonomial{1,0}([0x1])),
         Term(3, SimpleMonomial{1,0}([0x2]))]
    @test terms(SimplePolynomial(polynomial([1, x^2, x, 2x^2]), representation=:sparse)) ==
        [Term(1, SimpleMonomial{1,0}([0x0])),
         Term(1, SimpleMonomial{1,0}([0x1])),
         Term(3, SimpleMonomial{1,0}([0x2]))]
    @test terms(SimplePolynomial(polynomial([1, x^2, x, 2x^2]), representation=:dense, max_power=typemax(UInt16))) ==
        [Term(1, SimpleMonomial{1,0}([0x0000])),
         Term(1, SimpleMonomial{1,0}([0x0001])),
         Term(3, SimpleMonomial{1,0}([0x0002]))]

    @test term(SimplePolynomial(x + x^2 - x)) isa AbstractTerm

    DynamicPolynomials.@polycvar y
    p = SimplePolynomial(3x^2 * y^4 + 2x)
    @test terms(p)[1] == Term(2, SimpleMonomial{1,1}([0x1], [0x0], [0x0]))
    @test terms(p)[end] == Term(3, SimpleMonomial{1,1}([0x2], [0x4], [0x0]))
    typetests(p)
    typetests([p, SimplePolynomial(x + y)])

    @test coefficient(SimplePolynomial(2x + 4y^2 + 3), SimpleMonomial{1,1}([0x0], [0x2], [0x0])) == 4
    @test coefficient(SimplePolynomial(2x + 4y^2 + 3), SimpleMonomial{1,1}([0x2], [0x0], [0x0])) == 0

    # no coefficient with variable selection, requires zero(::SimplePolynomial)
    p = SimplePolynomial(x^2 - x^2)
    @test maxdegree(p) == 0
    @test maxdegree(p, SimpleRealVariable{1,1}(1)) == 0
    @test maxdegree(p, SimpleComplexVariable{1,1}(1)) == 0
    @test mindegree(p) == 0
    @test mindegree(p, SimpleRealVariable{1,1}(1)) == 0
    @test mindegree(p, SimpleComplexVariable{1,1}(1)) == 0
    @test extdegree(p) == (0, 0)
    @test extdegree(p, SimpleRealVariable{1,1}(1)) == (0, 0)
    @test extdegree(p, SimpleComplexVariable{1,1}(1)) == (0, 0)
    q = SimplePolynomial(x * y + 2 + x^2 * y + x + y)
    @test maxdegree(q) == 3
    @test maxdegree(q, SimpleRealVariable{1,1}(1)) == 2
    @test maxdegree(q, SimpleComplexVariable{1,1}(1)) == 1
    @test mindegree(q) == 0
    @test mindegree(q, SimpleRealVariable{1,1}(1)) == 0
    @test mindegree(q, SimpleComplexVariable{1,1}(1)) == 0
    @test extdegree(q) == (0, 3)
    @test extdegree(q, SimpleRealVariable{1,1}(1)) == (0, 2)
    @test extdegree(q, SimpleComplexVariable{1,1}(1)) == (0, 1)

    @test coefficients(SimplePolynomial(x * y + 2 + 3x^2 * y + 4x + 6y)) == [2, 6, 4, 1, 3]
    @test coefficients(SimplePolynomial(x * y + 2 + 3x^2 * y + 4x + 6y),
        [SimpleMonomial{1,1}(UInt8[1], UInt8[0], UInt8[0]),
         SimpleMonomial{1,1}(UInt8[1], UInt8[2], UInt8[0]),
         SimpleMonomial{1,1}(UInt8[1], UInt8[1], UInt8[0]),
         SimpleMonomial{1,1}(UInt8[2], UInt8[1], UInt8[0]),
         SimpleMonomial{1,1}(UInt8[0], UInt8[1], UInt8[0]),
         SimpleMonomial{1,1}(UInt8[3], UInt8[0], UInt8[0])
        ]) == [4, 0, 1, 3, 6, 0]
    @test monomials(SimplePolynomial(4x^2 * y + x * y + 2x + 3))[1:1] == [constant_monomial(SimplePolynomial(x * y))]

    for p in [SimplePolynomial(polynomial([4, 9], [x, x * x])), SimplePolynomial(polynomial([9, 4], [x * x, x]))]
        @test coefficients(p) == [4, 9]
        @test monomials(p)[1] == SimpleMonomial{1,0}(UInt8[1])
        @test monomials(p)[2] == SimpleMonomial{1,0}(UInt8[2])
        @test monomials(p)[1:2][1] == SimpleMonomial{1,0}(UInt8[1])
        @test monomials(p)[1:2][2] == SimpleMonomial{1,0}(UInt8[2])
    end

    @test SimplePolynomial(x + y)' == SimplePolynomial(x + conj(y))
    @test transpose(SimplePolynomial(x + y)) == SimplePolynomial(x + y)
    @test transpose(SimplePolynomial(Term([1 2; 3 4], x^1))) == SimplePolynomial(Term([1 3; 2 4], x^1))

    # none of the rest
end
# maybe promotion?
@testset "Monomial index and iterator" begin
    @test SimplePolynomials.monomial_count(5, 4) == 126
    mons = monomials(3, 2, 0:4)
    @test length(mons) == 330
    @test monomial_index(mons[35]) == 35
    @test monomial_index(mons[257]) == 257
    alloc_test(let mons=mons; () -> monomial_index(mons[56]) end, 0)
    # mons[17] = z₁z₂, mons[25] = x₂z̄₁, mons[205] = x₂z₁z₂z̄₁
    @test monomial_index(mons[17], mons[25]) == 205
    # mons[7] = x₂, mons[287] = x₁x₂z₂²
    @test monomial_index(mons[7], SimpleMonomial{3,2}(sparsevec(UInt8[1], UInt8[1], 3),
                                                      sparsevec(UInt8[2], UInt8[2], 2),
                                                      sparsevec(UInt8[], UInt8[], 2))) == 287
    alloc_test(let mons=mons; () -> monomial_index(mons[7], mons[9], mons[15]) end, 0)

    @test monomial_index(SimpleRealVariable{3,5}(2)) == 13
    @test monomial_index(SimpleComplexVariable{3,5}(4)) == 8
    @test monomial_index(SimpleComplexVariable{3,5}(3, true)) == 4

    # mons[35] = x₁x₂, mons[113] = x₁x₂²
    @test monomial_index(mons[35], SimpleRealVariable{3,2}(2)) == 113
    # mons[110] = x₁x₂z₂
    @test monomial_index(mons[35], SimpleComplexVariable{3,2}(2)) == 110
    # mons[108] = x₁x₂z̄₂
    @test monomial_index(mons[35], SimpleComplexVariable{3,2}(2, true)) == 108

    @testset failfast=true "Monomial iterator and index consistency with DynamicPolynomials" begin
        DynamicPolynomials.@polyvar x[1:3]
        multiit = Iterators.product(0x0:0x4, 0x0:0x4, 0x0:0x4)
        for mindeg in 0x0:0x4, maxdeg in 0x0:0x4, minmultideg in multiit, maxmultideg in multiit
            minm, maxm = collect(minmultideg), collect(maxmultideg)
            if mindeg > maxdeg || any(minmultideg .> maxmultideg)
                @test_throws ArgumentError MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minm, maxm)
            else
                mi = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minm, maxm)
                exp = exponents.(monomials(x, Int(mindeg):Int(maxdeg), m -> all(minm .≤ exponents(m) .≤ maxm)))
                @test collect(mi) == exp
                @test length(mi) == length(exp)
                powers = Vector{UInt8}(undef, 3)
                for (i, mipow) in enumerate(mi)
                    @test exponents_from_index!(powers, mi, i)
                    if powers != mipow
                        println(mindeg, " - ", maxdeg, " - ", minmultideg, " - ", maxmultideg, ": ", i)
                    end
                    @test powers == mipow
                end
                @test !exponents_from_index!(powers, mi, length(mi) +1)
            end
        end
        mi = MonomialIterator{Graded{LexOrder}}(0x0, 0x4, [0x0, 0x0, 0x0], [0x4, 0x4, 0x4], true)
        @test_throws ArgumentError exponents_from_index!(Vector{UInt8}(undef, 4), mi, 1)
        powers = Vector{UInt8}(undef, 3)
        for (i, mipow) in enumerate(mi)
            exponents_from_index!(powers, i)
            @test powers == mipow
        end
    end

    @testset "Ranged monomial iterator, LazyMonomials" begin
        mons = monomials(3, 0, 2:5, minmultideg=[1, 0, 2], maxmultideg=[7, 4, 3])
        lm = LazyMonomials(3, 0, 2:5, minmultideg=[1, 0, 2], maxmultideg=[7, 4, 3])
        @test mons == lm
        @test mons[3:7] == lm[3:7]
        @test mons[3:7] == @view(lm[3:7])
        @test mons[6] == lm[6]
    end
end