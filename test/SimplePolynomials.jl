# This testsuite uses some of MP's tests (and some more), but does so explicitly, since SimplePolynomials on purpose does not
# support a lot of the things that you would want to do with polynomials.
using Test
using LinearAlgebra, SparseArrays

using MultivariatePolynomials
const MP = MultivariatePolynomials
import DynamicPolynomials, Combinatorics

using PolynomialOptimization.SimplePolynomials
import PolynomialOptimization.SimplePolynomials: Absent
using PolynomialOptimization.SimplePolynomials.MultivariateExponents

testdir = dirname(pathof(MP)) * "/../test"
include("$testdir/utils.jl")

@testset "Exponents" begin
    @testset "ExponentsAll" begin
        ea = ExponentsAll{5,Int32}()
        @test stack(Iterators.take(Iterators.drop(ea, 5), 7)) == Int[
            1  0  0  0  0  0  0
            0  0  0  0  0  0  0
            0  0  0  0  1  1  2
            0  0  1  2  0  1  0
            0  2  1  0  1  0  0
        ]
        @test ea[Int32(8)] == [0, 0, 0, 1, 1]
        @test_throws BoundsError ea[Int32(0)]
        counts, success = index_counts(ea, 8)
        @test success
        # note: in a fresh Julia session, this should be the whole counts. However, if something has already run (or the test
        # suite is executed more than once), the previous runs will have populated the cache some more, so counts might have
        # more rows.
        @test counts[1:9, :] == Int32[   1   1   1  1 1 1
                                         6   5   4  3 2 1
                                        21  15  10  6 3 1
                                        56  35  20 10 4 1
                                       126  70  35 15 5 1
                                       252 126  56 21 6 1
                                       462 210  84 28 7 1
                                       792 330 120 36 8 1
                                      1287 495 165 45 9 1]
        alloc_test(let ea=ea; () -> index_counts(ea, 8) end, 0)

        ei = Vector{Int}(undef, 5)
        eiter = similar(ei)
        @test_throws DimensionMismatch iterate!(@view(eiter[1:3]), ea)
        copyto!(eiter, first(ea))
        for (i, eitervec, eitervec2) in zip(Int32(1):Int32(500), veciter(ea), veciter(ea, similar(ei)))
            alloc_test(let ea=ea, ei=ei, i=i; () -> copyto!(ei, exponents_from_index(ea, i)) end, 0)
            alloc_test(let ea=ea, ei=ei; () -> exponents_to_index(ea, ei) end, 0)
            @test exponents_to_index(ea, ei) === i
            @test eiter == ei
            @test iterate!(eiter, ea)
            @test eitervec == ei
            @test eitervec2 == ei
        end
        @test degree_from_index(ea, Int32(2000)) == 9
    end
    @testset "ExponentsDegree" begin
        ed = ExponentsDegree{5,Int32}(2:7)
        @test length(ed) == 786
        @test length(unsafe, ed) == 786
        @test stack(Iterators.take(Iterators.drop(ed, 5), 7)) == Int[
            0  0  0  0  0  1  1
            0  1  1  1  2  0  0
            2  0  0  1  0  0  0
            0  0  1  0  0  0  1
            0  1  0  0  0  1  0
        ]
        @test ed[Int32(8)] == [0, 1, 0, 1, 0]
        @test_throws BoundsError ed[Int32(0)]
        @test_throws BoundsError ed[Int32(787)]
        # Checking index_counts does not make much sense, it is shared with the generic verison. Just ensure that no higher
        # degrees are created than what we had before (which was one higher than for 2000 -> 10)
        counts, success = index_counts(ed, 11)
        @test !success

        ei = Vector{Int}(undef, 5)
        copyto!(ei, exponents_from_index(ed, one(Int32)))
        @test ei == [0, 0, 0, 0, 2]
        eiter = similar(ei)
        @test_throws DimensionMismatch iterate!(@view(eiter[1:3]), ed)
        copyto!(eiter, first(ed))
        for (i, eitervec, eitervec2) in zip(Int32(1):Int32(786), veciter(ed), veciter(ed, similar(ei)))
            alloc_test(let ed=ed, ei=ei, i=i; () -> copyto!(ei, exponents_from_index(ed, i)) end, 0)
            alloc_test(let ed=ed, ei=ei; () -> exponents_to_index(ed, ei) end, 0)
            @test exponents_to_index(ed, ei) === i
            @test eiter == ei
            @test iterate!(eiter, ed) == (i != 786)
            @test eitervec == ei
            @test eitervec2 == ei
        end
        @test degree_from_index(ed, Int32(786)) == 7
        @test degree_from_index(ed, Int32(787)) == 8
    end
    @testset "ExponentsMultideg" begin
        emd = ExponentsMultideg{5,Int32}(3:7, [1, 0, 0, 1, 0], [5, 3, 6, 7, 2])
        @test length(emd) == 223
        @test length(unsafe, emd) == 223
        @test stack(Iterators.take(Iterators.drop(emd, 5), 7)) == Int[
            1  1  1  1  1  1  1
            0  0  0  0  0  0  1
            0  0  0  1  1  2  0
            1  2  3  1  2  1  1
            2  1  0  1  0  0  1
        ]
        @test emd[Int32(8)] == [1, 0, 0, 3, 0]
        @test_throws BoundsError emd[Int32(0)]
        @test_throws BoundsError emd[Int32(224)]
        @test !index_counts(emd, 8)[2]
        counts, success = index_counts(emd, 7)
        @test success
        @test counts == Int32[  0    0   0   0  1  0
                                0    1   1   1  2  0
                                1    5   4   3  3  0
                                6   15  10   6  3  0
                               21   34  19   9  3  0
                               55   64  31  12  3  0
                              119  106  46  15  3  0
                              224  160  64  18  3  0]

        ei = Vector{Int}(undef, 5)
        copyto!(ei, exponents_from_index(emd, one(Int32)))
        @test ei == [1, 0, 0, 1, 1]
        eiter = similar(ei)
        @test_throws DimensionMismatch iterate!(@view(eiter[1:3]), emd)
        copyto!(eiter, first(emd))
        for (i, eitervec, eitervec2) in zip(Int32(1):Int32(223), veciter(emd), veciter(emd, similar(ei)))
            alloc_test(let emd=emd, ei=ei, i=i; () -> copyto!(ei, exponents_from_index(emd, i)) end, 0)
            alloc_test(let emd=emd, ei=ei; () -> exponents_to_index(emd, ei) end, 0)
            @test exponents_to_index(emd, ei) === i
            @test eiter == ei
            @test iterate!(eiter, emd) == (i != 223)
            @test eitervec == ei
            @test eitervec2 == ei
        end
        @test degree_from_index(emd, Int32(223)) == 7
        @test degree_from_index(emd, Int32(224)) == 8
    end
end

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
        z = SimpleComplexVariable{0,typemax(t)>>1}(i)
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

    @test nvariables(SimpleMonomialVector{3,4}([0 2; 0 0; 1 1], [0 2; 1 0; 0 0; 1 0], [0 0; 0 0; 1 0; 0 0])) == 11

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
    @test_throws ArgumentError monomials(2, 0, 0x1:0x0)
    X = [SimpleMonomial{2,0}(UInt8[0, 2]),
         SimpleMonomial{2,0}(UInt8[1, 1]),
         SimpleMonomial{2,0}(UInt8[2, 0])]
    monos = monomials(2, 0, 0x2:0x2)
    @test length(monos) == length(X)
    for (x, m) in zip(X, monos)
        @test m == x
    end
    monos = monomials(2, 0, 0x2:0x2, representation=:sparse)
    @test length(monos) == length(X)
    X = [SimpleMonomial{2,0}(sparsevec(UInt8[2], UInt8[2], 2)),
         SimpleMonomial{2,0}(sparsevec(UInt8[1, 2], UInt8[1, 1], 2)),
         SimpleMonomial{2,0}(sparsevec(UInt8[1], UInt8[2], 2))]
    for (x, m) in zip(X, monos)
        @test m == x
    end
    # we don't provide monomial_vector_type

    for md in (identity, sparse)
        X = SimpleMonomialVector{2,0}(md(UInt8[1 0 1; 0 0 1]))
        @test X == collect(X)
        @test nvariables(X) == 2
        @test variables(X)[1] == SimpleRealVariable{2,0}(1)
        @test variables(X)[2] == SimpleRealVariable{2,0}(2)
        @test X[2:3][1] == SimpleMonomial{2,0}([0x1, 0x0])
        @test X[2:3][2] == SimpleMonomial{2,0}([0x1, 0x1])
    end

    @test merge_monomial_vectors(SimpleMonomialVector{2,0,UInt}, [
        SimpleMonomialVector{2,0}([1; 1;; 1; 0]), SimpleMonomialVector{2,0}([2; 1;; 1; 0])
    ]) == SimpleMonomialVector{2,0}([1; 0;; 1; 1;; 2; 1])
    @test @inferred(merge_monomial_vectors(SimpleMonomialVector{2,0,UInt,Matrix{UInt}}, [
        SimpleMonomialVector{2,0}([1; 1;; 1; 0]), SimpleMonomialVector{2,0}(sparse([2; 1;; 1; 0]))
    ])) == SimpleMonomialVector{2,0}([1; 0;; 1; 1;; 2; 1])
    @test @inferred(merge_monomial_vectors(SimpleMonomialVector{2,0,UInt,SparseMatrixCSC{UInt,UInt8}}, [
        SimpleMonomialVector{2,0}([1; 1;; 1; 0]), SimpleMonomialVector{2,0}(sparse([2; 1;; 1; 0]))
    ])) == SimpleMonomialVector{2,0}([1; 0;; 1; 1;; 2; 1])

    @test monomials(1, 0, 0x1:0x3) == SimpleMonomialVector{1,0}(UInt8[1 2 3])

    @testset "monomials" begin
        @test monomials(3, 0, 0x0:0x3) == SimpleMonomialVector{3,0}(
            copy(transpose(UInt8[
                0 0 0
                0 0 1
                0 1 0
                1 1 1
                1 2 0
                2 0 1
                0 2 1
                0 3 0
                2 0 0
                0 2 0
                1 0 1
                0 0 3
                1 0 0
                0 0 2
                0 1 1
                1 1 0
                0 1 2
                1 0 2
                2 1 0
                3 0 0
            ]))
        )
    end

    @test_throws ArgumentError monomials(1, 0, -1:0)

    # new tests
    @test effective_nvariables(monomials(2, 3, 0x0:0x2)) == 8
    @test effective_nvariables(monomials(2, 3, 0x0:0x1), monomials(2, 3, 0:2, representation=:sparse)) == 8
    @test effective_nvariables(monomials(2, 3, 0x0:0x0)) == 0
    @test effective_nvariables(SimpleMonomialVector{2,0}([0 1; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([1 2; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([0 1; 1 0])) == 2
    @test effective_nvariables(SimpleMonomialVector{2,0}([0 1; 0 0]), SimpleMonomialVector{2,0}([0 1; 0 0])) == 1
    @test effective_nvariables(SimpleMonomialVector{2,0}([0 1; 0 0]), SimpleMonomialVector{2,0}([0 1; 0 1])) == 2
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
    @test terms(SimplePolynomial(polynomial([1, x^2, x, 2x^2]), representation=:dense, max_exponent=typemax(UInt16))) ==
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
                @test_throws ArgumentError MonomialIterator(mindeg, maxdeg, minm, maxm)
            else
                mi = MonomialIterator(mindeg, maxdeg, minm, maxm)
                exp = MP.exponents.(monomials(x, Int(mindeg):Int(maxdeg), m -> all(minm .≤ MP.exponents(m) .≤ maxm)))
                @test collect(mi) == exp
                @test length(mi) == length(exp)
                exponents = Vector{UInt8}(undef, 3)
                for (i, mipow) in enumerate(mi)
                    @test exponents_from_index!(exponents, mi, i)
                    mind = monomial_index(exponents, mi)
                    if exponents != mipow || mind != i
                        println(mindeg, " - ", maxdeg, " - ", minmultideg, " - ", maxmultideg, ": index ", i,
                            " gives exponents ", exponents, " which claim index ", mind)
                    end
                    @test exponents == mipow
                    @test mind == i
                end
                @test !exponents_from_index!(exponents, mi, length(mi) +1)
                @test iszero(monomial_index(maxm .+ one(eltype(maxm)), mi))
            end
        end
        mi = MonomialIterator(0x0, 0x4, [0x0, 0x0, 0x0], [0x4, 0x4, 0x4], ownexponents)
        @test_throws ArgumentError exponents_from_index!(Vector{UInt8}(undef, 4), mi, 1)
        exponents = Vector{UInt8}(undef, 3)
        for (i, mipow) in enumerate(mi)
            exponents_from_index!(exponents, i)
            @test exponents == mipow
        end
    end

    @testset "Ranged monomial iterator, LazyMonomials" begin
        mons = monomials(3, 0, 0x2:0x5, minmultideg=UInt8[1, 0, 2], maxmultideg=UInt8[7, 4, 3])
        lm = LazyMonomials{3,0}(0x2:0x5, minmultideg=UInt8[1, 0, 2], maxmultideg=UInt8[7, 4, 3])
        @test mons == lm
        @test mons[3:7] == lm[3:7]
        @test mons[3:7] == @view(lm[3:7])
        @test mons[6] == lm[6]
        ind = [1, 2, 4, 7, 8]
        @test mons[ind] == lm[ind]
        @test mons[ind] == @view(lm[ind])
        @test mons[3:7][[1, 3, 4]] == mons[[3, 5, 6]]
        @test mons[ind][2:4] == mons[ind[2:4]]
    end

    @testset "LazyMonomials effective variables" begin
        lm = LazyMonomials{3,0}(0x2:0x5, minmultideg=UInt8[1, 0, 2], maxmultideg=UInt8[7, 4, 3])
        @test collect(effective_variables(lm)) == SimpleVariable{3,0}.(1:3)
        @test effective_nvariables(lm) == 3
        lm = LazyMonomials{3,0}(0x2:0x5, minmultideg=UInt8[1, 0, 4], maxmultideg=UInt8[7, 4, 8])
        @test collect(effective_variables(lm)) == SimpleVariable{3,0}.([1, 3])
        @test effective_nvariables(lm) == 2
        lm = LazyMonomials{4,0}(0:5, minmultideg=[2, 0, 0, 0], maxmultideg=[4, 5, 6, 7])
        for rstart in 1:length(lm), rend in rstart:length(lm)
            sub = @view(lm[rstart:rend])
            correct_efvars = Set{SimpleVariable{4,0}}()
            for mon in sub, (var, pow) in mon
                push!(correct_efvars, var)
            end
            efvars = collect(effective_variables(sub))
            @test efvars == sort!(collect(correct_efvars))
            @test effective_nvariables(sub) == length(correct_efvars)
        end
        for subs in Combinatorics.powerset(1:length(lm), 0, 4)
            sub = @view(lm[subs])
            correct_efvars = Set{SimpleVariable{4,0}}()
            for mon in sub, (var, pow) in mon
                push!(correct_efvars, var)
            end
            efvars = collect(effective_variables(sub))
            @test efvars == sort!(collect(correct_efvars))
            @test effective_nvariables(sub) == length(correct_efvars)
        end
    end
end