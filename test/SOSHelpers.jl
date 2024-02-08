using Test
using MultivariatePolynomials, PolynomialOptimization, PolynomialOptimization.SimplePolynomials
using PolynomialOptimization: sos_add_matrix!, SOSPSDIterable, FastKey, sos_add_equality!, sort_along!

# These are the low-level tests. For every kind of possible problem, we must check that the appropriate solver methods are
# called with exactly the right data.
# We must check multiple cases:
# - only real-valued monomials in the grouping, regardless of whether the solver supports complex PSD. Both with real-valued
#   and complex-valued constraint terms (though of course, the constraint itself must be real-valued)
# - mixing them, where the solver supports complex PSD
# - mixing them, where the solver doesn't support complex PSD
# both for SOS and equality constraints.
# In the following setups:
# - scalar constraint polynomial where
#   - constraint polynomial only contains a single term
#   - constraint polynomial contains multiple terms
#   - also check that if the constraint polynomial is not real-valued, an assertion is raised
#   - grouping has length 1, 2 (with quadratic cone supported and unsupported), ≥ 3
# - matrix-valued constraint polynomial where
#   - constraint polynomial only contains a single term
#   - constraint polynomial contains multiple terms
#   - also check that if the constraint polynomial is not real-valued on the diagonal, an assertion is raised; explicitly check
#     complex-valued non-diagonal constraints, which must be allowed
#   - grouping has length 1 (with quadratic cone supported and unsupported), ≥ 2
# For PSD constraints, check different sos_solver_psd_indextype:
# - tuple with [1] linear indexing with some type that for sure would require conversion, say UInt16
#              [2] :U and :L
#              [3] some weird offset in a different type than [1], say 42
# - tuple with [1] matrix indexing with two different types, say (UInt32, UInt16)
#              [2] :U and :L
#              [3] some weird offset in a different type than both of [1], say 42
# - symbol :L, :LS, :U, :US, :F

# These are our test cases, always in the ring with 4 reals, 2 complexes:

# scalar
# 01: grouping = [x₂²x₃], constraint = 17x₁x₃²            - scalar scalar, fallback to scalar vector
# 03: grouping = [x₂²x₃], constraint = 24x₂z₁z̄₁           - scalar scalar, fallback to scalar vector
# 04: grouping = [x₂²x₃], constraint = (17+8im)x₁x₃²      - assertion failure
# 05: grouping = [x₂²x₃], constraint = 3im*z₁z̄₁           - assertion failure
# 06: grouping = [x₂²x₃], constraint = 2x₁ + 8x₂x₄²       - scalar vector
# 07: grouping = [x₂²x₃], constraint = 3(z₁ + z̄₁)         - scalar vector
# 08: grouping = [x₂²x₃], constraint = 7im(z₁z̄₂² - z̄₁z₂²) - scalar vector
# 09: grouping = [x₂²x₃], constraint = 5im(z₁z̄₂² + z₁z̄₁)  - assertion failure (z₁z̄₁ is detectable)
# 10: grouping = [x₂²x₃], constraint = 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁ - scalar vector
# 11: grouping = [x₂²z₁], constraint = 17x₁x₃²            - scalar scalar and scalar vector
# 13: grouping = [x₂²z₁], constraint = 24x₂z₁z̄₁           - scalar scalar and scalar vector
# 14: grouping = [x₂²z₁], constraint = (17+8im)x₁x₃²      - assertion failure
# 15: grouping = [x₂²z₁], constraint = 3im*z₁z̄₁           - assertion failure
# 16: grouping = [x₂²z₁], constraint = 2x₁ + 8x₂x₄²       - scalar vector
# 17: grouping = [x₂²z₁], constraint = 3(z₁ + z̄₁)         - scalar vector
# 18: grouping = [x₂²z₁], constraint = 7im(z₁z̄₂² - z̄₁z₂²) - scalar vector
# 19: grouping = [x₂²z₁], constraint = 5im(z₁z̄₂² + z₁z̄₁)  - assertion failure (z₁z̄₁ is detectable)
# 20: grouping = [x₂²z₁], constraint = 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁ - scalar vector

# quadratic
# 21: grouping = [x₁x₄², x₂²x₃³], constraint = 17x₁x₃²            - quadratic scalar one rhs, fallback to vector
# 23: grouping = [x₁x₄², x₂²x₃³], constraint = 24x₂z₁z̄₁           - quadratic scalar one rhs, fallback to vector
# 24: grouping = [x₁x₄², x₂²x₃³], constraint = (17+8im)x₁x₃²      - assertion failure
# 25: grouping = [x₁x₄², x₂²x₃³], constraint = 3im*z₁z̄₁           - assertion failure
# 26: grouping = [x₁x₄², x₂²x₃³], constraint = 2x₁ + 8x₂x₄²       - quadratic vector one rhs
# 27: grouping = [x₁x₄², x₂²x₃³], constraint = 3(z₁ + z̄₁)         - quadratic vector one rhs
# 28: grouping = [x₁x₄², x₂²x₃³], constraint = 7im(z₁z̄₂² - z̄₁z₂²) - quadratic vector one rhs
# 29: grouping = [x₁x₄², x₂²x₃³], constraint = 5im(z₁z̄₂² + z₁z̄₁)  - assertion failure (z₁z̄₁ is detectable)
# 30: grouping = [x₁x₄², x₂²x₃³], constraint = 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁ - quadratic vector two rhs
# 31: grouping = [x₁x₄², x₂²z₁],  constraint = 17x₁x₃²            - quadratic scalar two rhs, fallback to vector
# 33: grouping = [x₁x₄², x₂²z₁],  constraint = 24x₂z₁z̄₁           - quadratic scalar two rhs, fallback to vector
# 34: grouping = [x₁x₄², x₂²z₁],  constraint = (17+8im)x₁x₃²      - assertion failure
# 35: grouping = [x₁x₄², x₂²z₁],  constraint = 3im*z₁z̄₁           - assertion failure
# 36: grouping = [x₁x₄², x₂²z₁],  constraint = 2x₁ + 8x₂x₄²       - quadratic vector two rhs
# 37: grouping = [x₁x₄², x₂²z₁],  constraint = 3(z₁ + z̄₁)         - quadratic vector two rhs
# 38: grouping = [x₁x₄², x₂²z₁],  constraint = 7im(z₁z̄₂² - z̄₁z₂²) - quadratic vector two rhs
# 39: grouping = [x₁x₄², x₂²z₁],  constraint = 5im(z₁z̄₂² + z₁z̄₁)  - assertion failure (z₁z̄₁ is detectable)
# 40: grouping = [x₁x₄², x₂²z₁],  constraint = 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁ - quadratic vector two rhs
# 41: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8x₂²
#                                       8x₂²       6x₂x₄²]        - quadratic scalar one rhs, fallback to vector
# 42: grouping = [x₂²z₂], constraint = [0 0; 0 0]                 - nothing
# 43: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  0
#                                       0          6x₂x₄²]        - scalar scalar, fallback to vector (2x)
# 44: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  0
#                                       0          0]             - scalar scalar, fallback to vector
# 45: grouping = [x₂²z₂], constraint = [0  0
#                                       0  6x₂x₄²]                - scalar scalar, fallback to vector
# 46: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8x₂²
#                                       8x₂²       0]             - quadratic simplified scalar, fallbacks to vector, quadratic vector one rhs
# 47: grouping = [x₂²z₂], constraint = [0     8x₂²
#                                       8x₂²  6x₂x₄²]             - quadratic simplified scalar, fallbacks to vector, quadratic vector one rhs
# 48: grouping = [x₂²z₂], constraint = [17x₁z₁  8x₂²
#                                       8x₂²    6x₂x₄²]           - assertion failure
# 49: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8x₂²
#                                       8x₂²       6x₄²z₂]        - assertion failure
# 50: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8im*x₂²
#                                       -8im*x₂²   6x₂x₄²]        - quadratic scalar one rhs, fallback to vector
# 51: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8z₂²
#                                       8z̄₂²       6x₂x₄²]        - quadratic scalar two rhs, fallback to vector
# 52: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁  8im*z₂²
#                                       -8im*z̄₂²   6x₂x₄²]        - quadratic scalar two rhs, fallback to vector
# 53: grouping = [x₂²z₂], constraint = [17x₃²z₁z̄₁    (8+2im)*z₂²
#                                       (8-2im)*z̄₂²  6x₂x₄²]      - quadratic vector two rhs
# 54: grouping = [x₂²z₂], constraint = [5x₂²+17x₃²z₁z̄₁                (8-2im)*z̄₂²-8z₂²-3im*x₁+z₁z̄₁
#                                       (8+2im)*z₂²-8z̄₂²+3im*x₁+z₁z̄₁  6x₂x₄²]         - quadratic vector two rhs
# 55: grouping = [x₂²z₂], constraint = [5x₂²z₁+17im*x₃²z̄₁+5x₂²z̄₁-17im*x₃²z₁ (8-2im)*z̄₂²+(8+2im)z₂²
#                                       (8-2im)*z̄₂²+(8+2im)z₂²              6x₂x₄²]              - quadratic vector one rhs

# We always use a matrix-valued constraint, as this fallback will then work for the scalar case also.
# Exception is the very first problem, which serves as a test of the fallback.
# 61: grouping = [1, x₂, x₁x₄², x₂²x₃³], constraint = 3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂
# 62: grouping = [1, x₂, x₁x₄², x₂²x₃³], constraint = [3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂  17x₄                     7x₁+8z₂z̄₂
#                                                      17x₄                               0                        23im*x₃z₂-23im*x₃z̄₂
#                                                      7x₁+8z₂z̄₂                          -23im*x₃z̄₂+23im*x₃z₂     6x₂x₄]
# 63: grouping = [1, x₂, x₁x₄², x₂²x₃³], constraint = [3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂  (8-2im)*z̄₂²-8z₂²-3im*x₁+z₁z̄₁  7x₁+8z₂z̄₂
#                                                      (8+2im)*z₂²-8z̄₂²+3im*x₁+z₁z̄₁       0                             23im*x₃z₂-23im*x₃z̄₂
#                                                      7x₁+8z₂z̄₂                          -23im*x₃z̄₂+23im*x₃z₂          6x₂x₄]

abstract type SolverSetup end

mutable struct SolverSetupScalar{SupportScalar} <: SolverSetup
    lastcall::Symbol
    const instance::Int
end

mutable struct SolverSetupQuadratic{SupportScalar} <: SolverSetup
    lastcall::Symbol
    const instance::Int
end

mutable struct SolverSetupQuadraticSimplified{SupportScalar} <: SolverSetup
    lastcall::Symbol
    const instance::Int
end

abstract type SolverSetupPSD{Triangle,SupportsComplex} <: SolverSetup end

mutable struct SolverSetupPSDDictLinear{Index1,Triangle,SupportsComplex} <: SolverSetupPSD{Triangle,SupportsComplex}
    lastcall::Symbol
    const instance::Int
end

mutable struct SolverSetupPSDDictExplicit{Index1,Index2,Triangle,SupportsComplex} <: SolverSetupPSD{Triangle,SupportsComplex}
    lastcall::Symbol
    const instance::Int
end

mutable struct SolverSetupPSDLinear{Triangle,SupportsComplex} <: SolverSetupPSD{Triangle,SupportsComplex}
    lastcall::Symbol
    const instance::Int
end

mutable struct SolverSetupFree <: SolverSetup
    lastcall::Symbol
    available::UInt
    used::UInt
    lastnum::Int
    const instance::Int

    SolverSetupFree(lastcall::Symbol, instance::Int) = new(lastcall, zero(UInt), zero(UInt), typemin(Int), instance)
end

# make sure we have a very distinctive mindex type
const monomial_index_shift = 48
PolynomialOptimization.sos_solver_mindex(::SolverSetup, monomials::SimpleMonomial...) =
    BigInt(monomial_index(monomials...) + monomial_index_shift)

const sospsd_offset = UInt8(17)
sos_get_tri(::SolverSetupPSD{Triangle}) where {Triangle} = Triangle

PolynomialOptimization.sos_solver_psd_indextype(::SolverSetupPSDDictLinear{I,T}) where {I,T} = (I, T, sospsd_offset)
PolynomialOptimization.sos_solver_psd_indextype(::SolverSetupPSDDictExplicit{I1,I2,T}) where {I1,I2,T} =
    (Tuple{I1,I2}, T, sospsd_offset)
PolynomialOptimization.sos_solver_psd_indextype(::SolverSetupPSDLinear{T}) where {T} = T
PolynomialOptimization.sos_solver_psd_supports_complex(::SolverSetupPSD{<:Any,C}) where {C} = C

# just to check the tests
gi(computed_index, computed_value, (x₁, x₂, x₃, x₄), supposed_value::Int16) =
    @test monomial_index(SimpleMonomial{4,2}([x₁, x₂, x₃, x₄], [0, 0], [0, 0])) + monomial_index_shift == computed_index &&
        supposed_value === computed_value
function gi(reim::Symbol, computed_index, computed_value, (x₁, x₂, x₃, x₄, z₁, z₂, z̄₁, z̄₂), supposed_value::Int16)
    m = SimpleMonomial{4,2}([x₁, x₂, x₃, x₄], [z₁, z₂], [z̄₁, z̄₂])
    if reim === :re
        @test monomial_index(PolynomialOptimization.canonicalize(m)) + monomial_index_shift == computed_index &&
            supposed_value === computed_value
    elseif reim === :im
        @assert(!isreal(m))
        @test monomial_index(conj(PolynomialOptimization.canonicalize(m))) + monomial_index_shift == computed_index &&
            (PolynomialOptimization.iscanonical(m) ? supposed_value : -supposed_value) === computed_value
    else
        @assert(false)
    end
end


function PolynomialOptimization.sos_solver_add_scalar!(state::SolverSetupScalar{true}, index::BigInt, value::Int16)
    @test state.lastcall === :none || (state.instance == 43 && state.lastcall === :add_scalar)
    if state.instance == 1
        # x₂²x₃ * 17x₁x₃² * x₂²x₃ = 17x₁x₂⁴x₃⁴
        gi(index, value, (1, 4, 4, 0), Int16(17))
    elseif state.instance == 3
        # x₂²x₃ * 24x₂z₁z̄₁ * x₂²x₃ = 24x₂⁵x₃²z₁z̄₁
        gi(:re, index, value, (0, 5, 2, 0, 1, 0, 1, 0), Int16(24))
    elseif state.instance == 11
        # x₂²z₁ * 17x₁x₃² * x₂²z̄₁ = 17x₁x₂⁴x₃²z₁z̄₁
        gi(:re, index, value, (1, 4, 2, 0, 1, 0, 1, 0), Int16(17))
    elseif state.instance == 13
        # x₂²z₁ * 24x₂z₁z̄₁ * x₂²z̄₁ = 24x₂⁵z₁²z̄₁²
        gi(:re, index, value, (0, 5, 0, 0, 2, 0, 2, 0), Int16(24))
    elseif state.instance == 43
        # x₂²z₂ * [17x₃²z₁z̄₁  0     ] * x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z₁z̄₁  0          ]
        #         [0          6x₂x₄²]           [0                 6x₂⁵x₄²z₂z̄₂]
        if state.lastcall === :none
            gi(:re, index, value, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        else
            state.lastcall = :add_scalar2
            gi(:re, index, value, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
            return
        end
    elseif state.instance == 44
        # x₂²z₂ * [17x₃²z₁z̄₁  0] * x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z₁z̄₁  0]
        #         [0          0]           [0                 0]
        gi(:re, index, value, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
    elseif state.instance == 45
        # x₂²z₂ * [0  0     ] * x₂²z̄₂ = [0  0          ]
        #         [0  6x₂x₄²]           [0  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index, value, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
    else
        @test false
    end
    state.lastcall = :add_scalar
end

function PolynomialOptimization.sos_solver_add_scalar!(state::SolverSetupScalar{false}, index::AbstractVector{BigInt},
    value::AbstractVector{Int16})
    @test length(index) == length(value)
    @test state.lastcall === :none || (state.instance == 43 && state.lastcall === :add_scalar)
    if state.instance == 1
        @test length(index) == 1
        # x₂²x₃ * 17x₁x₃² * x₂²x₃ = 17x₁x₂⁴x₃⁴
        gi(index[1], value[1], (1, 4, 4, 0), Int16(17))
    elseif state.instance == 3
        @test length(index) == 1
        # x₂²x₃ * 24x₂z₁z̄₁ * x₂²x₃ = 24x₂⁵x₃²z₁z̄₁
        gi(:re, index[1], value[1], (0, 5, 2, 0, 1, 0, 1, 0), Int16(24))
    elseif state.instance == 6
        @test length(index) == 2
        # x₂²x₃ * (2x₁ + 8x₂x₄²) * x₂²x₃ = 2x₁x₂⁴x₃² + 8x₂⁵x₃²x₄²
        gi(index[1], value[1], (1, 4, 2, 0), Int16(2))
        gi(index[2], value[2], (0, 5, 2, 2), Int16(8))
    elseif state.instance == 7
        @test length(index) == 1
        # x₂²x₃ * 3(z₁ + z̄₁) * x₂²x₃ = 3x₂⁴x₃²z₁ + 3x₂⁴x₃²z̄₁
        gi(:re, index[1], value[1], (0, 4, 2, 0, 1, 0, 0, 0), Int16(6))
    elseif state.instance == 8
        @test length(index) == 1
        # x₂²x₃ * 7im(z₁z̄₂² - z̄₁z₂²) * x₂²x₃ = 7im*x₂⁴x₃²z₁z̄₂² - 7im*x₂⁴x₃²z̄₁z₂²
        gi(:im, index[1], value[1], (0, 4, 2, 0, 1, 0, 0, 2), Int16(-14))
    elseif state.instance == 10
        @test length(index) == 6
        # x₂²x₃ * (3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁) * x₂²x₃
        # = 3x₁x₂⁴x₃² + 5x₂⁵x₃²x₄² + 4x₂⁴x₃²z₂z̄₂ + 7x₂⁴x₃²z₁²z̄₂ + 7x₂⁴x₃²z̄₁²z₂ + (4+8im)*x₂⁴x₃²z₁ + (4-8im)*x₂⁴x₃²z̄₁
        gi(index[1], value[1], (1, 4, 2, 0), Int16(3))
        gi(index[2], value[2], (0, 5, 2, 2), Int16(5))
        gi(:re, index[3], value[3], (0, 4, 2, 0, 0, 1, 0, 1), Int16(4))
        gi(:re, index[4], value[4], (0, 4, 2, 0, 2, 0, 0, 1), Int16(14))
        gi(:re, index[5], value[5], (0, 4, 2, 0, 1, 0, 0, 0), Int16(8))
        gi(:im, index[6], value[6], (0, 4, 2, 0, 1, 0, 0, 0), Int16(-16))
    elseif state.instance == 11
        @test length(index) == 1
        # x₂²z₁ * 17x₁x₃² * x₂²z̄₁ = 17x₁x₂⁴x₃²z₁z̄₁
        gi(:re, index[1], value[1], (1, 4, 2, 0, 1, 0, 1, 0), Int16(17))
    elseif state.instance == 13
        @test length(index) == 1
        # x₂²z₁ * 24x₂z₁z̄₁ * x₂²z̄₁ = 24x₂⁵z₁²z̄₁²
        gi(:re, index[1], value[1], (0, 5, 0, 0, 2, 0, 2, 0), Int16(24))
    elseif state.instance == 16
        @test length(index) == 2
        # x₂²z₁ * (2x₁ + 8x₂x₄²) * x₂²z̄₁ = 2x₁x₂⁴z₁z̄₁ + 8x₂⁵x₄²z₁z̄₁
        gi(:re, index[1], value[1], (1, 4, 0, 0, 1, 0, 1, 0), Int16(2))
        gi(:re, index[2], value[2], (0, 5, 0, 2, 1, 0, 1, 0), Int16(8))
    elseif state.instance == 17
        @test length(index) == 1
        # x₂²z₁ * 3(z₁ + z̄₁) * x₂²z̄₁ = 3x₂⁴z₁²z̄₁ + 3x₂⁴z₁z̄₁²
        gi(:re, index[1], value[1], (0, 4, 0, 0, 2, 0, 1, 0), Int16(6))
    elseif state.instance == 18
        @test length(index) == 1
        # x₂²z₁ * 7im(z₁z̄₂² - z̄₁z₂²) * x₂²z̄₁ = 7im*x₂⁴z₁²z̄₁z̄₂² - 7im*x₂⁴z₁z₂²z̄₁²
        gi(:im, index[1], value[1], (0, 4, 0, 0, 2, 0, 1, 2), Int16(-14))
    elseif state.instance == 20
        @test length(index) == 6
        # x₂²z₁ * (3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁) * x₂²z̄₁
        # = 3x₁x₂⁴z₁z̄₁ + 5x₂⁵x₄²z₁z̄₁ + 4x₂⁴z₁z₂z̄₁z̄₂ + 7x₂⁴z₁³z̄₁z̄₂ + 7x₂⁴z₁z₂z̄₁³ + (4+8im)*x₂⁴z₁²z̄₁ + (4-8im)*x₂⁴z₁z̄₁²
        #   5623         32171         9249           17443         17461         4648               4653
        gi(:re, index[1], value[1], (1, 4, 0, 0, 1, 0, 1, 0), Int16(3))
        gi(:re, index[2], value[2], (0, 5, 0, 2, 1, 0, 1, 0), Int16(5))
        gi(:re, index[3], value[3], (0, 4, 0, 0, 1, 1, 1, 1), Int16(4))
        gi(:re, index[4], value[4], (0, 4, 0, 0, 3, 0, 1, 1), Int16(14))
        gi(:re, index[5], value[5], (0, 4, 0, 0, 2, 0, 1, 0), Int16(8))
        gi(:im, index[6], value[6], (0, 4, 0, 0, 2, 0, 1, 0), Int16(-16))
    elseif state.instance == 43
        @test length(index) == 1
        # x₂²z₂ * [17x₃²z₁z̄₁  0     ] * x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z₁z̄₁  0          ]
        #         [0          6x₂x₄²]           [0                 6x₂⁵x₄²z₂z̄₂]
        if state.lastcall === :none
            gi(:re, index[1], value[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        else
            state.lastcall = :add_scalar2
            gi(:re, index[1], value[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
            return
        end
    elseif state.instance == 44
        @test length(index) == 1
        # x₂²z₂ * [17x₃²z₁z̄₁  0] * x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z₁z̄₁  0]
        #         [0          0]           [0                 0]
        gi(:re, index[1], value[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
    elseif state.instance == 45
        @test length(index) == 1
        # x₂²z₂ * [0  0     ] * x₂²z̄₂ = [0  0          ]
        #         [0  6x₂x₄²]           [0  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index[1], value[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
    else
        @test false
    end
    state.lastcall = :add_scalar
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadratic{true}, index₁::BigInt, value₁::Int16,
    index₂::BigInt, value₂::Int16, (index₃, value₃)::Tuple{BigInt,Int16})
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 21
        # [x₁x₄²; x₂²x₃³] * 17x₁x₃² * [x₁x₄² x₂²x₃³] = 17 [x₁³x₃²x₄⁴      x₁²x₂²x₃⁵x₄²
        #                                                  x₁²x₂²x₃⁵x₄²   x₁x₂⁴x₃⁸]
        # ↪ (17x₁³x₃²x₄⁴) * (17x₁x₂⁴x₃⁸) ≥ (17x₁²x₂²x₃⁵x₄²)²
        gi(index₁, value₁, (3, 0, 2, 4), Int16(17))
        gi(index₂, value₂, (1, 4, 8, 0), Int16(17))
        gi(index₃, value₃, (2, 2, 5, 2), Int16(17))
    elseif state.instance == 23
        # [x₁x₄²; x₂²x₃³] * 24x₂z₁z̄₁ * [x₁x₄² x₂²x₃³] = 24 [x₁²x₂x₄⁴z₁z̄₁      x₁x₂³x₃³x₄²z₁z̄₁
        #                                                   x₁x₂³x₃³x₄²z₁z̄₁   x₂⁵x₃⁶z₁z̄₁]
        # ↪ (24x₁²x₂x₄⁴z₁z̄₁) * (24x₂⁵x₃⁶z₁z̄₁) ≥ (24x₁x₂³x₃³x₄²z₁z̄₁)²
        gi(:re, index₁, value₁, (2, 1, 0, 4, 1, 0, 1, 0), Int16(24))
        gi(:re, index₂, value₂, (0, 5, 6, 0, 1, 0, 1, 0), Int16(24))
        gi(:re, index₃, value₃, (1, 3, 3, 2, 1, 0, 1, 0), Int16(24))
    elseif state.instance == 41
        # x₂²z₂ [17x₃²z₁z̄₁  8x₂²  ] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8x₂⁶z₂z̄₂
        #       [8x₂²       6x₂x₄²]          8x₂⁶z₂z̄₂           6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁, value₁, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂, value₂, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃, value₃, (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 50
        # x₂²z₂ [17x₃²z₁z̄₁  8im*x₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8im*x₂⁶z₂z̄₂
        #       [-8im*x₂²   6x₂x₄² ]          -8im*x₂⁶z₂z̄₂       6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁, value₁, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂, value₂, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃, value₃, (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadratic{true}, index₁::BigInt, value₁::Int16,
    index₂::BigInt, value₂::Int16, (index₃, value₃)::Tuple{BigInt,Int16}, (index₄, value₄)::Tuple{BigInt,Int16})
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 31
        # [x₁x₄²; x₂²z₁] * 17x₁x₃² * [x₁x₄² x₂²z̄₁] = 17 [x₁³x₃²x₄⁴        x₁²x₂²x₃²x₄²z̄₁
        #                                                x₁²x₂²x₃²x₄²z₁   x₁x₂⁴x₃²z₁z̄₁]
        # ↪ (17x₁³x₃²x₄⁴) * (17x₁x₂⁴x₃²z₁z̄₁) ≥ (17Re(x₁²x₂²x₃²x₄²z̄₁))² + (17Im(x₁²x₂²x₃²x₄²z̄₁))²
        gi(index₁, value₁, (3, 0, 2, 4), Int16(17))
        gi(:re, index₂, value₂, (1, 4, 2, 0, 1, 0, 1, 0), Int16(17))
        gi(:re, index₃, value₃, (2, 2, 2, 2, 0, 0, 1, 0), Int16(17))
        gi(:im, index₄, value₄, (2, 2, 2, 2, 0, 0, 1, 0), Int16(17))
    elseif state.instance == 33
        # [x₁x₄²; x₂²z₁] * 24x₂z₁z̄₁ * [x₁x₄² x₂²z̄₁] = 24 [x₁²x₂x₄⁴z₁z̄₁    x₁x₂³x₄²z₁z̄₁²
        #                                                 x₁x₂³x₄²z₁²z̄₁   x₂⁵z₁²z̄₁²]
        # ↪ (24x₁²x₂x₄⁴z₁z̄₁) * (24x₂⁵z₁²z̄₁²) ≥ (24Re(x₁x₂³x₄²z₁z̄₁²))² + (24Im(x₁x₂³x₄²z₁z̄₁²))²
        gi(:re, index₁, value₁, (2, 1, 0, 4, 1, 0, 1, 0), Int16(24))
        gi(:re, index₂, value₂, (0, 5, 0, 0, 2, 0, 2, 0), Int16(24))
        gi(:re, index₃, value₃, (1, 3, 0, 2, 1, 0, 2, 0), Int16(24))
        gi(:im, index₄, value₄, (1, 3, 0, 2, 1, 0, 2, 0), Int16(24))
    elseif state.instance == 51
        # x₂²z₂ [17x₃²z₁z̄₁  8z₂²  ] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8x₂⁴z₂³z̄₂
        #       [8z̄₂²       6x₂x₄²]          8x₂⁴z₂z̄₂³          6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁, value₁, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂, value₂, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃, value₃, (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
        gi(:im, index₄, value₄, (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
    elseif state.instance == 52
        # x₂²z₂ [17x₃²z₁z̄₁  8im*z₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  8im*x₂⁴z₂³z̄₂
        #       [-8im*z̄₂²   6x₂x₄² ]          -8im*x₂⁴z₂³z̄₂     6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁, value₁, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂, value₂, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:im, index₃, value₃, (0, 4, 0, 0, 0, 3, 0, 1), Int16(-8))
        gi(:re, index₄, value₄, (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadratic{false},
    index₁::AbstractVector{BigInt}, value₁::AbstractVector{Int16}, index₂::AbstractVector{BigInt},
    value₂::AbstractVector{Int16}, (index₃, value₃)::Tuple{AbstractVector{BigInt},AbstractVector{Int16}})
    @test length(index₁) == length(value₁) && length(index₂) == length(value₂) && length(index₃) == length(value₃)
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 21
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # [x₁x₄²; x₂²x₃³] * 17x₁x₃² * [x₁x₄² x₂²x₃³] = 17 [x₁³x₃²x₄⁴    x₁²x₂²x₃⁵x₄²
        #                                                  *            x₁x₂⁴x₃⁸]
        # ↪ (17x₁³x₃²x₄⁴) * (17x₁x₂⁴x₃⁸) ≥ (17x₁²x₂²x₃⁵x₄²)²
        gi(index₁[1], value₁[1], (3, 0, 2, 4), Int16(17))
        gi(index₂[1], value₂[1], (1, 4, 8, 0), Int16(17))
        gi(index₃[1], value₃[1], (2, 2, 5, 2), Int16(17))
    elseif state.instance == 23
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # [x₁x₄²; x₂²x₃³] * 24x₂z₁z̄₁ * [x₁x₄² x₂²x₃³] = 24 [x₁²x₂x₄⁴z₁z̄₁    x₁x₂³x₃³x₄²z₁z̄₁
        #                                                   *               x₂⁵x₃⁶z₁z̄₁]
        # ↪ (24x₁²x₂x₄⁴z₁z̄₁) * (24x₂⁵x₃⁶z₁z̄₁) ≥ (24x₁x₂³x₃³x₄²z₁z̄₁)²
        gi(:re, index₁[1], value₁[1], (2, 1, 0, 4, 1, 0, 1, 0), Int16(24))
        gi(:re, index₂[1], value₂[1], (0, 5, 6, 0, 1, 0, 1, 0), Int16(24))
        gi(:re, index₃[1], value₃[1], (1, 3, 3, 2, 1, 0, 1, 0), Int16(24))
    elseif state.instance == 26
        @test length(index₁) == length(index₂) == length(index₃) == 2
        # [x₁x₄²; x₂²x₃³] * (2x₁ + 8x₂x₄²) * [x₁x₄² x₂²x₃³] = [2x₁³x₄⁴+8x₁²x₂x₄⁶    2x₁²x₂²x₃³x₄²+8x₁x₂³x₃³x₄⁴
        #                                                      *                    2x₁x₂⁴x₃⁶+8x₂⁵x₃⁶x₄²]
        # ↪ (2x₁³x₄⁴+8x₁²x₂x₄⁶) * (2x₁x₂⁴x₃⁶+8x₂⁵x₃⁶x₄²) ≥ (2x₁²x₂²x₃³x₄²+8x₁x₂³x₃³x₄⁴)²
        gi(index₁[1], value₁[1], (3, 0, 0, 4), Int16(2))
        gi(index₁[2], value₁[2], (2, 1, 0, 6), Int16(8))
        gi(index₂[1], value₂[1], (1, 4, 6, 0), Int16(2))
        gi(index₂[2], value₂[2], (0, 5, 6, 2), Int16(8))
        gi(index₃[1], value₃[1], (2, 2, 3, 2), Int16(2))
        gi(index₃[2], value₃[2], (1, 3, 3, 4), Int16(8))
    elseif state.instance == 27
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # [x₁x₄²; x₂²x₃³] * 3(z₁ + z̄₁) * [x₁x₄² x₂²x₃³] = 3[x₁²x₄⁴z₁+x₁²x₄⁴z̄₁   x₁x₂²x₃³x₄²z₁+x₁x₂²x₃³x₄²z̄₁
        #                                                   *                   x₂⁴x₃⁶z₁+x₂⁴x₃⁶z̄₁]
        # ↪ (6Re(x₁²x₄⁴z₁)) * (6Re(x₂⁴x₃⁶z₁)) ≥ (6Re(x₁x₂²x₃³x₄²z₁))²
        gi(:re, index₁[1], value₁[1], (2, 0, 0, 4, 1, 0, 0, 0), Int16(6))
        gi(:re, index₂[1], value₂[1], (0, 4, 6, 0, 1, 0, 0, 0), Int16(6))
        gi(:re, index₃[1], value₃[1], (1, 2, 3, 2, 1, 0, 0, 0), Int16(6))
    elseif state.instance == 28
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # [x₁x₄²; x₂²x₃³] * 7im(z₁z̄₂² - z̄₁z₂²) * [x₁x₄² x₂²x₃³] = 7im[x₁²x₄⁴z₁z̄₂²-x₁²x₄⁴z₂²z̄₁   x₁x₂²x₃³x₄²z₁z̄₂²-x₁x₂²x₃³x₄²z₂²z̄₁
        #                                                             *                         x₂⁴x₃⁶z₁z̄₂²-x₂⁴x₃⁶z₂²z̄₁]
        # ↪ (-14Im(x₁²x₄⁴z₁z̄₂²)) * (-14Im(x₂⁴x₃⁶z₁z̄₂²)) ≥ (-14Im(x₁x₂²x₃³x₄²z₁z̄₂²))²
        gi(:im, index₁[1], value₁[1], (2, 0, 0, 4, 1, 0, 0, 2), Int16(-14))
        gi(:im, index₂[1], value₂[1], (0, 4, 6, 0, 1, 0, 0, 2), Int16(-14))
        gi(:im, index₃[1], value₃[1], (1, 2, 3, 2, 1, 0, 0, 2), Int16(-14))
    elseif state.instance == 30
        @test length(index₁) == length(index₂) == length(index₃) == 6
        # [x₁x₄²; x₂²x₃³] * (3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁) * [x₁x₄² x₂²x₃³]
        # ↪ (3x₁³x₄⁴ + 5x₁²x₂x₄⁶ + 4x₁²x₄⁴z₂z̄₂ + 14Re(x₁²x₄⁴z₁²z̄₂) + 8Re(x₁²x₄⁴z₁) - 16Im(x₁²x₄⁴z₁))
        #  * (3x₁x₂⁴x₃⁶ + 5x₂⁵x₃⁶x₄² + 4x₂⁴x₃⁶z₂z̄₂ + 14Re(x₂⁴x₃⁶z₁²z̄₂) + 8Re(x₂⁴x₃⁶z₁) - 16Im(x₂⁴x₃⁶z₁))
        #  ≥ (3x₁²x₂²x₃³x₄² + 5x₁x₂³x₃³x₄⁴ + 4x₁x₂²x₃³x₄²z₂z̄₂ + 14Re(x₁x₂²x₃³x₄²z₁²z̄₂) + 8Re(x₁x₂²x₃³x₄²z₁) - 16Im(x₁x₂²x₃³x₄²z₁))²
        gi(index₁[1], value₁[1], (3, 0, 0, 4), Int16(3))
        gi(index₁[2], value₁[2], (2, 1, 0, 6), Int16(5))
        gi(:re, index₁[3], value₁[3], (2, 0, 0, 4, 0, 1, 0, 1), Int16(4))
        gi(:re, index₁[4], value₁[4], (2, 0, 0, 4, 2, 0, 0, 1), Int16(14))
        gi(:re, index₁[5], value₁[5], (2, 0, 0, 4, 1, 0, 0, 0), Int16(8))
        gi(:im, index₁[6], value₁[6], (2, 0, 0, 4, 1, 0, 0, 0), Int16(-16))

        gi(index₂[1], value₂[1], (1, 4, 6, 0), Int16(3))
        gi(index₂[2], value₂[2], (0, 5, 6, 2), Int16(5))
        gi(:re, index₂[3], value₂[3], (0, 4, 6, 0, 0, 1, 0, 1), Int16(4))
        gi(:re, index₂[4], value₂[4], (0, 4, 6, 0, 2, 0, 0, 1), Int16(14))
        gi(:re, index₂[5], value₂[5], (0, 4, 6, 0, 1, 0, 0, 0), Int16(8))
        gi(:im, index₂[6], value₂[6], (0, 4, 6, 0, 1, 0, 0, 0), Int16(-16))

        gi(index₃[1], value₃[1], (2, 2, 3, 2), Int16(3))
        gi(index₃[2], value₃[2], (1, 3, 3, 4), Int16(5))
        gi(:re, index₃[3], value₃[3], (1, 2, 3, 2, 0, 1, 0, 1), Int16(4))
        gi(:re, index₃[4], value₃[4], (1, 2, 3, 2, 2, 0, 0, 1), Int16(14))
        gi(:re, index₃[5], value₃[5], (1, 2, 3, 2, 1, 0, 0, 0), Int16(8))
        gi(:im, index₃[6], value₃[6], (1, 2, 3, 2, 1, 0, 0, 0), Int16(-16))
    elseif state.instance == 41
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # x₂²z₂ [17x₃²z₁z̄₁  8x₂²  ] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8x₂⁶z₂z̄₂
        #       [8x₂²       6x₂x₄²]          8x₂⁶z₂z̄₂           6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 46
        @test length(index₁) == length(index₃) == 1 && isempty(index₂)
        # x₂²z₂ [17x₃²z₁z̄₁  8x₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  8x₂⁶z₂z̄₂
        #       [8x₂²       0   ]          8x₂⁶z₂z̄₂          0]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₃[1], value₃[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 47
        # due to special case fallback, is shifted to index₁
        @test length(index₁) == length(index₃) == 1 && isempty(index₂)
        # x₂²z₂ [0     8x₂²  ] x₂²z̄₂ = [0         8x₂⁶z₂z̄₂
        #       [8x₂²  6x₂x₄²]          8x₂⁶z₂z̄₂  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 50
        @test length(index₁) == length(index₂) == length(index₃) == 1
        # x₂²z₂ [17x₃²z₁z̄₁  8im*x₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8im*x₂⁶z₂z̄₂
        #       [-8im*x₂²   6x₂x₄² ]          -8im*x₂⁶z₂z̄₂       6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 55
        @test length(index₁) == length(index₃) == 2 && length(index₂) == 1
        # x₂²z₂ [5x₂²z₁+17im*x₃²z̄₁+5x₂²z̄₁-17im*x₃²z₁  (8-2im)*z̄₂²+(8+2im)*z₂²] x₂²z̄₂ = [5x₂⁶z₁z₂z̄₂+17im*x₂⁴x₃²z₂z̄₁z̄₂+5x₂⁶z₂z̄₁z̄₂-17im*x₂⁴x₃²z₁z₂z̄₂  (8-2im)x₂⁴z₂z̄₂³+(8+2im)x₂⁴z₂³z̄₂
        #       [*                                    6x₂x₄²                 ]                                                                     6x₂⁵x₄²z₂z̄₂]
        gi(:im, index₁[1], value₁[1], (0, 4, 2, 0, 0, 1, 1, 1), Int16(-34))
        gi(:re, index₁[2], value₁[2], (0, 6, 0, 0, 1, 1, 0, 1), Int16(10))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 4, 0, 0, 0, 1, 0, 3), Int16(16))
        gi(:im, index₃[2], value₃[2], (0, 4, 0, 0, 0, 1, 0, 3), Int16(4))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadratic{false}, index₁::AbstractVector{BigInt},
    value₁::AbstractVector{Int16}, index₂::AbstractVector{BigInt}, value₂::AbstractVector{Int16},
    (index₃, value₃)::Tuple{AbstractVector{BigInt},AbstractVector{Int16}},
    (index₄, value₄)::Tuple{AbstractVector{BigInt},AbstractVector{Int16}})
    @test length(index₁) == length(value₁) && length(index₂) == length(value₂) && length(index₃) == length(value₃) &&
        length(index₄) == length(value₄)
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 31
        @test length(index₁) == length(index₂) == length(index₃) == length(index₄) == 1
        # [x₁x₄²; x₂²z₁] * 17x₁x₃² * [x₁x₄² x₂²z̄₁] = 17 [x₁³x₃²x₄⁴  x₁²x₂²x₃²x₄²z̄₁
        #                                                *          x₁x₂⁴x₃²z₁z̄₁]
        # ↪ (17x₁³x₃²x₄⁴) * (17x₁x₂⁴x₃²z₁z̄₁) ≥ (17Re(x₁²x₂²x₃²x₄²z̄₁))² + (17Im(x₁²x₂²x₃²x₄²z̄₁))²
        gi(index₁[1], value₁[1], (3, 0, 2, 4), Int16(17))
        gi(:re, index₂[1], value₂[1], (1, 4, 2, 0, 1, 0, 1, 0), Int16(17))
        gi(:re, index₃[1], value₃[1], (2, 2, 2, 2, 0, 0, 1, 0), Int16(17))
        gi(:im, index₄[1], value₄[1], (2, 2, 2, 2, 0, 0, 1, 0), Int16(17))
    elseif state.instance == 33
        @test length(index₁) == length(index₂) == length(index₃) == length(index₄) == 1
        # [x₁x₄²; x₂²z₁] * 24x₂z₁z̄₁ * [x₁x₄² x₂²z̄₁] = 24 [x₁²x₂x₄⁴z₁z̄₁  x₁x₂³x₄²z₁z̄₁²
        #                                                 *             x₂⁵z₁²z̄₁²]
        # ↪ (24x₁²x₂x₄⁴z₁z̄₁) * (24x₂⁵z₁²z̄₁²) ≥ (24Re(x₁x₂³x₄²z₁z̄₁²))² + (24Im(x₁x₂³x₄²z₁z̄₁²))²
        gi(:re, index₁[1], value₁[1], (2, 1, 0, 4, 1, 0, 1, 0), Int16(24))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 0, 2, 0, 2, 0), Int16(24))
        gi(:re, index₃[1], value₃[1], (1, 3, 0, 2, 1, 0, 2, 0), Int16(24))
        gi(:im, index₄[1], value₄[1], (1, 3, 0, 2, 1, 0, 2, 0), Int16(24))
    elseif state.instance == 36
        @test length(index₁) == length(index₂) == length(index₃) == length(index₄) == 2
        # [x₁x₄²; x₂²z₁] * (2x₁ + 8x₂x₄²) * [x₁x₄² x₂²z̄₁] = [2x₁³x₄⁴+8x₁²x₂x₄⁶  2x₁²x₂²x₄²z̄₁+8x₁x₂³x₄⁴z̄₁
        #                                                    *                  2x₁x₂⁴z₁z̄₁+8x₂⁵x₄²z₁z̄₁]
        # ↪ (2x₁³x₄⁴+8x₁²x₂x₄⁶) * (2x₁x₂⁴z₁z̄₁+8x₂⁵x₄²z₁z̄₁)
        #  ≥ (Re(2x₁²x₂²x₄²z̄₁+8x₁x₂³x₄⁴z̄₁))² + (Im(2x₁²x₂²x₄²z̄₁+8x₁x₂³x₄⁴z̄₁))²
        gi(index₁[1], value₁[1], (3, 0, 0, 4), Int16(2))
        gi(index₁[2], value₁[2], (2, 1, 0, 6), Int16(8))
        gi(:re, index₂[1], value₂[1], (1, 4, 0, 0, 1, 0, 1, 0), Int16(2))
        gi(:re, index₂[2], value₂[2], (0, 5, 0, 2, 1, 0, 1, 0), Int16(8))
        gi(:re, index₃[1], value₃[1], (2, 2, 0, 2, 0, 0, 1, 0), Int16(2))
        gi(:re, index₃[2], value₃[2], (1, 3, 0, 4, 0, 0, 1, 0), Int16(8))
        gi(:im, index₄[1], value₄[1], (2, 2, 0, 2, 0, 0, 1, 0), Int16(2))
        gi(:im, index₄[2], value₄[2], (1, 3, 0, 4, 0, 0, 1, 0), Int16(8))
    elseif state.instance == 37
        @test length(index₁) == length(index₂) == length(index₄) == 1 && length(index₃) == 2
        # [x₁x₄²; x₂²z₁] * 3(z₁ + z̄₁) * [x₁x₄² x₂²z̄₁] = 3[x₁²x₄⁴z₁+x₁²x₄⁴z̄₁  x₁x₂²x₄²z₁z̄₁+x₁x₂²x₄²z̄₁²
        #                                                 *                  x₂⁴z₁²z̄₁+x₂⁴z₁z̄₁²]
        # ↪ (6Re(x₁²x₄⁴z₁)) * (6Re(x₂⁴z₁²z̄₁)) ≥ (6Re(x₁x₂²x₄²z₁z̄₁+x₁x₂²x₄²z̄₁²))² + (6Im(x₁x₂²x₄²z₁z̄₁+x₁x₂²x₄²z̄₁²))²
        gi(:re, index₁[1], value₁[1], (2, 0, 0, 4, 1, 0, 0, 0), Int16(6))
        gi(:re, index₂[1], value₂[1], (0, 4, 0, 0, 2, 0, 1, 0), Int16(6))
        gi(:re, index₃[1], value₃[1], (1, 2, 0, 2, 0, 0, 2, 0), Int16(3)) # z̄₁ is the canonical version
        gi(:re, index₃[2], value₃[2], (1, 2, 0, 2, 1, 0, 1, 0), Int16(3))
        gi(:im, index₄[1], value₄[1], (1, 2, 0, 2, 0, 0, 2, 0), Int16(3))
    elseif state.instance == 38
        @test length(index₁) == length(index₂) == 1 && length(index₃) == length(index₄) == 2
        # [x₁x₄²; x₂²z₁] * 7im(z₁z̄₂² - z̄₁z₂²) * [x₁x₄² x₂²z̄₁] = 7im[x₁²x₄⁴z₁z̄₂²-x₁²x₄⁴z₂²z̄₁  x₁x₂²x₄²z₁z̄₁z̄₂²-x₁x₂²x₄²z₂²z̄₁²
        #                                                           *                        x₂⁴z₁²z̄₁z̄₂²-x₂⁴z₁z₂²z̄₁²]
        # ↪ (-14Im(x₁²x₄⁴z₁z̄₂²)) * (-14Im(x₂⁴z₁²z̄₁z̄₂²))
        #  ≥ (-7Im(x₁x₂²x₄²z₁z̄₁z̄₂²-x₁x₂²x₄²z₂²z̄₁²))² + (7Re(x₁x₂²x₄²z₁z̄₁z̄₂²-x₁x₂²x₄²z₂²z̄₁²))^2
        gi(:im, index₁[1], value₁[1], (2, 0, 0, 4, 1, 0, 0, 2), Int16(-14))
        gi(:im, index₂[1], value₂[1], (0, 4, 0, 0, 2, 0, 1, 2), Int16(-14))
        gi(:im, index₃[1], value₃[1], (1, 2, 0, 2, 0, 2, 2, 0), Int16(7))
        gi(:im, index₃[2], value₃[2], (1, 2, 0, 2, 1, 0, 1, 2), Int16(-7))
        gi(:re, index₄[1], value₄[1], (1, 2, 0, 2, 0, 2, 2, 0), Int16(-7))
        gi(:re, index₄[2], value₄[2], (1, 2, 0, 2, 1, 0, 1, 2), Int16(7))
    elseif state.instance == 40
        @test length(index₁) == length(index₂) == 6 && length(index₃) == length(index₄) == 8
        # [x₁x₄²; x₂²z₁] * (3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁) * [x₁x₄² x₂²z̄₁]
        # ↪ (3x₁³x₄⁴ + 5x₁²x₂x₄⁶ + 4x₁²x₄⁴z₂z̄₂ + 14Re(x₁²x₄⁴z₁²z̄₂) + 8Re(x₁²x₄⁴z₁) - 16Im(x₁²x₄⁴z₁))
        #  * (3x₁x₂⁴z₁z̄₁ + 5x₂⁵x₄²z₁z̄₁ + 4x₂⁴z₁z₂z̄₁z̄₂ + 14Re(x₂⁴z₁³z̄₁z̄₂) + 8Re(x₂⁴z₁²z̄₁) - 16Im(x₂⁴z₁²z̄₁))
        #  ≥ (3Re(x₁²x₂²x₄²z̄₁) + 5Re(x₁x₂³x₄⁴z̄₁) + 4Re(x₁x₂²x₄²z₂z̄₁z̄₂) + 7Re(x₁x₂²x₄²z₁²z̄₁z̄₂) + 7Re(x₁x₂²x₄²z₂z̄₁³) +
        #     4Re(x₁x₂²x₄²z₁z̄₁) + 4Re(x₁x₂²x₄²z̄₁²) + 8Im(x₁x₂²x₄²z̄₁²))²
        #  + (3Im(x₁²x₂²x₄²z̄₁) + 5Im(x₁x₂³x₄⁴z̄₁) + 4Im(x₁x₂²x₄²z₂z̄₁z̄₂) + 7Im(x₁x₂²x₄²z₁²z̄₁z̄₂) + 7Im(x₁x₂²x₄²z₂z̄₁³) +
        #     8Re(x₁x₂²x₄²z₁z̄₁) + 4Im(x₁x₂²x₄²z̄₁²) - 8Re(x₁x₂²x₄²z̄₁²))²
        gi(index₁[1], value₁[1], (3, 0, 0, 4), Int16(3))
        gi(index₁[2], value₁[2], (2, 1, 0, 6), Int16(5))
        gi(:re, index₁[3], value₁[3], (2, 0, 0, 4, 0, 1, 0, 1), Int16(4))
        gi(:re, index₁[4], value₁[4], (2, 0, 0, 4, 2, 0, 0, 1), Int16(14))
        gi(:re, index₁[5], value₁[5], (2, 0, 0, 4, 1, 0, 0, 0), Int16(8))
        gi(:im, index₁[6], value₁[6], (2, 0, 0, 4, 1, 0, 0, 0), Int16(-16))

        gi(:re, index₂[1], value₂[1], (1, 4, 0, 0, 1, 0, 1, 0), Int16(3))
        gi(:re, index₂[2], value₂[2], (0, 5, 0, 2, 1, 0, 1, 0), Int16(5))
        gi(:re, index₂[3], value₂[3], (0, 4, 0, 0, 1, 1, 1, 1), Int16(4))
        gi(:re, index₂[4], value₂[4], (0, 4, 0, 0, 3, 0, 1, 1), Int16(14))
        gi(:re, index₂[5], value₂[5], (0, 4, 0, 0, 2, 0, 1, 0), Int16(8))
        gi(:im, index₂[6], value₂[6], (0, 4, 0, 0, 2, 0, 1, 0), Int16(-16))

        gi(:re, index₃[1], value₃[1], (2, 2, 0, 2, 0, 0, 1, 0), Int16(3))
        gi(:re, index₃[2], value₃[2], (1, 3, 0, 4, 0, 0, 1, 0), Int16(5))
        gi(:re, index₃[3], value₃[3], (1, 2, 0, 2, 0, 1, 1, 1), Int16(4))
        gi(:re, index₃[4], value₃[4], (1, 2, 0, 2, 0, 1, 3, 0), Int16(7)) # flip as the second is the canonical one
        gi(:re, index₃[5], value₃[5], (1, 2, 0, 2, 2, 0, 1, 1), Int16(7))
        gi(:re, index₃[6], value₃[6], (1, 2, 0, 2, 0, 0, 2, 0), Int16(4))
        gi(:re, index₃[7], value₃[7], (1, 2, 0, 2, 1, 0, 1, 0), Int16(4)) # also flip
        gi(:im, index₃[8], value₃[8], (1, 2, 0, 2, 0, 0, 2, 0), Int16(8))

        gi(:im, index₄[1], value₄[1], (2, 2, 0, 2, 0, 0, 1, 0), Int16(3))
        gi(:im, index₄[2], value₄[2], (1, 3, 0, 4, 0, 0, 1, 0), Int16(5))
        gi(:im, index₄[3], value₄[3], (1, 2, 0, 2, 0, 1, 1, 1), Int16(4))
        gi(:im, index₄[4], value₄[4], (1, 2, 0, 2, 0, 1, 3, 0), Int16(7)) # flip as the second is the canonical one
        gi(:im, index₄[5], value₄[5], (1, 2, 0, 2, 2, 0, 1, 1), Int16(7))
        gi(:im, index₄[6], value₄[6], (1, 2, 0, 2, 0, 0, 2, 0), Int16(4))
        gi(:re, index₄[7], value₄[7], (1, 2, 0, 2, 0, 0, 2, 0), Int16(-8)) # also flip
        gi(:re, index₄[8], value₄[8], (1, 2, 0, 2, 1, 0, 1, 0), Int16(8))
    elseif state.instance == 51
        @test length(index₁) == length(index₂) == length(index₃) == length(index₄) == 1
        # x₂²z₂ [17x₃²z₁z̄₁  8z₂²  ] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂   8x₂⁴z₂³z̄₂
        #       [*          6x₂x₄²]          *                  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
        gi(:im, index₄[1], value₄[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
    elseif state.instance == 52
        @test length(index₁) == length(index₂) == length(index₃) == length(index₄) == 1
        # x₂²z₂ [17x₃²z₁z̄₁  8im*z₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  8im*x₂⁴z₂³z̄₂
        #       [*          6x₂x₄² ]          *                 6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:im, index₃[1], value₃[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(-8))
        gi(:re, index₄[1], value₄[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
    elseif state.instance == 53
        @test length(index₁) == length(index₂) == 1 && length(index₃) == length(index₄) == 2
        # x₂²z₂ [17x₃²z₁z̄₁  (8+2im)*z₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  (8+2im)*x₂⁴z₂³z̄₂
        #       [*          6x₂x₄²     ]          *                 6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
        gi(:im, index₃[2], value₃[2], (0, 4, 0, 0, 0, 3, 0, 1), Int16(-2))
        gi(:re, index₄[1], value₄[1], (0, 4, 0, 0, 0, 3, 0, 1), Int16(2))
        gi(:im, index₄[2], value₄[2], (0, 4, 0, 0, 0, 3, 0, 1), Int16(8))
    elseif state.instance == 54
        @test length(index₁) == 2 && length(index₂) == 1 && length(index₃) == length(index₄) == 3
        # x₂²z₂ [5x₂²+17x₃²z₁z̄₁  (8-2im)*z̄₂²-8z₂²-3im*x₁+z₁z̄₁] x₂²z̄₂ = [5x₂⁶z₂z̄₂+17x₂⁴x₃²z₁z₂z̄₁z̄₂  (8-2im)*x₂⁴z₂z̄₂³-8x₂⁴z₂³̄z₂²-3im*x₁x₂⁴z₂z̄₂+x₂⁴z₁z₂z̄₁z̄₂
        #       [*               6x₂x₄²                      ]                                     6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₁[1], value₁[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(5))
        gi(:re, index₁[2], value₁[2], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₃[1], value₃[1], (0, 4, 0, 0, 0, 1, 0, 3), Int16(0)) # 8z̄₂² - 8z₂²; but we don't delete zeros
        gi(:im, index₃[2], value₃[2], (0, 4, 0, 0, 0, 1, 0, 3), Int16(2))
        gi(:re, index₃[3], value₃[3], (0, 4, 0, 0, 1, 1, 1, 1), Int16(1))
        gi(:im, index₄[1], value₄[1], (0, 4, 0, 0, 0, 1, 0, 3), Int16(16)) # must be accumulated
        gi(:re, index₄[2], value₄[2], (0, 4, 0, 0, 0, 1, 0, 3), Int16(-2))
        gi(:re, index₄[3], value₄[3], (1, 4, 0, 0, 0, 1, 0, 1), Int16(-3))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadraticSimplified{true},
    index₊::BigInt, value₊::Int16, (index₂, value₂)::Tuple{BigInt,Int16})
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 46
        # x₂²z₂ [17x₃²z₁z̄₁  8x₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  8x₂⁶z₂z̄₂
        #       [8x₂²       0   ]          8x₂⁶z₂z̄₂          0]
        gi(:re, index₊, value₊, (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂, value₂, (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 47
        # x₂²z₂ [0     8x₂²  ] x₂²z̄₂ = [0         8x₂⁶z₂z̄₂
        #       [8x₂²  6x₂x₄²]          8x₂⁶z₂z̄₂  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₊, value₊, (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₂, value₂, (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_quadratic!(state::SolverSetupQuadraticSimplified{false},
    index₊::AbstractVector{BigInt}, value₊::AbstractVector{Int16},
    (index₂, value₂)::Tuple{AbstractVector{BigInt},AbstractVector{Int16}})
    @test length(index₊) == length(value₊) && length(index₂) == length(value₂)
    @test state.lastcall === :none
    state.lastcall = :add_quadratic
    if state.instance == 46
        @test length(index₊) == length(index₂) == 1
        # x₂²z₂ [17x₃²z₁z̄₁  8x₂²] x₂²z̄₂ = [17x₂⁴x₃²z₁z₂z̄₁z̄₂  8x₂⁶z₂z̄₂
        #       [8x₂²       0   ]          8x₂⁶z₂z̄₂          0]
        gi(:re, index₊[1], value₊[1], (0, 4, 2, 0, 1, 1, 1, 1), Int16(17))
        gi(:re, index₂[1], value₂[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    elseif state.instance == 47
        @test length(index₊) == length(index₂) == 1
        # x₂²z₂ [0     8x₂²  ] x₂²z̄₂ = [0         8x₂⁶z₂z̄₂
        #       [8x₂²  6x₂x₄²]          8x₂⁶z₂z̄₂  6x₂⁵x₄²z₂z̄₂]
        gi(:re, index₊[1], value₊[1], (0, 5, 0, 2, 0, 1, 0, 1), Int16(6))
        gi(:re, index₂[1], value₂[1], (0, 6, 0, 0, 0, 1, 0, 1), Int16(8))
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_psd!(state::SolverSetupPSDDictLinear, dim::Int,
    data::Dict{FastKey{BigInt},<:Tuple{AbstractVector{Int32},AbstractVector{Int16}}})
    tri = sos_get_tri(state)
    lower = tri === :L
    @assert(lower || tri === :U)
    @test state.lastcall === :none
    state.lastcall = :add_psd
    # To properly compare, let's first sort all of the indices. We don't want to test the internal ordering - which is
    # irrelevant for the actual problem - and it would be too much effort to synchronize it with the code generation for this
    # test in Mathematica.
    dropkeys = FastKey{BigInt}[]
    for (key, (pos, val)) in data
        # In the complex → real case, cancellation may happen. Mathematica will do it implicitly, but the code generation will
        # potentially give zero-index values.
        delpos = findall(iszero, val)
        deleteat!(pos, delpos)
        deleteat!(val, delpos)
        if isempty(val)
            push!(dropkeys, key)
        else
            sort_along!(pos, val)
        end
    end
    delete!.((data,), dropkeys)
    if state.instance == 61
        @test dim == 4
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (Int32[17], Int16[-2]),
            FastKey(BigInt(52)) => (Int32[17], Int16[3]),
            FastKey(BigInt(65)) => (Int32[17], Int16[5]),
            FastKey(BigInt(79)) => (Int32[18], Int16[-2]),
            FastKey(BigInt(81)) => (Int32[18], Int16[3]),
            FastKey(BigInt(84)) => (Int32[17], Int16[3]),
            FastKey(BigInt(157)) => (Int32[18], Int16[5]),
            FastKey(BigInt(171)) => (lower ? Int32[21] : Int32[19], Int16[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[21] : Int32[19], Int16[3]),
            FastKey(BigInt(176)) => (Int32[18], Int16[3]),
            FastKey(BigInt(403)) => (lower ? Int32[21] : Int32[19], Int16[5]),
            FastKey(BigInt(422)) => (lower ? Int32[21] : Int32[19], Int16[3]),
            FastKey(BigInt(454)) => (lower ? Int32[19] : Int32[20], Int16[-2]),
            FastKey(BigInt(456)) => (lower ? Int32[19] : Int32[20], Int16[3]),
            FastKey(BigInt(1068)) => (lower ? Int32[19] : Int32[20], Int16[5]),
            FastKey(BigInt(1162)) => (lower ? Int32[22] : Int32[21], Int16[-2]),
            FastKey(BigInt(1164)) => (lower ? Int32[22] : Int32[21], Int16[3]),
            FastKey(BigInt(1181)) => (lower ? Int32[19] : Int32[20], Int16[3]),
            FastKey(BigInt(2170)) => (lower ? Int32[20] : Int32[23], Int16[-2]),
            FastKey(BigInt(2172)) => (lower ? Int32[20] : Int32[23], Int16[3]),
            FastKey(BigInt(2574)) => (lower ? Int32[22] : Int32[21], Int16[5]),
            FastKey(BigInt(2687)) => (lower ? Int32[22] : Int32[21], Int16[3]),
            FastKey(BigInt(4544)) => (lower ? Int32[20] : Int32[23], Int16[5]),
            FastKey(BigInt(4678)) => (lower ? Int32[23] : Int32[24], Int16[-2]),
            FastKey(BigInt(4680)) => (lower ? Int32[23] : Int32[24], Int16[3]),
            FastKey(BigInt(4683)) => (lower ? Int32[20] : Int32[23], Int16[3]),
            FastKey(BigInt(5813)) => (lower ? Int32[24] : Int32[22], Int16[-2]),
            FastKey(BigInt(5815)) => (lower ? Int32[24] : Int32[22], Int16[3]),
            FastKey(BigInt(9263)) => (lower ? Int32[23] : Int32[24], Int16[5]),
            FastKey(BigInt(9402)) => (lower ? Int32[23] : Int32[24], Int16[3]),
            FastKey(BigInt(11405)) => (lower ? Int32[24] : Int32[22], Int16[5]),
            FastKey(BigInt(11860)) => (lower ? Int32[24] : Int32[22], Int16[3]),
            FastKey(BigInt(20439)) => (Int32[25], Int16[-2]),
            FastKey(BigInt(20441)) => (Int32[25], Int16[3]),
            FastKey(BigInt(36384)) => (Int32[25], Int16[5]),
            FastKey(BigInt(36903)) => (Int32[25], Int16[3]),
            FastKey(BigInt(55253)) => (Int32[26], Int16[-2]),
            FastKey(BigInt(55255)) => (Int32[26], Int16[3]),
            FastKey(BigInt(92465)) => (Int32[26], Int16[5]),
            FastKey(BigInt(93270)) => (Int32[26], Int16[3]),
        )
    elseif state.instance == 62
        @test dim == 12
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (Int32[17], Int16[-2]),
            FastKey(BigInt(52)) => (Int32[17], Int16[3]),
            FastKey(BigInt(54)) => (Int32[18], Int16[17]),
            FastKey(BigInt(57)) => (lower ? Int32[19] : Int32[20], Int16[10]),
            FastKey(BigInt(61)) => (lower ? Int32[19] : Int32[20], Int16[8]),
            FastKey(BigInt(65)) => (Int32[17], Int16[5]),
            FastKey(BigInt(75)) => (lower ? Int32[30] : Int32[21], Int16[-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20] : Int32[23], Int16[-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20] : Int32[23], Int16[3]),
            FastKey(BigInt(83)) => (lower ? Int32[21, 31, 40] : Int32[22, 24, 27], lower ? Int16[17, 17, 6] : Int16[6, 17, 17]),
            FastKey(BigInt(84)) => (Int32[17], Int16[3]),
            FastKey(BigInt(92)) => (lower ? Int32[22, 41] : Int32[25, 32], Int16[10, 10]),
            FastKey(BigInt(153)) => (lower ? Int32[22, 41] : Int32[25, 32], Int16[8, 8]),
            FastKey(BigInt(157)) => (lower ? Int32[20] : Int32[23], Int16[5]),
            FastKey(BigInt(167)) => (lower ? Int32[33, 42] : Int32[29, 33], Int16[-24, -24]),
            FastKey(BigInt(171)) => (lower ? Int32[50] : Int32[26], Int16[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[50] : Int32[26], Int16[3]),
            FastKey(BigInt(175)) => (lower ? Int32[43, 51] : Int32[30, 34], lower ? Int16[6, 17] : Int16[17, 6]),
            FastKey(BigInt(176)) => (lower ? Int32[20] : Int32[23], Int16[3]),
            FastKey(BigInt(205)) => (lower ? Int32[52] : Int32[35], Int16[10]),
            FastKey(BigInt(399)) => (lower ? Int32[52] : Int32[35], Int16[8]),
            FastKey(BigInt(403)) => (lower ? Int32[50] : Int32[26], Int16[5]),
            FastKey(BigInt(413)) => (lower ? Int32[60] : Int32[36], Int16[-24]),
            FastKey(BigInt(421)) => (lower ? Int32[67] : Int32[37], Int16[6]),
            FastKey(BigInt(422)) => (lower ? Int32[50] : Int32[26], Int16[3]),
            FastKey(BigInt(454)) => (lower ? Int32[23] : Int32[38], Int16[-2]),
            FastKey(BigInt(456)) => (lower ? Int32[23] : Int32[38], Int16[3]),
            FastKey(BigInt(458)) => (lower ? Int32[24, 34] : Int32[39, 45], Int16[17, 17]),
            FastKey(BigInt(522)) => (lower ? Int32[25, 44] : Int32[40, 53], Int16[10, 10]),
            FastKey(BigInt(1064)) => (lower ? Int32[25, 44] : Int32[40, 53], Int16[8, 8]),
            FastKey(BigInt(1068)) => (lower ? Int32[23] : Int32[38], Int16[5]),
            FastKey(BigInt(1108)) => (lower ? Int32[36, 45] : Int32[47, 54], Int16[-24, -24]),
            FastKey(BigInt(1162)) => (lower ? Int32[53] : Int32[41], Int16[-2]),
            FastKey(BigInt(1164)) => (lower ? Int32[53] : Int32[41], Int16[3]),
            FastKey(BigInt(1166)) => (lower ? Int32[46, 54, 61] : Int32[42, 48, 55], lower ? Int16[6, 17, 17] : Int16[17, 17, 6]),
            FastKey(BigInt(1181)) => (lower ? Int32[23] : Int32[38], Int16[3]),
            FastKey(BigInt(1286)) => (lower ? Int32[55, 68] : Int32[43, 56], Int16[10, 10]),
            FastKey(BigInt(2170)) => (lower ? Int32[26] : Int32[62], Int16[-2]),
            FastKey(BigInt(2172)) => (lower ? Int32[26] : Int32[62], Int16[3]),
            FastKey(BigInt(2174)) => (lower ? Int32[27, 37] : Int32[63, 72], Int16[17, 17]),
            FastKey(BigInt(2570)) => (lower ? Int32[55, 68] : Int32[43, 56], Int16[8, 8]),
            FastKey(BigInt(2574)) => (lower ? Int32[53] : Int32[41], Int16[5]),
            FastKey(BigInt(2614)) => (lower ? Int32[63, 69] : Int32[50, 57], Int16[-24, -24]),
            FastKey(BigInt(2672)) => (lower ? Int32[70] : Int32[58], Int16[6]),
            FastKey(BigInt(2687)) => (lower ? Int32[53] : Int32[41], Int16[3]),
            FastKey(BigInt(2693)) => (lower ? Int32[28, 47] : Int32[64, 83], Int16[10, 10]),
            FastKey(BigInt(4540)) => (lower ? Int32[28, 47] : Int32[64, 83], Int16[8, 8]),
            FastKey(BigInt(4544)) => (lower ? Int32[26] : Int32[62], Int16[5]),
            FastKey(BigInt(4554)) => (lower ? Int32[39, 48] : Int32[74, 84], Int16[-24, -24]),
            FastKey(BigInt(4678)) => (lower ? Int32[56] : Int32[65], Int16[-2]),
            FastKey(BigInt(4680)) => (lower ? Int32[56] : Int32[65], Int16[3]),
            FastKey(BigInt(4682)) => (lower ? Int32[49, 57, 64] : Int32[66, 75, 85], lower ? Int16[6, 17, 17] : Int16[17, 17, 6]),
            FastKey(BigInt(4683)) => (lower ? Int32[26] : Int32[62], Int16[3]),
            FastKey(BigInt(5663)) => (lower ? Int32[58, 71] : Int32[67, 86], Int16[10, 10]),
            FastKey(BigInt(5813)) => (lower ? Int32[74] : Int32[44], Int16[-2]),
            FastKey(BigInt(5815)) => (lower ? Int32[74] : Int32[44], Int16[3]),
            FastKey(BigInt(5817)) => (lower ? Int32[75] : Int32[51], Int16[17]),
            FastKey(BigInt(6223)) => (lower ? Int32[76] : Int32[59], Int16[10]),
            FastKey(BigInt(9259)) => (lower ? Int32[58, 71] : Int32[67, 86], Int16[8, 8]),
            FastKey(BigInt(9263)) => (lower ? Int32[56] : Int32[65], Int16[5]),
            FastKey(BigInt(9273)) => (lower ? Int32[66, 72] : Int32[77, 87], Int16[-24, -24]),
            FastKey(BigInt(9401)) => (lower ? Int32[73] : Int32[88], Int16[6]),
            FastKey(BigInt(9402)) => (lower ? Int32[56] : Int32[65], Int16[3]),
            FastKey(BigInt(11401)) => (lower ? Int32[76] : Int32[59], Int16[8]),
            FastKey(BigInt(11405)) => (lower ? Int32[74] : Int32[44], Int16[5]),
            FastKey(BigInt(11536)) => (lower ? Int32[81] : Int32[60], Int16[-24]),
            FastKey(BigInt(11790)) => (lower ? Int32[85] : Int32[61], Int16[6]),
            FastKey(BigInt(11860)) => (lower ? Int32[74] : Int32[44], Int16[3]),
            FastKey(BigInt(20439)) => (lower ? Int32[77] : Int32[68], Int16[-2]),
            FastKey(BigInt(20441)) => (lower ? Int32[77] : Int32[68], Int16[3]),
            FastKey(BigInt(20443)) => (lower ? Int32[78, 82] : Int32[69, 78], Int16[17, 17]),
            FastKey(BigInt(22426)) => (lower ? Int32[79, 86] : Int32[70, 89], Int16[10, 10]),
            FastKey(BigInt(36380)) => (lower ? Int32[79, 86] : Int32[70, 89], Int16[8, 8]),
            FastKey(BigInt(36384)) => (lower ? Int32[77] : Int32[68], Int16[5]),
            FastKey(BigInt(36424)) => (lower ? Int32[84, 87] : Int32[80, 90], Int16[-24, -24]),
            FastKey(BigInt(36888)) => (lower ? Int32[88] : Int32[91], Int16[6]),
            FastKey(BigInt(36903)) => (lower ? Int32[77] : Int32[68], Int16[3]),
            FastKey(BigInt(55253)) => (lower ? Int32[89] : Int32[71], Int16[-2]),
            FastKey(BigInt(55255)) => (lower ? Int32[89] : Int32[71], Int16[3]),
            FastKey(BigInt(55257)) => (lower ? Int32[90] : Int32[81], Int16[17]),
            FastKey(BigInt(63728)) => (lower ? Int32[91] : Int32[92], Int16[10]),
            FastKey(BigInt(92461)) => (lower ? Int32[91] : Int32[92], Int16[8]),
            FastKey(BigInt(92465)) => (lower ? Int32[89] : Int32[71], Int16[5]),
            FastKey(BigInt(92475)) => (Int32[93], Int16[-24]),
            FastKey(BigInt(93269)) => (Int32[94], Int16[6]),
            FastKey(BigInt(93270)) => (lower ? Int32[89] : Int32[71], Int16[3]),
        )
    elseif state.instance == 63
        @test dim == 24
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (lower ? Int32[17,239] : Int32[17,107], Int16[-2,-2]),
            FastKey(BigInt(52)) => (lower ? Int32[17,239] : Int32[17,107], Int16[3,3]),
            FastKey(BigInt(57)) => (lower ? Int32[19,30,52,241] : Int32[20,96,108,134], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(58)) => (lower ? Int32[30,52] : Int32[96,108], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(61)) => (lower ? Int32[19,241] : Int32[20,134], Int16[8,8]),
            FastKey(BigInt(63)) => (lower ? Int32[18,30,52,240] : Int32[18,96,108,120], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(65)) => (lower ? Int32[17,18,239,240] : Int32[17,18,107,120], Int16[5,2,5,2]),
            FastKey(BigInt(75)) => (lower ? Int32[42,252] : Int32[21,135], Int16[-24,-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20,242] : Int32[23,149], Int16[-2,-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20,242] : Int32[23,149], Int16[3,3]),
            FastKey(BigInt(83)) => (lower ? Int32[64,262] : Int32[22,136], Int16[6,6]),
            FastKey(BigInt(84)) => (lower ? Int32[17,239] : Int32[17,107], Int16[3,3]),
            FastKey(BigInt(92)) => (lower ? Int32[22,33,55,65,96,115,244,263] : Int32[25,32,99,111,138,151,153,182], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(150)) => (lower ? Int32[33,55,96,115] : Int32[99,111,138,153], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(153)) => (lower ? Int32[22,65,244,263] : Int32[25,32,151,182], Int16[8,8,8,8]),
            FastKey(BigInt(155)) => (lower ? Int32[21,33,43,55,96,115,243,253] : Int32[24,27,99,111,138,150,153,165], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(157)) => (lower ? Int32[20,21,43,242,243,253] : Int32[23,24,27,149,150,165], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(167)) => (lower ? Int32[45,66,255,264] : Int32[29,33,167,183], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(171)) => (lower ? Int32[86,272] : Int32[26,152], Int16[-2,-2]),
            FastKey(BigInt(173)) => (lower ? Int32[86,272] : Int32[26,152], Int16[3,3]),
            FastKey(BigInt(175)) => (lower ? Int32[67,265] : Int32[34,184], Int16[6,6]),
            FastKey(BigInt(176)) => (lower ? Int32[20,242] : Int32[23,149], Int16[3,3]),
            FastKey(BigInt(205)) => (lower ? Int32[88,99,118,274] : Int32[35,141,156,185], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(396)) => (lower ? Int32[99,118] : Int32[141,156], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(399)) => (lower ? Int32[88,274] : Int32[35,185], Int16[8,8]),
            FastKey(BigInt(401)) => (lower ? Int32[87,99,118,273] : Int32[30,141,156,168], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(403)) => (lower ? Int32[86,87,272,273] : Int32[26,30,152,168], Int16[5,2,5,2]),
            FastKey(BigInt(413)) => (lower ? Int32[108,282] : Int32[36,186], Int16[-24,-24]),
            FastKey(BigInt(421)) => (lower ? Int32[127,289] : Int32[37,187], Int16[6,6]),
            FastKey(BigInt(422)) => (lower ? Int32[86,272] : Int32[26,152], Int16[3,3]),
            FastKey(BigInt(454)) => (lower ? Int32[23,245] : Int32[38,200], Int16[-2,-2]),
            FastKey(BigInt(456)) => (lower ? Int32[23,245] : Int32[38,200], Int16[3,3]),
            FastKey(BigInt(522)) => (lower ? Int32[25,36,58,68,153,169,247,266] : Int32[40,53,102,114,189,202,207,239], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(1061)) => (lower ? Int32[36,58,153,169] : Int32[102,114,189,207], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(1064)) => (lower ? Int32[25,68,247,266] : Int32[40,53,202,239], Int16[8,8,8,8]),
            FastKey(BigInt(1066)) => (lower ? Int32[24,36,46,58,153,169,246,256] : Int32[39,45,102,114,189,201,207,219], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(1068)) => (lower ? Int32[23,24,46,245,246,256] : Int32[38,39,45,200,201,219], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(1108)) => (lower ? Int32[48,69,258,267] : Int32[47,54,221,240], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(1162)) => (lower ? Int32[89,275] : Int32[41,203], Int16[-2,-2]),
            FastKey(BigInt(1164)) => (lower ? Int32[89,275] : Int32[41,203], Int16[3,3]),
            FastKey(BigInt(1166)) => (lower ? Int32[70,268] : Int32[55,241], Int16[6,6]),
            FastKey(BigInt(1181)) => (lower ? Int32[23,245] : Int32[38,200], Int16[3,3]),
            FastKey(BigInt(1286)) => (lower ? Int32[91,102,121,128,156,172,277,290] : Int32[43,56,144,159,192,205,210,242], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(2170)) => (lower ? Int32[26,248] : Int32[62,260], Int16[-2,-2]),
            FastKey(BigInt(2172)) => (lower ? Int32[26,248] : Int32[62,260], Int16[3,3]),
            FastKey(BigInt(2567)) => (lower ? Int32[102,121,156,172] : Int32[144,159,192,210], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(2570)) => (lower ? Int32[91,128,277,290] : Int32[43,56,205,242], Int16[8,8,8,8]),
            FastKey(BigInt(2572)) => (lower ? Int32[90,102,109,121,156,172,276,283] : Int32[42,48,144,159,192,204,210,222], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(2574)) => (lower ? Int32[89,90,109,275,276,283] : Int32[41,42,48,203,204,222], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(2614)) => (lower ? Int32[111,129,285,291] : Int32[50,57,224,243], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(2672)) => (lower ? Int32[130,292] : Int32[58,244], Int16[6,6]),
            FastKey(BigInt(2687)) => (lower ? Int32[89,275] : Int32[41,203], Int16[3,3]),
            FastKey(BigInt(2693)) => (lower ? Int32[28,39,61,71,201,214,250,269] : Int32[64,83,105,117,249,262,270,305], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(4537)) => (lower ? Int32[39,61,201,214] : Int32[105,117,249,270], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(4540)) => (lower ? Int32[28,71,250,269] : Int32[64,83,262,305], Int16[8,8,8,8]),
            FastKey(BigInt(4542)) => (lower ? Int32[27,39,49,61,201,214,249,259] : Int32[63,72,105,117,249,261,270,282], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(4544)) => (lower ? Int32[26,27,49,248,249,259] : Int32[62,63,72,260,261,282], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(4554)) => (lower ? Int32[51,72,261,270] : Int32[74,84,284,306], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(4678)) => (lower ? Int32[92,278] : Int32[65,263], Int16[-2,-2]),
            FastKey(BigInt(4680)) => (lower ? Int32[92,278] : Int32[65,263], Int16[3,3]),
            FastKey(BigInt(4682)) => (lower ? Int32[73,271] : Int32[85,307], Int16[6,6]),
            FastKey(BigInt(4683)) => (lower ? Int32[26,248] : Int32[62,260], Int16[3,3]),
            FastKey(BigInt(5663)) => (lower ? Int32[94,105,124,131,204,217,280,293] : Int32[67,86,147,162,252,265,273,308], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(5813)) => (lower ? Int32[146,296] : Int32[44,206], Int16[-2,-2]),
            FastKey(BigInt(5815)) => (lower ? Int32[146,296] : Int32[44,206], Int16[3,3]),
            FastKey(BigInt(6223)) => (lower ? Int32[148,159,175,298] : Int32[59,195,213,245], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(9256)) => (lower ? Int32[105,124,204,217] : Int32[147,162,252,273], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(9259)) => (lower ? Int32[94,131,280,293] : Int32[67,86,265,308], Int16[8,8,8,8]),
            FastKey(BigInt(9261)) => (lower ? Int32[93,105,112,124,204,217,279,286] : Int32[66,75,147,162,252,264,273,285], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(9263)) => (lower ? Int32[92,93,112,278,279,286] : Int32[65,66,75,263,264,285], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(9273)) => (lower ? Int32[114,132,288,294] : Int32[77,87,287,309], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(9401)) => (lower ? Int32[133,295] : Int32[88,310], Int16[6,6]),
            FastKey(BigInt(9402)) => (lower ? Int32[92,278] : Int32[65,263], Int16[3,3]),
            FastKey(BigInt(11398)) => (lower ? Int32[159,175] : Int32[195,213], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(11401)) => (lower ? Int32[148,298] : Int32[59,245], Int16[8,8]),
            FastKey(BigInt(11403)) => (lower ? Int32[147,159,175,297] : Int32[51,195,213,225], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(11405)) => (lower ? Int32[146,147,296,297] : Int32[44,51,206,225], Int16[5,2,5,2]),
            FastKey(BigInt(11536)) => (lower ? Int32[165,303] : Int32[60,246], Int16[-24,-24]),
            FastKey(BigInt(11790)) => (lower ? Int32[181,307] : Int32[61,247], Int16[6,6]),
            FastKey(BigInt(11860)) => (lower ? Int32[146,296] : Int32[44,206], Int16[3,3]),
            FastKey(BigInt(20439)) => (lower ? Int32[149,299] : Int32[68,266], Int16[-2,-2]),
            FastKey(BigInt(20441)) => (lower ? Int32[149,299] : Int32[68,266], Int16[3,3]),
            FastKey(BigInt(22426)) => (lower ? Int32[151,162,178,182,207,220,301,308] : Int32[70,89,198,216,255,268,276,311], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(36377)) => (lower ? Int32[162,178,207,220] : Int32[198,216,255,276], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(36380)) => (lower ? Int32[151,182,301,308] : Int32[70,89,268,311], Int16[8,8,8,8]),
            FastKey(BigInt(36382)) => (lower ? Int32[150,162,166,178,207,220,300,304] : Int32[69,78,198,216,255,267,276,288], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(36384)) => (lower ? Int32[149,150,166,299,300,304] : Int32[68,69,78,266,267,288], Int16[5,2,2,5,2,2]),
            FastKey(BigInt(36424)) => (lower ? Int32[168,183,306,309] : Int32[80,90,290,312], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(36888)) => (lower ? Int32[184,310] : Int32[91,313], Int16[6,6]),
            FastKey(BigInt(36903)) => (lower ? Int32[149,299] : Int32[68,266], Int16[3,3]),
            FastKey(BigInt(55253)) => (lower ? Int32[197,311] : Int32[71,269], Int16[-2,-2]),
            FastKey(BigInt(55255)) => (lower ? Int32[197,311] : Int32[71,269], Int16[3,3]),
            FastKey(BigInt(63728)) => (lower ? Int32[199,210,223,313] : Int32[92,258,279,314], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(92458)) => (lower ? Int32[210,223] : Int32[258,279], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(92461)) => (lower ? Int32[199,313] : Int32[92,314], Int16[8,8]),
            FastKey(BigInt(92463)) => (lower ? Int32[198,210,223,312] : Int32[81,258,279,291], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(92465)) => (lower ? Int32[197,198,311,312] : Int32[71,81,269,291], Int16[5,2,5,2]),
            FastKey(BigInt(92475)) => (lower ? Int32[213,315] : Int32[93,315], Int16[-24,-24]),
            FastKey(BigInt(93269)) => (lower ? Int32[226,316] : Int32[94,316], Int16[6,6]),
            FastKey(BigInt(93270)) => (lower ? Int32[197,311] : Int32[71,269], Int16[3,3]),
        )
    elseif state.instance == 71
        @test dim == 8
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (lower ? Int32[17,43] : Int32[17,31], Int16[-2,-2]),
            FastKey(BigInt(52)) => (lower ? Int32[17,43] : Int32[17,31], Int16[6,6]),
            FastKey(BigInt(65)) => (lower ? Int32[17,43] : Int32[17,31], Int16[8,8]),
            FastKey(BigInt(79)) => (lower ? Int32[18,44] : Int32[18,36], Int16[-2,-2]),
            FastKey(BigInt(81)) => (lower ? Int32[18,44] : Int32[18,36], Int16[6,6]),
            FastKey(BigInt(84)) => (lower ? Int32[17,43] : Int32[17,31], Int16[4,4]),
            FastKey(BigInt(157)) => (lower ? Int32[18,44] : Int32[18,36], Int16[8,8]),
            FastKey(BigInt(171)) => (lower ? Int32[25,47] : Int32[19,37], Int16[-2,-2]),
            FastKey(BigInt(173)) => (lower ? Int32[25,47] : Int32[19,37], Int16[6,6]),
            FastKey(BigInt(176)) => (lower ? Int32[18,44] : Int32[18,36], Int16[4,4]),
            FastKey(BigInt(270)) => (lower ? Int32[19,23,34,45] : Int32[20,29,38,42], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(273)) => (lower ? Int32[19,23,34,45] : Int32[20,29,38,42], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(275)) => (lower ? Int32[19,23,34,45] : Int32[20,29,38,42], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(277)) => (lower ? Int32[19,23,34,45] : Int32[20,29,38,42], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(403)) => (lower ? Int32[25,47] : Int32[19,37], Int16[8,8]),
            FastKey(BigInt(422)) => (lower ? Int32[25,47] : Int32[19,37], Int16[4,4]),
            FastKey(BigInt(647)) => (lower ? Int32[19,45] : Int32[20,42], Int16[4,4]),
            FastKey(BigInt(652)) => (lower ? Int32[23,34] : Int32[29,38], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(852)) => (lower ? Int32[26,30,35,48] : Int32[21,34,39,43], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(855)) => (lower ? Int32[26,30,35,48] : Int32[21,34,39,43], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(857)) => (lower ? Int32[26,30,35,48] : Int32[21,34,39,43], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(859)) => (lower ? Int32[26,30,35,48] : Int32[21,34,39,43], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(897)) => (lower ? Int32[19,45] : Int32[20,42], Int16[2,2]),
            FastKey(BigInt(899)) => (lower ? Int32[23,34] : Int32[29,38], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(1742)) => (lower ? Int32[20,24,39,46] : Int32[23,30,45,49], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(1746)) => (lower ? Int32[20,24,39,46] : Int32[23,30,45,49], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(1749)) => (lower ? Int32[20,24,39,46] : Int32[23,30,45,49], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(1751)) => (lower ? Int32[20,24,39,46] : Int32[23,30,45,49], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(1901)) => (lower ? Int32[26,48] : Int32[21,43], Int16[4,4]),
            FastKey(BigInt(1906)) => (lower ? Int32[30,35] : Int32[34,39], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(2151)) => (lower ? Int32[26,48] : Int32[21,43], Int16[2,2]),
            FastKey(BigInt(2153)) => (lower ? Int32[30,35] : Int32[34,39], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(3358)) => (lower ? Int32[32,50] : Int32[22,44], Int16[-2,-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[32,50] : Int32[22,44], Int16[6,6]),
            FastKey(BigInt(3734)) => (lower ? Int32[20,46] : Int32[23,49], Int16[4,4]),
            FastKey(BigInt(3741)) => (lower ? Int32[24,39] : Int32[30,45], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(4250)) => (lower ? Int32[27,31,40,49] : Int32[24,35,46,50], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(4254)) => (lower ? Int32[27,31,40,49] : Int32[24,35,46,50], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(4257)) => (lower ? Int32[27,31,40,49] : Int32[24,35,46,50], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(4259)) => (lower ? Int32[27,31,40,49] : Int32[24,35,46,50], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(4285)) => (lower ? Int32[20,46] : Int32[23,49], Int16[2,2]),
            FastKey(BigInt(4290)) => (lower ? Int32[24,39] : Int32[30,45], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(6936)) => (lower ? Int32[32,50] : Int32[22,44], Int16[8,8]),
            FastKey(BigInt(8303)) => (lower ? Int32[32,50] : Int32[22,44], Int16[4,4]),
            FastKey(BigInt(8453)) => (lower ? Int32[27,49] : Int32[24,50], Int16[4,4]),
            FastKey(BigInt(8460)) => (lower ? Int32[31,40] : Int32[35,46], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(9004)) => (lower ? Int32[27,49] : Int32[24,50], Int16[2,2]),
            FastKey(BigInt(9009)) => (lower ? Int32[31,40] : Int32[35,46], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(14609)) => (lower ? Int32[33,37,41,51] : Int32[25,41,47,51], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(14612)) => (lower ? Int32[33,37,41,51] : Int32[25,41,47,51], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(14614)) => (lower ? Int32[33,37,41,51] : Int32[25,41,47,51], lower ? Int16[-3,1,-1,-3] : Int16[-3,-1,1,-3]),
            FastKey(BigInt(14618)) => (lower ? Int32[33,37,41,51] : Int32[25,41,47,51], lower ? Int16[3,1,-1,3] : Int16[3,-1,1,3]),
            FastKey(BigInt(26806)) => (lower ? Int32[33,51] : Int32[25,51], Int16[4,4]),
            FastKey(BigInt(26811)) => (lower ? Int32[37,41] : Int32[41,47], lower ? Int16[-4,4] : Int16[4,-4]),
            FastKey(BigInt(29211)) => (lower ? Int32[33,51] : Int32[25,51], Int16[2,2]),
            FastKey(BigInt(29213)) => (lower ? Int32[37,41] : Int32[41,47], lower ? Int16[-2,2] : Int16[2,-2]),
            FastKey(BigInt(47934)) => (lower ? Int32[38,52] : Int32[26,52], Int16[-2,-2]),
            FastKey(BigInt(47938)) => (lower ? Int32[38,52] : Int32[26,52], Int16[6,6]),
            FastKey(BigInt(81397)) => (lower ? Int32[38,52] : Int32[26,52], Int16[8,8]),
            FastKey(BigInt(86070)) => (lower ? Int32[38,52] : Int32[26,52], Int16[4,4]),
        )
    elseif state.instance == 72
        @test dim == 24
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (lower ? Int32[17,239] : Int32[17,107], Int16[-2,-2]),
            FastKey(BigInt(52)) => (lower ? Int32[17,239] : Int32[17,107], Int16[6,6]),
            FastKey(BigInt(54)) => (lower ? Int32[18,240] : Int32[18,120], Int16[18,18]),
            FastKey(BigInt(57)) => (lower ? Int32[19,241] : Int32[20,134], Int16[10,10]),
            FastKey(BigInt(61)) => (lower ? Int32[19,241] : Int32[20,134], Int16[8,8]),
            FastKey(BigInt(65)) => (lower ? Int32[17,239] : Int32[17,107], Int16[8,8]),
            FastKey(BigInt(75)) => (lower ? Int32[42,252] : Int32[21,135], Int16[-24,-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20,242] : Int32[23,149], Int16[-2,-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20,242] : Int32[23,149], Int16[6,6]),
            FastKey(BigInt(83)) => (lower ? Int32[21,43,64,243,253,262] : Int32[22,24,27,136,150,165], lower ? Int16[18,18,6,18,18,6] : Int16[6,18,18,6,18,18]),
            FastKey(BigInt(84)) => (lower ? Int32[17,239] : Int32[17,107], Int16[4,4]),
            FastKey(BigInt(92)) => (lower ? Int32[22,65,244,263] : Int32[25,32,151,182], Int16[10,10,10,10]),
            FastKey(BigInt(153)) => (lower ? Int32[22,65,244,263] : Int32[25,32,151,182], Int16[8,8,8,8]),
            FastKey(BigInt(157)) => (lower ? Int32[20,242] : Int32[23,149], Int16[8,8]),
            FastKey(BigInt(167)) => (lower ? Int32[45,66,255,264] : Int32[29,33,167,183], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(171)) => (lower ? Int32[86,272] : Int32[26,152], Int16[-2,-2]),
            FastKey(BigInt(173)) => (lower ? Int32[86,272] : Int32[26,152], Int16[6,6]),
            FastKey(BigInt(175)) => (lower ? Int32[67,87,265,273] : Int32[30,34,168,184], lower ? Int16[6,18,6,18] : Int16[18,6,18,6]),
            FastKey(BigInt(176)) => (lower ? Int32[20,242] : Int32[23,149], Int16[4,4]),
            FastKey(BigInt(205)) => (lower ? Int32[88,274] : Int32[35,185], Int16[10,10]),
            FastKey(BigInt(270)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(273)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(275)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(277)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(280)) => (lower ? Int32[24,46,246,256] : Int32[39,45,201,219], Int16[9,9,9,9]),
            FastKey(BigInt(282)) => (lower ? Int32[36,58,153,169] : Int32[102,114,189,207], lower ? Int16[9,9,-9,-9] : Int16[-9,-9,9,9]),
            FastKey(BigInt(399)) => (lower ? Int32[88,274] : Int32[35,185], Int16[8,8]),
            FastKey(BigInt(403)) => (lower ? Int32[86,272] : Int32[26,152], Int16[8,8]),
            FastKey(BigInt(413)) => (lower ? Int32[108,282] : Int32[36,186], Int16[-24,-24]),
            FastKey(BigInt(421)) => (lower ? Int32[127,289] : Int32[37,187], Int16[6,6]),
            FastKey(BigInt(422)) => (lower ? Int32[86,272] : Int32[26,152], Int16[4,4]),
            FastKey(BigInt(455)) => (lower ? Int32[25,68,247,266] : Int32[40,53,202,239], Int16[5,5,5,5]),
            FastKey(BigInt(457)) => (lower ? Int32[37,80,154,185] : Int32[103,128,190,227], lower ? Int16[5,5,-5,-5] : Int16[-5,-5,5,5]),
            FastKey(BigInt(640)) => (lower ? Int32[25,68,247,266] : Int32[40,53,202,239], Int16[4,4,4,4]),
            FastKey(BigInt(647)) => (lower ? Int32[23,245] : Int32[38,200], Int16[4,4]),
            FastKey(BigInt(648)) => (lower ? Int32[37,80,154,185] : Int32[103,128,190,227], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(652)) => (lower ? Int32[35,152] : Int32[101,188], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(726)) => (lower ? Int32[60,81,171,186] : Int32[116,129,209,228], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(729)) => (lower ? Int32[60,81,171,186] : Int32[116,129,209,228], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(731)) => (lower ? Int32[48,69,258,267] : Int32[47,54,221,240], Int16[12,12,12,12]),
            FastKey(BigInt(733)) => (lower ? Int32[48,69,258,267] : Int32[47,54,221,240], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(852)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(855)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(857)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(859)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(862)) => (lower ? Int32[70,90,109,268,276,283] : Int32[42,48,55,204,222,241], lower ? Int16[3,9,9,3,9,9] : Int16[9,9,3,9,9,3]),
            FastKey(BigInt(864)) => (lower ? Int32[82,102,121,156,172,187] : Int32[130,144,159,192,210,229], lower ? Int16[3,9,9,-9,-9,-3] : Int16[-3,-9,-9,9,9,3]),
            FastKey(BigInt(897)) => (lower ? Int32[23,245] : Int32[38,200], Int16[2,2]),
            FastKey(BigInt(899)) => (lower ? Int32[35,152] : Int32[101,188], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(1163)) => (lower ? Int32[91,128,277,290] : Int32[43,56,205,242], Int16[5,5,5,5]),
            FastKey(BigInt(1165)) => (lower ? Int32[103,140,157,188] : Int32[145,176,193,230], lower ? Int16[5,5,-5,-5] : Int16[-5,-5,5,5]),
            FastKey(BigInt(1742)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(1746)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(1749)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(1751)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(1762)) => (lower ? Int32[27,49,249,259] : Int32[63,72,261,282], Int16[9,9,9,9]),
            FastKey(BigInt(1767)) => (lower ? Int32[39,61,201,214] : Int32[105,117,249,270], lower ? Int16[9,9,-9,-9] : Int16[-9,-9,9,9]),
            FastKey(BigInt(1894)) => (lower ? Int32[91,128,277,290] : Int32[43,56,205,242], Int16[4,4,4,4]),
            FastKey(BigInt(1901)) => (lower ? Int32[89,275] : Int32[41,203], Int16[4,4]),
            FastKey(BigInt(1902)) => (lower ? Int32[103,140,157,188] : Int32[145,176,193,230], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(1906)) => (lower ? Int32[101,155] : Int32[143,191], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(1980)) => (lower ? Int32[123,141,174,189] : Int32[161,177,212,231], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(1983)) => (lower ? Int32[123,141,174,189] : Int32[161,177,212,231], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(1985)) => (lower ? Int32[111,129,285,291] : Int32[50,57,224,243], Int16[12,12,12,12]),
            FastKey(BigInt(1987)) => (lower ? Int32[111,129,285,291] : Int32[50,57,224,243], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(2116)) => (lower ? Int32[130,292] : Int32[58,244], Int16[3,3]),
            FastKey(BigInt(2118)) => (lower ? Int32[142,190] : Int32[178,232], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(2151)) => (lower ? Int32[89,275] : Int32[41,203], Int16[2,2]),
            FastKey(BigInt(2153)) => (lower ? Int32[101,155] : Int32[143,191], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(2491)) => (lower ? Int32[28,71,250,269] : Int32[64,83,262,305], Int16[5,5,5,5]),
            FastKey(BigInt(2496)) => (lower ? Int32[40,83,202,227] : Int32[106,131,250,293], lower ? Int16[5,5,-5,-5] : Int16[-5,-5,5,5]),
            FastKey(BigInt(3358)) => (lower ? Int32[146,296] : Int32[44,206], Int16[-2,-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[146,296] : Int32[44,206], Int16[6,6]),
            FastKey(BigInt(3374)) => (lower ? Int32[147,297] : Int32[51,225], Int16[18,18]),
            FastKey(BigInt(3723)) => (lower ? Int32[28,71,250,269] : Int32[64,83,262,305], Int16[4,4,4,4]),
            FastKey(BigInt(3730)) => (lower ? Int32[40,83,202,227] : Int32[106,131,250,293], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(3734)) => (lower ? Int32[26,248] : Int32[62,260], Int16[4,4]),
            FastKey(BigInt(3741)) => (lower ? Int32[38,200] : Int32[104,248], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(3788)) => (lower ? Int32[63,84,216,228] : Int32[119,132,272,294], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(3792)) => (lower ? Int32[63,84,216,228] : Int32[119,132,272,294], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(3795)) => (lower ? Int32[51,72,261,270] : Int32[74,84,284,306], Int16[12,12,12,12]),
            FastKey(BigInt(3797)) => (lower ? Int32[51,72,261,270] : Int32[74,84,284,306], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(4250)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(4254)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(4257)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(4259)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(4270)) => (lower ? Int32[73,93,112,271,279,286] : Int32[66,75,85,264,285,307], lower ? Int16[3,9,9,3,9,9] : Int16[9,9,3,9,9,3]),
            FastKey(BigInt(4275)) => (lower ? Int32[85,105,124,204,217,229] : Int32[133,147,162,252,273,295], lower ? Int16[3,9,9,-9,-9,-3] : Int16[-3,-9,-9,9,9,3]),
            FastKey(BigInt(4285)) => (lower ? Int32[26,248] : Int32[62,260], Int16[2,2]),
            FastKey(BigInt(4290)) => (lower ? Int32[38,200] : Int32[104,248], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(4970)) => (lower ? Int32[148,298] : Int32[59,245], Int16[10,10]),
            FastKey(BigInt(5461)) => (lower ? Int32[94,131,280,293] : Int32[67,86,265,308], Int16[5,5,5,5]),
            FastKey(BigInt(5466)) => (lower ? Int32[106,143,205,230] : Int32[148,179,253,296], lower ? Int16[5,5,-5,-5] : Int16[-5,-5,5,5]),
            FastKey(BigInt(6929)) => (lower ? Int32[148,298] : Int32[59,245], Int16[8,8]),
            FastKey(BigInt(6936)) => (lower ? Int32[146,296] : Int32[44,206], Int16[8,8]),
            FastKey(BigInt(7288)) => (lower ? Int32[165,303] : Int32[60,246], Int16[-24,-24]),
            FastKey(BigInt(8093)) => (lower ? Int32[181,307] : Int32[61,247], Int16[6,6]),
            FastKey(BigInt(8303)) => (lower ? Int32[146,296] : Int32[44,206], Int16[4,4]),
            FastKey(BigInt(8442)) => (lower ? Int32[94,131,280,293] : Int32[67,86,265,308], Int16[4,4,4,4]),
            FastKey(BigInt(8449)) => (lower ? Int32[106,143,205,230] : Int32[148,179,253,296], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(8453)) => (lower ? Int32[92,278] : Int32[65,263], Int16[4,4]),
            FastKey(BigInt(8460)) => (lower ? Int32[104,203] : Int32[146,251], lower ? Int16[4,-4] : Int16[-4,4]),
            FastKey(BigInt(8507)) => (lower ? Int32[126,144,219,231] : Int32[164,180,275,297], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(8511)) => (lower ? Int32[126,144,219,231] : Int32[164,180,275,297], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(8514)) => (lower ? Int32[114,132,288,294] : Int32[77,87,287,309], Int16[12,12,12,12]),
            FastKey(BigInt(8516)) => (lower ? Int32[114,132,288,294] : Int32[77,87,287,309], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(8989)) => (lower ? Int32[133,295] : Int32[88,310], Int16[3,3]),
            FastKey(BigInt(8994)) => (lower ? Int32[145,232] : Int32[181,298], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(9004)) => (lower ? Int32[92,278] : Int32[65,263], Int16[2,2]),
            FastKey(BigInt(9009)) => (lower ? Int32[104,203] : Int32[146,251], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(14609)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(14612)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(14614)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-3,1,-1,-3] : Int16[-3,-1,1,-3]),
            FastKey(BigInt(14618)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[3,1,-1,3] : Int16[3,-1,1,3]),
            FastKey(BigInt(14642)) => (lower ? Int32[150,166,300,304] : Int32[69,78,267,288], Int16[9,9,9,9]),
            FastKey(BigInt(14644)) => (lower ? Int32[162,178,207,220] : Int32[198,216,255,276], lower ? Int16[-9,-9,9,9] : Int16[9,9,-9,-9]),
            FastKey(BigInt(19058)) => (lower ? Int32[151,182,301,308] : Int32[70,89,268,311], Int16[5,5,5,5]),
            FastKey(BigInt(19060)) => (lower ? Int32[163,194,208,233] : Int32[199,236,256,299], lower ? Int16[-5,-5,5,5] : Int16[5,5,-5,-5]),
            FastKey(BigInt(26790)) => (lower ? Int32[151,182,301,308] : Int32[70,89,268,311], Int16[4,4,4,4]),
            FastKey(BigInt(26800)) => (lower ? Int32[163,194,208,233] : Int32[199,236,256,299], lower ? Int16[-4,-4,4,4] : Int16[4,4,-4,-4]),
            FastKey(BigInt(26806)) => (lower ? Int32[149,299] : Int32[68,266], Int16[4,4]),
            FastKey(BigInt(26811)) => (lower ? Int32[161,206] : Int32[197,254], lower ? Int16[-4,4] : Int16[4,-4]),
            FastKey(BigInt(27050)) => (lower ? Int32[180,195,222,234] : Int32[218,237,278,300], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(27053)) => (lower ? Int32[180,195,222,234] : Int32[218,237,278,300], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(27055)) => (lower ? Int32[168,183,306,309] : Int32[80,90,290,312], Int16[12,12,12,12]),
            FastKey(BigInt(27059)) => (lower ? Int32[168,183,306,309] : Int32[80,90,290,312], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(29085)) => (lower ? Int32[184,310] : Int32[91,313], Int16[3,3]),
            FastKey(BigInt(29087)) => (lower ? Int32[196,235] : Int32[238,301], lower ? Int16[-3,3] : Int16[3,-3]),
            FastKey(BigInt(29211)) => (lower ? Int32[149,299] : Int32[68,266], Int16[2,2]),
            FastKey(BigInt(29213)) => (lower ? Int32[161,206] : Int32[197,254], lower ? Int16[-2,2] : Int16[2,-2]),
            FastKey(BigInt(47934)) => (lower ? Int32[197,311] : Int32[71,269], Int16[-2,-2]),
            FastKey(BigInt(47938)) => (lower ? Int32[197,311] : Int32[71,269], Int16[6,6]),
            FastKey(BigInt(47988)) => (lower ? Int32[198,312] : Int32[81,291], Int16[18,18]),
            FastKey(BigInt(59069)) => (lower ? Int32[199,313] : Int32[92,314], Int16[10,10]),
            FastKey(BigInt(81375)) => (lower ? Int32[199,313] : Int32[92,314], Int16[8,8]),
            FastKey(BigInt(81397)) => (lower ? Int32[197,311] : Int32[71,269], Int16[8,8]),
            FastKey(BigInt(81582)) => (lower ? Int32[213,315] : Int32[93,315], Int16[-24,-24]),
            FastKey(BigInt(86000)) => (lower ? Int32[226,316] : Int32[94,316], Int16[6,6]),
            FastKey(BigInt(86070)) => (lower ? Int32[197,311] : Int32[71,269], Int16[4,4]),
        )
    elseif state.instance == 73
        @test dim == 24
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Int16}}}(
            FastKey(BigInt(50)) => (lower ? Int32[17,239] : Int32[17,107], Int16[-2,-2]),
            FastKey(BigInt(52)) => (lower ? Int32[17,239] : Int32[17,107], Int16[6,6]),
            FastKey(BigInt(57)) => (lower ? Int32[19,30,52,241] : Int32[20,96,108,134], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(58)) => (lower ? Int32[30,52] : Int32[96,108], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(61)) => (lower ? Int32[19,241] : Int32[20,134], Int16[8,8]),
            FastKey(BigInt(63)) => (lower ? Int32[18,30,52,240] : Int32[18,96,108,120], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(65)) => (lower ? Int32[17,18,239,240] : Int32[17,18,107,120], Int16[8,2,8,2]),
            FastKey(BigInt(75)) => (lower ? Int32[42,252] : Int32[21,135], Int16[-24,-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20,242] : Int32[23,149], Int16[-2,-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20,242] : Int32[23,149], Int16[6,6]),
            FastKey(BigInt(83)) => (lower ? Int32[64,262] : Int32[22,136], Int16[6,6]),
            FastKey(BigInt(84)) => (lower ? Int32[17,239] : Int32[17,107], Int16[4,4]),
            FastKey(BigInt(92)) => (lower ? Int32[22,33,55,65,96,115,244,263] : Int32[25,32,99,111,138,151,153,182], lower ? Int16[10,6,-6,10,6,-6,10,10] : Int16[10,10,-6,6,-6,10,6,10]),
            FastKey(BigInt(150)) => (lower ? Int32[33,55,96,115] : Int32[99,111,138,153], lower ? Int16[1,-1,1,-1] : Int16[-1,1,-1,1]),
            FastKey(BigInt(153)) => (lower ? Int32[22,65,244,263] : Int32[25,32,151,182], Int16[8,8,8,8]),
            FastKey(BigInt(155)) => (lower ? Int32[21,33,43,55,96,115,243,253] : Int32[24,27,99,111,138,150,153,165], lower ? Int16[-1,8,-1,-8,8,-8,-1,-1] : Int16[-1,-1,-8,8,-8,-1,8,-1]),
            FastKey(BigInt(157)) => (lower ? Int32[20,21,43,242,243,253] : Int32[23,24,27,149,150,165], Int16[8,2,2,8,2,2]),
            FastKey(BigInt(167)) => (lower ? Int32[45,66,255,264] : Int32[29,33,167,183], Int16[-24,-24,-24,-24]),
            FastKey(BigInt(171)) => (lower ? Int32[86,272] : Int32[26,152], Int16[-2,-2]),
            FastKey(BigInt(173)) => (lower ? Int32[86,272] : Int32[26,152], Int16[6,6]),
            FastKey(BigInt(175)) => (lower ? Int32[67,265] : Int32[34,184], Int16[6,6]),
            FastKey(BigInt(176)) => (lower ? Int32[20,242] : Int32[23,149], Int16[4,4]),
            FastKey(BigInt(205)) => (lower ? Int32[88,99,118,274] : Int32[35,141,156,185], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(270)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(273)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(275)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(277)) => (lower ? Int32[23,35,152,245] : Int32[38,101,188,200], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(396)) => (lower ? Int32[99,118] : Int32[141,156], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(399)) => (lower ? Int32[88,274] : Int32[35,185], Int16[8,8]),
            FastKey(BigInt(401)) => (lower ? Int32[87,99,118,273] : Int32[30,141,156,168], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(403)) => (lower ? Int32[86,87,272,273] : Int32[26,30,152,168], Int16[8,2,8,2]),
            FastKey(BigInt(413)) => (lower ? Int32[108,282] : Int32[36,186], Int16[-24,-24]),
            FastKey(BigInt(421)) => (lower ? Int32[127,289] : Int32[37,187], Int16[6,6]),
            FastKey(BigInt(422)) => (lower ? Int32[86,272] : Int32[26,152], Int16[4,4]),
            FastKey(BigInt(455)) => (lower ? Int32[25,36,58,68,153,169,247,266] : Int32[40,53,102,114,189,202,207,239], lower ? Int16[5,3,-3,5,3,-3,5,5] : Int16[5,5,-3,3,-3,5,3,5]),
            FastKey(BigInt(457)) => (lower ? Int32[24,37,46,80,154,185,246,256] : Int32[39,45,103,128,190,201,219,227], lower ? Int16[-3,5,3,5,-5,-5,-3,3] : Int16[3,-3,-5,-5,5,3,-3,5]),
            FastKey(BigInt(636)) => (lower ? Int32[24,36,46,169,246,256] : Int32[39,45,102,201,207,219], lower ? Int16[4,1,-4,-1,4,-4] : Int16[-4,4,-1,-4,1,4]),
            FastKey(BigInt(640)) => (lower ? Int32[25,68,247,266] : Int32[40,53,202,239], Int16[4,4,4,4]),
            FastKey(BigInt(643)) => (lower ? Int32[24,46,58,153,246,256] : Int32[39,45,114,189,201,219], lower ? Int16[-4,4,-1,1,-4,4] : Int16[4,-4,1,-1,4,-4]),
            FastKey(BigInt(645)) => (lower ? Int32[36,46,58,153,169,256] : Int32[39,102,114,189,201,207], lower ? Int16[-4,1,4,-4,4,1] : Int16[1,4,-4,4,1,-4]),
            FastKey(BigInt(647)) => (lower ? Int32[23,24,46,245,246,256] : Int32[38,39,45,200,201,219], Int16[4,1,1,4,1,1]),
            FastKey(BigInt(648)) => (lower ? Int32[37,80,154,185] : Int32[103,128,190,227], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(650)) => (lower ? Int32[24,36,58,153,169,246] : Int32[45,102,114,189,207,219], lower ? Int16[-1,4,-4,4,-4,-1] : Int16[-1,-4,4,-4,4,-1]),
            FastKey(BigInt(652)) => (lower ? Int32[35,36,58,152,153,169] : Int32[101,102,114,188,189,207], lower ? Int16[4,1,1,-4,-1,-1] : Int16[-4,-1,-1,4,1,1]),
            FastKey(BigInt(726)) => (lower ? Int32[60,81,171,186] : Int32[116,129,209,228], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(729)) => (lower ? Int32[60,81,171,186] : Int32[116,129,209,228], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(731)) => (lower ? Int32[48,69,258,267] : Int32[47,54,221,240], Int16[12,12,12,12]),
            FastKey(BigInt(733)) => (lower ? Int32[48,69,258,267] : Int32[47,54,221,240], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(852)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(855)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(857)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(859)) => (lower ? Int32[89,101,155,275] : Int32[41,143,191,203], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(862)) => (lower ? Int32[70,268] : Int32[55,241], Int16[3,3]),
            FastKey(BigInt(864)) => (lower ? Int32[82,187] : Int32[130,229], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(897)) => (lower ? Int32[23,245] : Int32[38,200], Int16[2,2]),
            FastKey(BigInt(899)) => (lower ? Int32[35,152] : Int32[101,188], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(1163)) => (lower ? Int32[91,102,121,128,156,172,277,290] : Int32[43,56,144,159,192,205,210,242], lower ? Int16[5,3,-3,5,3,-3,5,5] : Int16[5,5,-3,3,-3,5,3,5]),
            FastKey(BigInt(1165)) => (lower ? Int32[90,103,109,140,157,188,276,283] : Int32[42,48,145,176,193,204,222,230], lower ? Int16[-3,5,3,5,-5,-5,-3,3] : Int16[3,-3,-5,-5,5,3,-3,5]),
            FastKey(BigInt(1742)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(1746)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(1749)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(1751)) => (lower ? Int32[26,38,200,248] : Int32[62,104,248,260], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(1890)) => (lower ? Int32[90,102,109,172,276,283] : Int32[42,48,144,204,210,222], lower ? Int16[4,1,-4,-1,4,-4] : Int16[-4,4,-1,-4,1,4]),
            FastKey(BigInt(1894)) => (lower ? Int32[91,128,277,290] : Int32[43,56,205,242], Int16[4,4,4,4]),
            FastKey(BigInt(1897)) => (lower ? Int32[90,109,121,156,276,283] : Int32[42,48,159,192,204,222], lower ? Int16[-4,4,-1,1,-4,4] : Int16[4,-4,1,-1,4,-4]),
            FastKey(BigInt(1899)) => (lower ? Int32[102,109,121,156,172,283] : Int32[42,144,159,192,204,210], lower ? Int16[-4,1,4,-4,4,1] : Int16[1,4,-4,4,1,-4]),
            FastKey(BigInt(1901)) => (lower ? Int32[89,90,109,275,276,283] : Int32[41,42,48,203,204,222], Int16[4,1,1,4,1,1]),
            FastKey(BigInt(1902)) => (lower ? Int32[103,140,157,188] : Int32[145,176,193,230], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(1904)) => (lower ? Int32[90,102,121,156,172,276] : Int32[48,144,159,192,210,222], lower ? Int16[-1,4,-4,4,-4,-1] : Int16[-1,-4,4,-4,4,-1]),
            FastKey(BigInt(1906)) => (lower ? Int32[101,102,121,155,156,172] : Int32[143,144,159,191,192,210], lower ? Int16[4,1,1,-4,-1,-1] : Int16[-4,-1,-1,4,1,1]),
            FastKey(BigInt(1980)) => (lower ? Int32[123,141,174,189] : Int32[161,177,212,231], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(1983)) => (lower ? Int32[123,141,174,189] : Int32[161,177,212,231], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(1985)) => (lower ? Int32[111,129,285,291] : Int32[50,57,224,243], Int16[12,12,12,12]),
            FastKey(BigInt(1987)) => (lower ? Int32[111,129,285,291] : Int32[50,57,224,243], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(2116)) => (lower ? Int32[130,292] : Int32[58,244], Int16[3,3]),
            FastKey(BigInt(2118)) => (lower ? Int32[142,190] : Int32[178,232], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(2151)) => (lower ? Int32[89,275] : Int32[41,203], Int16[2,2]),
            FastKey(BigInt(2153)) => (lower ? Int32[101,155] : Int32[143,191], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(2491)) => (lower ? Int32[28,39,61,71,201,214,250,269] : Int32[64,83,105,117,249,262,270,305], lower ? Int16[5,3,-3,5,3,-3,5,5] : Int16[5,5,-3,3,-3,5,3,5]),
            FastKey(BigInt(2496)) => (lower ? Int32[27,40,49,83,202,227,249,259] : Int32[63,72,106,131,250,261,282,293], lower ? Int16[-3,5,3,5,-5,-5,-3,3] : Int16[3,-3,-5,-5,5,3,-3,5]),
            FastKey(BigInt(3358)) => (lower ? Int32[146,296] : Int32[44,206], Int16[-2,-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[146,296] : Int32[44,206], Int16[6,6]),
            FastKey(BigInt(3718)) => (lower ? Int32[27,39,49,214,249,259] : Int32[63,72,105,261,270,282], lower ? Int16[4,1,-4,-1,4,-4] : Int16[-4,4,-1,-4,1,4]),
            FastKey(BigInt(3723)) => (lower ? Int32[28,71,250,269] : Int32[64,83,262,305], Int16[4,4,4,4]),
            FastKey(BigInt(3727)) => (lower ? Int32[27,49,61,201,249,259] : Int32[63,72,117,249,261,282], lower ? Int16[-8,8,-2,2,-8,8] : Int16[8,-8,2,-2,8,-8]),
            FastKey(BigInt(3730)) => (lower ? Int32[40,83,202,227] : Int32[106,131,250,293], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(3732)) => (lower ? Int32[27,39,61,201,214,249] : Int32[72,105,117,249,270,282], lower ? Int16[-1,4,-4,4,-4,-1] : Int16[-1,-4,4,-4,4,-1]),
            FastKey(BigInt(3734)) => (lower ? Int32[26,27,49,248,249,259] : Int32[62,63,72,260,261,282], Int16[4,1,1,4,1,1]),
            FastKey(BigInt(3741)) => (lower ? Int32[38,39,61,200,201,214] : Int32[104,105,117,248,249,270], lower ? Int16[4,1,1,-4,-1,-1] : Int16[-4,-1,-1,4,1,1]),
            FastKey(BigInt(3788)) => (lower ? Int32[63,84,216,228] : Int32[119,132,272,294], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(3792)) => (lower ? Int32[63,84,216,228] : Int32[119,132,272,294], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(3795)) => (lower ? Int32[51,72,261,270] : Int32[74,84,284,306], Int16[12,12,12,12]),
            FastKey(BigInt(3797)) => (lower ? Int32[51,72,261,270] : Int32[74,84,284,306], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(4250)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(4254)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(4257)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[-3,-1,1,-3] : Int16[-3,1,-1,-3]),
            FastKey(BigInt(4259)) => (lower ? Int32[92,104,203,278] : Int32[65,146,251,263], lower ? Int16[3,-1,1,3] : Int16[3,1,-1,3]),
            FastKey(BigInt(4270)) => (lower ? Int32[73,271] : Int32[85,307], Int16[3,3]),
            FastKey(BigInt(4275)) => (lower ? Int32[85,229] : Int32[133,295], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(4285)) => (lower ? Int32[26,248] : Int32[62,260], Int16[2,2]),
            FastKey(BigInt(4290)) => (lower ? Int32[38,200] : Int32[104,248], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(4970)) => (lower ? Int32[148,159,175,298] : Int32[59,195,213,245], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(5461)) => (lower ? Int32[94,105,124,131,204,217,280,293] : Int32[67,86,147,162,252,265,273,308], lower ? Int16[5,3,-3,5,3,-3,5,5] : Int16[5,5,-3,3,-3,5,3,5]),
            FastKey(BigInt(5466)) => (lower ? Int32[93,106,112,143,205,230,279,286] : Int32[66,75,148,179,253,264,285,296], lower ? Int16[-3,5,3,5,-5,-5,-3,3] : Int16[3,-3,-5,-5,5,3,-3,5]),
            FastKey(BigInt(6925)) => (lower ? Int32[159,175] : Int32[195,213], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(6929)) => (lower ? Int32[148,298] : Int32[59,245], Int16[8,8]),
            FastKey(BigInt(6932)) => (lower ? Int32[147,159,175,297] : Int32[51,195,213,225], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(6936)) => (lower ? Int32[146,147,296,297] : Int32[44,51,206,225], Int16[8,2,8,2]),
            FastKey(BigInt(7288)) => (lower ? Int32[165,303] : Int32[60,246], Int16[-24,-24]),
            FastKey(BigInt(8093)) => (lower ? Int32[181,307] : Int32[61,247], Int16[6,6]),
            FastKey(BigInt(8303)) => (lower ? Int32[146,296] : Int32[44,206], Int16[4,4]),
            FastKey(BigInt(8437)) => (lower ? Int32[93,105,112,217,279,286] : Int32[66,75,147,264,273,285], lower ? Int16[4,1,-4,-1,4,-4] : Int16[-4,4,-1,-4,1,4]),
            FastKey(BigInt(8442)) => (lower ? Int32[94,131,280,293] : Int32[67,86,265,308], Int16[4,4,4,4]),
            FastKey(BigInt(8446)) => (lower ? Int32[93,112,124,204,279,286] : Int32[66,75,162,252,264,285], lower ? Int16[-8,8,-2,2,-8,8] : Int16[8,-8,2,-2,8,-8]),
            FastKey(BigInt(8449)) => (lower ? Int32[106,143,205,230] : Int32[148,179,253,296], lower ? Int16[4,4,-4,-4] : Int16[-4,-4,4,4]),
            FastKey(BigInt(8451)) => (lower ? Int32[93,105,124,204,217,279] : Int32[75,147,162,252,273,285], lower ? Int16[-1,4,-4,4,-4,-1] : Int16[-1,-4,4,-4,4,-1]),
            FastKey(BigInt(8453)) => (lower ? Int32[92,93,112,278,279,286] : Int32[65,66,75,263,264,285], Int16[4,1,1,4,1,1]),
            FastKey(BigInt(8460)) => (lower ? Int32[104,105,124,203,204,217] : Int32[146,147,162,251,252,273], lower ? Int16[4,1,1,-4,-1,-1] : Int16[-4,-1,-1,4,1,1]),
            FastKey(BigInt(8507)) => (lower ? Int32[126,144,219,231] : Int32[164,180,275,297], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(8511)) => (lower ? Int32[126,144,219,231] : Int32[164,180,275,297], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(8514)) => (lower ? Int32[114,132,288,294] : Int32[77,87,287,309], Int16[12,12,12,12]),
            FastKey(BigInt(8516)) => (lower ? Int32[114,132,288,294] : Int32[77,87,287,309], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(8989)) => (lower ? Int32[133,295] : Int32[88,310], Int16[3,3]),
            FastKey(BigInt(8994)) => (lower ? Int32[145,232] : Int32[181,298], lower ? Int16[3,-3] : Int16[-3,3]),
            FastKey(BigInt(9004)) => (lower ? Int32[92,278] : Int32[65,263], Int16[2,2]),
            FastKey(BigInt(9009)) => (lower ? Int32[104,203] : Int32[146,251], lower ? Int16[2,-2] : Int16[-2,2]),
            FastKey(BigInt(14609)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-1,3,-3,-1] : Int16[-1,-3,3,-1]),
            FastKey(BigInt(14612)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-1,-3,3,-1] : Int16[-1,3,-3,-1]),
            FastKey(BigInt(14614)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[-3,1,-1,-3] : Int16[-3,-1,1,-3]),
            FastKey(BigInt(14618)) => (lower ? Int32[149,161,206,299] : Int32[68,197,254,266], lower ? Int16[3,1,-1,3] : Int16[3,-1,1,3]),
            FastKey(BigInt(19058)) => (lower ? Int32[151,162,178,182,207,220,301,308] : Int32[70,89,198,216,255,268,276,311], lower ? Int16[5,3,-3,5,3,-3,5,5] : Int16[5,5,-3,3,-3,5,3,5]),
            FastKey(BigInt(19060)) => (lower ? Int32[150,163,166,194,208,233,300,304] : Int32[69,78,199,236,256,267,288,299], lower ? Int16[3,-5,-3,-5,5,5,3,-3] : Int16[-3,3,5,5,-5,-3,3,-5]),
            FastKey(BigInt(26786)) => (lower ? Int32[150,166,178,207,300,304] : Int32[69,78,216,255,267,288], lower ? Int16[-4,4,-1,1,-4,4] : Int16[4,-4,1,-1,4,-4]),
            FastKey(BigInt(26790)) => (lower ? Int32[151,182,301,308] : Int32[70,89,268,311], Int16[4,4,4,4]),
            FastKey(BigInt(26793)) => (lower ? Int32[150,162,166,220,300,304] : Int32[69,78,198,267,276,288], lower ? Int16[4,1,-4,-1,4,-4] : Int16[-4,4,-1,-4,1,4]),
            FastKey(BigInt(26795)) => (lower ? Int32[150,162,178,207,220,300] : Int32[78,198,216,255,276,288], lower ? Int16[1,-4,4,-4,4,1] : Int16[1,4,-4,4,-4,1]),
            FastKey(BigInt(26800)) => (lower ? Int32[163,194,208,233] : Int32[199,236,256,299], lower ? Int16[-4,-4,4,4] : Int16[4,4,-4,-4]),
            FastKey(BigInt(26804)) => (lower ? Int32[162,166,178,207,220,304] : Int32[69,198,216,255,267,276], lower ? Int16[4,-1,-4,4,-4,-1] : Int16[-1,-4,4,-4,-1,4]),
            FastKey(BigInt(26806)) => (lower ? Int32[149,150,166,299,300,304] : Int32[68,69,78,266,267,288], Int16[4,1,1,4,1,1]),
            FastKey(BigInt(26811)) => (lower ? Int32[161,162,178,206,207,220] : Int32[197,198,216,254,255,276], lower ? Int16[-4,-1,-1,4,1,1] : Int16[4,1,1,-4,-1,-1]),
            FastKey(BigInt(27050)) => (lower ? Int32[180,195,222,234] : Int32[218,237,278,300], lower ? Int16[-12,-12,12,12] : Int16[12,12,-12,-12]),
            FastKey(BigInt(27053)) => (lower ? Int32[180,195,222,234] : Int32[218,237,278,300], lower ? Int16[12,12,-12,-12] : Int16[-12,-12,12,12]),
            FastKey(BigInt(27055)) => (lower ? Int32[168,183,306,309] : Int32[80,90,290,312], Int16[12,12,12,12]),
            FastKey(BigInt(27059)) => (lower ? Int32[168,183,306,309] : Int32[80,90,290,312], Int16[-12,-12,-12,-12]),
            FastKey(BigInt(29085)) => (lower ? Int32[184,310] : Int32[91,313], Int16[3,3]),
            FastKey(BigInt(29087)) => (lower ? Int32[196,235] : Int32[238,301], lower ? Int16[-3,3] : Int16[3,-3]),
            FastKey(BigInt(29211)) => (lower ? Int32[149,299] : Int32[68,266], Int16[2,2]),
            FastKey(BigInt(29213)) => (lower ? Int32[161,206] : Int32[197,254], lower ? Int16[-2,2] : Int16[2,-2]),
            FastKey(BigInt(47934)) => (lower ? Int32[197,311] : Int32[71,269], Int16[-2,-2]),
            FastKey(BigInt(47938)) => (lower ? Int32[197,311] : Int32[71,269], Int16[6,6]),
            FastKey(BigInt(59069)) => (lower ? Int32[199,210,223,313] : Int32[92,258,279,314], lower ? Int16[10,6,-6,10] : Int16[10,-6,6,10]),
            FastKey(BigInt(81370)) => (lower ? Int32[210,223] : Int32[258,279], lower ? Int16[1,-1] : Int16[-1,1]),
            FastKey(BigInt(81375)) => (lower ? Int32[199,313] : Int32[92,314], Int16[8,8]),
            FastKey(BigInt(81379)) => (lower ? Int32[198,210,223,312] : Int32[81,258,279,291], lower ? Int16[-1,8,-8,-1] : Int16[-1,-8,8,-1]),
            FastKey(BigInt(81397)) => (lower ? Int32[197,198,311,312] : Int32[71,81,269,291], Int16[8,2,8,2]),
            FastKey(BigInt(81582)) => (lower ? Int32[213,315] : Int32[93,315], Int16[-24,-24]),
            FastKey(BigInt(86000)) => (lower ? Int32[226,316] : Int32[94,316], Int16[6,6]),
            FastKey(BigInt(86070)) => (lower ? Int32[197,311] : Int32[71,269], Int16[4,4]),
        )
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_psd!(state::SolverSetupPSDDictExplicit, dim::Int,
    data::Dict{FastKey{BigInt},<:Tuple{AbstractVector{Int8},AbstractVector{Int32},V}}) where {V<:AbstractVector{Int16}}
    tri = sos_get_tri(state)
    lower = sos_get_tri(state) === :L
    @assert(lower || tri === :U)
    @test state.lastcall === :none
    state.lastcall = :add_psd
    # Let's not make an extra test case out of this. Instead, we convert the indices to linear indices and call the other
    # method.
    data_conv = sizehint!(Dict{FastKey{BigInt},Tuple{Vector{Int32},V}}(), length(data))
    for (k, (rows, cols, vals)) in data
        newkeys = Vector{Int32}(undef, length(rows))
        for (i, (row, col)) in enumerate(zip(rows, cols))
            @inbounds newkeys[i] = lower ? (Int(sospsd_offset) - (Int(col) - Int(sospsd_offset)) *
                                                                 (Int(col) - 2dim - Int(sospsd_offset) -1) ÷ 2 +
                                            Int(row) - Int(col)) :
                                           ((Int(col) - Int(sospsd_offset)) * (1 + Int(col) - Int(sospsd_offset)) ÷ 2 +
                                            Int(row))
        end
        data_conv[k] = (newkeys, vals)
    end
    PolynomialOptimization.sos_solver_add_psd!(
        SolverSetupPSDDictLinear{Int32,tri,PolynomialOptimization.sos_solver_psd_supports_complex(state)}(:none, state.instance),
        dim, data_conv
    )
end

function PolynomialOptimization.sos_solver_add_psd!(state::SolverSetupPSDLinear, dim::Int, data::SOSPSDIterable{BigInt,Int16})
    @test state.lastcall === :none
    state.lastcall = :add_psd
end

function Base.deleteat!(fv::PolynomialOptimization.FastVec, inds)
    deleteat!(fv.data, inds)
    fv.len -= length(inds)
    return fv
end

function PolynomialOptimization.sos_solver_add_psd_complex!(state::SolverSetupPSDDictLinear{<:Any,<:Any,true}, dim::Int,
    data::Dict{FastKey{BigInt},<:Tuple{AbstractVector{Int32},AbstractVector{Complex{Int16}}}})
    tri = sos_get_tri(state)
    lower = tri === :L
    @assert(lower || tri === :U)
    @test state.lastcall === :none
    state.lastcall = :add_psd_complex
    dropkeys = FastKey{BigInt}[]
    for (key, (pos, val)) in data
        # In the complex case, cancellation may happen. Mathematica will do it implicitly, but the code generation will
        # potentially give zero-index values.
        delpos = findall(iszero, val)
        deleteat!(pos, delpos)
        deleteat!(val, delpos)
        if isempty(val)
            push!(dropkeys, key)
        else
            sort_along!(pos, val)
        end
    end
    delete!.((data,), dropkeys)
    if state.instance == 63
        @test dim == 12
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Complex{Int16}}}}(
            FastKey(BigInt(50)) => (Int32[17], Complex{Int16}[-2]),
            FastKey(BigInt(52)) => (Int32[17], Complex{Int16}[3]),
            FastKey(BigInt(57)) => (lower ? Int32[18, 19] : Int32[18, 20], lower ? Complex{Int16}[-6im, 10] : Complex{Int16}[6im, 10]),
            FastKey(BigInt(58)) => (Int32[18], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(61)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[8]),
            FastKey(BigInt(63)) => (Int32[18], lower ? Complex{Int16}[-1 - 8im] : Complex{Int16}[-1 + 8im]),
            FastKey(BigInt(65)) => (Int32[17, 18], Complex{Int16}[5, 2]),
            FastKey(BigInt(75)) => (lower ? Int32[30] : Int32[21], Complex{Int16}[-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[3]),
            FastKey(BigInt(83)) => (lower ? Int32[40] : Int32[22], Complex{Int16}[6]),
            FastKey(BigInt(84)) => (Int32[17], Complex{Int16}[3]),
            FastKey(BigInt(92)) => (lower ? Int32[21, 22, 31, 41] : Int32[24, 25, 27, 32], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(150)) => (lower ? Int32[21, 31] : Int32[24, 27], Complex{Int16}[-im,im]),
            FastKey(BigInt(153)) => (lower ? Int32[22, 41] : Int32[25, 32], Complex{Int16}[8, 8]),
            FastKey(BigInt(155)) => (lower ? Int32[21, 31] : Int32[24, 27], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(157)) => (lower ? Int32[20, 21, 31] : Int32[23, 24, 27], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(167)) => (lower ? Int32[33, 42] : Int32[29, 33], Complex{Int16}[-24, -24]),
            FastKey(BigInt(171)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[3]),
            FastKey(BigInt(175)) => (lower ? Int32[43] : Int32[34], Complex{Int16}[6]),
            FastKey(BigInt(176)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[3]),
            FastKey(BigInt(205)) => (lower ? Int32[51, 52] : Int32[30, 35], lower ? Complex{Int16}[-6im, 10] : Complex{Int16}[6im, 10]),
            FastKey(BigInt(396)) => (lower ? Int32[51] : Int32[30], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(399)) => (lower ? Int32[52] : Int32[35], Complex{Int16}[8]),
            FastKey(BigInt(401)) => (lower ? Int32[51] : Int32[30], lower ? Complex{Int16}[-1 - 8im] : Complex{Int16}[-1 + 8im]),
            FastKey(BigInt(403)) => (lower ? Int32[50, 51] : Int32[26, 30], Complex{Int16}[5, 2]),
            FastKey(BigInt(413)) => (lower ? Int32[60] : Int32[36], Complex{Int16}[-24]),
            FastKey(BigInt(421)) => (lower ? Int32[67] : Int32[37], Complex{Int16}[6]),
            FastKey(BigInt(422)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[3]),
            FastKey(BigInt(454)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[-2]),
            FastKey(BigInt(456)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[3]),
            FastKey(BigInt(522)) => (lower ? Int32[24, 25, 34, 44] : Int32[39, 40, 45, 53], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(1061)) => (lower ? Int32[24, 34] : Int32[39, 45], Complex{Int16}[-im,im]),
            FastKey(BigInt(1064)) => (lower ? Int32[25, 44] : Int32[40, 53], Complex{Int16}[8, 8]),
            FastKey(BigInt(1066)) => (lower ? Int32[24, 34] : Int32[39, 45], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(1068)) => (lower ? Int32[23, 24, 34] : Int32[38, 39, 45], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(1108)) => (lower ? Int32[36, 45] : Int32[47, 54], Complex{Int16}[-24, -24]),
            FastKey(BigInt(1162)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[-2]),
            FastKey(BigInt(1164)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[3]),
            FastKey(BigInt(1166)) => (lower ? Int32[46] : Int32[55], Complex{Int16}[6]),
            FastKey(BigInt(1181)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[3]),
            FastKey(BigInt(1286)) => (lower ? Int32[54, 55, 61, 68] : Int32[42, 43, 48, 56], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(2170)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[-2]),
            FastKey(BigInt(2172)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[3]),
            FastKey(BigInt(2567)) => (lower ? Int32[54, 61] : Int32[42, 48], Complex{Int16}[-im,im]),
            FastKey(BigInt(2570)) => (lower ? Int32[55, 68] : Int32[43, 56], Complex{Int16}[8, 8]),
            FastKey(BigInt(2572)) => (lower ? Int32[54, 61] : Int32[42, 48], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(2574)) => (lower ? Int32[53, 54, 61] : Int32[41, 42, 48], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(2614)) => (lower ? Int32[63, 69] : Int32[50, 57], Complex{Int16}[-24, -24]),
            FastKey(BigInt(2672)) => (lower ? Int32[70] : Int32[58], Complex{Int16}[6]),
            FastKey(BigInt(2687)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[3]),
            FastKey(BigInt(2693)) => (lower ? Int32[27, 28, 37, 47] : Int32[63, 64, 72, 83], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(4537)) => (lower ? Int32[27, 37] : Int32[63, 72], Complex{Int16}[-im,im]),
            FastKey(BigInt(4540)) => (lower ? Int32[28, 47] : Int32[64, 83], Complex{Int16}[8, 8]),
            FastKey(BigInt(4542)) => (lower ? Int32[27, 37] : Int32[63, 72], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(4544)) => (lower ? Int32[26, 27, 37] : Int32[62, 63, 72], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(4554)) => (lower ? Int32[39, 48] : Int32[74, 84], Complex{Int16}[-24, -24]),
            FastKey(BigInt(4678)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[-2]),
            FastKey(BigInt(4680)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[3]),
            FastKey(BigInt(4682)) => (lower ? Int32[49] : Int32[85], Complex{Int16}[6]),
            FastKey(BigInt(4683)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[3]),
            FastKey(BigInt(5663)) => (lower ? Int32[57, 58, 64, 71] : Int32[66, 67, 75, 86], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(5813)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[-2]),
            FastKey(BigInt(5815)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[3]),
            FastKey(BigInt(6223)) => (lower ? Int32[75, 76] : Int32[51, 59], lower ? Complex{Int16}[-6im, 10] : Complex{Int16}[6im, 10]),
            FastKey(BigInt(9256)) => (lower ? Int32[57, 64] : Int32[66, 75], Complex{Int16}[-im,im]),
            FastKey(BigInt(9259)) => (lower ? Int32[58, 71] : Int32[67, 86], Complex{Int16}[8, 8]),
            FastKey(BigInt(9261)) => (lower ? Int32[57, 64] : Int32[66, 75], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(9263)) => (lower ? Int32[56, 57, 64] : Int32[65, 66, 75], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(9273)) => (lower ? Int32[66, 72] : Int32[77, 87], Complex{Int16}[-24, -24]),
            FastKey(BigInt(9401)) => (lower ? Int32[73] : Int32[88], Complex{Int16}[6]),
            FastKey(BigInt(9402)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[3]),
            FastKey(BigInt(11398)) => (lower ? Int32[75] : Int32[51], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(11401)) => (lower ? Int32[76] : Int32[59], Complex{Int16}[8]),
            FastKey(BigInt(11403)) => (lower ? Int32[75] : Int32[51], lower ? Complex{Int16}[-1 - 8im] : Complex{Int16}[-1 + 8im]),
            FastKey(BigInt(11405)) => (lower ? Int32[74, 75] : Int32[44, 51], Complex{Int16}[5, 2]),
            FastKey(BigInt(11536)) => (lower ? Int32[81] : Int32[60], Complex{Int16}[-24]),
            FastKey(BigInt(11790)) => (lower ? Int32[85] : Int32[61], Complex{Int16}[6]),
            FastKey(BigInt(11860)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[3]),
            FastKey(BigInt(20439)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[-2]),
            FastKey(BigInt(20441)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[3]),
            FastKey(BigInt(22426)) => (lower ? Int32[78, 79, 82, 86] : Int32[69, 70, 78, 89], Complex{Int16}[-6im, 10, 6im, 10]),
            FastKey(BigInt(36377)) => (lower ? Int32[78, 82] : Int32[69, 78], Complex{Int16}[-im,im]),
            FastKey(BigInt(36380)) => (lower ? Int32[79, 86] : Int32[70, 89], Complex{Int16}[8, 8]),
            FastKey(BigInt(36382)) => (lower ? Int32[78, 82] : Int32[69, 78], Complex{Int16}[-1 - 8im, -1 + 8im]),
            FastKey(BigInt(36384)) => (lower ? Int32[77, 78, 82] : Int32[68, 69, 78], Complex{Int16}[5, 2, 2]),
            FastKey(BigInt(36424)) => (lower ? Int32[84, 87] : Int32[80, 90], Complex{Int16}[-24, -24]),
            FastKey(BigInt(36888)) => (lower ? Int32[88] : Int32[91], Complex{Int16}[6]),
            FastKey(BigInt(36903)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[3]),
            FastKey(BigInt(55253)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[-2]),
            FastKey(BigInt(55255)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[3]),
            FastKey(BigInt(63728)) => (lower ? Int32[90, 91] : Int32[81, 92], lower ? Complex{Int16}[-6im, 10] : Complex{Int16}[6im, 10]),
            FastKey(BigInt(92458)) => (lower ? Int32[90] : Int32[81], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(92461)) => (lower ? Int32[91] : Int32[92], Complex{Int16}[8]),
            FastKey(BigInt(92463)) => (lower ? Int32[90] : Int32[81], lower ? Complex{Int16}[-1 - 8im] : Complex{Int16}[-1 + 8im]),
            FastKey(BigInt(92465)) => (lower ? Int32[89, 90] : Int32[71, 81], Complex{Int16}[5, 2]),
            FastKey(BigInt(92475)) => (Int32[93], Complex{Int16}[-24]),
            FastKey(BigInt(93269)) => (Int32[94], Complex{Int16}[6]),
            FastKey(BigInt(93270)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[3]),
        )
    elseif state.instance == 71
        @test dim == 4
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Complex{Int16}}}}(
            FastKey(BigInt(50)) => (Int32[17], Complex{Int16}[-2]),
            FastKey(BigInt(52)) => (Int32[17], Complex{Int16}[6]),
            FastKey(BigInt(65)) => (Int32[17], Complex{Int16}[8]),
            FastKey(BigInt(79)) => (Int32[18], Complex{Int16}[-2]),
            FastKey(BigInt(81)) => (Int32[18], Complex{Int16}[6]),
            FastKey(BigInt(84)) => (Int32[17], Complex{Int16}[4]),
            FastKey(BigInt(157)) => (Int32[18], Complex{Int16}[8]),
            FastKey(BigInt(171)) => (lower ? Int32[21] : Int32[19], Complex{Int16}[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[21] : Int32[19], Complex{Int16}[6]),
            FastKey(BigInt(176)) => (Int32[18], Complex{Int16}[4]),
            FastKey(BigInt(270)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(273)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(275)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(277)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(403)) => (lower ? Int32[21] : Int32[19], Complex{Int16}[8]),
            FastKey(BigInt(422)) => (lower ? Int32[21] : Int32[19], Complex{Int16}[4]),
            FastKey(BigInt(647)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[4]),
            FastKey(BigInt(652)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(852)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(855)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(857)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(859)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(897)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[2]),
            FastKey(BigInt(899)) => (lower ? Int32[19] : Int32[20], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(1742)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(1746)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(1749)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(1751)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(1901)) => (lower ? Int32[22] : Int32[21], Complex{Int16}[4]),
            FastKey(BigInt(1906)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(2151)) => (lower ? Int32[22] : Int32[21], Complex{Int16}[2]),
            FastKey(BigInt(2153)) => (lower ? Int32[22] : Int32[21], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(3358)) => (lower ? Int32[24] : Int32[22], Complex{Int16}[-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[24] : Int32[22], Complex{Int16}[6]),
            FastKey(BigInt(3734)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[4]),
            FastKey(BigInt(3741)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(4250)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(4254)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(4257)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(4259)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(4285)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[2]),
            FastKey(BigInt(4290)) => (lower ? Int32[20] : Int32[23], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(6936)) => (lower ? Int32[24] : Int32[22], Complex{Int16}[8]),
            FastKey(BigInt(8303)) => (lower ? Int32[24] : Int32[22], Complex{Int16}[4]),
            FastKey(BigInt(8453)) => (lower ? Int32[23] : Int32[24], Complex{Int16}[4]),
            FastKey(BigInt(8460)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(9004)) => (lower ? Int32[23] : Int32[24], Complex{Int16}[2]),
            FastKey(BigInt(9009)) => (lower ? Int32[23] : Int32[24], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(14609)) => (Int32[25], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(14612)) => (Int32[25], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(14614)) => (Int32[25], lower ? Complex{Int16}[-3-im] : Complex{Int16}[-3+im]),
            FastKey(BigInt(14618)) => (Int32[25], lower ? Complex{Int16}[3-im] : Complex{Int16}[3+im]),
            FastKey(BigInt(26806)) => (Int32[25], Complex{Int16}[4]),
            FastKey(BigInt(26811)) => (Int32[25], lower ? Complex{Int16}[4im] : Complex{Int16}[-4im]),
            FastKey(BigInt(29211)) => (Int32[25], Complex{Int16}[2]),
            FastKey(BigInt(29213)) => (Int32[25], lower ? Complex{Int16}[2im] : Complex{Int16}[-2im]),
            FastKey(BigInt(47934)) => (Int32[26], Complex{Int16}[-2]),
            FastKey(BigInt(47938)) => (Int32[26], Complex{Int16}[6]),
            FastKey(BigInt(81397)) => (Int32[26], Complex{Int16}[8]),
            FastKey(BigInt(86070)) => (Int32[26], Complex{Int16}[4]),
        )
    elseif state.instance == 72
        @test dim == 12
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Complex{Int16}}}}(
            FastKey(BigInt(50)) => (Int32[17], Complex{Int16}[-2]),
            FastKey(BigInt(52)) => (Int32[17], Complex{Int16}[6]),
            FastKey(BigInt(54)) => (Int32[18], Complex{Int16}[18]),
            FastKey(BigInt(57)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[10]),
            FastKey(BigInt(61)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[8]),
            FastKey(BigInt(65)) => (Int32[17], Complex{Int16}[8]),
            FastKey(BigInt(75)) => (lower ? Int32[30] : Int32[21], Complex{Int16}[-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[6]),
            FastKey(BigInt(83)) => (lower ? Int32[21,31,40] : Int32[22,24,27], lower ? Complex{Int16}[18,18,6] : Complex{Int16}[6,18,18]),
            FastKey(BigInt(84)) => (Int32[17], Complex{Int16}[4]),
            FastKey(BigInt(92)) => (lower ? Int32[22,41] : Int32[25,32], Complex{Int16}[10,10]),
            FastKey(BigInt(153)) => (lower ? Int32[22,41] : Int32[25,32], Complex{Int16}[8,8]),
            FastKey(BigInt(157)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[8]),
            FastKey(BigInt(167)) => (lower ? Int32[33,42] : Int32[29,33], Complex{Int16}[-24,-24]),
            FastKey(BigInt(171)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[6]),
            FastKey(BigInt(175)) => (lower ? Int32[43,51] : Int32[30,34], lower ? Complex{Int16}[6,18] : Complex{Int16}[18,6]),
            FastKey(BigInt(176)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[4]),
            FastKey(BigInt(205)) => (lower ? Int32[52] : Int32[35], Complex{Int16}[10]),
            FastKey(BigInt(270)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(273)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(275)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(277)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(280)) => (lower ? Int32[24,34] : Int32[39,45], Complex{Int16}[9,9]),
            FastKey(BigInt(282)) => (lower ? Int32[24,34] : Int32[39,45], lower ? Complex{Int16}[-9im,-9im] : Complex{Int16}[9im,9im]),
            FastKey(BigInt(399)) => (lower ? Int32[52] : Int32[35], Complex{Int16}[8]),
            FastKey(BigInt(403)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[8]),
            FastKey(BigInt(413)) => (lower ? Int32[60] : Int32[36], Complex{Int16}[-24]),
            FastKey(BigInt(421)) => (lower ? Int32[67] : Int32[37], Complex{Int16}[6]),
            FastKey(BigInt(422)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[4]),
            FastKey(BigInt(455)) => (lower ? Int32[25,44] : Int32[40,53], Complex{Int16}[5,5]),
            FastKey(BigInt(457)) => (lower ? Int32[25,44] : Int32[40,53], lower ? Complex{Int16}[-5im,-5im] : Complex{Int16}[5im,5im]),
            FastKey(BigInt(640)) => (lower ? Int32[25,44] : Int32[40,53], Complex{Int16}[4,4]),
            FastKey(BigInt(647)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[4]),
            FastKey(BigInt(648)) => (lower ? Int32[25,44] : Int32[40,53], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(652)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(726)) => (lower ? Int32[36,45] : Int32[47,54], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(729)) => (lower ? Int32[36,45] : Int32[47,54], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(731)) => (lower ? Int32[36,45] : Int32[47,54], Complex{Int16}[12,12]),
            FastKey(BigInt(733)) => (lower ? Int32[36,45] : Int32[47,54], Complex{Int16}[-12,-12]),
            FastKey(BigInt(852)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(855)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(857)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(859)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(862)) => (lower ? Int32[46,54,61] : Int32[42,48,55], lower ? Complex{Int16}[3,9,9] : Complex{Int16}[9,9,3]),
            FastKey(BigInt(864)) => (lower ? Int32[46,54,61] : Int32[42,48,55], lower ? Complex{Int16}[-3im,-9im,-9im] : Complex{Int16}[9im,9im,3im]),
            FastKey(BigInt(897)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[2]),
            FastKey(BigInt(899)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(1163)) => (lower ? Int32[55,68] : Int32[43,56], Complex{Int16}[5,5]),
            FastKey(BigInt(1165)) => (lower ? Int32[55,68] : Int32[43,56], lower ? Complex{Int16}[-5im,-5im] : Complex{Int16}[5im,5im]),
            FastKey(BigInt(1742)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(1746)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(1749)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(1751)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(1762)) => (lower ? Int32[27,37] : Int32[63,72], Complex{Int16}[9,9]),
            FastKey(BigInt(1767)) => (lower ? Int32[27,37] : Int32[63,72], lower ? Complex{Int16}[-9im,-9im] : Complex{Int16}[9im,9im]),
            FastKey(BigInt(1894)) => (lower ? Int32[55,68] : Int32[43,56], Complex{Int16}[4,4]),
            FastKey(BigInt(1901)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[4]),
            FastKey(BigInt(1902)) => (lower ? Int32[55,68] : Int32[43,56], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(1906)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(1980)) => (lower ? Int32[63,69] : Int32[50,57], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(1983)) => (lower ? Int32[63,69] : Int32[50,57], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(1985)) => (lower ? Int32[63,69] : Int32[50,57], Complex{Int16}[12,12]),
            FastKey(BigInt(1987)) => (lower ? Int32[63,69] : Int32[50,57], Complex{Int16}[-12,-12]),
            FastKey(BigInt(2116)) => (lower ? Int32[70] : Int32[58], Complex{Int16}[3]),
            FastKey(BigInt(2118)) => (lower ? Int32[70] : Int32[58], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(2151)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[2]),
            FastKey(BigInt(2153)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(2491)) => (lower ? Int32[28,47] : Int32[64,83], Complex{Int16}[5,5]),
            FastKey(BigInt(2496)) => (lower ? Int32[28,47] : Int32[64,83], lower ? Complex{Int16}[-5im,-5im] : Complex{Int16}[5im,5im]),
            FastKey(BigInt(3358)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[6]),
            FastKey(BigInt(3374)) => (lower ? Int32[75] : Int32[51], Complex{Int16}[18]),
            FastKey(BigInt(3723)) => (lower ? Int32[28,47] : Int32[64,83], Complex{Int16}[4,4]),
            FastKey(BigInt(3730)) => (lower ? Int32[28,47] : Int32[64,83], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(3734)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[4]),
            FastKey(BigInt(3741)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(3788)) => (lower ? Int32[39,48] : Int32[74,84], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(3792)) => (lower ? Int32[39,48] : Int32[74,84], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(3795)) => (lower ? Int32[39,48] : Int32[74,84], Complex{Int16}[12,12]),
            FastKey(BigInt(3797)) => (lower ? Int32[39,48] : Int32[74,84], Complex{Int16}[-12,-12]),
            FastKey(BigInt(4250)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(4254)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(4257)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(4259)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(4270)) => (lower ? Int32[49,57,64] : Int32[66,75,85], lower ? Complex{Int16}[3,9,9] : Complex{Int16}[9,9,3]),
            FastKey(BigInt(4275)) => (lower ? Int32[49,57,64] : Int32[66,75,85], lower ? Complex{Int16}[-3im,-9im,-9im] : Complex{Int16}[9im,9im,3im]),
            FastKey(BigInt(4285)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[2]),
            FastKey(BigInt(4290)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(4970)) => (lower ? Int32[76] : Int32[59], Complex{Int16}[10]),
            FastKey(BigInt(5461)) => (lower ? Int32[58,71] : Int32[67,86], Complex{Int16}[5,5]),
            FastKey(BigInt(5466)) => (lower ? Int32[58,71] : Int32[67,86], lower ? Complex{Int16}[-5im,-5im] : Complex{Int16}[5im,5im]),
            FastKey(BigInt(6929)) => (lower ? Int32[76] : Int32[59], Complex{Int16}[8]),
            FastKey(BigInt(6936)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[8]),
            FastKey(BigInt(7288)) => (lower ? Int32[81] : Int32[60], Complex{Int16}[-24]),
            FastKey(BigInt(8093)) => (lower ? Int32[85] : Int32[61], Complex{Int16}[6]),
            FastKey(BigInt(8303)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[4]),
            FastKey(BigInt(8442)) => (lower ? Int32[58,71] : Int32[67,86], Complex{Int16}[4,4]),
            FastKey(BigInt(8449)) => (lower ? Int32[58,71] : Int32[67,86], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(8453)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[4]),
            FastKey(BigInt(8460)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-4im] : Complex{Int16}[4im]),
            FastKey(BigInt(8507)) => (lower ? Int32[66,72] : Int32[77,87], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(8511)) => (lower ? Int32[66,72] : Int32[77,87], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(8514)) => (lower ? Int32[66,72] : Int32[77,87], Complex{Int16}[12,12]),
            FastKey(BigInt(8516)) => (lower ? Int32[66,72] : Int32[77,87], Complex{Int16}[-12,-12]),
            FastKey(BigInt(8989)) => (lower ? Int32[73] : Int32[88], Complex{Int16}[3]),
            FastKey(BigInt(8994)) => (lower ? Int32[73] : Int32[88], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(9004)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[2]),
            FastKey(BigInt(9009)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(14609)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(14612)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(14614)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-3-im] : Complex{Int16}[-3+im]),
            FastKey(BigInt(14618)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[3-im] : Complex{Int16}[3+im]),
            FastKey(BigInt(14642)) => (lower ? Int32[78,82] : Int32[69,78], Complex{Int16}[9,9]),
            FastKey(BigInt(14644)) => (lower ? Int32[78,82] : Int32[69,78], lower ? Complex{Int16}[9im,9im] : Complex{Int16}[-9im,-9im]),
            FastKey(BigInt(19058)) => (lower ? Int32[79,86] : Int32[70,89], Complex{Int16}[5,5]),
            FastKey(BigInt(19060)) => (lower ? Int32[79,86] : Int32[70,89], lower ? Complex{Int16}[5im,5im] : Complex{Int16}[-5im,-5im]),
            FastKey(BigInt(26790)) => (lower ? Int32[79,86] : Int32[70,89], Complex{Int16}[4,4]),
            FastKey(BigInt(26800)) => (lower ? Int32[79,86] : Int32[70,89], lower ? Complex{Int16}[4im,4im] : Complex{Int16}[-4im,-4im]),
            FastKey(BigInt(26806)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[4]),
            FastKey(BigInt(26811)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[4im] : Complex{Int16}[-4im]),
            FastKey(BigInt(27050)) => (lower ? Int32[84,87] : Int32[80,90], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(27053)) => (lower ? Int32[84,87] : Int32[80,90], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(27055)) => (lower ? Int32[84,87] : Int32[80,90], Complex{Int16}[12,12]),
            FastKey(BigInt(27059)) => (lower ? Int32[84,87] : Int32[80,90], Complex{Int16}[-12,-12]),
            FastKey(BigInt(29085)) => (lower ? Int32[88] : Int32[91], Complex{Int16}[3]),
            FastKey(BigInt(29087)) => (lower ? Int32[88] : Int32[91], lower ? Complex{Int16}[3im] : Complex{Int16}[-3im]),
            FastKey(BigInt(29211)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[2]),
            FastKey(BigInt(29213)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[2im] : Complex{Int16}[-2im]),
            FastKey(BigInt(47934)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[-2]),
            FastKey(BigInt(47938)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[6]),
            FastKey(BigInt(47988)) => (lower ? Int32[90] : Int32[81], Complex{Int16}[18]),
            FastKey(BigInt(59069)) => (lower ? Int32[91] : Int32[92], Complex{Int16}[10]),
            FastKey(BigInt(81375)) => (lower ? Int32[91] : Int32[92], Complex{Int16}[8]),
            FastKey(BigInt(81397)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[8]),
            FastKey(BigInt(81582)) => (Int32[93], Complex{Int16}[-24]),
            FastKey(BigInt(86000)) => (Int32[94], Complex{Int16}[6]),
            FastKey(BigInt(86070)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[4]),
        )
    elseif state.instance == 73
        @test dim == 12
        @test data == Dict{FastKey{BigInt},Tuple{Vector{Int32},Vector{Complex{Int16}}}}(
            FastKey(BigInt(50)) => (Int32[17], Complex{Int16}[-2]),
            FastKey(BigInt(52)) => (Int32[17], Complex{Int16}[6]),
            FastKey(BigInt(57)) => (lower ? Int32[18,19] : Int32[18,20], lower ? Complex{Int16}[-6im,10] : Complex{Int16}[6im,10]),
            FastKey(BigInt(58)) => (Int32[18], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(61)) => (lower ? Int32[19] : Int32[20], Complex{Int16}[8]),
            FastKey(BigInt(63)) => (Int32[18], lower ? Complex{Int16}[-1-8im] : Complex{Int16}[-1+8im]),
            FastKey(BigInt(65)) => (Int32[17,18], Complex{Int16}[8,2]),
            FastKey(BigInt(75)) => (lower ? Int32[30] : Int32[21], Complex{Int16}[-24]),
            FastKey(BigInt(79)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[-2]),
            FastKey(BigInt(81)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[6]),
            FastKey(BigInt(83)) => (lower ? Int32[40] : Int32[22], Complex{Int16}[6]),
            FastKey(BigInt(84)) => (Int32[17], Complex{Int16}[4]),
            FastKey(BigInt(92)) => (lower ? Int32[21,22,31,41] : Int32[24,25,27,32], Complex{Int16}[-6im,10,6im,10]),
            FastKey(BigInt(150)) => (lower ? Int32[21,31] : Int32[24,27], Complex{Int16}[-im,im]),
            FastKey(BigInt(153)) => (lower ? Int32[22,41] : Int32[25,32], Complex{Int16}[8,8]),
            FastKey(BigInt(155)) => (lower ? Int32[21,31] : Int32[24,27], Complex{Int16}[-1-8im,-1+8im]),
            FastKey(BigInt(157)) => (lower ? Int32[20,21,31] : Int32[23,24,27], Complex{Int16}[8,2,2]),
            FastKey(BigInt(167)) => (lower ? Int32[33,42] : Int32[29,33], Complex{Int16}[-24,-24]),
            FastKey(BigInt(171)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[-2]),
            FastKey(BigInt(173)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[6]),
            FastKey(BigInt(175)) => (lower ? Int32[43] : Int32[34], Complex{Int16}[6]),
            FastKey(BigInt(176)) => (lower ? Int32[20] : Int32[23], Complex{Int16}[4]),
            FastKey(BigInt(205)) => (lower ? Int32[51,52] : Int32[30,35], lower ? Complex{Int16}[-6im,10] : Complex{Int16}[6im,10]),
            FastKey(BigInt(270)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(273)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(275)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(277)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(396)) => (lower ? Int32[51] : Int32[30], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(399)) => (lower ? Int32[52] : Int32[35], Complex{Int16}[8]),
            FastKey(BigInt(401)) => (lower ? Int32[51] : Int32[30], lower ? Complex{Int16}[-1-8im] : Complex{Int16}[-1+8im]),
            FastKey(BigInt(403)) => (lower ? Int32[50,51] : Int32[26,30], Complex{Int16}[8,2]),
            FastKey(BigInt(413)) => (lower ? Int32[60] : Int32[36], Complex{Int16}[-24]),
            FastKey(BigInt(421)) => (lower ? Int32[67] : Int32[37], Complex{Int16}[6]),
            FastKey(BigInt(422)) => (lower ? Int32[50] : Int32[26], Complex{Int16}[4]),
            FastKey(BigInt(455)) => (lower ? Int32[24,25,34,44] : Int32[39,40,45,53], Complex{Int16}[-3im,5,3im,5]),
            FastKey(BigInt(457)) => (lower ? Int32[24,25,34,44] : Int32[39,40,45,53], lower ? Complex{Int16}[-3,-5im,3,-5im] : Complex{Int16}[3,5im,-3,5im]),
            FastKey(BigInt(636)) => (lower ? Int32[24,34] : Int32[39,45], lower ? Complex{Int16}[4-im,-4] : Complex{Int16}[-4,4+im]),
            FastKey(BigInt(640)) => (lower ? Int32[25,44] : Int32[40,53], Complex{Int16}[4,4]),
            FastKey(BigInt(643)) => (lower ? Int32[24,34] : Int32[39,45], lower ? Complex{Int16}[-4,4+im] : Complex{Int16}[4-im,-4]),
            FastKey(BigInt(645)) => (lower ? Int32[24,34] : Int32[39,45], lower ? Complex{Int16}[4im,1-4im] : Complex{Int16}[1+4im,-4im]),
            FastKey(BigInt(647)) => (lower ? Int32[23,24,34] : Int32[38,39,45], Complex{Int16}[4,1,1]),
            FastKey(BigInt(648)) => (lower ? Int32[25,44] : Int32[40,53], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(650)) => (lower ? Int32[24,34] : Int32[39,45], lower ? Complex{Int16}[-1-4im,4im] : Complex{Int16}[-4im,-1+4im]),
            FastKey(BigInt(652)) => (lower ? Int32[23,24,34] : Int32[38,39,45], lower ? Complex{Int16}[-4im,-im,-im] : Complex{Int16}[4im,im,im]),
            FastKey(BigInt(726)) => (lower ? Int32[36,45] : Int32[47,54], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(729)) => (lower ? Int32[36,45] : Int32[47,54], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(731)) => (lower ? Int32[36,45] : Int32[47,54], Complex{Int16}[12,12]),
            FastKey(BigInt(733)) => (lower ? Int32[36,45] : Int32[47,54], Complex{Int16}[-12,-12]),
            FastKey(BigInt(852)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(855)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(857)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(859)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(862)) => (lower ? Int32[46] : Int32[55], Complex{Int16}[3]),
            FastKey(BigInt(864)) => (lower ? Int32[46] : Int32[55], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(897)) => (lower ? Int32[23] : Int32[38], Complex{Int16}[2]),
            FastKey(BigInt(899)) => (lower ? Int32[23] : Int32[38], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(1163)) => (lower ? Int32[54,55,61,68] : Int32[42,43,48,56], Complex{Int16}[-3im,5,3im,5]),
            FastKey(BigInt(1165)) => (lower ? Int32[54,55,61,68] : Int32[42,43,48,56], lower ? Complex{Int16}[-3,-5im,3,-5im] : Complex{Int16}[3,5im,-3,5im]),
            FastKey(BigInt(1742)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(1746)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(1749)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(1751)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(1890)) => (lower ? Int32[54,61] : Int32[42,48], lower ? Complex{Int16}[4-im,-4] : Complex{Int16}[-4,4+im]),
            FastKey(BigInt(1894)) => (lower ? Int32[55,68] : Int32[43,56], Complex{Int16}[4,4]),
            FastKey(BigInt(1897)) => (lower ? Int32[54,61] : Int32[42,48], lower ? Complex{Int16}[-4,4+im] : Complex{Int16}[4-im,-4]),
            FastKey(BigInt(1899)) => (lower ? Int32[54,61] : Int32[42,48], lower ? Complex{Int16}[4im,1-4im] : Complex{Int16}[1+4im,-4im]),
            FastKey(BigInt(1901)) => (lower ? Int32[53,54,61] : Int32[41,42,48], Complex{Int16}[4,1,1]),
            FastKey(BigInt(1902)) => (lower ? Int32[55,68] : Int32[43,56], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(1904)) => (lower ? Int32[54,61] : Int32[42,48], lower ? Complex{Int16}[-1-4im,4im] : Complex{Int16}[-4im,-1+4im]),
            FastKey(BigInt(1906)) => (lower ? Int32[53,54,61] : Int32[41,42,48], lower ? Complex{Int16}[-4im,-im,-im] : Complex{Int16}[4im,im,im]),
            FastKey(BigInt(1980)) => (lower ? Int32[63,69] : Int32[50,57], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(1983)) => (lower ? Int32[63,69] : Int32[50,57], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(1985)) => (lower ? Int32[63,69] : Int32[50,57], Complex{Int16}[12,12]),
            FastKey(BigInt(1987)) => (lower ? Int32[63,69] : Int32[50,57], Complex{Int16}[-12,-12]),
            FastKey(BigInt(2116)) => (lower ? Int32[70] : Int32[58], Complex{Int16}[3]),
            FastKey(BigInt(2118)) => (lower ? Int32[70] : Int32[58], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(2151)) => (lower ? Int32[53] : Int32[41], Complex{Int16}[2]),
            FastKey(BigInt(2153)) => (lower ? Int32[53] : Int32[41], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(2491)) => (lower ? Int32[27,28,37,47] : Int32[63,64,72,83], Complex{Int16}[-3im,5,3im,5]),
            FastKey(BigInt(2496)) => (lower ? Int32[27,28,37,47] : Int32[63,64,72,83], lower ? Complex{Int16}[-3,-5im,3,-5im] : Complex{Int16}[3,5im,-3,5im]),
            FastKey(BigInt(3358)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[-2]),
            FastKey(BigInt(3361)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[6]),
            FastKey(BigInt(3718)) => (lower ? Int32[27,37] : Int32[63,72], lower ? Complex{Int16}[4-im,-4] : Complex{Int16}[-4,4+im]),
            FastKey(BigInt(3723)) => (lower ? Int32[28,47] : Int32[64,83], Complex{Int16}[4,4]),
            FastKey(BigInt(3727)) => (lower ? Int32[27,37] : Int32[63,72], lower ? Complex{Int16}[-8,8+2im] : Complex{Int16}[8-2im,-8]),
            FastKey(BigInt(3730)) => (lower ? Int32[28,47] : Int32[64,83], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(3732)) => (lower ? Int32[27,37] : Int32[63,72], lower ? Complex{Int16}[-1-4im,4im] : Complex{Int16}[-4im,-1+4im]),
            FastKey(BigInt(3734)) => (lower ? Int32[26,27,37] : Int32[62,63,72], Complex{Int16}[4,1,1]),
            FastKey(BigInt(3741)) => (lower ? Int32[26,27,37] : Int32[62,63,72], lower ? Complex{Int16}[-4im,-im,-im] : Complex{Int16}[4im,im,im]),
            FastKey(BigInt(3788)) => (lower ? Int32[39,48] : Int32[74,84], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(3792)) => (lower ? Int32[39,48] : Int32[74,84], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(3795)) => (lower ? Int32[39,48] : Int32[74,84], Complex{Int16}[12,12]),
            FastKey(BigInt(3797)) => (lower ? Int32[39,48] : Int32[74,84], Complex{Int16}[-12,-12]),
            FastKey(BigInt(4250)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(4254)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(4257)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-3+im] : Complex{Int16}[-3-im]),
            FastKey(BigInt(4259)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[3+im] : Complex{Int16}[3-im]),
            FastKey(BigInt(4270)) => (lower ? Int32[49] : Int32[85], Complex{Int16}[3]),
            FastKey(BigInt(4275)) => (lower ? Int32[49] : Int32[85], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(4285)) => (lower ? Int32[26] : Int32[62], Complex{Int16}[2]),
            FastKey(BigInt(4290)) => (lower ? Int32[26] : Int32[62], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(4970)) => (lower ? Int32[75,76] : Int32[51,59], lower ? Complex{Int16}[-6im,10] : Complex{Int16}[6im,10]),
            FastKey(BigInt(5461)) => (lower ? Int32[57,58,64,71] : Int32[66,67,75,86], Complex{Int16}[-3im,5,3im,5]),
            FastKey(BigInt(5466)) => (lower ? Int32[57,58,64,71] : Int32[66,67,75,86], lower ? Complex{Int16}[-3,-5im,3,-5im] : Complex{Int16}[3,5im,-3,5im]),
            FastKey(BigInt(6925)) => (lower ? Int32[75] : Int32[51], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(6929)) => (lower ? Int32[76] : Int32[59], Complex{Int16}[8]),
            FastKey(BigInt(6932)) => (lower ? Int32[75] : Int32[51], lower ? Complex{Int16}[-1-8im] : Complex{Int16}[-1+8im]),
            FastKey(BigInt(6936)) => (lower ? Int32[74,75] : Int32[44,51], Complex{Int16}[8,2]),
            FastKey(BigInt(7288)) => (lower ? Int32[81] : Int32[60], Complex{Int16}[-24]),
            FastKey(BigInt(8093)) => (lower ? Int32[85] : Int32[61], Complex{Int16}[6]),
            FastKey(BigInt(8303)) => (lower ? Int32[74] : Int32[44], Complex{Int16}[4]),
            FastKey(BigInt(8437)) => (lower ? Int32[57,64] : Int32[66,75], lower ? Complex{Int16}[4-im,-4] : Complex{Int16}[-4,4+im]),
            FastKey(BigInt(8442)) => (lower ? Int32[58,71] : Int32[67,86], Complex{Int16}[4,4]),
            FastKey(BigInt(8446)) => (lower ? Int32[57,64] : Int32[66,75], lower ? Complex{Int16}[-8,8+2im] : Complex{Int16}[8-2im,-8]),
            FastKey(BigInt(8449)) => (lower ? Int32[58,71] : Int32[67,86], lower ? Complex{Int16}[-4im,-4im] : Complex{Int16}[4im,4im]),
            FastKey(BigInt(8451)) => (lower ? Int32[57,64] : Int32[66,75], lower ? Complex{Int16}[-1-4im,4im] : Complex{Int16}[-4im,-1+4im]),
            FastKey(BigInt(8453)) => (lower ? Int32[56,57,64] : Int32[65,66,75], Complex{Int16}[4,1,1]),
            FastKey(BigInt(8460)) => (lower ? Int32[56,57,64] : Int32[65,66,75], lower ? Complex{Int16}[-4im,-im,-im] : Complex{Int16}[4im,im,im]),
            FastKey(BigInt(8507)) => (lower ? Int32[66,72] : Int32[77,87], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(8511)) => (lower ? Int32[66,72] : Int32[77,87], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(8514)) => (lower ? Int32[66,72] : Int32[77,87], Complex{Int16}[12,12]),
            FastKey(BigInt(8516)) => (lower ? Int32[66,72] : Int32[77,87], Complex{Int16}[-12,-12]),
            FastKey(BigInt(8989)) => (lower ? Int32[73] : Int32[88], Complex{Int16}[3]),
            FastKey(BigInt(8994)) => (lower ? Int32[73] : Int32[88], lower ? Complex{Int16}[-3im] : Complex{Int16}[3im]),
            FastKey(BigInt(9004)) => (lower ? Int32[56] : Int32[65], Complex{Int16}[2]),
            FastKey(BigInt(9009)) => (lower ? Int32[56] : Int32[65], lower ? Complex{Int16}[-2im] : Complex{Int16}[2im]),
            FastKey(BigInt(14609)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-1-3im] : Complex{Int16}[-1+3im]),
            FastKey(BigInt(14612)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-1+3im] : Complex{Int16}[-1-3im]),
            FastKey(BigInt(14614)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[-3-im] : Complex{Int16}[-3+im]),
            FastKey(BigInt(14618)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[3-im] : Complex{Int16}[3+im]),
            FastKey(BigInt(19058)) => (lower ? Int32[78,79,82,86] : Int32[69,70,78,89], Complex{Int16}[-3im,5,3im,5]),
            FastKey(BigInt(19060)) => (lower ? Int32[78,79,82,86] : Int32[69,70,78,89], lower ? Complex{Int16}[3,5im,-3,5im] : Complex{Int16}[-3,-5im,3,-5im]),
            FastKey(BigInt(26786)) => (lower ? Int32[78,82] : Int32[69,78], lower ? Complex{Int16}[-4,4+im] : Complex{Int16}[4-im,-4]),
            FastKey(BigInt(26790)) => (lower ? Int32[79,86] : Int32[70,89], Complex{Int16}[4,4]),
            FastKey(BigInt(26793)) => (lower ? Int32[78,82] : Int32[69,78], lower ? Complex{Int16}[4-im,-4] : Complex{Int16}[-4,4+im]),
            FastKey(BigInt(26795)) => (lower ? Int32[78,82] : Int32[69,78], lower ? Complex{Int16}[1+4im,-4im] : Complex{Int16}[4im,1-4im]),
            FastKey(BigInt(26800)) => (lower ? Int32[79,86] : Int32[70,89], lower ? Complex{Int16}[4im,4im] : Complex{Int16}[-4im,-4im]),
            FastKey(BigInt(26804)) => (lower ? Int32[78,82] : Int32[69,78], lower ? Complex{Int16}[-4im,-1+4im] : Complex{Int16}[-1-4im,4im]),
            FastKey(BigInt(26806)) => (lower ? Int32[77,78,82] : Int32[68,69,78], Complex{Int16}[4,1,1]),
            FastKey(BigInt(26811)) => (lower ? Int32[77,78,82] : Int32[68,69,78], lower ? Complex{Int16}[4im,im,im] : Complex{Int16}[-4im,-im,-im]),
            FastKey(BigInt(27050)) => (lower ? Int32[84,87] : Int32[80,90], lower ? Complex{Int16}[12im,12im] : Complex{Int16}[-12im,-12im]),
            FastKey(BigInt(27053)) => (lower ? Int32[84,87] : Int32[80,90], lower ? Complex{Int16}[-12im,-12im] : Complex{Int16}[12im,12im]),
            FastKey(BigInt(27055)) => (lower ? Int32[84,87] : Int32[80,90], Complex{Int16}[12,12]),
            FastKey(BigInt(27059)) => (lower ? Int32[84,87] : Int32[80,90], Complex{Int16}[-12,-12]),
            FastKey(BigInt(29085)) => (lower ? Int32[88] : Int32[91], Complex{Int16}[3]),
            FastKey(BigInt(29087)) => (lower ? Int32[88] : Int32[91], lower ? Complex{Int16}[3im] : Complex{Int16}[-3im]),
            FastKey(BigInt(29211)) => (lower ? Int32[77] : Int32[68], Complex{Int16}[2]),
            FastKey(BigInt(29213)) => (lower ? Int32[77] : Int32[68], lower ? Complex{Int16}[2im] : Complex{Int16}[-2im]),
            FastKey(BigInt(47934)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[-2]),
            FastKey(BigInt(47938)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[6]),
            FastKey(BigInt(59069)) => (lower ? Int32[90,91] : Int32[81,92], lower ? Complex{Int16}[-6im,10] : Complex{Int16}[6im,10]),
            FastKey(BigInt(81370)) => (lower ? Int32[90] : Int32[81], lower ? Complex{Int16}[-im] : Complex{Int16}[im]),
            FastKey(BigInt(81375)) => (lower ? Int32[91] : Int32[92], Complex{Int16}[8]),
            FastKey(BigInt(81379)) => (lower ? Int32[90] : Int32[81], lower ? Complex{Int16}[-1-8im] : Complex{Int16}[-1+8im]),
            FastKey(BigInt(81397)) => (lower ? Int32[89,90] : Int32[71,81], Complex{Int16}[8,2]),
            FastKey(BigInt(81582)) => (Int32[93], Complex{Int16}[-24]),
            FastKey(BigInt(86000)) => (Int32[94], Complex{Int16}[6]),
            FastKey(BigInt(86070)) => (lower ? Int32[89] : Int32[71], Complex{Int16}[4]),
        )
    else
        @test false
    end
end

function PolynomialOptimization.sos_solver_add_psd_complex!(state::SolverSetupPSDDictExplicit{<:Any,<:Any,<:Any,true},
    dim::Int, data::Dict{FastKey{BigInt},<:Tuple{AbstractVector{Int8},AbstractVector{Int32},V}}) where {V<:AbstractVector{Complex{Int16}}}
    tri = sos_get_tri(state)
    lower = sos_get_tri(state) === :L
    @assert(lower || tri === :U)
    @test state.lastcall === :none
    state.lastcall = :add_psd_complex
    # Let's not make an extra test case out of this. Instead, we convert the indices to linear indices and call the other
    # method.
    data_conv = sizehint!(Dict{FastKey{BigInt},Tuple{Vector{Int32},V}}(), length(data))
    for (k, (rows, cols, vals)) in data
        newkeys = Vector{Int32}(undef, length(rows))
        for (i, (row, col)) in enumerate(zip(rows, cols))
            @inbounds newkeys[i] = lower ? (Int(sospsd_offset) - (Int(col) - Int(sospsd_offset)) *
                                                                 (Int(col) - 2dim - Int(sospsd_offset) -1) ÷ 2 +
                                            Int(row) - Int(col)) :
                                           ((Int(col) - Int(sospsd_offset)) * (1 + Int(col) - Int(sospsd_offset)) ÷ 2 +
                                            Int(row))
        end
        data_conv[k] = (newkeys, vals)
    end
    PolynomialOptimization.sos_solver_add_psd_complex!(
        SolverSetupPSDDictLinear{Int32,tri,true}(:none, state.instance),
        dim, data_conv
    )
end

function PolynomialOptimization.sos_solver_add_psd_complex!(state::SolverSetupPSDLinear{<:Any,true}, dim::Int,
    data::SOSPSDIterable{BigInt,Int16})
    @test state.lastcall === :none
    state.lastcall = :add_psd_complex
end

function sostest(instance, state_type::Type{<:SolverSetup}, expect, fn, args...)
    state = state_type(:none, instance)
    fn(state, args...)
    @test state.lastcall === expect
end

function sostest_error(err, state_type::Type{<:SolverSetup}, fn, args...)
    state = state_type(:none, -1)
    @test_throws err fn(state, args...)
    @test state.lastcall === :none || state.lastcall === :add_free_prepare
end

function sostest_psd(instance, complex, fn, args...)
    for state_type in (SolverSetupPSDDictLinear{Int32,:L}, SolverSetupPSDDictLinear{Int32,:U},
                       SolverSetupPSDDictExplicit{Int8,Int32,:L}, SolverSetupPSDDictExplicit{Int8,Int32,:U},
                    #    SolverSetupPSDLinear{:L}, SolverSetupPSDLinear{:LS}, SolverSetupPSDLinear{:U},
                    #    SolverSetupPSDLinear{:US}, SolverSetupPSDLinear{:F}
                      )
        for state_type_complex in (complex ? (state_type{true}, state_type{false}) : (state_type{Any},))
            state = state_type_complex(:none, instance)
            fn(state, args...)
            @test state.lastcall === (state_type_complex === state_type{true} ? :add_psd_complex : :add_psd)
        end
    end
end

function PolynomialOptimization.sos_solver_add_free_prepare!(state::SolverSetupFree, num::Int)
    @test state.lastcall === :none
    state.lastcall = :add_free_prepare
    state.available += num
    state.lastnum = num
    return state.used
end

function PolynomialOptimization.sos_solver_add_free!(state::SolverSetupFree, eqstate::UInt, index::AbstractVector{BigInt},
    value::AbstractVector{Int16}, obj::Bool)
    @test (iszero(eqstate) && state.lastcall === :add_free_prepare) || state.lastcall === :add_free
    state.lastcall = :add_free
    @test eqstate === state.used
    @test eqstate < state.available
    index = collect(index) # just so that we can sort if there are StackVecs
    value = collect(value)
    sort_along!(index, value)
    if state.instance == 81
        @test (eqstate == 0 && index == BigInt[57] && value == Int16[5]) ||
            (eqstate == 1 && index == BigInt[93] && value == Int16[5]) ||
            (eqstate == 2 && index == BigInt[213] && value == Int16[5])
    elseif state.instance == 82
        @test (eqstate == 0 && index == BigInt[57] && value == Int16[5]) ||
            (eqstate == 1 && index == BigInt[87] && value == Int16[5]) ||
            (eqstate == 2 && index == BigInt[89] && value == Int16[5]) ||
            (eqstate == 3 && index == BigInt[185] && value == Int16[5])
    elseif state.instance == 83
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,8]) ||
            (eqstate == 1 && index == BigInt[87,89] && value == Int16[6,8]) ||
            (eqstate == 2 && index == BigInt[207,209] && value == Int16[6,8])
    elseif state.instance == 84
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,8]) ||
            (eqstate == 1 && index == BigInt[60,65,67] && value == Int16[6,12,8]) ||
            (eqstate == 2 && index == BigInt[60,65,67] && value == Int16[-8,16,6]) ||
            (eqstate == 3 && index == BigInt[106,111] && value == Int16[6,8])
    elseif state.instance == 85
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,8]) ||
            (eqstate == 1 && index == BigInt[59,62,64,66] && value == Int16[6,6,8,8]) ||
            (eqstate == 2 && index == BigInt[59,62,64,66] && value == Int16[-8,8,-6,6]) ||
            (eqstate == 3 && index == BigInt[99,107] && value == Int16[6,8])
    elseif state.instance == 86
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,4]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[4,-3]) ||
            (eqstate == 2 && index == BigInt[87,89] && value == Int16[3,4]) ||
            (eqstate == 3 && index == BigInt[87,89] && value == Int16[4,-3]) ||
            (eqstate == 4 && index == BigInt[207,209] && value == Int16[3,4]) ||
            (eqstate == 5 && index == BigInt[207,209] && value == Int16[4,-3])
    elseif state.instance == 87
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,4]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[4,-3]) ||
            (eqstate == 2 && index == BigInt[60,65,67] && value == Int16[3,6,4]) ||
            (eqstate == 3 && index == BigInt[60,65,67] && value == Int16[4,8,-3]) ||
            (eqstate == 4 && index == BigInt[60,65,67] && value == Int16[-4,8,3]) ||
            (eqstate == 5 && index == BigInt[60,65,67] && value == Int16[3,-6,4]) ||
            (eqstate == 6 && index == BigInt[106,111] && value == Int16[3,4]) ||
            (eqstate == 7 && index == BigInt[106,111] && value == Int16[4,-3])
    elseif state.instance == 88
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,4]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[4,-3]) ||
            (eqstate == 2 && index == BigInt[59,62,64,66] && value == Int16[3,3,4,4]) ||
            (eqstate == 3 && index == BigInt[59,62,64,66] && value == Int16[4,4,-3,-3]) ||
            (eqstate == 4 && index == BigInt[59,62,64,66] && value == Int16[-4,4,-3,3]) ||
            (eqstate == 5 && index == BigInt[59,62,64,66] && value == Int16[3,-3,-4,4]) ||
            (eqstate == 6 && index == BigInt[99,107] && value == Int16[3,4]) ||
            (eqstate == 7 && index == BigInt[99,107] && value == Int16[4,-3])
    elseif state.instance == 91
        @test !obj
        @test (eqstate == 0 && index == BigInt[57] && value == Int16[6])
    elseif state.instance == 92
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,-4])
    elseif state.instance == 93
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,-2]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[2,3])
    elseif state.instance == 94
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[7,-2]) ||
            (eqstate == 1 && index == BigInt[51,53,55] && value == Int16[2,-1,2])
    elseif state.instance == 95
        @test !obj
        @test (eqstate == 0 && index == BigInt[49,58,61] && value == Int16[-10,-2,8])
    elseif state.instance == 101
        @test !obj
        @test (eqstate == 0 && index == BigInt[57] && value == Int16[6]) ||
            (eqstate == 1 && index == BigInt[93] && value == Int16[6]) ||
            (eqstate == 2 && index == BigInt[205] && value == Int16[6]) ||
            (eqstate == 3 && index == BigInt[213] && value == Int16[6]) ||
            (eqstate == 4 && index == BigInt[535] && value == Int16[6]) ||
            (eqstate == 5 && index == BigInt[1215] && value == Int16[6])
    elseif state.instance == 102
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,-4]) ||
            (eqstate == 1 && index == BigInt[87,89] && value == Int16[6,-4]) ||
            (eqstate == 2 && index == BigInt[172,174] && value == Int16[6,-4]) ||
            (eqstate == 3 && index == BigInt[207,209] && value == Int16[6,-4]) ||
            (eqstate == 4 && index == BigInt[502,504] && value == Int16[6,-4]) ||
            (eqstate == 5 && index == BigInt[1000,1002] && value == Int16[6,-4])
    elseif state.instance == 103
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,-2]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[2,3]) ||
            (eqstate == 2 && index == BigInt[87,89] && value == Int16[3,-2]) ||
            (eqstate == 3 && index == BigInt[87,89] && value == Int16[2,3]) ||
            (eqstate == 4 && index == BigInt[172,174] && value == Int16[3,-2]) ||
            (eqstate == 5 && index == BigInt[172,174] && value == Int16[2,3]) ||
            (eqstate == 6 && index == BigInt[207,209] && value == Int16[3,-2]) ||
            (eqstate == 7 && index == BigInt[207,209] && value == Int16[2,3]) ||
            (eqstate == 8 && index == BigInt[502,504] && value == Int16[3,-2]) ||
            (eqstate == 9 && index == BigInt[502,504] && value == Int16[2,3]) ||
            (eqstate == 10 && index == BigInt[1000,1002] && value == Int16[3,-2]) ||
            (eqstate == 11 && index == BigInt[1000,1002] && value == Int16[2,3])
    elseif state.instance == 104
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[7,-2]) ||
            (eqstate == 1 && index == BigInt[51,53,55] && value == Int16[2,-1,2]) ||
            (eqstate == 2 && index == BigInt[87,89] && value == Int16[7,-2]) ||
            (eqstate == 3 && index == BigInt[87,89,91] && value == Int16[2,-1,2]) ||
            (eqstate == 4 && index == BigInt[172,174] && value == Int16[7,-2]) ||
            (eqstate == 5 && index == BigInt[172,174,176] && value == Int16[2,-1,2]) ||
            (eqstate == 6 && index == BigInt[207,209] && value == Int16[7,-2]) ||
            (eqstate == 7 && index == BigInt[207,209,211] && value == Int16[2,-1,2]) ||
            (eqstate == 8 && index == BigInt[502,504] && value == Int16[7,-2]) ||
            (eqstate == 9 && index == BigInt[502,504,506] && value == Int16[2,-1,2]) ||
            (eqstate == 10 && index == BigInt[1000,1002] && value == Int16[7,-2]) ||
            (eqstate == 11 && index == BigInt[1000,1002,1004] && value == Int16[2,-1,2])
    elseif state.instance == 105
        @test !obj
        @test (eqstate == 0 && index == BigInt[49,58,61] && value == Int16[-10,-2,8]) ||
            (eqstate == 1 && index == BigInt[57,178,181] && value == Int16[-10,-2,8]) ||
            (eqstate == 2 && index == BigInt[85,396,399] && value == Int16[-10,-2,8]) ||
            (eqstate == 3 && index == BigInt[93,508,511] && value == Int16[-10,-2,8]) ||
            (eqstate == 4 && index == BigInt[205,1188,1191] && value == Int16[-10,-2,8]) ||
            (eqstate == 5 && index == BigInt[423,2232,2235] && value == Int16[-10,-2,8])
    elseif state.instance == 111
        @test !obj
        @test (eqstate == 0 && index == BigInt[57] && value == Int16[6]) ||
            (eqstate == 1 && index == BigInt[93] && value == Int16[6]) ||
            (eqstate == 2 && index == BigInt[213] && value == Int16[6]) ||
            (eqstate == 3 && index == BigInt[178] && value == Int16[6]) ||
            (eqstate == 4 && index == BigInt[183] && value == Int16[6]) ||
            (eqstate == 5 && index == BigInt[508] && value == Int16[6]) ||
            (eqstate == 6 && index == BigInt[513] && value == Int16[6]) ||
            (eqstate == 7 && index == BigInt[1015] && value == Int16[6])
    elseif state.instance == 112
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[6,-4]) ||
            (eqstate == 1 && index == BigInt[87,89] && value == Int16[6,-4]) ||
            (eqstate == 2 && index == BigInt[207,209] && value == Int16[6,-4]) ||
            (eqstate == 3 && index == BigInt[95,102,104,109] && value == Int16[6,6,-4,-4]) ||
            (eqstate == 4 && index == BigInt[95,102,104,109] && value == Int16[4,-4,-6,6]) ||
            (eqstate == 5 && index == BigInt[425,432,434,439] && value == Int16[6,6,-4,-4]) ||
            (eqstate == 6 && index == BigInt[425,432,434,439] && value == Int16[4,-4,-6,6]) ||
            (eqstate == 7 && index == BigInt[556,574] && value == Int16[6,-4])
    elseif state.instance == 113
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[3,-2]) ||
            (eqstate == 1 && index == BigInt[51,53] && value == Int16[2,3]) ||
            (eqstate == 2 && index == BigInt[87,89] && value == Int16[3,-2]) ||
            (eqstate == 3 && index == BigInt[87,89] && value == Int16[2,3]) ||
            (eqstate == 4 && index == BigInt[207,209] && value == Int16[3,-2]) ||
            (eqstate == 5 && index == BigInt[207,209] && value == Int16[2,3]) ||
            (eqstate == 6 && index == BigInt[95,102,104,109] && value == Int16[3,3,-2,-2]) ||
            (eqstate == 7 && index == BigInt[95,102,104,109] && value == Int16[2,2,3,3]) ||
            (eqstate == 8 && index == BigInt[95,102,104,109] && value == Int16[2,-2,-3,3]) ||
            (eqstate == 9 && index == BigInt[95,102,104,109] && value == Int16[-3,3,-2,2]) ||
            (eqstate == 10 && index == BigInt[425,432,434,439] && value == Int16[3,3,-2,-2]) ||
            (eqstate == 11 && index == BigInt[425,432,434,439] && value == Int16[2,2,3,3]) ||
            (eqstate == 12 && index == BigInt[425,432,434,439] && value == Int16[2,-2,-3,3]) ||
            (eqstate == 13 && index == BigInt[425,432,434,439] && value == Int16[-3,3,-2,2]) ||
            (eqstate == 14 && index == BigInt[556,574] && value == Int16[3,-2]) ||
            (eqstate == 15 && index == BigInt[556,574] && value == Int16[2,3])
    elseif state.instance == 114
        @test !obj
        @test (eqstate == 0 && index == BigInt[51,53] && value == Int16[7,-2]) ||
            (eqstate == 1 && index == BigInt[51,53,55] && value == Int16[2,-1,2]) ||
            (eqstate == 2 && index == BigInt[87,89] && value == Int16[7,-2]) ||
            (eqstate == 3 && index == BigInt[87,89,91] && value == Int16[2,-1,2]) ||
            (eqstate == 4 && index == BigInt[207,209] && value == Int16[7,-2]) ||
            (eqstate == 5 && index == BigInt[207,209,211] && value == Int16[2,-1,2]) ||
            (eqstate == 6 && index == BigInt[95,102,104,109] && value == Int16[7,7,-2,-2]) ||
            (eqstate == 7 && index == BigInt[95,102,104,109,129] && value == Int16[2,2,-1,-1,2]) ||
            (eqstate == 8 && index == BigInt[95,102,104,109] && value == Int16[2,-2,-7,7]) ||
            (eqstate == 9 && index == BigInt[95,102,104,109,134] && value == Int16[1,-1,-2,2,2]) ||
            (eqstate == 10 && index == BigInt[425,432,434,439] && value == Int16[7,7,-2,-2]) ||
            (eqstate == 11 && index == BigInt[425,432,434,439,459] && value == Int16[2,2,-1,-1,2]) ||
            (eqstate == 12 && index == BigInt[425,432,434,439] && value == Int16[2,-2,-7,7]) ||
            (eqstate == 13 && index == BigInt[425,432,434,439,464] && value == Int16[1,-1,-2,2,2]) ||
            (eqstate == 14 && index == BigInt[556,574] && value == Int16[7,-2]) ||
            (eqstate == 15 && index == BigInt[556,574,679] && value == Int16[2,-1,2])
    elseif state.instance == 115
        @test !obj
        @test (eqstate == 0 && index == BigInt[49,58,61] && value == Int16[-10,-2,8]) ||
            (eqstate == 1 && index == BigInt[57,178,181] && value == Int16[-10,-2,8]) ||
            (eqstate == 2 && index == BigInt[93,508,511] && value == Int16[-10,-2,8]) ||
            (eqstate == 3 && index == BigInt[58,214,219,223] && value == Int16[-10,-2,8,-4]) ||
            (eqstate == 4 && index == BigInt[63,226,228] && value == Int16[-10,8,-2]) ||
            (eqstate == 5 && index == BigInt[178,1006,1011,1015] && value == Int16[-10,-2,8,-4]) ||
            (eqstate == 6 && index == BigInt[183,1018,1020] && value == Int16[-10,8,-2]) ||
            (eqstate == 7 && index == BigInt[223,1349,1354] && value == Int16[-10,-2,8])
    else
        @test false
    end
    return state.used += 1
end

function PolynomialOptimization.sos_solver_add_free_finalize!(state::SolverSetupFree, num::Int, eqstate::UInt)
    @test state.lastcall === :add_free
    state.lastcall = :add_free_finalize
    @test eqstate === state.used
    @test state.used === state.available
    @test num === state.lastnum
end

@testset "Grouping of length 1 ($text)" for (grouping, text, offset) in (
    (SimpleMonomialVector{4,2}(UInt8[0; 2; 1; 0;;], UInt8[0; 0;;], UInt8[0; 0;;]), "real", 0),
    (SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;;], UInt8[0; 0;;], UInt8[1; 0;;]), "complex", 10)
)
    # the grouping is x₂²x₃ or x₂²z₁
    @testset "Single term constraint" begin
        # 1 - the constraint is 17x₁x₃²
        constraint = SimplePolynomial(Int16[17], SimpleMonomialVector{4,2}(UInt8[1; 0; 2; 0;;], UInt8[0; 0;;],
            UInt8[0; 0;;]))
        sostest(1 + offset, SolverSetupScalar{true}, :add_scalar, sos_add_matrix!, grouping, constraint)
        sostest(1 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
    end
    @testset "Single term constraint (complex)" begin
        # 3 - the constraint is 24x₂z₁z̄₁
        constraint = SimplePolynomial(Int16[24], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 0;;], UInt8[1; 0;;],
            UInt8[1; 0;;]))
        sostest(3 + offset, SolverSetupScalar{true}, :add_scalar, sos_add_matrix!, grouping, constraint)
        sostest(3 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
        # 4 - the constraint is (17+8im)x₁x₃²
        constraint = SimplePolynomial(Complex{Int16}[17+8im], SimpleMonomialVector{4,2}(UInt8[1; 0; 2; 0;;], UInt8[0; 0;;],
            UInt8[0; 0;;]))
        sostest_error(AssertionError, SolverSetupScalar{Any}, sos_add_matrix!, grouping, constraint)
        # 5 - the constraint is 3im*z₁z̄₁
        constraint = SimplePolynomial(Complex{Int16}[3im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;], UInt8[1; 0;;],
            UInt[1; 0;;]))
        sostest_error(AssertionError, SolverSetupScalar{Any}, sos_add_matrix!, grouping, constraint)
    end
    @testset "Multiple term constraint" begin
        # 6 - the constraint is 2x₁ + 8x₂x₄²
        constraint = SimplePolynomial(Int16[2, 8], SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;; 0; 1; 0; 2],
            UInt8[0; 0;; 0; 0], UInt8[0; 0;; 0; 0]))
        sostest(6 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
        # 7 - the constraint is 3(z₁ + z̄₁)
        constraint = SimplePolynomial(Int16[3, 3], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0],
            UInt8[1; 0;; 0; 0], UInt8[0; 0;; 1; 0]))
        sostest(7 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
        # 8 - the constraint is 7im(z₁z̄₂² - z̄₁z₂²)
        constraint = SimplePolynomial(Complex{Int16}[7im, -7im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 2),
            UInt8[1; 0;; 0; 2], UInt8[0; 2;; 1; 0]))
        sostest(8 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
        # 9 - the constraint is 5im(z₁z̄₂² + z₁z̄₁)
        constraint = SimplePolynomial(Complex{Int16}[5im, 5im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 2),
            UInt8[1; 0;; 1; 0], UInt8[0; 2;; 1; 0]))
        sostest_error(AssertionError, SolverSetupScalar{Any}, sos_add_matrix!, grouping, constraint)
        # 10 - the constraint is 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁
        constraint = SimplePolynomial(Complex{Int16}[3, 5, 4, 7, 7, 4+8im, 4-8im],
            SimpleMonomialVector{4,2}(
                UInt8[1; 0; 0; 0;; 0; 1; 0; 2;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0],
                UInt8[0; 0      ;; 0; 0      ;; 0; 1      ;; 2; 0      ;; 0; 1      ;; 1; 0      ;; 0; 0],
                UInt8[0; 0      ;; 0; 0      ;; 0; 1      ;; 0; 1      ;; 2; 0      ;; 0; 0      ;; 1; 0]))
        sostest(10 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
    end
end

@testset "Grouping of length 2 ($text)" for (grouping, text, offset) in (
    (SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 2;; 0; 2; 3; 0], UInt8[0; 0;; 0; 0], UInt8[0; 0;; 0; 0]), "real", 20),
    (SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 2;; 0; 2; 0; 0], UInt8[0; 0;; 1; 0], UInt8[0; 0;; 0; 0]), "complex", 30)
)
    # the grouping is [x₁x₄², x₂²x₃³] or [x₁x₄², x₂²z₁]
    @testset "Single term constraint" begin
        # 1 - the constraint is 17x₁x₃²
        constraint = SimplePolynomial(Int16[17], SimpleMonomialVector{4,2}(UInt8[1; 0; 2; 0;;], UInt8[0; 0;;],
            UInt8[0; 0;;]))
        sostest(1 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        sostest(1 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    end
    @testset "Single term constraint (complex)" begin
        # 3 - the constraint is 24x₂z₁z̄₁
        constraint = SimplePolynomial(Int16[24], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 0;;], UInt8[1; 0;;],
            UInt8[1; 0;;]))
        sostest(3 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        sostest(3 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        # 4 - the constraint is (17+8im)x₁x₃²
        constraint = SimplePolynomial(Complex{Int16}[17+8im], SimpleMonomialVector{4,2}(UInt8[1; 0; 2; 0;;], UInt8[0; 0;;],
            UInt8[0; 0;;]))
        sostest_error(AssertionError, SolverSetupQuadratic{Any}, sos_add_matrix!, grouping, constraint)
        # 5 - the constraint is 3im*z₁z̄₁
        constraint = SimplePolynomial(Complex{Int16}[3im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;], UInt8[1; 0;;],
            UInt[1; 0;;]))
        sostest_error(AssertionError, SolverSetupQuadratic{Any}, sos_add_matrix!, grouping, constraint)
    end
    @testset "Multiple term constraint" begin
        # 6 - the constraint is 2x₁ + 8x₂x₄²
        constraint = SimplePolynomial(Int16[2, 8], SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;; 0; 1; 0; 2],
            UInt8[0; 0;; 0; 0], UInt8[0; 0;; 0; 0]))
        sostest(6 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        # 7 - the constraint is 3(z₁ + z̄₁)
        constraint = SimplePolynomial(Int16[3, 3], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0],
            UInt8[1; 0;; 0; 0], UInt8[0; 0;; 1; 0]))
        sostest(7 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        # 8 - the constraint is 7im(z₁z̄₂² - z̄₁z₂²)
        constraint = SimplePolynomial(Complex{Int16}[7im, -7im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 2),
            UInt8[1; 0;; 0; 2], UInt8[0; 2;; 1; 0]))
        sostest(8 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
        # 9 - the constraint is 5im(z₁z̄₂² + z₁z̄₁)
        constraint = SimplePolynomial(Complex{Int16}[5im, 5im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 2),
            UInt8[1; 0;; 1; 0], UInt8[0; 2;; 1; 0]))
        sostest_error(AssertionError, SolverSetupQuadratic{Any}, sos_add_matrix!, grouping, constraint)
        # 10 - the constraint is 3x₁ + 5x₂x₄² + 4z₂z̄₂ + 7(z₁²z̄₂ + z̄₁²z₂) + (4 + 8im)z₁ + (4 - 8im)z̄₁
        constraint = SimplePolynomial(Complex{Int16}[3, 5, 4, 7, 7, 4+8im, 4-8im],
            SimpleMonomialVector{4,2}(
                UInt8[1; 0; 0; 0;; 0; 1; 0; 2;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0],
                UInt8[0; 0      ;; 0; 0      ;; 0; 1      ;; 2; 0      ;; 0; 1      ;; 1; 0      ;; 0; 0],
                UInt8[0; 0      ;; 0; 0      ;; 0; 1      ;; 0; 1      ;; 2; 0      ;; 0; 0      ;; 1; 0]))
        sostest(10 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    end
end

@testset "Grouping of length 1, matrix constraint" begin
    offset = 40
    grouping = SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;;], UInt8[0; 1;;], UInt8[0; 0;;])
    # the grouping is x₂²z₂
    c₁₁ = SimplePolynomial(Int16[17], SimpleMonomialVector{4,2}(UInt8[0; 0; 2; 0;;], UInt8[1; 0;;], UInt8[1; 0;;]))
    c₁₁c = SimplePolynomial(convert(Vector{Complex{Int16}}, coefficients(c₁₁)), monomials(c₁₁))
    c₁₂ = SimplePolynomial(Int16[8], SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    c₂₂ = SimplePolynomial(Int16[6], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 2;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    c₂₂c = SimplePolynomial(convert(Vector{Complex{Int16}}, coefficients(c₂₂)), monomials(c₂₂))
    z = SimplePolynomial(Int16[0], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 1), zeros(UInt8, 2, 1), zeros(UInt8, 2, 1)))
    # 1 - the constraint is [17x₃²z₁z̄₁  8x₂²
    #                        8x₂²       6x₂x₄²]
    constraint = [c₁₁ c₁₂; c₁₂ c₂₂]
    sostest(1 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(1 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 2 - the constraint is [0 0; 0 0]
    constraint = [z z; z z]
    sostest(2 + offset, SolverSetupQuadratic{Any}, :none, sos_add_matrix!, grouping, constraint)
    # 3 - the constraint is [17x₃²z₁z̄₁  0
    #                        0          6x₂x₄²]
    constraint = [c₁₁ z; z c₂₂]
    sostest(3 + offset, SolverSetupScalar{true}, :add_scalar2, sos_add_matrix!, grouping, constraint)
    sostest(3 + offset, SolverSetupScalar{false}, :add_scalar2, sos_add_matrix!, grouping, constraint)
    # 4 - the constraint is [17x₃²z₁z̄₁  0
    #                        0          0]
    constraint = [c₁₁ z; z z]
    sostest(4 + offset, SolverSetupScalar{true}, :add_scalar, sos_add_matrix!, grouping, constraint)
    sostest(4 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
    # 5 - the constraint is [0  0
    #                        0  6x₂x₄²]
    constraint = [z z; z c₂₂]
    sostest(5 + offset, SolverSetupScalar{true}, :add_scalar, sos_add_matrix!, grouping, constraint)
    sostest(5 + offset, SolverSetupScalar{false}, :add_scalar, sos_add_matrix!, grouping, constraint)
    # 6 - the constraint is [17x₃²z₁z̄₁  8x₂²
    #                        8x₂²       0]
    constraint = [c₁₁ c₁₂; c₁₂ z]
    sostest(6 + offset, SolverSetupQuadraticSimplified{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(6 + offset, SolverSetupQuadraticSimplified{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(6 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 7 - the constraint is [0     8x₂²
    #                        8x₂²  6x₂x₄²]
    constraint = [z c₁₂; c₁₂ c₂₂]
    sostest(7 + offset, SolverSetupQuadraticSimplified{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(7 + offset, SolverSetupQuadraticSimplified{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(7 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 8 - the constraint is [17x₃²z₁  8x₂²
    #                        8x₂²     6x₂x₄²]
    constraint = [SimplePolynomial(Int16[17], SimpleMonomialVector{4,2}(UInt8[1; 0; 2; 0;;], UInt8[1; 0;;],
                                                                        UInt8[0; 0;;])) c₁₂; c₁₂ c₂₂]
    sostest_error(AssertionError, SolverSetupQuadratic{Any}, sos_add_matrix!, grouping, constraint)
    # 9 - the constraint is [17x₃²z₁z̄₁  8x₂²
    #                        8x₂²       6x₄²z₂]
    constraint = [c₁₁ c₁₂; c₁₂ SimplePolynomial(Int16[6], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 0;;], UInt8[0; 1;;],
                                                                                    UInt8[0; 0;;]))]
    sostest_error(AssertionError, SolverSetupQuadratic{Any}, sos_add_matrix!, grouping, constraint)
    # 10 - the constraint is [17x₃²z₁z̄₁  8im*x₂²
    #                         -8im*x₂²   6x₂x₄²]
    constraint = let c₁₂=SimplePolynomial(Complex{Int16}[8im], SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;;],
                                                                                         UInt8[0; 0;;], UInt8[0; 0;;]))
        [c₁₁c c₁₂; conj(c₁₂) c₂₂c]
    end
    sostest(10 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(10 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 11 - the constraint is [17x₃²z₁z̄₁  8z₂²
    #                         8z̄₂²       6x₂x₄²]
    constraint = let c₁₂=SimplePolynomial(Int16[8], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;],
                                                                                UInt8[0; 2;;], UInt8[0; 0;;]))
        [c₁₁ c₁₂; conj(c₁₂) c₂₂]
    end
    sostest(11 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(11 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 12 - the constraint is [17x₃²z₁z̄₁  8im*z₂²
    #                         -8im*z̄₂²   6x₂x₄²]
    constraint = let c₁₂=SimplePolynomial(Complex{Int16}[8im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;],
                                                                                            UInt8[0; 2;;], UInt8[0; 0;;]))
        [c₁₁c c₁₂; conj(c₁₂) c₂₂c]
    end
    sostest(12 + offset, SolverSetupQuadratic{true}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    sostest(12 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 13 - the constraint is [17x₃²z₁z̄₁    (8+2im)*z₂²
    #                         (8-2im)*z̄₂²  6x₂x₄²]
    constraint = let c₁₂=SimplePolynomial(Complex{Int16}[8+2im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;],
                                                                                           UInt8[0; 2;;], UInt8[0; 0;;]))
        [c₁₁c c₁₂; conj(c₁₂) c₂₂c]
    end
    sostest(13 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 14 - the constraint is [5x₂²+17x₃²z₁z̄₁                (8-2im)*z̄₂²-8z₂²-3im*x₁+z₁z̄₁
    #                         (8+2im)*z₂²-8z̄₂²+3im*x₁+z₁z̄₁  6x₂x₄²]
    c₁₁c = SimplePolynomial(Complex{Int16}[5, 17], SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;; 0; 0; 2; 0],
                                                                             UInt8[0; 0;; 1; 0], UInt8[0; 0;; 1; 0]))
    constraint = let c₁₂=SimplePolynomial(Complex{Int16}[8-2im, -8, -3im, 1],
                                          SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0;; 1; 0; 0; 0;; 0; 0; 0; 0],
                                                                    UInt8[0; 0;; 0; 2;; 0; 0;; 1; 0],
                                                                    UInt8[0; 2;; 0; 0;; 0; 0;; 1; 0]))
        [c₁₁c c₁₂; conj(c₁₂) c₂₂c]
    end
    sostest(14 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
    # 15 - the constraint is [5x₂²z₁+17im*x₃²z̄₁+5x₂²z̄₁-17im*x₃²z₁ (8-2im)*z̄₂²+(8+2im)z₂²
    #                         (8-2im)*z̄₂²+(8+2im)z₂²              6x₂x₄²]
    c₁₁c = SimplePolynomial(Complex{Int16}[5, 17im, 5, -17im], SimpleMonomialVector{4,2}(UInt8[0; 2; 0; 0;; 0; 0; 2; 0;;
                                                                                               0; 2; 0; 0;; 0; 0; 2; 0],
                                                                                         UInt8[1; 0;; 0; 0;; 0; 0;; 1; 0],
                                                                                         UInt8[0; 0;; 1; 0;; 1; 0;; 0; 0]))
    constraint = let c₁₂=SimplePolynomial(Complex{Int16}[8-2im, 8+2im],
                                          SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0],
                                                                    UInt8[0; 0;; 0; 2], UInt8[0; 2;; 0; 0]))
        [c₁₁c c₁₂; conj(c₁₂) c₂₂c]
    end
    sostest(15 + offset, SolverSetupQuadratic{false}, :add_quadratic, sos_add_matrix!, grouping, constraint)
end

# TODO: fallback quadratic -> SOS

@testset "Real grouping of length 4" begin
    offset = 60
    grouping = SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 1; 0; 0;; 1; 0; 0; 2;; 0; 2; 3; 0],
                                         zeros(UInt8, 2, 4), zeros(UInt8, 2, 4))
    # the grouping is [1, x₂, x₁x₄², x₂²x₃³]
    c₁₁ = SimplePolynomial(Complex{Int16}[3, 5, -(2+3im), -(2-3im)],
                           SimpleMonomialVector{4,2}(UInt8[0; 1; 1; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0],
                                                     UInt8[0; 0;; 1; 0;; 0; 1;; 0; 0],
                                                     UInt8[0; 0;; 1; 0;; 0; 0;; 0; 1]))
    # 61 - the constraint is 3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂
    sostest_psd(1 + offset, false, sos_add_matrix!, grouping, c₁₁)

    # 62 - the constraint is [3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂  17x₄                     10x₁+8z₂z̄₂
    #                         17x₄                               0                        24im*x₃z₂-23im*x₃z̄₂
    #                         10x₁+8z₂z̄₂                          -24im*x₃z̄₂+23im*x₃z₂     6x₂x₄]
    c₁₂ = SimplePolynomial(Complex{Int16}[17], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 1;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    c₁₃ = SimplePolynomial(Complex{Int16}[10, 8],
                           SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;; 0; 0; 0; 0], UInt8[0; 0;; 0; 1], UInt8[0; 0;; 0; 1]))
    c₂₂ = SimplePolynomial(Complex{Int16}[], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 0), zeros(UInt8, 2, 0),
                                                                       zeros(UInt8, 2, 0)))
    c₂₃ = SimplePolynomial(Complex{Int16}[24im, -24im],
                           SimpleMonomialVector{4,2}(UInt8[0; 0; 1; 0;; 0; 0; 1; 0], UInt8[0; 1;; 0; 0], UInt8[0; 0;; 0; 1]))
    c₃₃ = SimplePolynomial(Complex{Int16}[6], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 1;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    sostest_psd(2 + offset, false, sos_add_matrix!, grouping, [c₁₁ c₁₂ c₁₃; conj(c₁₂) c₂₂ c₂₃; conj(c₁₃) conj(c₂₃) c₃₃])

    # 64 - the constraint is [3x₂x₃+5z₁z̄₁-(2+3im)*z₂-(2-3im)*z̄₂  (8-2im)*z̄₂²-8z₂²-6im*x₁+2z₁z̄₁  10x₁+8z₂z̄₂
    #                         (8+2im)*z₂²-8z̄₂²+6im*x₁+2z₁z̄₁       0                             24im*x₃z₂-24im*x₃z̄₂
    #                         10x₁+8z₂z̄₂                          -24im*x₃z̄₂+24im*x₃z₂           6x₂x₄]
    # Note the careful design of the constraint matrix. We use an integer value data type for our testing in order to make all
    # comparisons exact. But since the solver is expected to automatically double off-diagonals, our framework has to halve the
    # values. While this won't be an issue in the final result if all off-diagonals are themselves real-valued polynomials in
    # total, even then the intermediate calculation will require a floating point data type. However, conversion to the integer
    # will be automatically done if we choose all coefficient of monomials in off-diagonal entries to be even.
    c₁₂ = SimplePolynomial(Complex{Int16}[8-2im, -8, -6im, 2],
                           SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0;; 1; 0; 0; 0;; 0; 0; 0; 0],
                                                     UInt8[0; 0;; 0; 2;; 0; 0;; 1; 0],
                                                     UInt8[0; 2;; 0; 0;; 0; 0;; 1; 0]))
    sostest_psd(3 + offset, true, sos_add_matrix!, grouping, [c₁₁ c₁₂ c₁₃; conj(c₁₂) c₂₂ c₂₃; conj(c₁₃) conj(c₂₃) c₃₃])
end

@testset "Complex grouping of length 4" begin
    offset = 70
    grouping = SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 1; 0; 0;; 0; 0; 0; 2;; 0; 0; 3; 0],
                                         UInt8[0; 0;; 0; 0;; 1; 0;; 0; 2], zeros(UInt8, 2, 4))
    # the grouping is [1, x₂, z₁x₄², z₂²x₃³]
    c₁₁ = SimplePolynomial(Complex{Int16}[4, 8, -(2+6im), -(2-6im)],
                           SimpleMonomialVector{4,2}(UInt8[0; 1; 1; 0;; 0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 0; 0],
                                                     UInt8[0; 0;; 1; 0;; 0; 1;; 0; 0],
                                                     UInt8[0; 0;; 1; 0;; 0; 0;; 0; 1]))
    # 71 - the constraint is 4x₂x₃+8z₁z̄₁-(2+6im)*z₂-(2-6im)*z̄₂
    sostest_psd(1 + offset, true, sos_add_matrix!, grouping, c₁₁)

    # 72 - the constraint is [4x₂x₃+8z₁z̄₁-(2+6im)*z₂-(2-6im)*z̄₂  18x₄                     10x₁+8z₂z̄₂
    #                         18x₄                               0                        24im*x₃z₂-23im*x₃z̄₂
    #                         10x₁+8z₂z̄₂                          -24im*x₃z̄₂+23im*x₃z₂     6x₂x₄]
    c₁₂ = SimplePolynomial(Complex{Int16}[18], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 1;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    c₁₃ = SimplePolynomial(Complex{Int16}[10, 8],
                           SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;; 0; 0; 0; 0], UInt8[0; 0;; 0; 1], UInt8[0; 0;; 0; 1]))
    c₂₂ = SimplePolynomial(Complex{Int16}[], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 0), zeros(UInt8, 2, 0),
                                                                       zeros(UInt8, 2, 0)))
    c₂₃ = SimplePolynomial(Complex{Int16}[24im, -24im],
                           SimpleMonomialVector{4,2}(UInt8[0; 0; 1; 0;; 0; 0; 1; 0], UInt8[0; 1;; 0; 0], UInt8[0; 0;; 0; 1]))
    c₃₃ = SimplePolynomial(Complex{Int16}[6], SimpleMonomialVector{4,2}(UInt8[0; 1; 0; 1;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    sostest_psd(2 + offset, true, sos_add_matrix!, grouping, [c₁₁ c₁₂ c₁₃; conj(c₁₂) c₂₂ c₂₃; conj(c₁₃) conj(c₂₃) c₃₃])

    # 73 - the constraint is [4x₂x₃+8z₁z̄₁-(2+6im)*z₂-(2-6im)*z̄₂  (8-2im)*z̄₂²-8z₂²-6im*x₁+2z₁z̄₁  10x₁+8z₂z̄₂
    #                         (8+2im)*z₂²-8z̄₂²+6im*x₁+2z₁z̄₁       0                             24im*x₃z₂-24im*x₃z̄₂
    #                         10x₁+8z₂z̄₂                          -24im*x₃z̄₂+24im*x₃z₂           6x₂x₄]
    c₁₂ = SimplePolynomial(Complex{Int16}[8-2im, -8, -6im, 2],
                           SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0;; 1; 0; 0; 0;; 0; 0; 0; 0],
                                                     UInt8[0; 0;; 0; 2;; 0; 0;; 1; 0],
                                                     UInt8[0; 2;; 0; 0;; 0; 0;; 1; 0]))
    sostest_psd(3 + offset, true, sos_add_matrix!, grouping, [c₁₁ c₁₂ c₁₃; conj(c₁₂) c₂₂ c₂₃; conj(c₁₃) conj(c₂₃) c₃₃])
end

@testset "Equality constraints (elementary)" begin
    # these are basic tests, we don't check special cases
    offset = 80
    realgrouping = SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 1; 0; 0; 0], UInt8[0; 0;; 0; 0], UInt8[0; 0;; 0; 0])
    mixedgrouping = SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0], UInt8[0; 0;; 1; 0], UInt8[0; 0;; 0; 0])
    altgrouping = SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0], UInt8[0; 0;; 0; 1], UInt8[0; 0;; 0; 0])

    constraint = SimplePolynomial(Int16[5], SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    sostest(1 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, realgrouping, constraint)
    sostest(2 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, mixedgrouping, constraint)

    constraint = SimplePolynomial(Complex{Int16}[6+8im, 6-8im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 2),
        UInt8[0; 0;; 1; 0], UInt8[1; 0;; 0; 0]))
    sostest(3 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, realgrouping, constraint)
    sostest(4 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, mixedgrouping, constraint)
    sostest(5 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, altgrouping, constraint)

    constraint = SimplePolynomial(Complex{Int16}[6+8im], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 1), UInt8[0; 0;;],
        UInt8[1; 0;;]))
    sostest(6 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, realgrouping, constraint)
    sostest(7 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, mixedgrouping, constraint)
    sostest(8 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, altgrouping, constraint)
end

@testset failfast=true "Equality constraints ($text)" for (grouping, text, offset) in (
    (SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;], UInt8[0; 0;;], UInt8[0; 0;;]), "1-basis", 90),
    (SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 1; 0; 0; 0;; 0; 2; 0; 0], UInt8[0; 0;; 0; 0;; 0; 0],
        UInt8[0; 0;; 0; 0;; 0; 0]), "real basis", 100),
    (SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 1; 0; 0; 0;; 0; 0; 0; 0], UInt8[0; 0;; 0; 0;; 0; 2],
        UInt8[0; 0;; 0; 0;; 0; 0]), "complex basis", 110)
)
    constraint = SimplePolynomial(Int16[6], SimpleMonomialVector{4,2}(UInt8[1; 0; 0; 0;;], UInt8[0; 0;;], UInt8[0; 0;;]))
    sostest(1 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, grouping, constraint)

    constraint = SimplePolynomial(Complex{Int16}[6+4im, 6-4im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0],
        UInt8[1; 0;; 0; 0], UInt8[0; 0;; 1; 0]))
    sostest(2 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, grouping, constraint)

    constraint = SimplePolynomial(Complex{Int16}[6+4im], SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;;], UInt8[1; 0;;],
        UInt8[0; 0;;]))
    sostest(3 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, grouping, constraint)

    constraint = SimplePolynomial(Complex{Int16}[6+4im, 8, 2im],
        SimpleMonomialVector{4,2}(UInt8[0; 0; 0; 0;; 0; 0; 0; 0;; 0; 0; 1; 0], UInt8[1; 0;; 0; 0;; 0; 0],
            UInt8[0; 0;; 1; 0;; 0; 0]))
    sostest(4 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, grouping, constraint)

    constraint = SimplePolynomial(Int16[-10, -2, 8, -2], SimpleMonomialVector{4,2}(zeros(UInt8, 4, 4),
        UInt8[0; 0;; 0; 0;; 0; 1;; 0; 2], UInt8[0; 0;; 0; 2;; 0; 1;; 0; 0]))
    sostest(5 + offset, SolverSetupFree, :add_free_finalize, sos_add_equality!, grouping, constraint)
end