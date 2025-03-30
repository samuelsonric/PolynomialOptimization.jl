include("./Macros.jl")
include("./Mutation.jl")
include("./FastVector.jl")
using .FastVector
include("./StackVector.jl")
include("./SortAlong.jl")
include("./FastKey.jl")
include("./Allocations.jl")

# we must load this before MatrixPolynomials (so that effective_variables_in is known), but after FastVector, which is required
# by IntPolynomials.
include("../poly/IntPolynomials.jl")
using .IntPolynomials, .IntPolynomials.MultivariateExponents
using .IntPolynomials: IntMonomialOrConj
import .IntPolynomials.MultivariateExponents: iterate! # be careful to avoid duplication of methods; let's reuse this one

include("./MatrixPolynomials.jl")

"""
    issubset_sorted(a, b)

Equivalent to `issubset(a, b)`, but assumes that both `a` and `b` are sorted vectors.
"""
function issubset_sorted(a, b)
    i = firstindex(b)
    @inbounds for x in a
        i += searchsortedlast(@view(b[i:end]), x) -1
        if i < firstindex(b) || b[i] != x
            return false
        end
    end
    return true
end