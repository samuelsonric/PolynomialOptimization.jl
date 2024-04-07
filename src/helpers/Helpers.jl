include("./Mutation.jl")
include("./FastVector.jl")
using .FastVector
include("./StackVector.jl")
include("./SortAlong.jl")
include("./FastKey.jl")
include("./Allocations.jl")

# we must load this before MatrixPolynomials (so that effective_variables_in is known), but after FastVector, which is required
# by SimplePolynomials.
include("../poly/SimplePolynomials.jl")
using .SimplePolynomials

include("./MatrixPolynomials.jl")