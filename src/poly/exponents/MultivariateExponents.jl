module MultivariateExponents

using PolynomialOptimization: @assert

export unsafe

struct Unsafe end
const unsafe = Unsafe()

include("./AbstractExponents.jl")
include("./ExponentsIndices.jl")
include("./ExponentsAll.jl")
include("./ExponentsDegree.jl")
include("./ExponentsMultidegree.jl")

end