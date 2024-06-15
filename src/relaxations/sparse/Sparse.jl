"""
    AbstractRelaxationSparse{Prob} <: AbstractRelaxation{Prob}

An `AbstractRelaxationSparse` is a relaxation of a polynomial optimization problem that applies sparsity methods to reduce the
size of the associated problem, possibly at the expense of lowering the objective bound.
"""
abstract type AbstractRelaxationSparse{Prob<:Problem} <: AbstractRelaxation{Prob} end

include("./Chordal.jl")
include("./Correlative.jl")
include("./Term.jl")
