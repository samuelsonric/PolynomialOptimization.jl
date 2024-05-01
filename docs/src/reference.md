# Reference

```@meta
CurrentModule = PolynomialOptimization
```

## Problem definition
```@docs
POProblem
poly_problem
variables
nvariables
isreal
```

## Optimization
```@docs
poly_optimize
```

## Working with problem solutions
All functions listed here requires that the problem be optimized before.
```@docs
POResult
optimality_certificate
moment_matrix
MomentVector
```

## Relaxations
```@docs
AbstractPORelaxation
degree
basis
groupings
iterate!(::AbstractPORelaxation)
Core.Type(::POProblem, ::Tuple{Vararg{Any}})
RelaxationGroupings
AbstractRelaxationDegree
RelaxationDense
RelaxationNewton
RelaxationCustom
AbstractRelaxationSparse
RelaxationSparsityCorrelative
```

## Helper functions used for some relaxations
```@docs
Newton.halfpolytope
Newton.halfpolytope_from_file
chordal_completion!
chordal_cliques!
@allocdiff
```

```@meta
CurrentModule = PolynomialOptimization.FastVector
```
## FastVector
To improve the speed in some implementation details, `PolynomialOptimization` provides a "fast" vector type. This is basically
just a wrapper around the stdlib `Vector`; however, it actually takes proper advantage of sizehints. The fact that Julia does
this badly has been known for quite some time ([#24909](https://github.com/JuliaLang/julia/issues/24909)), but the default
implementation has not changed. Our own `FastVec` is a bit more specific than the
[PushVector](https://github.com/tpapp/PushVectors.jl), but also allows for more aggressive optimization.
```@docs
FastVec
Base.sizehint!(::FastVec, ::Integer)
Base.empty!(::FastVec)
prepare_push!
Base.push!(::FastVec{V}, el) where {V}
unsafe_push!
Base.append!(::FastVec{V}, ::AbstractVector) where {V}
unsafe_append!
Base.prepend!(::FastVec{V}, ::AbstractVector) where {V}
unsafe_prepend!
Base.similar(::FastVec{V}) where {V}
Base.copyto!(::FastVec, ::Integer, ::FastVec, ::Integer, ::Integer)
finish!
```
