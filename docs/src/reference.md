# Reference

```@meta
CurrentModule = PolynomialOptimization
```

## Problem definition
```@docs
PolyOptProblem
poly_problem
newton_halfpolytope
EqualityMethod
variables
nvariables
degree
length
isreal
```

## Optimization
```@docs
sparse_optimize
poly_optimize
```

## Working with problem solutions
All functions listed here requires that the problem be optimized before.
```@docs
poly_solutions
poly_solutions_heuristic
poly_all_solutions
poly_solution_badness
optimality_certificate
moment_matrix
last_moments
```

## Sparsity
```@docs
SparseAnalysisState
SparsityNone
SparsityCorrelative
SparsityTerm
SparsityTermBlock
SparsityTermCliques
SparsityCorrelativeTerm
sparse_iterate!
sparse_groupings
sparse_problem
TermMode
```

## Chordal graphs
```@docs
chordal_completion!
chordal_cliques!
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
prepare_push!
Base.push!(::FastVec{V}, el) where {V}
unsafe_push!
Base.append!(::FastVec{V}, ::AbstractVector) where {V}
unsafe_append!
Base.prepend!(::FastVec{V}, ::AbstractVector) where {V}
unsafe_prepend!
Base.similar(::FastVec{V}) where {V}
finish!
```
