# Reference

```@meta
CurrentModule = PolynomialOptimization
```

## Problem definition
```@docs
PolyOptProblem
poly_problem
newton_polytope
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

## SketchyCGAL
While the solver was implemented for the purpose of being used within `PolynomialOptimization`, it also works as a standalone
routine (and could in principle be a separate package). SketchyCGAL is a solver that scales very well for large problem sizes
and is based on the assumption that the optimal solution has low rank; indeed, in polynomial optimizations, if there is an
optimal point for the problem that can be encoded in the chosen basis, then this automatically gives rise to a rank-one
semidefinite encoding of this point.
```@docs
sketchy_cgal
SketchyCGALStatus
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
