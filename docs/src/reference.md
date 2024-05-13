# Reference

```@meta
CurrentModule = PolynomialOptimization
```

# Optimization reference
This reference page lists all functions that are relevant for polynomial optimization. All functions listed here are exported
by the package.

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

## Newton polytope construction (manually)
```@docs
Newton.halfpolytope
Newton.halfpolytope_from_file
```