# Introduction

`PolynomialOptimization` is a Julia package that allows to easily optimize large-scale polynomial optimization problems.
It builds on `MultivariatePolynomials` to provide a concise interface for the specification of the problem and allows to
directly control the problem's sparsity using correlative sparsity, (chordal) term sparsity, and a combination of both. It also
supports complex-valued problems and positive semidefinite constraints, and allows to extract solutions even for sparse
problems.
It _directly_ interfaces the solvers [Mosek](https://www.mosek.com/) (version 10 strongly preferred, less features available
with version 9), [COSMO](https://oxfordcontrol.github.io/COSMO.jl/stable/),
[Hypatia](https://github.com/chriscoey/Hypatia.jl), and [COPT](https://www.shanshu.ai/copt) without using `JuMP`. Despite
`JuMP` being very performant for a modelling framework, it introduces a significant overhead that is omitted in this way.

## Overview
```@contents
Depth=3
```

## Index
```@index
Modules=[PolynomialOptimization]
```