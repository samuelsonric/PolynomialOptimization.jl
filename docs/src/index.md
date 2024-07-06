# Introduction

`PolynomialOptimization` is a Julia package that allows to easily optimize large-scale polynomial optimization problems.
It builds on `MultivariatePolynomials` to provide a concise interface for the specification of the problem and allows to
directly control the problem's sparsity using correlative sparsity, (chordal) term sparsity, and a combination of both. It also
supports complex-valued problems and positive semidefinite constraints, and allows to extract solutions even for sparse
problems.
It provides a solver interface specifically designed for the optimization problems arising in polynomial optimization. This
interface makes it easy to implement new solvers. On purpose, `MathOptInterface`/`JuMP` is not employed; despite `JuMP` being
very performant for a modelling framework, it introduces a significant overhead that is omitted in this way.
The following solvers are supported:
- [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl)
- [COPT](https://www.shanshu.ai/copt)
- [Hypatia](https://github.com/jump-dev/Hypatia.jl)
- [Mosek](https://www.mosek.com/)
- [SCS](https://github.com/cvxgrp/scs)

## Overview
```@contents
Depth=3
```

## Index
```@index
Modules=[PolynomialOptimization]
```