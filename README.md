[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://projekter.github.io/PolynomialOptimization.jl/dev)
[![Build status (Github Actions)](https://github.com/projekter/PolynomialOptimization.jl/workflows/CI/badge.svg)](https://github.com/projekter/PolynomialOptimization.jl/actions)
[![codecov.io](http://codecov.io/github/projekter/PolynomialOptimization.jl/coverage.svg?branch=main)](http://codecov.io/github/projekter/PolynomialOptimization.jl?branch=main)

# PolynomialOptimization.jl

`PolynomialOptimization` is a Julia package that allows to easily optimize large-scale polynomial optimization problems.
It builds on `MultivariatePolynomials` to provide a concise interface for the specification of the problem and allows to apply
many kinds of sparsity methods. It also fully supports complex-valued problems and positive semidefinite constraints and allows
to extract solutions even for sparse problems.
It _directly_ interfaces the solvers [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl),
[COPT](https://www.shanshu.ai/copt), [Hypatia](https://github.com/jump-dev/Hypatia.jl), [Mosek](https://www.mosek.com/), and
[SCS](https://github.com/cvxgrp/scs) without using `JuMP`, avoiding this bottleneck so that indeed almost all the time is spent
in the solver, not in the problem formulation.

## About this branch

This branch provides an interface to the [LANCELOT sovler](https://github.com/ralna/GALAHAD). LANCELOT is a sophisticated
non-convex solver which can now be invoked using the solver `:LANCELOT`. Note that contrary to all other methods, this will
*not* result in an optimization that is based on convex relaxations; no global guarantees are made. Futher notes:

- We call `LANCELOT_simple`, which does not make use of all the potential sophistication LANCELOT provides as regards the use
  of variables.
- We use `Galahad.jl` for the pre-packaged binary to access LANCELOT (and a mechanism provide your own library, if you wish),
  though we use our own wrapper for LANCELOT. Since LANCELOT currently only has a Fortran interface, we need to rely on
  compiler internals to call it. Memory corruption will definitely occur if the library was compiled with a version of GFortran
  smaller than 9 or e.g., Intel Fortran. Both use different formats for the array descriptors.