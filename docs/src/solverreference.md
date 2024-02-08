```@meta
CurrentModule = PolynomialOptimization
```

# Solver reference
`PolynomialOptimization` mostly uses external solvers that have Julia bindings or are implemented in Julia, but it also
provides own solver implementations particularly for the purpose of polynomial optimization.
!!! warning
    The package does not introduce any hard dependencies on the external solvers. Therefore, you may or may not decide to
    install them on your system. Instead, the solvers are addressed as weak dependencies. This means that *you have to load the
    solver Julia package manually* before you are able to use it for optimization.
`PolynomialOptimization` also provides an interface that can be implemented if another solver should be supported. This
consists of just a few methods for the various functions.

!!! info "MathOptInterface"
    Why do we re-invent the wheel and don't just use `MathOptInterface`? This would immediately bring compatibility with a huge
    number of solvers that are available under Julia.

    This would certainly be possible. It might even be a not-too-bad idea to provide an automatic fallback to
    `MathOptInterface` in the future for solvers that are not supported natively.

    However, `MathOptInterface` is enormous in its feature range. Considering that, it is amazingly fast. But in
    `PolynomialOptimization`, only a very small subset of all the features is required, and a lot of additional assumptions on
    how the problem is constructed can be made. This allows to use the solver's API in a most efficient way, which typically
    is not the way in which implementations of `MathOptInterface` would address the solver.

    Additionally, in some situations, a lot of very similar sub-problems need to be solved; in these cases,
    `PolynomialOptimization`'s own interface allows to keep the optimizer task alive and just do the tiny modification instead
    of setting things up from the start again - which for `MathOptInterface` only works if a solver is implemented in a
    particular way.

    This focus on efficiency is very important as (relevant) polynomial optimization problems are huge. It is just not possible
    to waste time and memory in bookkeeping that is not really needed.

## [`poly_optimize`](@ref)
The optimization of polynomial problems requires a solver that understands linear and semidefinite constraints.

### Solver interface
_Note that none of the functions listed in this section are exported._
In general, a solver implementation can do whatever it wants; it just needs to implement the [`poly_optimize`](@ref) method
with the appropriate `Val`-wrapped solver method as its first parameter. However, it is very helpful to just do some basic
setup such as creating the solver object in this function and delegate all the work of setting up the actual problem to
[`sos_setup!`](@ref).
The solution is contained in the dual variables of the constraints; it can be converted to a `FullMonomialVector` that is
required by the solution extraction functions by [`sos_solution`](@ref).
```@docs
sos_setup!
sos_solution
```

There are some functions that should be implemented to tell the framework what kind of data the solver expects and which cones
are supported; these should return constants. There is also a method that directly converts monomials to their constraint
indices as required by the solver.
```@docs
sos_solver_psd_indextype
sos_solver_supports_quadratic
sos_solver_supports_complex_psd
sos_solver_mindex
```

Then, to actually feed data to the solver, the following methods (or a subset as previously indicated) must be implemented.
```@docs
sos_solver_add_scalar!
sos_solver_add_quadratic!
sos_solver_add_psd!
sos_solver_add_psd_complex!
sos_solver_add_free_prepare!
sos_solver_add_free!
sos_solver_add_free_finalize!
sos_solver_fix_constraints!
```

Once a solver has been implemented, it should add its solver symbol to the vector `PolynomialOptimization.solver_methods`,
which enables this solver to be chosen automatically.

## [`newton_halfpolynomial`](@ref)
Finding the Newton halfpolynomial requires a linear solver that supports problem modification for quick reoptimization.

### Solver interface
The following functions need to be implemented so that a solver is available via [`newton_halfpolytope`](@ref). The
preprocessing functions can be omitted if no preprocessing capabilities should be provided.
```@docs
newton_polytope_preproc_quick
newton_polytope_preproc_remove
newton_halfpolytope_alloc_global
newton_halfpolytope_alloc_local
newton_halfpolytope_clonetask
newton_polytope_do_worker
newton_halfpolytope_do_prepare
```

Once a solver has been implemented, it should add its solver symbol to the vector `PolynomialOptimization.newton_methods`,
which enables this solver to be chosen automatically.

## Automatic tightening
Automatic tightening of a polynomial optimization problem requires a linear solver that finds a solution to a system of linear
equations that minimizes the ℓ₁-norm (better yet, the ℓ₀-norm, if you can implement this). The solver is only called if the
number of rows is smaller than the number of columns; else, the solution is calculated using SPQR's direct solver.

### Solver interface
The following function needs to be implemented so that a solver is available via automatic tightening.
```@docs
tighten_minimize_l1
```

Once a solver has been implemented, it should add its solver symbol to the vector `PolynomialOptimization.tightening_methods`,
which enables this solver to be chosen automatically.