# Walkthrough

We start the Julia session by including the required packages.
```Julia
julia> using PolynomialOptimization
julia> using DynamicPolynomials
```
Note that we import `PolynomialOptimization` before `DynamicPolynomials` (and possibly `MultivariatePolynomials`). This is
important as currently, we hack into those package to provide support for complex-valued polynomials. Hopefully, this will soon
be integrated into the packages themselves.

## A simple unconstrained problem
### Constructing the problem
Next, we define some simple optimization problem.
```Julia
julia> @polyvar x[1:3];
julia> prob = poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1]^2*x[2]^2 + x[1]^2*x[3]^2 + x[2]^2*x[3]^2 + x[2]*x[3], 2)
Real-valued polynomial optimization hierarchy of degree 2 in 3 variable(s)
Objective: x₁⁴ + x₁²x₂² + x₁²x₃² + x₂⁴ + x₂²x₃² + x₃⁴ + x₂x₃ + 1
Size of full basis: 10
```
This is a very simple problem: We have three variables and want to minimize an unconstrained objective function, using a
Lasserre relaxation of order `2`. In fact, `2` is the minimal degree: By passing `0` for the degree, the minimal degree will be
determined automatically.
```Julia
julia> prob = poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1]^2*x[2]^2 + x[1]^2*x[3]^2 + x[2]^2*x[3]^2 + x[2]*x[3], 0)
[ Info: Automatically selecting minimal degree 2 for the relaxation
Real-valued polynomial optimization hierarchy of degree 2 in 3 variable(s)
Objective: x₁⁴ + x₁²x₂² + x₁²x₃² + x₂⁴ + x₂²x₃² + x₃⁴ + x₂x₃ + 1
Size of full basis: 10
```

### Densely solving the problem
Since this problem is so small, we can solve it directly without any sparsity consideration.
```Julia
julia> poly_optimize(:MosekSOS, prob)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.9166666483392931)
```
The solution that we found was indeed optimal and the value is `0.9166...`.
Note that "optimal" here means that the solver converged for the given problem. However, the _given problem_ is the second
order Lasserre relaxation; there is no guarantee that this order indeed corresponds to the optimum of the _original_ problem,
only that it is a lower bound.

### Checking optimality
There are two different ways to check whether the given bound is optimal for the original problem. Obviously, if we find a
point $x$ such that the objective evaluated at $x$ gives our bound, then this bound must have been optimal. Of course, the
difficulty now lies in finding the point. `PolynomialOptimization` implements a state of the art
[solution extraction algorithm](https://doi.org/10.1016/j.laa.2017.04.015), which can relatively quickly (the cost is
essentially that of performing an SVD on the moment matrix) obtain solutions. This will only be guaranteed to work if the
problem was indeed optimal, there were finitely many solutions in the first place, and a "good" moment matrix is obtained
(i.e., a dense matrix and no low-rank solver was employed) - but there is an alternative which might work well in case these
conditions are not satisfied (apart from optimality, obviously), more on this below.
The function [`poly_solutions`](@ref) gives an iterator that delivers all the (potential) solutions one at a time in an
arbitrary order.
Alternatively, [`poly_all_solutions`](@ref) directly calculates all the solutions and grades them according to how much they
violate the bound or constraints, if any were given. The solutions are then return in a best-to-worst order.
When calling [`poly_optimize`](@ref), the keyword argument `solutions` can be set to `true`, which will automatically call
[`poly_all_solutions`](@ref):
```Julia
julia> poly_optimize(:MosekSOS, prob, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.9166666483392931, [([-2.3921125960809997e-18, -0.40826233531625955, 0.4082623353161486], 1.8721902916851718e-8), ([-3.771718492228676e-18, 0.4082623353162627, -0.40826233531615175], 1.8721902916851718e-8)])
```
Now, the third argument of the return value contains a vector with all solutions along with their badnesses (which can also be
calculated manually using [`poly_solution_badness`](@ref)). Since here, the badness is of the order of $10^{-8}$, i.e.,
numerically zero, the points are indeed valid global minima.

Note that the solution extraction functions just give a vector of numbers; to assign these numbers to variables, the function
[`variables`](@ref) can be applied to the problem. This is particularly useful if there are multiple variables created at
different times, occurring differently in the constraints, such that the order is not clear beforehand.
```Julia
julia> variables(prob)
3-element Vector{PolyVar{true}}:
 x₁
 x₂
 x₃
```

A second way to check for optimality is to use the flat extension/truncation criterion originally due to
[Curto and Fialkow](https://doi.org/10.1090/S0002-9947-00-02472-7) and improved by [Nie](https://arxiv.org/abs/1106.2384v2).
This is a sufficient criterion for optimality, and it can be manually checked by calling [`optimality_certificate`](@ref) on
the problem (it will only work with non-sparse problems). The function will return `:Optimal` if the given minimum value can
be certified to be optimal; else it will return `:Unknown`. The invocation of this function can also be automated by setting
the keyword argument `certificate` to `true` when invoking [`poly_optimize`](@ref). However, note that this is just a
sufficient criterion, and the solution might be optimal even if it is violated:
```Julia
julia> optimality_certificate(prob)
:Unknown
```
In this case, we'd need to go to the third level in the hierarchy in order to get a positive certificate. Calculating the
certificate will involve calculating the ranks of several matrices (and is more complicated in the complex case) and is
therefore also disabled by default.

### Using the Newton polytope
The current example is an unconstrained optimization problem; hence, the size of the full basis, which is 10, may be larger
than actually necessary. It is not a simple problem to determine the relevant basis elements in general; but unconstrained
problems allow for the Newton polytope technique. Using it is as simple as only passing the objective (leaving out the degree)
to [`poly_problem`](@ref):
```Julia
julia> poly_problem(1 + x[1]^4 + x[2]^4 + x[3]^4 + x[1]^2*x[2]^2 + x[1]^2*x[3]^2 + x[2]^2*x[3]^2 + x[2]*x[3])
Real-valued polynomial optimization hierarchy of degree 2 in 3 variable(s)
Objective: x₁⁴ + x₁²x₂² + x₁²x₃² + x₂⁴ + x₂²x₃² + x₃⁴ + x₂x₃ + 1
Size of full basis: 10
```
In this case, no basis reduction was possible. However, in other cases, this can work. For example, if you want to inspect the
Newton polytope of the polynomials whose squares might make up a certain objective, you  can call [`newton_polytope`](@ref)
directly:
```Julia
julia> @polyvar x y
julia> newton_polytope(x^4*y^2 + x^2*y^4 - 3x^2*y^2 +1)
4-element Vector{Monomial{true}}:
 x²y
 xy²
 xy
 1
```
This reveals that, were the Motzkin representable by a sum of squares, the equality
```math
x^4 y^2 + x^2 y^4 - 3 x^2 y^2 + 1 = \sum_i (\alpha_i x^2 y + \beta_i x y^2 + \gamma_i x y + \delta_i)^2
```
would have to hold; but expanding the right-hand side will lead to the coefficient ``\sum_i \gamma_i^2`` in front of the
monomial ``x^2 y^2``, which cannot be negative; hence, the Motzkin polynomial is not a sum of squares.

Note that the calculation of the Newton polytope currently requires Mosek.

In case you already happen to know a (better) choice of basis, you may provide this basis to [`poly_problem`](@ref).

### Applying sparsity
There are four kinds of sparsity implemented in `PolynomialOptimization`:
- [`SparsityNone`](@ref)
- [`SparsityTermBlock`](@ref)
- [`SparsityTermCliques`](@ref)
- [`SparsityCorrelativeTerm`](@ref)
Applying the analysis as simple as passing the problem to the respective type. For this particular problem, there is no
correlative sparsity:
```Julia
julia> SparsityCorrelative(prob)
SparsityCorrelative with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [10 => 1]
```
So there is only a single clique, leading to a basis of size `10`. However, there is term sparsity:
```Julia
julia> tbs = SparsityTermBlock(prob)
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [5 => 1, 2 => 1, 1 => 3]
```
We get a basis of size `5`, one of size `2`, and three bases of size `1` (here, by _basis_ we mean a set of monomials that
indexes the moment/SOS matrices). `PolynomialOptimization` will model these by a `5 × 5` semidefinite matrix, a rotated
second-order cone, as well as three linear constraints. This is much cheaper than a `10 × 10` semidefinite matrix.
Let's optimize the sparse problem:
```Julia
julia> sparse_optimize(:MosekSOS, tbs, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.916666648342404, [([0.0, 0.4082608082296415, -0.4082685270025584], 1.8900428111479073e-8), ([0.0, -0.4082608082296415, 0.4082685270025584], 1.8900428111479073e-8)])
```
Again, we get the same optimal value, so introducing the sparsity did not make our relaxation worse (which is _per se_ not
guaranteed), and we are still able to get the same optimal solutions.

Note that perhaps surprisingly, `PolynomialOptimization` can still deliver good optimal points despite the fact that term
sparsity was in effect. The usual extraction algorithms will fail, as for every moment ``m``, they also require the moment
``m x`` to be present for all variables ``x`` - term sparsity will fail to provide this.
Hence, [`poly_all_solutions`](@ref) will automatically switch to a different heuristical solution extraction algorithm that is
always successful in the simple case of a rank-1 moment matrix, but can also often also give good results in the more general
case such as here. As a rule of thumb, if all the solutions encoded in the moment matrix differ only by the signs or phases of
individual components, the heuristic will be successful. Still, the fact that the moments may encode multiple solutions may be
an issue that can prevent successfully obtaining a solution vector. We will introduce a way to bypass this problem below.

Assume that our term sparsity gave a worse bound than the dense case (which in general we would not know, since the dense
problem is typically far too large to be solved; but we just don't get a proper optimal point, though this could also be due to
the Lasserre hierarchy level being insufficient). Then, we could try to iterate the term sparsity hierarchy, keeping the same
level in the Lasserre hierarchy.
```Julia
julia> sparse_iterate!(tbs)
SparsityTermBlock with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [5 => 1, 2 => 2, 1 => 1]

julia> sparse_iterate!(tbs)
```
In general, we simply use `sparse_iterate!` to move to the next higher level (note how two of the linear constraints were
merged into a quadratic constraint). If `sparse_iterate!` returns the new sparsity object, something changed and we might try
to optimize the new sparse problem, getting a potentially better bound. If `sparse_iterate!` returns `nothing`, the hierachy
terminated and nothing more can (for term sparsity: must) be done (as the last level for term sparsity is as good as the dense
problem).

We can also try what happens if we use term sparsity with chordal cliques instead of connected components:
```Julia
julia> tcs = SparsityTermCliques(prob)
SparsityTermCliques with 0 constraint(s)
Variable cliques:
  x[1], x[2], x[3]
Block sizes:
  [4 => 1, 2 => 2, 1 => 3]

julia> sparse_optimize(:MosekSOS, tcs, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.9166666618078856, [([0.0, 0.40826934273815213, -0.40827645479440017], 6.103714644822844e-9), ([0.0, -0.40826934273815213, 0.40827645479440017], 6.103714644822844e-9)])
```
So again, we get the same optimal result, can extract a solution point with even better accuary, and the problem was smaller,
as we now have just a basis of size `4` instead of `5` (at the cost of another quadratic constraint, which is much cheaper than
larger semidefinite matrices).

### Details on the optimization process
The first parameter for [`sparse_optimize`](@ref) or [`poly_optimize`](@ref) is the solver/method that is used to optimize
the problem. The following methods are currently supported:
- `:MosekSOS`: for real-valued problems, requires Mosek 9+ and uses a SOS approach. This is typically both the fastest and a
  very precise method and should be the way to go, unless the problem is complex-valued.
- `:MosekMoment`: for any kind of problem, requires Mosek 10+ and uses a moment-matrix approach. This is precise and moderately
  fast. It is more prone to failure in case of close-to-illposed problems; sometimes, this is an issue of the presolver, which
  can be turned off by passing `MSK_IPAR_PRESOLVE_USE="MSK_PRESOLVE_MODE_OFF"` to [`poly_optimize`](@ref) or
  [`sparse_optimize`](@ref).
- `:COSMOMoment`: for real-valued problems, requires COSMO and uses a moment-matrix approach. This is imprecise and not too
  fast, but can scale to very large sizes.
- `:HypatiaMoment`: for any kind of problem, requires Hypatia. This is moderately precise and not too fast. Complex-valued
  problems are modeled uses Hypatia's complex PSD cone.
  !!! tip
      Note that by default, a sparse solver is used (unless the problem was constructed with a `factor_coercive` different from
      one). This is typically a good idea for large systems with not too much monomials. However, if you have a very dense
      system, the sparse solver will take forever; better pass `dense=true` to the optimization routine. This will then be much
      faster (and always much more accurate).
- `:SpecBMSOS`: for real-valued problems, requires Mosek or Hypatia as subsolvers and uses a SOS approach. This is more precise
  than COSMO and not very fast for small-scale problems, but scales very well. Note that some default parameters for the SpecBM
  solver are assumed here (in particular, `r_past` and `r_current`) that can be played with.
  !!! tip
      In case a sphere or ball constraint with radius ``R`` is present for all variables, setting the parameter
      ``\rho = (1 + R)^{d -1}`` (where ``d`` is the [`degree`](@ref) of the problem) and disabling `adaptiveρ = false` can be a
      good idea.
  This solver does not come as an extra package, but was written specifically for `PolynomialOptimization`. For details, see
  the [`specbm_primal`](@ref) documentation.
  !!! info
      The dense solution extraction mechanism will not work very well with this solver, as it truncates the rank of the full
      moment matrix. However, this will usually lead to a result of almost rank 1, for which the solution extraction heuristic
      works quite well. Therefore, if you pass `solutions=true` to the optimization function, it will automatically invoke the
      heuristic function. However, if you  want to use `poly_all_solutions` after the optimization was already done, remember
      to set the `heuristic` parameter manually.

Note that by passing the keyword argument `verbose=true` to the optimization function, we get some more insight into what
happens behind the hood. Let's redo the last optimization.
```Julia
julia> sparse_optimize(:MosekSOS, tcs, solution=true, verbose=true)
Determining groupings...
Determined grouping in 6.52e-5 seconds
Clique merging disabled. Block sizes:
[4 => 1, 2 => 2, 1 => 3]
Starting optimization
Setup complete in 0.0002685 seconds
Problem
  Name                   :
  Objective sense        : maximize
  Type                   : CONIC (conic optimization problem)
  Constraints            : 20
  Affine conic cons.     : 0
  Disjunctive cons.      : 0
  Cones                  : 2
  Scalar variables       : 10
  Matrix variables       : 1
  Integer variables      : 0

Optimizer started.
Presolve started.
Linear dependency checker started.
Linear dependency checker terminated.
Eliminator started.
Freed constraints in eliminator : 0
Eliminator terminated.
Eliminator - tries                  : 1                 time                   : 0.00
Lin. dep.  - tries                  : 1                 time                   : 0.00
Lin. dep.  - number                 : 0
Presolve terminated. Time: 0.00
Problem
  Name                   :
  Objective sense        : maximize
  Type                   : CONIC (conic optimization problem)
  Constraints            : 20
  Affine conic cons.     : 0
  Disjunctive cons.      : 0
  Cones                  : 2
  Scalar variables       : 10
  Matrix variables       : 1
  Integer variables      : 0

Optimizer  - threads                : 8
Optimizer  - solved problem         : the primal
Optimizer  - Constraints            : 11
Optimizer  - Cones                  : 3
Optimizer  - Scalar variables       : 11                conic                  : 8
Optimizer  - Semi-definite variables: 1                 scalarized             : 10
Factor     - setup time             : 0.00              dense det. time        : 0.00
Factor     - ML order time          : 0.00              GP order time          : 0.00
Factor     - nonzeros before factor : 60                after factor           : 60
Factor     - dense dim.             : 0                 flops                  : 8.46e+02
ITE PFEAS    DFEAS    GFEAS    PRSTATUS   POBJ              DOBJ              MU       TIME
0   1.0e+00  1.0e+00  1.0e+00  0.00e+00   0.000000000e+00   0.000000000e+00   1.0e+00  0.00
1   2.5e-01  2.5e-01  1.2e-01  8.79e-01   4.603493496e-01   5.031535243e-01   2.5e-01  0.00
2   2.9e-02  2.9e-02  4.1e-03  9.93e-01   8.774106262e-01   8.861354829e-01   2.9e-02  0.00
3   6.2e-03  6.2e-03  3.8e-04  1.14e+00   9.137178230e-01   9.155191208e-01   6.2e-03  0.00
4   9.0e-04  9.0e-04  2.2e-05  1.05e+00   9.161039142e-01   9.163311344e-01   9.0e-04  0.00
5   4.5e-05  4.5e-05  2.5e-07  1.02e+00   9.166506402e-01   9.166607688e-01   4.5e-05  0.00
6   4.3e-07  4.3e-07  2.3e-10  1.00e+00   9.166662556e-01   9.166663522e-01   4.3e-07  0.00
7   4.7e-08  4.7e-08  8.2e-12  1.00e+00   9.166666155e-01   9.166666264e-01   4.7e-08  0.00
8   5.4e-09  5.4e-09  3.2e-13  1.00e+00   9.166666618e-01   9.166666631e-01   5.4e-09  0.00
Optimizer terminated. Time: 0.02

Optimization complete
[ Info: The signs/phases of the following variables are still unknown: x[2], x[3]
Extracted all signs/phases, found 2 possible solution(s)
Checking validity of the solutions
(Mosek.MSK_SOL_STA_OPTIMAL, 0.9166666618078856, [([0.0, 0.40826934273815213, -0.40827645479440017], 6.103714644822844e-9), ([0.0, -0.40826934273815213, 0.40827645479440017], 6.103714644822844e-9)])
```
So first, `PolynomialOptimization` will determine the bases for the matrices according to the sparsity pattern.
At this step, if the optional keyword argument `clique_merging` is set to `true` (default is `false`), an attempt will be made
to merge bases if their heuristic cost for treating them separately would be worse than joining them (this concept is nicely
explained in the [COSMO documenation](https://oxfordcontrol.github.io/COSMO.jl/stable/decomposition/#Clique-merging)). In
general, doing clique merging will lead to faster optimizations; however, the merging process itself can be quite costly and in
fact for large problems might cost much more time than it gains - hence, it is turned off by default.
After this step is done, the Mosek task (or any other optimizer structure, which we all address directly without `JuMP`) is
constructed; then Mosek runs. Having completed the optimization, `PolynomialOptimization` tries to extract solutions from the
moments that were available. It finds appropriate magnitudes for the variables, but there may be sign ambiguities, which can
indeed be resolved.

We can also manually have a look at the moments that were available for solution extraction:
```Julia
julia> last_moments(prob)
Dict{Monomial{true}, Float64} with 11 entries:
  x₃²    => 0.16669
  x₂⁴    => 0.0277835
  1      => 1.0
  x₃⁴    => 0.0277854
  x₁²x₂² => 5.28163e-9
  x₁²    => 3.22222e-8
  x₂²x₃² => 0.0277845
  x₁²x₃² => 5.61816e-9
  x₂x₃   => -0.166687
  x₁⁴    => -3.13486e-9
  x₂²    => 0.166684
```
This reveals how the package is able to return solutions without having access to the full moment matrix. There are values for
the squares of the variables available, so we can deduce two possible candidates for the original variables - at least, if the
values assigned to the moments are consistent.

### Always extracting a solution
The fact that `PolynomialOptimization` was unable to extract a (valid) solution can either mean that the relaxation was
insufficient and did not converge to the optimum of the _actual_ problem, or that there are multiple solutions which are too
difficult for the heuristic to grasp.
There is a simple remedy of this problem: By introducing a small, linear perturbation to the objective, the solution will
almost surely be unique; so if the extracted solution is bad, this means that the relaxation was insufficient.
Of course, now the solution is not a solution to the original problem, but the perturbed one - but assuming a robust problem,
the returned optimal point will be close to the actual global optimum. It can therefore then be used as an initial point to
another nonlinear solver that will deliver the true global optimum.
`PolynomialOptimization` makes adding a perturbation easy: Just call [`poly_problem`](@ref) with the keyword parameter
`perturbation=...`, where the magnitude of the perturbation should be specified (typically between `1e-3` and `1e-6` is a good
guess).
Note that adding a perturbation may degrade sparsity. For this, you may also give a vector of the same length as the number of
variables, specifying a different perturbation magnitude for each variable (or just disabling the perturbation by passing `0`).

## Constraints

### Equality constraints
Equality constraints are implemented in three different ways in this package. All of them are available by passing the keyword
argument `zero` to [`poly_problem`](@ref), which constrains those polynomials to be zero. The method can be chosen by
using the `equality_method` keyword, which can either be specified for all constraints together or even chosen individually for
each constraint. Valid choices for the methods are documented in [`EqualityMethod`](@ref).
```Julia
julia> @polyvar x[1:2];
julia> poly_optimize(:MosekSOS, poly_problem(-(x[1] -1)^2 - (x[1] - x[2])^2 - (x[2] -3)^2, 1,
                                             zero=[(x[1] - 1.5)^2 + (x[2] - 2.5)^2 - .5]), solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, -3.999999989793517, [([0.9999767219767653, 2.9999767173404606], 1.500750634875203e-10)])
```
Note that when grading the quality of a solution, the package will determine the violation of the constraints as well as how
far the actual value is away from what it should be, and return the worst of all violations.
In the real-valued case, `PolynomialOptimization` is able to calculate a Gröbner basis based on the equality constraints. This
may lead to a reduction of the total degree that is required for the optimization. However, the calculation itself may incur a
substantial overhead and also make the problem more difficult than necessary. Use `equality_method=emCalculateGröbner` to
enable Gröbner-based methods (for which the `SemialgebraicSets` package is used; given that with the new 0.5 version, different
monomial orderings may be imposed, it can pay off to experiment).

Ideas for an interface to the `Oscar` package also exists, which should in principle allow to use `AbstractAlgebra` polynomials
from rings with a given term order and to delegate the task of finding a Gröbner basis to the best algorithms available.
However, other optimizations may not work so well with `Oscar` as with `DynamicPolynomials`, which might not make switching the
backend worth the effort (in particular, there is no difference between monomials, terms, and polynomials).

Finally note that just because a Gröbner basis method was used, this does not imply that the equality constraints are
automatically implied in any solution. (This basically amounts to the fact that solving polynomials systems can be done in two
steps: first calculate a Gröbner basis which disentangled complicated variable relationsships; second, perform simple
backsubstitution steps - "simple" compared to the original equations, but not so simple that they could be enforced in convex
optimization.) If you are only interested in the global optimal value, this does not matter. However, for the solution
extraction process, the results may not obey the equality constraints. Passing `add_gröbner=true` to `poly_problem`, the
backsubstitution step will be explicitly accounted for by adding all Gröbner basis elements to the problem. This can have a bad
impact on the solution time (and also completely counter the reason to use Gröbner basis methods in the first place), so it is
disabled by default.

### Inequality constraints
Inequality constraints are implemented using Putinar's Positivstellensatz or localizing matrices. They can be specified by
passing the keyword argument `nonneg` to [`poly_problem`](@ref), which constraints those polynomials to be greater or
equal to zero.
```Julia
julia> @polyvar x[1:2];
julia> poly_optimize.(:MosekSOS, [poly_problem(-(x[1]-1)^2 - (x[1]-x[2])^2 - (x[2]-3)^2, i,
                                               nonneg=[1-(x[1]-1)^2, 1-(x[1]-x[2])^2, 1-(x[2]-3)^2]) for i in 1:2],
                      solutions=true)
2-element Vector{Tuple{Mosek.Solsta, Float64, Vector{Tuple{Vector{Float64}, Float64}}}}:
 (Mosek.MSK_SOL_STA_OPTIMAL, -2.999999999626025, [([1.6532933768701226, 2.224919010979244], 1.6457013582516082)])
 (Mosek.MSK_SOL_STA_OPTIMAL, -2.00000000172626, [([0.9999999975925994, 1.9999999931047932], 1.3790413255776457e-8), ([1.9999999906975054, 2.9999999590933153], 8.353962677176696e-8), ([1.9999999917961266, 1.9999999522274106], 9.554518065613138e-8)])
```
This is an example where the first relaxation level is not optimal, but the second is, which can clearly be seen from the
vanishing badnesses of the solutions in the latter case.

### PSD constraints
`PolynomialOptimization` also supports conditions that constrain a matrix that is made up of polynomials to be positive
semidefinite. They can be specified by passing the keyword argument `psd` to [`poly_problem`](@ref); note that the
matrices must be symmetric/hermitian.
```Julia
julia> @polyvar x[1:2];
julia> poly_optimize.(:MosekSOS, [poly_problem(-x[1]^2 - x[2]^2, i, zero=[x[1]+x[2]-1],
                                               psd=[[1-4x[1]*x[2]  x[1]; x[1]  4-x[1]^2-x[2]^2]]) for i in 1:2],
                      solutions=true)
2-element Vector{Tuple{Mosek.Solsta, Float64, Vector{Tuple{Vector{Float64}, Float64}}}}:
 (Mosek.MSK_SOL_STA_OPTIMAL, -3.999999998038271, [([-0.5811321995597589, 1.581132199559759], 1.1623063321884262)])
 (Mosek.MSK_SOL_STA_OPTIMAL, -3.9048915352982325, [([-0.8047780449611077, 1.8047780449611077], 4.207317205739969e-8)])
```
At second level, we get the optimal solution.

### Improving the optimization without changing the level
The problem can be further tightened by a careful analysis, as [Nie](https://doi.org/10.1007/s10107-018-1276-2) noted, by
rewriting the Lagrange multipliers as polynomials - which will not modify the problem if the minimum is attained at a critical
point (but not that non-critical global minima will be missed).
`PolynomialOptimization` is able to automatically analyze the problem and add the tightening constraints (Mosek is required at
the moment). For this, simply pass `tighter=true` to `poly_problem`. This will result in a preprocessing that adds
constraints, so expect the problem to grow. To see the progress during the preprocessing stage, use `verbose=true`.
It may be the case that the required tightening polynomials are of a higher degree than allowed, in which case an error message
is printed. However, once the required degree is reached, increasing the degree will not change the tightening any more.
Also note that integer-valued coefficients will necessarily be converted to floating point during the processes of tightening.
This might fail in some corner cases; then simply convert the types manually.
Complex-valued problems are not supported at the moment; and PSD constraints will be skipped during the tightening.
```Julia
julia> @polyvar x y;
julia> poly_optimize(:MosekSOS, poly_problem(x^4*y^2 + x^2*y^4 - 3x^2*y^2 +1, 6), solutions=true)
(Mosek.MSK_SOL_STA_PRIM_ILLPOSED_CER, -4.505910348833973e-9, [([0.0, 0.0], 1.0000000045059103)])
```
Without tightening, all the orders from third to sixth of a minimization of the Motzkin polynomial are ill-posed. The first
valid formulation is of seventh order. However, adding the tightening equalities (here, as there are no additional constraints,
this just means to add the condition $\nabla\mathrm{objective} = 0$), the fifth order is already sufficient:
```Julia
julia> prob = poly_problem(x^4*y^2 + x^2*y^4 - 3x^2*y^2 +1, 5, tighter=true)
Real-valued polynomial optimization hierarchy of degree 5 in 2 variable(s)
Objective: x⁴y² + x²y⁴ - 3.0x²y² + 1.0
2 constraints
1: 0 = 4.0x³y² + 2.0xy⁴ - 6.0xy²
2: 0 = 2.0x⁴y + 4.0x²y³ - 6.0x²y
Size of full basis: 21

julia> poly_optimize(:MosekSOS, prob, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, 7.566557217530723e-8, [([1.0000006443510576, 1.0000006443510574], 7.566058993846192e-8)])
```

### Helping convergence
Another way to modify the problem is to exploit a prefactor in the objective.
[Mai et al.](https://doi.org/10.1007/s10107-021-01634-1) showed that by changing the objective from ``f(x)`` to
``\theta^k(x) \bigl(f(x) + \epsilon \theta^d(x)\bigr)``, where ``\theta(x) = 1 + \Vert x\Vert^2`` there is a bound on ``k``
that guarantees membership in the degree-``(k + d)`` sums-of-squares cone. In contrast to other known bounds, this one is very
easy to calculate, and it is not exponential (at least for unconstrained problems...). It holds even for noncompact feasible
sets, in contrast to Putinar's result.
The price to pay is that the objective itself is of course modified and therefore the optimal value of the problem is only in a
neighborhood of the original problem.

By using the `noncompact=(ϵ, k)` when constructing the problem using [`poly_problem`](@ref), this is done automatically.
Let us apply this to the Motzkin case:
```Julia
julia> prob = poly_problem(x^4*y^2 + x^2*y^4 - 3x^2*y^2 +1, 0, noncompact=(1e-5, 1))
[ Info: Automatically selecting minimal degree 4 for the relaxation
Real-valued polynomial optimization hierarchy of degree 4 in 2 variable(s)
Objective: 1.0e-5x⁸ + 1.00004x⁶y² + 2.00006x⁴y⁴ + 1.00004x²y⁶ + 1.0e-5y⁸ + 4.0e-5x⁶ - 1.99988x⁴y² - 1.99988x²y⁴ + 4.0e-5y⁶ + 6.000000000000001e-5x⁴ - 2.9998799999999997x²y² + 6.000000000000001e-5y⁴ + 1.00004x² + 1.00004y² + 1.00001
Objective was scaled by the prefactor x² + y² + 1.0
Size of full basis: 15

julia> poly_optimize(:MosekSOS, prob)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.00026998172528386815)
```
Indeed, now a basis of degree 4 was sufficient to find that the minimum value looks pretty nonnegative. However, this is hard
to quantify, as for this, we'd have to extract a solution from the perturbed `prob`. The algorithm to do this is not
implemented at the moment, as it would require the successive construction and solution of multiple polynomial optimization
problems, which is not very efficient.

## Complex-valued problems
`PolynomialOptimization` fully supports the [complex-valued Lasserre hierarchy](https://doi.org/10.1137/15M1034386), including
its [sparse analysis](https://doi.org/10.1007/s10957-021-01975-z). For this, simply use `@polycvar` instead of `@polyvar` to
declare your variables as complex. Note that this extension to `DynamicPolynomials`, which was written specifically for the use
in `PolynomialOptimization`, has not yet made it into the original package; you need a
[development version](https://github.com/projekter/DynamicPolynomials.jl) for this.
Use `conj` at your discretion, but note that `real` and `imag` should not be used in the problem description! Instead, use
`(z + conj(z))/2` for the real and `im*(conj(z) - z)/2` for the imaginary part, as well as `z*conj(z)` for the absolute value
square.

As soon as [`poly_problem`](@ref) detects complex variables, it switches to the complex-valued hierarchy. Note that
equality constraints will no longer be handled by Gröbner basis methods, but instead as two separate inequality constraints.
For complex-valued optimizations, only the methods `:MosekMoment` (which requires Mosek 10+) and `:HypatiaMoment` are
available.
```Julia
julia> @polycvar z;
julia> prob = poly_problem(z + conj(z), 1, zero=[z*conj(z)-1]);
julia> poly_optimize(:MosekMoment, prob, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, -2.0, Tuple{Vector{ComplexF64}, Float64}[([-0.9999999999999998 + 0.0im], 4.440892098500626e-16)])
```
The dense solution extraction mechanism also works in the complex case.

Let's try a more complicated example from the paper on the complex-valued Lasserre hierarchy (example 4.1):
```Julia
julia> @polycvar z[1:2];
julia> prob = poly_problem(3 - z[1]*conj(z[1]) - .5im*z[1]*conj(z[2])^2 + .5im*z[2]^2*conj(z[1]), 3,
                           zero=[z[1]*conj(z[1])-.25z[1]^2-.25conj(z[1])^2-1, # abs(z₁)^2 - z₁^2/4 - conj(z₁)^2/4 = 1
                                 z[1]*conj(z[1])+z[2]*conj(z[2])-3, # abs(z₁)^2 + abs(z₂)^2 = 3
                                 im*z[2]-im*conj(z[2])], # i z₂ - i conj(z₂) = 0
                           nonneg=[z[2]+conj(z[2])]); # z₂ + conj(z₂) ≥ 0
julia> poly_optimize(:MosekMoment, prob, solutions=true)
(Mosek.MSK_SOL_STA_OPTIMAL, 0.42817479499599975, Tuple{Vector{ComplexF64}, Float64}[([2.695688878989238e-18 - 0.8164965277154765im, 1.5275251790901432 + 5.333185915395668e-18im], 2.474741958025106e-7)])
```
Indeed, this solution gives the same objective value and satisfies the constraints, so we found the optimum!

And finally something with matrices:
```Julia
julia> @polycvar z[1:2];
julia> poly_optimize(:MosekMoment, poly_problem(-x[1]*conj(x[1]) - x[2]*conj(x[2]), 2,
                                                psd=[[1-2*(x[1]*x[2]+conj(x[1]*x[2]))  x[1]
                                                      conj(x[1])  4-x[1]*conj(x[1])-x[2]*conj(x[2])]]))
(Mosek.MSK_SOL_STA_OPTIMAL, -3.999999918874431, [([0.000631210850055003, 1.99999988011182], 3.1932358310108857e-7), ([-0.000631210850055003, -1.99999988011182], 3.1932358310108857e-7)])
```

Note that the implementation of the solution extraction algorithm also works in the complex case even though the moment matrix
is no longer of Hankel form; the theory is powerful enough to handle this "minor detail." The built-in heuristic will still try
to find good solutions and can sometimes do so even in the case of multiple solutions if they only differ in the phase of
variables.