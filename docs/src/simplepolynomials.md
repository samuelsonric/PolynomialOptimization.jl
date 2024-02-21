```@meta
CurrentModule = PolynomialOptimization.SimplePolynomials
```

# SimplePolynomials
`PolynomialOptimization` allows data in the form of any implementation that supports the
[`MultivariatePolynomials`](https://github.com/JuliaAlgebra/MultivariatePolynomials.jl) interface. However, it does not keep
the data in this way, which would not be particularly efficient. Instead, it is converted into an internal format, the
`SimplePolynomial`. It offers very compact storage (though it does not pack exponents as
[`SIMDPolynomials`](https://github.com/YingboMa/SIMDPolynomials.jl) does) and is particularly focused on being as
allocation-free as possible - which means that once the original polynomials were created, at no further stage in processing
the polynomials or monomial bases will any allocations be done.

This means that `SimplePolynomial`s (and all their related types) are immutable; multiple objects may share the same memory,
they may even directly use the arguments that are passed to the constructors without copying. Additionally, they don't support
any arithmetic operations - no addition, no multiplication, .... However, there is a very special support for multiplication:
instead of returning the resulting monomial, just the _index_ of this monomial with respect to a particular fixed ordering is
calculated.

This makes `SimplePolynomials` very specialized for the particular needs of `PolynomialOptimization`; however, it is wrapped in
its own subpackage and can be loaded independently of the main package. Don't do the conversion manually and then pass the
converted polynomials to [`poly_problem`](@ref) - when the polynomial problem is initialized, depending on the keyword
arguments, some arithmetic operations still need to be carried out using the full interface of `MultivariatePolynomials`, not
just the restricted subset that `SimplePolynomials` provides.

## Types for the MultivariatePolynomials interface
```@docs
SimpleVariable
SimpleMonomial
SimpleMonomialVector
SimplePolynomial
```

## Additional types and functions
```@docs
monomial_count
monomial_index
MonomialIterator
RangedMonomialIterator
MultivariatePolynomials.monomials
LazyMonomials
effective_nvariables
exponents_from_index!
expoennts_from_index_prepare
ownpowers
```