using Documenter
using PolynomialOptimization
using MultivariatePolynomials
using DynamicPolynomials

makedocs(sitename="PolynomialOptimization.jl",
    modules=[PolynomialOptimization, PolynomialOptimization.SimplePolynomials,
        PolynomialOptimization.SimplePolynomials.MultivariateExponents, PolynomialOptimization.FastVector,
        PolynomialOptimization.Newton, PolynomialOptimization.Solver],
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "index.md",
        "guide.md",
        "reference.md",
        "solverreference.md",
        "auxreference.md",
        "simplepolynomials.md"
    ],
    warnonly=:missing_docs)