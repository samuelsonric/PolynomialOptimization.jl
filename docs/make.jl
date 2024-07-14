using Documenter
using PolynomialOptimization, GALAHAD
using MultivariatePolynomials, DynamicPolynomials

makedocs(sitename="PolynomialOptimization.jl",
    modules=filter(m -> startswith(string(m), "PolynomialOptimization"), Docs.modules),
    format=Documenter.HTML(prettyurls=false),
    pages=[
        "index.md",
        "guide.md",
        "reference.md",
        "solverreference.md",
        "auxreference.md",
        "simplepolynomials.md"
    ],
    warnonly=:missing_docs,
    doctest=false
)