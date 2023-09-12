struct MonomialComplexContainer{M}
    mon::M
    re::Bool
end

MultivariatePolynomials.degree(c::MonomialComplexContainer) = degree(c.mon)

Base.hash(m::MonomialComplexContainer, u::UInt) = hash((m.mon, m.re), u)

Base.:(==)(m1::MonomialComplexContainer, m2::MonomialComplexContainer) =
    (m1.re == m2.re) && (m1.mon == m2.mon)

function MultivariatePolynomials._show(io::IO, mime, m::MonomialComplexContainer)
    print(io, m.re ? "Re(" : "Im(")
    MultivariatePolynomials._show(io, mime, m.mon)
    print(io, ")")
end