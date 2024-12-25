"""
    RepresentationPSD <: RepresentationMethod

Model the constraint "σ ⪰ 0" as a positive semidefinite cone membership, `σ ∈ PSD`. This is the strongest possible model, but
the most resource-intensive.
"""
struct RepresentationPSD end

"""
    RepresentationSDD([u]; complex=true) <: RepresentationMethod

Model the constraint "σ ⪰ 0" as a membership in the scaled diagonally dominant cone, ``\\sigma = u^\\dagger Q u`` for some
`Q ∈ SDD`. The matrix `u` is by default an identity of any dimension; however, usually, care must be taken to have a matrix
of suitable dimension.
The membership `Q ∈ SDD` is achieved using the scaled diagonally dominant (dual) cone directly or (rotated) quadratic cones.

If σ is a Hermitian matrix, a complex-valued scaled diagonally dominant (dual) cone will be used, if supported. If not,
fallbacks to the (rotated) quadratic cones are used; however, if `complex=false` and the ordinary scaled diagonally dominant
(dual) cone is supported, rewrite the matrix as a real one and then use the real-valued cone. This is usually never advisable,
as the rotated quadratic cone always works on the complex data.
Note that if rewritten, `u` must be real-valued and have twice the side dimension of the complex-valued matrix.
"""
struct RepresentationSDD{M,Complex}
    u::M

    RepresentationSDD(u::M=I; complex=true) where {M} = new{M,complex}(u)
end

@doc raw"""
    RepresentationDD([u]; complex=true) <: RepresentationMethod

Model the constraint "σ ⪰ 0" as a membership in the diagonally dominant cone, ``\sigma = u^\dagger Q u`` for some `Q ∈ DD`.
The matrix `u` is by default an identity of any dimension; however, usually, care must be taken to have a matrix of suitable
dimension.
The membership `Q ∈ DD` is achieved using the diagonally dominant (dual) cone directly, ``\ell_1``- or ``\ell_\infty``-norm
cones or linear inequalities; slack variables will be added as necessary.

If σ is a Hermitian matrix, a complex-valued diagonally dominant (dual) cone will be used, if supported. If not, fallbacks will
first try quadratic cones on the complex-valued data, and if this is also not supported, rewrite the matrix as a real one and
then apply the real-valued DD constraint.
By setting the `complex` parameter to `false`, the rewriting to a real matrix will always be used, regardless of complex-valued
solver support.
Note that if rewritten, `u` must be real-valued and have twice the side dimension of the complex-valued matrix.
"""
struct RepresentationDD{M,Complex}
    u::M

    RepresentationDD(u::M=I; complex=true) where {M} = new{M,complex}(u)
end

"""
    RepresentationMethod

Union type that defines how the optimizer constraint σ ⪰ 0 is interpreted. Usually, "⪰" means positive semidefinite;
however, there are various other possibilities giving rise to weaker results, but scale more favorably. The following methods
are supported:
- [`RepresentationPSD`](@ref)
- [`RepresentationSDD`](@ref)
- [`RepresentationDD`](@ref)
"""
const RepresentationMethod = Union{RepresentationPSD,<:RepresentationSDD,<:RepresentationDD}