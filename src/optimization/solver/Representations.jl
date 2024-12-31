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

    RepresentationSDD(u::M=I; complex::Bool=true) where {M} = new{M,complex}(u)
end

(::Type{<:RepresentationSDD{<:Any,complex}})(u) where {complex} = RepresentationSDD(u; complex)

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

    RepresentationDD(u::M=I; complex::Bool=true) where {M} = new{M,complex}(u)
end

(::Type{<:RepresentationDD{<:Any,complex}})(u) where {complex} = RepresentationDD(u; complex)

"""
    RepresentationMethod{M,Complex}

Union type that defines how the optimizer constraint σ ⪰ 0 is interpreted. Usually, "⪰" means positive semidefinite;
however, there are various other possibilities giving rise to weaker results, but scale more favorably. The following methods
are supported:
- [`RepresentationPSD`](@ref)
- [`RepresentationSDD`](@ref)
- [`RepresentationDD`](@ref)

See also [`RepresentationNondiagI`](@ref).
"""
const RepresentationMethod{M,Complex} = Union{RepresentationPSD,RepresentationSDD{M,Complex},RepresentationDD{M,Complex}}

"""
    RepresentationNondiagI(r::Type{RepresentationDD,RepresentationSDD}; complex=true)

Default callable that will instantiate a correctly-sized representation of type `r` with an identity rotation that is, however,
not recognized as a diagonal rotation. Use this type in the first call to [`poly_optimize`](@ref) if you want to re-optimize
the problem afterwards using arbitrary rotations.
"""
struct RepresentationNondiagI{R<:Union{RepresentationDD,RepresentationSDD},Complex}
    RepresentationNondiagI(r::R; complex::Bool=true) where {R<:Type{<:Union{RepresentationDD,RepresentationSDD}}} =
        new{r,complex}()
end

(::RepresentationNondiagI{R,complex})(_, dim) where {R,complex} = R(Matrix(I, dim, dim); complex)

struct RepresentationChangedError <: Exception end

struct Rerepresent{C<:SOSCertificate,F}
    info::Vector{Vector{Symbol}}
    cert::C
    fn::F
    fix_structure::Bool
end

Rerepresent(r::Rerepresent, fix_structure::Bool) = Rerepresent(r.info, r.cert, r.fn, fix_structure)

function (r::Rerepresent)((type, index, grouping), dim)
    idx = id_to_index(r.cert.relaxation, (type, index, grouping))
    oldtype = r.info[idx][grouping]
    if oldtype in Solver.INFO_PSD
        oldrep = RepresentationPSD
        complex = false # just to make it compatible with the checks below
        diagonal = true
    else
        complex = oldtype in Solver.INFO_COMPLEX
        diagonal = oldtype in Solver.INFO_DIAG
        if oldtype in Solver.INFO_SDD
            oldrep = diagonal ? RepresentationSDD{<:Union{<:UniformScaling,<:Diagonal},complex} :
                                RepresentationSDD{<:Matrix,complex}
        else
            @assert(oldtype in Solver.INFO_DD)
            oldrep = diagonal ? RepresentationDD{<:Union{<:UniformScaling,<:Diagonal},complex} :
                                RepresentationDD{<:Matrix,complex}
        end
    end
    newrep = r.fn((type, index, grouping), dim, oldrep, r.cert.data[idx][grouping])::RepresentationMethod
    if r.fix_structure
        ((oldtype in Solver.INFO_PSD && !(newtype isa RepresentationPSD)) ||
         (oldtype in Solver.INFO_SDD && !(newrep isa RepresentationSDD)) ||
         (oldtype in Solver.INFO_DD && !(newrep isa RepresentationDD)) ||
         (complex && newrep isa RepresentationMethod{<:Any,false}))
        throw(RepresentationChangedError())
    end
    newdiag = newrep isa RepresentationMethod{<:Union{<:UniformScaling,<:Diagonal}}
    !r.fix_structure || diagonal == newdiag || throw(RepresentationChangedError())
    return newrep
end