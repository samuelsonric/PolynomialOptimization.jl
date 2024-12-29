export poly_optimize, optimality_certificate, RepresentationPSD, RepresentationDD, RepresentationSDD, RepresentationNondiagI

include("./Result.jl")
include("./MomentMatrix.jl")
include("./SOSCertificate.jl")
include("./OptimalityCertificate.jl")
include("./solver/Solver.jl")
using .Solver: default_solver_method, monomial_count, RepresentationMethod, RepresentationPSD, RepresentationDD,
    RepresentationSDD, RepresentationNondiagI
import .Solver: poly_optimize
using StandardPacked: SPMatrix, SPMatrixUpper, SPMatrixLower

_val_of(::Val{S}) where {S} = S

"""
    poly_optimize([method, ]relaxation::AbstractRelaxation; verbose=false,
        representation=RepresentationPSD(), kwargs...)

Optimize a relaxed polynomial optimization problem that was construced via [`poly_problem`](@ref) and then wrapped into an
[`AbstractRelaxation`](@ref). Returns a [`Result`](@ref) object.

Instead of modeling the moment/SOS matrices as positive semidefinite, other representations such as the (scaled) diagonally
dominant description are also possible. The `representation` parameter can be used to define a representation that is employed
for the individual groupings. This may either be an instance of a [`RepresentationMethod`](@ref) - which requires the method to
be independent of the dimension of the grouping - or a callable. In the latter case, it will be passed as a first parameter an
identifier[^1] of the current conic variable, and as a second parameter the side dimension of its matrix. The method must then
return a [`RepresentationMethod`](@ref) instance.

`verbose=true` will enable logging; this will print basic information about the relaxation itself as well as instruct the
solver to output a detailed log. The PSD block sizes reported accurately represent the side dimensions of semidefinite
variables and how many of these variables appear. The free block sizes are only very loose upper bounds on the maximal number
of equality constraints that will be constructed by multiplying two elements from a block, as duplicates will be ignored.
Any additional keyword argument is passed on to the solver.

For a list of supported methods, see [the solver reference](@ref solvers_poly_optimize). If `method` is omitted, the default
solver is used. Note that this depends on the loaded solver packages, and possibly also their loading order if no preferred
solver has been loaded.

See also [`RepresentationNondiagI`](@ref).

[^1]: This identifier will be a tuple, where the first element is a symbol - either `:objective`, `:nonneg`, or `:psd` - to
      indicate the general reason why the variable is there. The second element is an `Int` denoting the index of the
      constraint (and will be undefined for the objective, but still present to avoid extra compilation). The last element
      is an `Int` denoting the index of the grouping within the constraint/objective.
"""
function poly_optimize(@nospecialize(v::Val), relaxation::AbstractRelaxation; verbose::Bool=false,
    representation=RepresentationPSD(), kwargs...)
    otime = @elapsed begin
        @verbose_info("Beginning optimization...")
        groups = groupings(relaxation) # This is instantaneous, as the groupings were already calculated when the relaxation
                                       # was constructed.
        if verbose
            bs = StatsBase.countmap(length.(groups.obj))
            @unroll for constrs in (groups.nonnegs, groups.psds)
                for constr in constrs
                    mergewith!(+, bs, StatsBase.countmap(length.(constr)))
                end
            end
            print("PSD block sizes:\n  ", sort!(collect(bs), rev=true))
            if !isempty(groups.zeros)
                empty!(bs)
                for constr in groups.zeros
                    mergewith!(+, bs, StatsBase.countmap(length.(constr)))
                end
                print("\nFree block sizes:\n  ", sort!(collect(bs), rev=true))
            end
            println("\nStarting solver...")
        end
        result = poly_optimize(v, relaxation, groups; verbose, representation, kwargs...)
    end
    return Result(relaxation, _val_of(v), otime, result...)
end

"""
    poly_optimize([method, ]problem::Problem[, degree::Int]; kwargs...)

Construct a [`Relaxation.Dense`](@ref) by default.
"""
poly_optimize(v::Val, problem::Problem, rest...; kwargs...) =
    poly_optimize(v, Relaxation.Dense(problem, rest...); kwargs...)

poly_optimize(s::Symbol, rest...; kwrest...) = poly_optimize(Val(s), rest...; kwrest...)

function poly_optimize(args...; kwargs...)
    if !isempty(args) && args[1] isa Val
        error("Unknown solver method specified. Are the required solver packages loaded?")
    end
    method = default_solver_method()
    @info("No solver method specified: choosing $method")
    poly_optimize(Val(method), args...; kwargs...)
end

const warned_diagonal = Ref(false)

function iterate_representation((type, index, grouping), _, oldrep::Type{<:RepresentationMethod}, oldsos)
    @assert(!(oldrep <: RepresentationPSD))
    # We need to perfom a Cholesky decomposition in the form oldsos = Uᵀ * U, where U is upper triangular.
    # We have multiple issues here:
    # - oldsos might be a Hermitian matrix with the lower triangle specified, so cholesky will return the L * Lᵀ form.
    # - oldsos might be a SPMatrix, so that the pptrf is called, which is numerically less stable.
    # - oldsos might have tiny negative eigenvalues, strictly speaking making the Cholesky decomposition undefined.
    # Individually, these issues are unproblematic; just take the adjoint, don't worry about the instability, disable the
    # check. However, combined, cholesky(oldsos, check=false).U can lead to extremely bad quality, even to worsening from
    # iteration to iteration. Therefore, we must do the more laborious thing: We make sure we can apply the more stable full
    # storage methods, natively use the upper triangle. For the negative eigenvalues, we really just ignore them.
    if oldsos isa (Symmetric{T,Matrix{T}} where {T<:Real}) || oldsos isa (Hermitian{T,Matrix{T}} where {T})
        if oldsos.uplo == 'U'
            fullm = Hermitian(parent(oldsos), :U) # just for type stability
        else
            fullm = Hermitian(LinearAlgebra.copytri!(parent(oldsos), 'L', true), :U)
        end
    elseif oldsos isa Matrix
        fullm = Hermitian(oldsos, :U)
    elseif oldsos isa SPMatrixUpper
        fullm = Hermitian(convert(Matrix{eltype(oldsos)}, oldsos), :U)
    elseif oldsos isa SPMatrixLower
        fullm = Hermitian(LinearAlgebra.copytri!(convert(Matrix{eltype(oldsos)}, oldsos), 'L', true), :U)
    else
        fullm = Hermitian(Matrix(oldsos)::Matrix{eltype(oldsos)}, :U)
    end
    newrot = cholesky(fullm, check=false).U
    if oldrep <: RepresentationMethod{<:Union{<:UniformScaling,<:Diagonal}} && !(newrot isa Union{<:UniformScaling,<:Diagonal})
        # we just force it to be diagonal, but let's issue a warning once
        if !warned_diagonal[]
            @warn("The reoptimization was done enforcing a diagonal rotation. This may lead to suboptimal result or numerical \
                   problems. Use a non-diagonal rotation in the initial optimization (e.g., RepresentationNondiagI) to allow \
                   for arbitrary rotations. This warning will only be shown once in the current Julia session.")
            warned_diagonal[] = true
        end
        return oldrep(Diagonal(newrot))
    else
        return oldrep(newrot)
    end
end

struct _Rerepresent{C<:SOSCertificate,F}
    info::Vector{Vector{Symbol}}
    cert::C
    fn::F
end

function (r::_Rerepresent)((type, index, grouping), dim)
    idx = id_to_index(r.cert.relaxation, (type, index, grouping))
    oldtype = r.info[idx][grouping]
    if oldtype in Solver.INFO_PSD
        # nothing can be changed
        return RepresentationPSD()
    end
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
    newrep = r.fn((type, index, grouping), dim, oldrep, r.cert.data[idx][grouping])
    if (oldtype in Solver.INFO_SDD && !(newrep isa RepresentationSDD)) ||
        (oldtype in Solver.INFO_DD && !(newrep isa RepresentationDD)) ||
        (complex && newrep isa RepresentationMethod{<:Any,false})
        error("The representation of individual types must not change")
    end
    newdiag = newrep isa RepresentationMethod{<:Union{<:UniformScaling,<:Diagonal}}
    diagonal == newdiag || error("A representation must not change from a diagonal to a nondiagonal type or vice versa")
    return newrep
end

"""
    poly_optimize(result::Result; [representation, ]kwargs...)

Re-optimizes a previously optimized polynomial optimization problem. This is usually pointless, as the employed optimizers will
find globally optimal solutions. However, this method allows to change the representation used for the constraints (or
objective) in a very limited way: It is possible to adjust the rotation of the DD and SDD representation. If `representation`
is a callable, it will now receive as a third parameter the type of the [`RepresentationMethod`](@ref) used before for this
constraint[^2], and as a fourth parameter the associated SOS matrix from the previous optimization.

The default implementation will take this SOS matrix, perform a Cholesky decomposition, and return the result as the new
rotation for the DD or SDD cone. However, note that only structure-preserving rotations are possible; i.e., if the rotation was
diagonal before, it must be diagonal afterwards. In order to allow for non-diagonal rotations, start using the
[`RepresentationNondiagI`](@ref) representation in the first optimization.

!!! warning
    The internal state of the previous solver run will be re-used whenever possible. Therefore, no further data may be queried
    from the previous result afterwards. While `result` will still be able to offer information about the relaxation, method,
    time, status, and objective value, moment matrices can only be accessed if they were already cached (i.e., accessed)
    before. Existing SOS certificates of the previous result will still be available, but new ones may not be constructed.

[^2]: Roughly, as the exact type is not known. For sure, it will be possible to distinguish between [`RepresentationDD`](@ref)
      and [`RepresentationSDD`](@ref) (the callback will not be invoked for [`RepresentationPSD`](@ref), as nothing can be
      changed there). The matrix type will not be concrete, but either `Union{<:UniformScaling,<:Diagonal}` if a diagonal
      representation was used before, or `Matrix` else. The complex identification will be `true` if a complex-valued cone was
      used and `false` else (where during specification, it could also have been `true` for real-valued data, which would
      simply be ignored). Despite not being concrete, these two possible union types can be used as constructors accepting the
      new rotation matrix as parameter.
"""
function poly_optimize(result::Result; representation=iterate_representation, verbose::Bool=false, kwargs...)
    ismissing(result.state) && throw(ArgumentError("The given result does not contain any data."))
    @verbose_info("Beginning re-optimization...")
    oldstate = result.state
    if !(representation isa RepresentationMethod)
        representation = _Rerepresent([[s for (s, _) in x] for x in Solver.extract_info(result.state)], SOSCertificate(result),
            representation)
    end
    relaxation = result.relaxation
    otime = @elapsed begin
        new_result = poly_optimize(Val(result.method), result.state, relaxation, Relaxation.groupings(relaxation);
            verbose, representation, kwargs...)
    end
    if new_result isa Solver._Copied
        new_result = new_result.data
    else
        result.state = missing
    end
    return Result(relaxation, result.method, otime, new_result...)
end