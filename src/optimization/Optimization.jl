export poly_optimize, optimality_certificate, RepresentationPSD, RepresentationDD, RepresentationSDD, RepresentationNondiagI,
    IterateRepresentation

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

struct IterateRepresentation
    keep_diagonal::Bool

    @doc """
        IterateRepresentation(; keep_diagonal=false)

    Default iteration method for DD and SDD representations. This is will perform a Cholesky decomposition of the old SOS
    matrix and use it as the new rotation, ensuring that results never get worse (at least in theory; since a positive definite
    SOS matrix is only guaranteed up to a certain tolerance, bad things could still happen).

    Note that this breaks existing diagonal structures; by setting `keep_diagonal` to `true`, only the diagonal of the Cholesky
    factor will be taken (with no theoretical guarantees, not even about convergence).

    See also [`poly_optimize`](@ref poly_optimize(::Result))
    """
    IterateRepresentation(; keep_diagonal::Bool=false) = new(keep_diagonal)
end

function (i::IterateRepresentation)((type, index, grouping), _, oldrep::Type{<:RepresentationMethod}, oldsos)
    oldrep <: RepresentationPSD && return RepresentationPSD()
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
    if i.keep_diagonal && oldrep <: RepresentationMethod{<:Union{<:UniformScaling,<:Diagonal}} &&
        !(newrot isa Union{<:UniformScaling,<:Diagonal})
        return oldrep(Diagonal(newrot)) # well, maybe we should care...
    else
        return oldrep(newrot)
    end
end

"""
    poly_optimize(result::Result; [representation=IterateRepresentation(), ]kwargs...)

Re-optimizes a previously optimized polynomial optimization problem. This is usually pointless, as the employed optimizers will
find globally optimal solutions. However, this method allows to change the representation used for the constraints (or
objective). If `representation` is a callable, it will now receive as a third parameter the type of the
[`RepresentationMethod`](@ref) used before for this constraint[^2], and as a fourth parameter the associated SOS matrix from
the previous optimization. For efficiency reasons, this should only be used for changes that preserve the structure of the
representation (i.e., whether it was PSD/DD/SDD and if its rotation was diagonal or not). If a structure non-preserving change
is made, the problem needs to be constructed from scratch. For non-diagonal rotations, consider using
[`RepresentationNondiagI`](@ref) in the first optimization.

!!! warning
    The internal state of the previous solver run will be re-used whenever possible. Therefore, no further data may be queried
    from the previous result afterwards, unless a re-optimization from scratch was necessary. While `result` will still be able
    to offer information about the relaxation, method, time, status, and objective value, moment matrices can only be accessed
    if they were already cached (i.e., accessed) before. Existing SOS certificates of the previous result will still be
    available, but new ones may not be constructed.

See also [`IterateRepresentation`](@ref).

[^2]: Roughly, as the exact type is not known. For sure, it will be possible to distinguish between [`RepresentationDD`](@ref)
      and [`RepresentationSDD`](@ref) (the callback will not be invoked for [`RepresentationPSD`](@ref), as nothing can be
      changed there). The matrix type will not be concrete, but either `Union{<:UniformScaling,<:Diagonal}` if a diagonal
      representation was used before, or `Matrix` else. The complex identification will be `true` if a complex-valued cone was
      used and `false` else (where during specification, it could also have been `true` for real-valued data, which would
      simply be ignored). Despite not being concrete, these two possible union types can be used as constructors accepting the
      new rotation matrix as parameter.
"""
function poly_optimize(result::Result; representation=IterateRepresentation(), verbose::Bool=false, kwargs...)
    ismissing(result.state) && throw(ArgumentError("The given result does not contain any data."))
    @verbose_info("Beginning re-optimization...")
    oldstate = result.state
    if !(representation isa RepresentationMethod)
        representation = Solver.Rerepresent([[s for (s, _) in x] for x in Solver.extract_info(result.state)],
            SOSCertificate(result), representation, true)
    end
    relaxation = result.relaxation
    local otime
    try
        otime = @elapsed begin
            new_result = poly_optimize(Val(result.method), result.state, relaxation, Relaxation.groupings(relaxation);
                verbose, representation, kwargs...)
        end
    catch e
        if e isa Solver.RepresentationChangedError && representation isa Solver.Rerepresent
            @warn("The representation of at least one grouping changed, either in type or diagonally. The problem has to be \
                   set-up from the beginning. If this is due to a change in diagonality, consider using \
                   RepresentationNondiagI as initial value.")
            representation = Solver.Rerepresent(representation, false)
            otime = @elapsed begin
                new_result = Solver._Copied(poly_optimize(Val(result.method), relaxation, Relaxation.groupings(relaxation),
                    verbose, representation, kwargs...))
            end
        else
            rethrow(e)
        end
    end
    if new_result isa Solver._Copied
        new_result = new_result.data
    else
        result.state = missing
    end
    return Result(relaxation, result.method, otime, new_result...)
end