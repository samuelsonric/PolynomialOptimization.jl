module Newton

using MultivariatePolynomials, ..SimplePolynomials, ..FastVector, SparseArrays, Printf
import BufferedStreams
using ..SimplePolynomials: SimpleRealPolynomial, SimpleComplexPolynomial, SimpleComplexMonomial, smallest_unsigned
using PolynomialOptimization: @assert, @verbose_info, @capture, haveMPI, matrix_delete_end!, resizable_array, resizable_copy,
    keepcol!, FastKey, RelaxationGroupings

export halfpolytope, halfpolytope_from_file

_effective_nvars(::SimplePolynomial{<:Any,Nr,0}) where {Nr} = Nr
_effective_nvars(::SimplePolynomial{<:Any,0,Nc}) where {Nc} = Nc
_effective_nvars(::SimplePolynomial) = error("Mixing real- and complex-valued variables prevents Newton polytope methods")

"""
    halfpolytope(method, poly; verbose=false, preprocess_quick=true,
        preprocess_randomized=false, preprocess_fine=false, preprocess=nothing,
        filepath=nothing, parameters...)

Calculates the Newton polytope for the sum of squares optimization of a given objective, which is half the Newton polytope of
the objective itself. This requires the availability of a linear solver. For a list of supported solvers, see
[the solver reference](@ref solvers_poly_optimize).

There are three preprocessing methods which can be turned on individually or collectively using `preprocess`; depending on the
problem, they may reduce the amount of time that is required to construct the convex hull of the full Newton polytope:
- `preprocess_quick` is the Akl-Toussaint heuristic. Every monomial will be checked against a linear program that scales as the
  number of variables in the objective. This is enabled by default.
- `preprocess_randomized` performs a reduction of the possible number of monomials that comprise the convex hull by picking
  smaller random subsets of them and eliminating entries in the subset that can be expressed by other entries. This is a good
  idea if the number of candidate monomials for the vertices of the convex hull is huge (so that `preprocess_fine` will take
  too long) but also very redundant. The final polish can be done by enabling both this and the following preprocessing option.
  Randomized reduction will use multithreading if possible.
- `preprocess_fine` performs an extensive reduction of the possible number of monomials that comprise the convex hull. Every
  monomial will be checked against a linear program that scales as the number of monomials in the objective (though it might
  become more efficient when monomials are ruled out).
After preprocessing is done, the monomials in the half Newton polytope are constructed efficiently subject to a simple
min/max-degree constraint using [`MonomialIterator`](@ref) and taken over into the basis if they are contained in the convex
polytope whose vertices were determined based on the objective and preprocessing; this is done by performing a linear program
for each candidate monomial.

The `parameters` will be passed on to the linear solver in every case (preprocessing and construction).

!!! info "Multithreading"
    For large initial sets of monomials (≥ 10⁴), the final construction will use multithreading if possible. Make sure to start
    Julia with an appropriate number of threads configured.

!!! tip "Distributed computing"
    This function is capable of using MPI for multi-node distributed computing. For this, make sure to start Julia using
    `mpiexec`, appropriately configured; then load the `MPI` package in addition to `PolynomialOptimization` (this is required
    for distributed computing to work). If `MPI.Init` was not called before, `PolynomialOptimization` will do it for you.
    This function is compatible with the MPI thread level `MPI.THREAD_FUNNELED` if multithreading is used in combination with
    MPI. Currently, only the main function will use MPI, not the preprocessing.

    Note that the function will assume that each MPI worker has the same number of threads available. Further note that Julia's
    GC works in a multithreaded context using the SIGSEG signal. This is known to cause problems among all MPI backends, which
    can usually be fixed by using the most recent version of MPI and setting some environment variables. Not all of these
    settings are incorporated into the MPI package yet. For OpenMPI and Intel MPI, set `ENV["IPATH_NO_BACKTRACE"] = "1"`.

!!! warning "Verbose output"
    The `verbose` option generates very helpful output to observe the current progress. It also works in a multithreaded and
    distributed context. However, consider the fact that providing these messages requires additional computational and
    communication effort and should not be enabled when speed matters.

!!! tip "Interrupting the computation/Large outputs"
    If you expect the final Newton basis to be very large, so that keeping everything in memory (potentially in parallel) might
    be troublesome, the option `filepath` allows to instead write the output to a file. This is also useful if the process of
    determining the polytope is aborted, as it can be resumed from its current state (also in a multithreaded or
    multiprocessing context) if the same file name is passed to `filepath`, provided the Julia configuration (number of
    threads, number of processes) was the same at any time. Make sure to always delete the output files if you compute the with
    a different configuration or the results will probably be corrupt!

    Using this option will create one (or multiple, if multithreading/multiprocessing is used) file that has the file name
    `filepath` with the extension `.out`, and for every `.out` file also a corresponding `.prog` file that captures the current
    status. The `.out` file(s) will hold the resulting basis in a binary format, the `.prog` file is a small indicator required
    for resuming the operation after an abort.
    This function will only `true` when it is finished and the data was stored to a file; it will _not_ load the actual data.
    To do so, use [`halfpolytope_from_file`](@ref) in a separate step, which can also tell you exactly how much memory will be
    required for this operation.

See also [`halfpolytope_from_file`](@ref).
"""
function halfpolytope(method::Symbol, poly::SimplePolynomial; kwargs...)
    P = typeof(poly)
    mhd = maxhalfdegree(poly)
    T = smallest_unsigned(2mhd)
    mons = LazyMonomials{_effective_nvars(poly),0}(zero(T):T(mhd), powers=ownpowers)
    MV = typeof(mons)
    return halfpolytope(Val(method), poly, Val(haveMPI[]); zero=P[], nonneg=P[], psd=Matrix{P}[],
        groupings=RelaxationGroupings(
            [mons], Vector{MV}[], Vector{MV}[], Vector{MV}[], Vector{variable_union_type(poly)}[]
        ), kwargs...)
end
function halfpolytope(method::Symbol, poly::AbstractPolynomialLike; verbose::Bool=false, kwargs...)
    out = halfpolytope(method, SimplePolynomial(poly); verbose, kwargs...)
    if out isa SimpleMonomialVector
        conv_time = @elapsed begin
            real_vars = variable_union_type(poly)[]
            complex_vars = similar(real_vars)
            for v in variables(poly)
                if isreal(v)
                    push!(real_vars, v)
                elseif isconj(v)
                    vo = conj(v)
                    vo ∈ complex_vars || push!(complex_vars, vo)
                else
                    push!(complex_vars, v)
                end
            end
            mv = monomial_vector(FakeMonomialVector(out, real_vars, complex_vars))
        end
        @verbose_info("Converted monomials back to a $(typeof(mv)) with length $(length(mv)) in $conv_time seconds")
        return mv
    else
        return out
    end
end
halfpolytope(objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    halfpolytope(default_newton_method(), objective; kwargs...)

function halfpolytope(V, objective::P, ::Val{false}; verbose::Bool=false, filepath::Union{<:AbstractString,Nothing}=nothing,
    zero::AbstractVector{P}, nonneg::AbstractVector{P}, psd::AbstractVector{<:AbstractMatrix{P}},
    groupings::RelaxationGroupings, kwargs...) where {P<:SimpleRealPolynomial}
    parameters, coeffs = preproc(V, objective; verbose, zero, nonneg, psd, groupings, kwargs...)
    newton_time = @elapsed candidates = let
        analysis = analyze(coeffs)
        # We don't construct the monomials using monomials(). First, it's not the most efficient implementation underlying,
        # and we also don't want to create a huge list that is then filtered (what if there's no space for the huge list?).
        # However, since we implement the monomial iteration by ourselves, we must make some assumptions about the
        # variables - this is commuting only.
        iter = MonomialIterator(analysis..., ownpowers)
        num = length(iter)
        @verbose_info("Starting point selection among ", num, " possible monomials")
        nthreads, task, secondtask = prepare(V, coeffs, num, verbose; parameters...)
        execute(V, size(coeffs, 1), verbose, iter, num, nthreads, task, secondtask, filepath)
    end

    if isnothing(filepath)
        @verbose_info("Found ", length(candidates), " elements in the Newton halfpolytope in ", newton_time, " seconds")
        return candidates
    else
        @verbose_info("Found ", candidates[2], " elements in the Newton halfpolytope in ", newton_time,
            " seconds and stored the results to the given file")
        return true
    end
end

function halfpolytope(::Val{:complex}, objective::P, ::Any; verbose::Bool=false, zero::AbstractVector{P}=P[],
    nonneg::AbstractVector{P}=P[], psd::AbstractVector{<:AbstractMatrix{P}}=Matrix{P}[],
    degree::Int=maxhalfdegree(objective)) where {Nc,P<:SimpleComplexPolynomial{<:Any,Nc}}
    # For complex-valued polynomials, the SDP looks like dot(basis, M, basis); due to the conjugation of the first element,
    # this is a 1:1 mapping between elements in M and monomials - contrary to the non-unique real case. Given that the
    # polynomials must be real-valued, any monomial that is present in the objective will also be present with its conjugate.
    # So we just have to look at the exponents_complex, ignoring exponents_conj, of each monomial, and if it is present, then
    # this monomial needs to be in the basis. This simple construction is the reason why this method is neither parallelized
    # nor has a distributed version.
    nv = Nc
    newton_time = @elapsed begin
        if isempty(zero) && isempty(nonneg) && isempty(psd)
            @verbose_info("Complex-valued Newton polytope without constraints: copying exponents")
            exps = unique(monomials(objective).exponents_complex, dims=2)
        else
            @verbose_info("Complex-valued Newton polytope: merging constraints")
            exps = merge_constraints(
                degree, SimplePolynomials.smallest_unsigned(monomial_count(2degree, nv)), objective, zero, nonneg, psd,
                Val(monomials(objective).exponents_real isa DenseMatrix), verbose
            )
        end
    end
    @verbose_info("Found ", size(exps, 2), " elements in the complex-valued \"Newton halfpolytope\" in ", newton_time,
        " seconds")
    return SimpleMonomialVector{0,nv}(exps, convert(typeof(exps), spzeros(eltype(exps), size(exps)...)))
end

const newton_methods = Symbol[]

function default_newton_method()
    isempty(newton_methods) &&
        error("No Newton method is available. Load a solver package that provides such a method (e.g., Mosek)")
    return first(newton_methods)
end

include("./helpers/Utils.jl")
include("./helpers/Sampling.jl")
include("./helpers/InitialStateIterator.jl")
include("./helpers/FakeMonomialVector.jl")
include("./Interface.jl")
include("./Constraints.jl")
include("./Preprocessing.jl")
include("./Worker.jl")
include("./Files.jl")

end