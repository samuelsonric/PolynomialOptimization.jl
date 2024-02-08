export MonomialIterator, newton_halfpolytope, newton_halfpolytope_from_file

"""
    newton_halfpolytope(method, poly; verbose=false, preprocess_quick=true,
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
    To do so, use [`newton_halfpolytope_from_file`](@ref) in a separate step, which can also tell you exactly how much memory
    will be required for this operation.

See also [`newton_halfpolytope_from_file`](@ref).
"""
newton_halfpolytope(method::Symbol, poly::SimplePolynomial; kwargs...) =
    newton_halfpolytope(Val(method), poly, Val(haveMPI[]); kwargs...)
function newton_halfpolytope(method::Symbol, poly::AbstractPolynomialLike; verbose::Bool=false, kwargs...)
    out = newton_halfpolytope(method, SimplePolynomial(poly); verbose, kwargs...)
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
newton_halfpolytope(objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_halfpolytope(default_newton_method(), objective; kwargs...)

# We provide some setter functions for SimpleDenseMonomialVector vector mainly due to the need for sorting when the vector is
# constructed in the parallelized Newton algorithm. This is the only reason - Simplexxx is not supposed to be mutable!
@inline function Base.setindex!(x::SimplePolynomials.SimpleRealDenseMonomialVectorOrView{Nr,P},
    val::SimplePolynomials.SimpleRealDenseMonomial{Nr,P}, i::Integer) where {Nr,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_real[:, i]), val.exponents_real)
    return val
end
@inline function Base.setindex!(x::SimplePolynomials.SimpleComplexDenseMonomialVectorOrView{Nc,P},
    val::SimplePolynomials.SimpleComplexDenseMonomial{Nc,P}, i::Integer) where {Nc,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_complex[:, i]), val.exponents_complex)
    @inbounds copyto!(@view(x.exponents_conj[:, i]), val.exponents_conj)
    return val
end
@inline function Base.setindex!(x::SimplePolynomials.SimpleDenseMonomialVectorOrView{Nc,P},
    val::SimplePolynomials.SimpleDenseMonomial{Nc,P}, i::Integer) where {Nc,P}
    @boundscheck checkbounds(x, i)
    @inbounds copyto!(@view(x.exponents_real[:, i]), val.exponents_real)
    @inbounds copyto!(@view(x.exponents_complex[:, i]), val.exponents_complex)
    @inbounds copyto!(@view(x.exponents_conj[:, i]), val.exponents_conj)
    return val
end

# The same goes for resizing, which is done after the monomial vector was sorted by calling unique!.
function Base.resize!(x::SimplePolynomials.SimpleRealMonomialVector{Nr,P,M}, len) where {Nr,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{Nr,0,P,M}(matrix_delete_end!(x.exponents_real, n))
end
function Base.resize!(x::SimplePolynomials.SimpleComplexMonomialVector{Nc,P,M}, len) where {Nc,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{0,Nc,P,M}(matrix_delete_end!(x.exponents_complex, n), matrix_delete_end!(x.exponents_conj, n))
end
function Base.resize!(x::SimplePolynomials.SimpleMonomialVector{Nr,Nc,P,M}, len) where {Nr,Nc,P<:Unsigned,M<:DenseMatrix}
    n = length(x) - len
    n < 0 && error("Cannot increase the size of a monomial vector")
    iszero(n) && return x
    return SimpleMonomialVector{Nr,Nc,P,M}(matrix_delete_end!(x.exponents_real, n), matrix_delete_end!(x.exponents_complex, n),
        matrix_delete_end!(x.exponents_conj, n))
end

#region Preprocessing for the full Newton polytope convex hull
"""
    newton_polytope_preproc_quick(V, coeffs::AbstractMatrix{<:Integer},
        vertexindices::Vector{Int}, verbose::Bool; parameters...)

Eliminate all the coefficients that by the Akl-Toussaint heuristic cannot be part of the convex hull anyway. The implementation
has to return a `Vector{Bool}` that for every column in `coeffs` indicates whether this particular column can be obtained as
the convex combinations of the columns indexed by `vertexindices` (`true` if the answer is yes).
"""
function newton_polytope_preproc_quick end

"""
    newton_polytope_preproc_remove(V, nv::Int, nc::Int, getvarcon::Function, verbose::Bool,
        singlethread::Bool; parameters...)

Removes all convex dependencies from the list of coefficients. There are `nc` coefficients, each with `nv` entries of Integer
type. The `AbstractVector` representing the `i`th coefficient (`1 ≤ i ≤ nc`) can be obtained by calling `getvarcon(i)`.
The implementation has to return a `Vector{Bool}` of length `nc` that for every coefficient contains `false` if and only if the
convex hull spanned by all coefficients is invariant under removal of this coefficient.
"""
function newton_polytope_preproc_remove end

function newton_polytope_preproc_prequick(V, coeffs, verbose; parameters...)
    nv = size(coeffs, 1)
    vertexindices = fill(1, 2nv)
    @inbounds lowestidx = @view(vertexindices[1:nv])
    @inbounds highestidx = @view(vertexindices[nv+1:2nv])
    # we might also add the points with the smallest/largest sum of all coordinates, or differences (but there are 2^nv ways to
    # combine, so let's skip it)
    @inbounds for (j, coeff) in enumerate(eachcol(coeffs))
        for (i, coeffᵢ) in enumerate(coeff)
            if coeffs[i, lowestidx[i]] > coeffᵢ
                lowestidx[i] = j
            end
            if coeffs[i, highestidx[i]] < coeffᵢ
                highestidx[i] = j
            end
        end
    end
    unique!(sort!(vertexindices))
    return newton_polytope_preproc_quick(V, coeffs, vertexindices, verbose; parameters...)
end

#region Sampling functions
# This is an adaptation of StatsBase.sample! with replace=false, however specifically adapted to the case where we are
# sampling indices from `a` whose value is `true`, we will accumulate them into `x` and set them to `false` in `a`.
# `total` must be the number of `true`s occuring in `a`, else the function output will be undefined.
function sample!(a::AbstractVector{Bool}, x::AbstractVector{<:Integer}, total::Integer=sum(a, init=0))
    k = length(x)
    iszero(k) && return x
    0 < total ≤ length(a) || error("Invalid total number")
    k ≤ total || error("Cannot draw more samples without replacement.")
    if k == 1
        @inbounds x[1] = sample!(a, total)
    elseif k == 2
        @inbounds (x[1], x[2]) = samplepair!(a, total)
    elseif total < 24k
        fisher_yates_sample!(a, x, total)
    else
        self_avoid_sample!(a, x, total)
    end
    return x
end

sample(a::AbstractVector{Bool}, n::Integer, total::Integer=sum(a, init=0)) = sample!(a, Vector{Int}(undef, n), total)

function sample!(a::AbstractVector{Bool}, total::Integer)
    idx = rand(1:total)
    for (i, aᵢ) in enumerate(a)
        if aᵢ
            if !isone(idx)
                idx -= 1
            else
                @inbounds a[i] = false
                return i
            end
        end
    end
    error("No valid sample vector")
end

function samplepair!(a::AbstractVector{Bool}, total::Integer)
    idx1 = rand(1:total)
    idx2 = rand(1:total-1)
    if idx1 == idx2
        idx2 = total
    elseif idx1 > idx2
        idx1, idx2 = idx2, idx1
    end
    idx2 -= idx1
    i₁ = 0
    for (i, aᵢ) in enumerate(a)
        if aᵢ
            if !isone(idx1)
                idx1 -= 1
            elseif !iszero(i₁)
                if !isone(idx2)
                    idx2 -= 1
                else
                    @inbounds a[i] = false
                    return i₁, i
                end
            else
                @inbounds a[i] = false
                i₁ = i
            end
        end
    end
    error("No valid sample vector")
end

function fisher_yates_sample!(a::AbstractVector{Bool}, x::AbstractVector{<:Integer}, total::Integer)
    0 < total ≤ length(a) || error("Invalid total number")
    k = length(x)
    k ≤ total || error("length(x) should not exceed total")
    inds = let inds=FastVec{Int}(buffer=total)
        for (i, aᵢ) in enumerate(a)
            aᵢ && unsafe_push!(inds, i)
        end
        finish!(inds)
    end
    @inbounds for i in 1:k
        j = rand(i:total)
        t = inds[j]
        inds[j] = inds[i]
        inds[i] = t
        x[i] = t
        a[t] = false
    end
    return x
end

function self_avoid_sample!(a::AbstractVector{Bool}, x::AbstractVector{<:Integer}, total::Integer)
    0 < total ≤ length(a) || error("Invalid total number")
    k = length(x)
    k ≤ total || error("length(x) should not exceed total")
    curidx = findlast(a)::eltype(eachindex(a))
    currelidx = total
    @inbounds for i in 1:k
        newidx = rand(1:total)
        # newidx is the relative index; search from the current position until we find it
        while newidx < currelidx
            curidx -= 1
            if a[curidx]
                currelidx -= 1
            end
        end
        while newidx > currelidx
            curidx += 1
            if a[curidx]
                currelidx += 1
            end
        end
        # then set the output
        x[i] = curidx
        a[curidx] = false
        total -= 1
        iszero(total) && break
        # and make sure that currelidx indeed corresponds to curidx: for this, we must increase curidx unless we are at the
        # last item.
        if currelidx != total
            while !a[curidx]
                curidx += 1
            end
        else
            while !a[curidx]
                curidx -= 1
            end
            currelidx -= 1
        end
    end
    return x
end
#endregion

function newton_polytope_preproc_randomized_taskfun(V, coeffs, nv, subset_size, required_coeffs, subset, done,
    event, stop; parameters...)
    @inbounds while !stop[]
        dropped = 0
        for (cfi, rem) in zip(subset, newton_polytope_preproc_remove(V, nv, subset_size, i -> @view(coeffs[:, subset[i]]),
                                                                     false, true; parameters...))
            required_coeffs[cfi] = rem
            if !rem
                dropped += 1
            end
        end
        put!(done, dropped)
        stop[] && return
        wait(event)
    end
end

function newton_polytope_preproc_randomized(V, coeffs, verbose; parameters...)
    nv, nc = size(coeffs)
    nthreads = Threads.nthreads()
    subset_size = min(1000, nc ÷ 20) # samples too small -> no chance of success; samples too large -> takes too long
    @assert(subset_size ≥ 1)
    required_coeffs = fill(true, nc)
    if nthreads * subset_size > nc
        nthreads = nc ÷ subset_size # we could squeeze in another incomplete thread, but this would make the code more
                                    # complicated than necessary without a real benefit
    end
    if isone(nthreads)
        let
            _subset = sample(required_coeffs, subset_size, nc)
            @inbounds while true
                totaldrop = 0
                for (cfi, rem) in zip(_subset, newton_polytope_preproc_remove(V, nv, subset_size,
                                                                              i -> @view(coeffs[:, _subset[i]]), false, false);
                                                                              parameters...)
                    required_coeffs[cfi] = rem
                    if !rem
                        totaldrop += 1
                    end
                end
                nc -= totaldrop
                if verbose
                    print("\33[2KStatus update: ", nc, " remaining extremal points, last drop was ", totaldrop, "\r")
                    flush(stdout)
                end
                (totaldrop < 20 || 10totaldrop < subset_size) && break
                subset_size > nc && break
                sample!(required_coeffs, _subset, nc)
            end
        end
    else
        ccall(:jl_enter_threaded_region, Cvoid, ())
        try
            # We divide our variables in randomized, disjoint (in the case of multi-threading) subsets and try to find
            # linear dependencies in them. Depending on our success rate, we'll try to increase the size of the subsets.
            allsubs = sample(required_coeffs, nthreads * subset_size, nc)
            done = Base.Channel{Int}(1)
            events = [Base.Event(true) for _ in 1:nthreads]
            stop = Ref(false)
            # Initialize all threads; as everything is already set up, they will directly start
            threads = Vector{Task}(undef, nthreads)
            for (tid, start) in enumerate(1:subset_size:subset_size*nthreads)
                @inbounds threads[tid] = Threads.@spawn newton_polytope_preproc_randomized_taskfun($V, $coeffs, $nv,
                   $subset_size, $required_coeffs, $(@view(allsubs[start:start+subset_size-1])), $done, $(events[tid]),
                   $stop; $parameters...)
            end
            while true
                # We wait for all threads to finish - if the have a sufficiently good coverage of the whole set with
                # all available threads, it would make no sense to directly re-start a thread, it would just get the
                # same set again. Instead, when all threads are finished (which happens approximately at the same
                # time), we can rehash the full set.
                totaldrop = 0
                for _ in 1:nthreads
                    totaldrop += take!(done)
                end
                nc -= totaldrop
                if verbose
                    print("\33[2KStatus update: ", nc, " remaining extremal points, last drop was ", totaldrop, "\r")
                    flush(stdout)
                end
                totalsize = nthreads * subset_size
                (totaldrop < 20 || 10totaldrop < totalsize) && break
                if totalsize > nc
                    nthreads = nc ÷ subset_size
                    iszero(nthreads) && break
                    stop[] = true
                    for event in events
                        notify(event)
                    end
                    totalsize = nthreads * subset_size
                end
                subs = @view(allsubs[1:totalsize])
                sample!(required_coeffs, subs, nc)
                for event in @view(events[1:nthreads])
                    notify(event) # don't start those beyond nthreads
                end
            end
            stop[] = true
            for event in events
                notify(event) # to properly finish all tasks
            end
        finally
            ccall(:jl_exit_threaded_region, Cvoid, ())
        end
    end
    return required_coeffs
end

function newton_polytope_preproc_merge_constraints_postproc(T, nv, mons_idx_set, degree, dense::Bool)
    mons_idx = sort!(collect(mons_idx_set))
    next_col = 1
    max_col = length(mons_idx)
    iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), T(degree), zeros(T, nv), fill(T(degree), nv), true)
    if dense
        coeffs = resizable_array(T, nv, max_col)
        @inbounds for (idx, powers) in enumerate(iter)
            if idx == mons_idx[next_col]
                copyto!(@view(coeffs[:, next_col]), powers)
                next_col += 1
                next_col > max_col && break
            end
        end
        @assert(next_col == max_col +1)
        return coeffs
    else
        colptr = resizable_array(UInt, max_col +1)
        rowval = FastVec{UInt}()
        nzval = FastVec{T}()
        @inbounds for (idx, powers) in enumerate(iter)
            if idx == mons_idx[next_col]
                colptr[next_col] = length(rowval) +1
                for (row, val) in enumerate(powers)
                    if !iszero(val)
                        push!(rowval, row)
                        push!(nzval, val)
                    end
                end
                next_col += 1
                next_col > max_col && break
            end
        end
        @assert(next_col == max_col +1)
        colptr[next_col] = length(rowval) +1
        return SparseMatrixCSC{T,UInt}(nv, max_col, colptr, finish!(rowval), finish!(nzval))
    end
end

function newton_polytope_preproc_merge_constraints(degree, indextype, objective::SimpleRealPolynomial, zero, nonneg, psd, dense::Bool)
    nv = nvariables(objective)
    T = SimplePolynomials.smallest_unsigned(2degree)
    mons_idx_set = sizehint!(Set{indextype}(), length(objective))
    # we start by storing the indices of the monomials only, which is the most efficient way for eliminating duplicates
    # afterwards
    for mon in monomials(objective)
        push!(mons_idx_set, monomial_index(mon))
    end
    # If there are constraints present, things are not so simple. We assume a Putinar certificate:
    # f ≥ 0 on {zero == 0, nonneg ≥ 0, psd ⪰ 0} ⇐ f = σ₀ + ∑ᵢ nonnegᵢ σᵢ + ∑ⱼ ⟨psdⱼ, Mⱼ⟩ + ∑ₖ zeroₖ pₖ
    #                                              where σ₀, σᵢ ∈ SOS, Mⱼ ∈ SOSmatrix, pₖ ∈ poly
    # This can simply be reformulated into f - ∑ᵢ nonnegᵢ σᵢ - ∑ⱼ ⟨psdⱼ, Mⱼ⟩ - ∑ₖ zeroₖ pₖ ∈ SOS, i.e., we can now apply
    # Newton methods to the polynomial with subtracted constraint certifiers. The variable degree influences how large
    # the σᵢ, Mⱼ, and pₖ will maximally be.
    # Note that since the coefficients of the σᵢ, Mⱼ, and pₖ are unknowns, we don't need to ask ourselves whether some
    # cancellation may occur - we don't know. So every monomial that is present in any of the constraints, multiplied
    # by any monomial of allowed degree for the multiplier, will give rise to an additional entry in the coeffs array.
    # This can quickly become disastrous if `degree` is high but the degree of the constraints is low (as then, the
    # prefactors have lots of entries), but it is not so harmful in the other regime.
    minmultideg = zeros(T, nv)
    maxmultideg = similar(minmultideg)
    powers₁ = similar(minmultideg)
    powers₂ = similar(minmultideg)
    monomial₁ = SimpleMonomial{nv,0,T,typeof(powers₁)}(powers₁, SimplePolynomials.absent, SimplePolynomials.absent)
    monomial₂ = SimpleMonomial{nv,0,T,typeof(powers₂)}(powers₂, SimplePolynomials.absent, SimplePolynomials.absent)

    # 1. zero constraints
    for zeroₖ in zero
        maxdeg = T(2(degree - maxhalfdegree(zeroₖ)))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        sizehint!(mons_idx_set, length(mons_idx_set) + length(iter) * length(zeroₖ))
        for t in zeroₖ
            monₜ = monomial(t)
            for _ in iter
                push!(mons_idx_set, monomial_index(monomial₁, monₜ))
            end
        end
    end

    # 2. nonneg constraints
    # Note that we can still exploit that the σⱼ must be sums of squares: despite being polynomials of degree deg(σⱼ)
    # with unknown coefficients, _some_ of these must for sure be zero, namely those that cannot be reached by
    # combining any two of the possible coefficients that are in the valid multidegree range of σⱼ.
    for nonnegᵢ in nonneg
        maxdeg = T(degree - maxhalfdegree(nonnegᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        len = length(iter)
        sizehint!(mons_idx_set, length(mons_idx_set) + (len * (len +1) ÷ 2) * length(nonnegᵢ))
        for t in nonnegᵢ
            monₜ = monomial(t)
            for _ in iter
                # no need to run over duplicates
                push!(mons_idx_set, monomial_index(monomial₁, conj(monomial₁), monₜ))
                @inbounds copyto!(powers₂, powers₁)
                for _ in InitialStateIterator(MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg,
                                                                                 maxmultideg, powers₂),
                                              moniter_state(powers₂))
                    push!(mons_idx_set, monomial_index(monomial₁, monomial₂, monₜ))
                end
            end
        end
    end

    # 3. psd constraints
    # Those are modeled in terms of SOS matrices. Given the basis u of `degree`, an m×m-matrix M is a SOS matrix iff
    # M(x) = (u ⊗ 1ₘ)ᵀ Z (u ⊗ 1ₘ) with Z ⪰ 0. Still, there is no duplication of the Z-coefficients (apart from
    # symmetry), so there is no way in which these unknown coefficients could potentially cancel: every entry in Z will
    # appear in exactly one cell in a triangle of M. So basically, all that we have to do is to apply the SOS
    # decomposition cell-wise.
    for psdᵢ in psd
        dim = LinearAlgebra.checksquare(psdᵢ)
        maxdeg = T(degree - maxhalfdegree(psdᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        len = length(iter)
        sizehint!(mons_idx_set, length(mons_idx_set) + (len * (len +1) ÷ 2) *
                                sum(@capture(length($psdᵢ[i, j]) for j in 1:dim for i in 1:j), init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for _ in iter
                    # no need to run over duplicates
                    push!(mons_idx_set, monomial_index(monomial₁, conj(monomial₁), monₜ))
                    @inbounds copyto!(powers₂, powers₁)
                    for _ in InitialStateIterator(MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg,
                                                                                     maxmultideg, powers₂),
                                                  moniter_state(powers₂))
                        push!(mons_idx_set, monomial_index(monomial₁, monomial₂, monₜ))
                    end
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    return newton_polytope_preproc_merge_constraints_postproc(T, nv, mons_idx_set, 2degree, dense)
end

_realify(m::SimpleComplexMonomial{Nc,P,V}) where {Nc,P<:Unsigned,V<:AbstractVector{P}} =
    SimpleMonomial{Nc,0,P,V}(m.exponents_complex, SimplePolynomials.absent, SimplePolynomials.absent)

function newton_polytope_preproc_merge_constraints(degree, indextype, objective::SimpleComplexPolynomial, zero, nonneg, psd, dense::Bool)
    # Note that in the complex-valued case there's no mixing - i.e., no real variables. And every monomial appears once in its
    # original form, once in its conjugate. We are only interested in the "original" (whichever it is), so we effectively treat
    # every monomial as real, discarding the conjugate part. (It would be possible to do it differently, but then in the end
    # when we reduce everything to the complex part dropping the conjugates, deleting duplicates would be necessary - in this
    # way, we don't even generate duplicates.)

    nv = nvariables(objective) ÷ 2 # don't double Nc
    T = SimplePolynomials.smallest_unsigned(2degree)
    mons_idx_set = sizehint!(Set{indextype}(), length(objective))

    for mon in monomials(objective)
        push!(mons_idx_set, monomial_index(_realify(mon)))
    end

    minmultideg = zeros(T, nv)
    maxmultideg = similar(minmultideg)
    powers₁ = similar(minmultideg)
    monomial₁ = SimpleMonomial{nv,0,T,typeof(powers₁)}(powers₁, SimplePolynomials.absent, SimplePolynomials.absent)

    # 1. zero constraints
    # 2. psd constraints
    for constraints in (zero, nonneg)
        for constrₖ in constraints
            maxdeg = T(degree - maxhalfdegree(constrₖ))
            fill!(maxmultideg, maxdeg)
            iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
            sizehint!(mons_idx_set, length(mons_idx_set) + 2length(iter) * length(constrₖ))
            for t in constrₖ
                monₜ = monomial(t)
                for _ in iter
                    push!(mons_idx_set,
                        monomial_index(monomial₁, _realify(monₜ)),
                        monomial_index(monomial₁, _realify(conj(monₜ)))
                    )
                end
            end
        end
    end

    # 3. psd constraints
    for psdᵢ in psd
        dim = LinearAlgebra.checksquare(psdᵢ)
        maxdeg = T(degree - maxhalfdegree(psdᵢ))
        fill!(maxmultideg, maxdeg)
        iter = MonomialIterator{Graded{LexOrder}}(Base.zero(T), maxdeg, minmultideg, maxmultideg, powers₁)
        sizehint!(mons_idx_set, length(mons_idx_set) + 2length(iter) * sum(length, psdᵢ, init=0))
        @inbounds for j in 1:dim, i in 1:j
            for t in psdᵢ[i, j]
                monₜ = monomial(t)
                for _ in iter
                    push!(mons_idx_set, monomial_index(monomial₁, _realify(monₜ)))
                    i == j || push!(mons_idx_set, monomial_index(monomial₁, _realify(conj(monₜ))))
                end
            end
        end
    end

    # now we need to re-cast the indices into the exponent-representations
    return newton_polytope_preproc_merge_constraints_postproc(T, nv, mons_idx_set, degree, dense)
end

function newton_polytope_preproc(V, objective::P; verbose::Bool=false, zero::AbstractVector{P}, nonneg::AbstractVector{P},
    psd::AbstractVector{<:AbstractMatrix{P}}, degree::Int, preprocess::Union{Nothing,Bool}=nothing,
    preprocess_quick::Bool=true, preprocess_randomized::Bool=false, preprocess_fine::Bool=false,
    warn_disable_randomization::Bool=true, parameters...) where {P<:SimpleRealPolynomial}
    if !isnothing(preprocess)
        preprocess_quick = preprocess_randomized = preprocess_fine = preprocess
    end
    @verbose_info("Determining Newton polytope (quick preprocessing: ", preprocess_quick, ", randomized preprocessing: ",
        preprocess_randomized, ", fine preprocessing: ", preprocess_fine, ")")
    nv = nvariables(objective)
    nc = length(objective)
    if !isempty(zero) || !isempty(nonneg) || !isempty(psd)
        coeffs = newton_polytope_preproc_merge_constraints(
            degree, SimplePolynomials.smallest_unsigned(monomial_count(2degree, nv)), objective, zero, nonneg, psd,
            monomials(objective).exponents_real isa DenseMatrix
        )
    else
        # shortcut, no need to temporarily go to the index-based version. Just copy the coefficient matrix (as preprocessing
        # will make changes to it)
        let objexps=monomials(objective).exponents_real
            if VERSION < v"1.11"
                # due to preprocessing, we might want to shrink this array. In Julia, shrinking is almost unsupported - a new
                # buffer of the smaller size is allocated and things are copied over. However, in case we actually wrap a
                # malloc-allocated buffer into an array, realloc is used properly.
                if objexps isa DenseMatrix
                    coeffs = resizable_copy(objexps)
                else
                    @assert(objexps isa SparseArrays.AbstractSparseMatrixCSC)
                    colptr = SparseArrays.getcolptr(objexps)
                    rowval = rowvals(objexps)
                    nzval = nonzeros(objexps)
                    @inbounds coeffs = typeof(objexps)(size(objexps)..., resizable_copy(colptr), resizable_copy(rowval),
                        resizable_copy(nzval))
                end
            else
                coeffs = copy(objexps)
            end
        end
    end
    if preprocess_quick
        @verbose_info("Removing redundancies from the convex hull - quick heuristic, ", nc, " initial candidates")
        preproc_time = @elapsed begin
            coeffs = keepcol!(coeffs, newton_polytope_preproc_prequick(V, coeffs, verbose; parameters...))
            nc = size(coeffs, 2)
        end
        @verbose_info("Found ", nc, " potential extremal points of the convex hull in ", preproc_time, " seconds")
    end
    if preprocess_randomized
        if nc ≥ 100
            @verbose_info("Removing redundancies from the convex hull - randomized, ", nc, " initial candidates")
            preproc_time = @elapsed begin
                coeffs = keepcol!(coeffs, newton_polytope_preproc_randomized(V, coeffs, verbose; parameters...))
                nc = size(coeffs, 2)
            end
            @verbose_info("Found ", nc, " extremal points of the convex hull via randomization in ", preproc_time, " seconds")
        else
            warn_disable_randomization &&
                @info("Removing redundancies from the convex hull via randomization was requested, but skipped due to the small size of the problem")
        end
    end
    if preprocess_fine
        # eliminate all the coefficients that are redundant themselves to make the linear system smaller
        @verbose_info("Removing redundancies from the convex hull - fine, ", nc, " initial candidates")
        preproc_time = @elapsed begin
            coeffs = keepcol!(coeffs, newton_polytope_preproc_remove(V, nv, nc, @capture(i -> @inbounds(@view $coeffs[:, i])),
                verbose, false; parameters...))
            nc = size(coeffs, 2)
        end
        @verbose_info("Found ", nc, " extremal points of the convex hull in ", preproc_time, " seconds")
    end
    return parameters, coeffs
end
#endregion

"""
    newton_polytope_do_worker(V, task, data_global, data_local, moniter::MonomialIterator,
        Δprogress::Ref{Int}, Δacceptance::Ref{Int}, add_callback::Function,
        iteration_callback::Union{Nothing,Function})

Iterates through `moniter` and for every monomial checks whether this set of exponents can be reached by a convex combination
of the coefficients as they are set up in `task`. If yes, `add_callback` must be called with the exponents as a parameter, and
`Δacceptance` should be incremented. In any case, `Δprogress` should be incremented. Additionally, after every check,
`iteration_callback` should be called with the exponents of the iteration as a parameter, if it is a function.
The `data` parameters contain the custom data that was previously generated using [`newton_halfpolytope_alloc_global`](@ref)
and [`newton_halfpolytope_alloc_local`](@ref). Only `data_local` may be mutated.
"""
function newton_polytope_do_worker end

function newton_polytope_do_taskfun(V, tid, task, ranges, nv, mindeg, maxdeg, data_global, cond, progresses, acceptances,
    allcandidates, notifier, init_time, init_progress, num, filestuff)
    # notifier: 0 - no notification; 1 - the next to get becomes the notifier; 2 - notifier is taken
    verbose = notifier[] != 0
    lastappend = time_ns()
    isnotifier = Ref(false) # necessary due to the capturing/boxing bug
    lastinfo = Ref{Int}(lastappend)
    data_local = newton_halfpolytope_alloc_local(V, nv)
    candidates = FastVec{typeof(maxdeg)}()
    @inbounds progress = Ref(progresses, tid)
    @inbounds acceptance = Ref(acceptances, tid)
    local curminrange, curmaxrange, iter
    if isnothing(filestuff)
        powers = Vector{typeof(maxdeg)}(undef, nv)
        fileprogress = nothing # required for capture below
        fileout = nothing
        cut = 0
    else
        fileprogress = filestuff[1]
        fileout = filestuff[2]
        cut = filestuff[3]
        if ismissing(filestuff[4])
            powers = Vector{typeof(maxdeg)}(undef, nv)
        else
            curminrange, curmaxrange, powers = filestuff[4]
            iter = InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers),
                moniter_state(powers))
            # @goto start - must be deferred as it would jump into a try block
        end
    end
    try
        !isnothing(filestuff) && !ismissing(filestuff[4]) && @goto start
        while true
            try
                curminrange, curmaxrange = take!(ranges)
            catch e
                e isa InvalidStateException && break
                rethrow(e)
            end
            iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, curminrange, curmaxrange, powers)
            @label start
            newton_polytope_do_worker(V, task, data_global, data_local, iter, progress, acceptance,
                @capture(p -> append!($candidates, p)),
                !verbose && isnothing(filestuff) ? nothing : @capture(p -> let
                    nextinfo = time_ns()
                    if nextinfo - $lastinfo[] > 1_000_000_000
                        if !isnothing($filestuff)
                            write($fileout, $candidates)
                            flush(fileout)
                            seekstart($fileprogress)
                            write(fileprogress, $progress[], $acceptance[], p)
                            flush(fileprogress)
                            empty!(candidates)
                        end
                        if $verbose
                            if !$isnotifier[] && notifier[] == 1
                                isnotifier[] = true
                                notifier[] = 2
                            end
                            if isnotifier[]
                                allprogress = sum(progresses, init=0)
                                allacceptance = sum(acceptances, init=0)
                                Δt = allprogress == $init_progress ? 1 : allprogress - init_progress
                                # ^ if a finished job is started, this might happen
                                rem_sec = round(Int, ((nextinfo - init_time) / 1_000_000_000Δt) * (num - allprogress))
                                @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                                    100allprogress / num, 100allacceptance / allprogress, rem_sec ÷ 60, rem_sec % 60)
                                flush(stdout)
                            end
                        end
                        lastinfo[] = nextinfo
                    end
                end)
            )
            # make sure that we update the main list regularly, but not ridiculously often
            nextappend = time_ns()
            if nextappend - lastappend > 10_000_000_000
                if isnothing(filestuff)
                    lock(cond)
                    try
                        append!(allcandidates, candidates)
                    finally
                        unlock(cond)
                    end
                else
                    write(fileout, candidates)
                    flush(fileout)
                    seekstart(fileprogress)
                    write(fileprogress, progress[], acceptance[], @view(curminrange[cut:end]))
                    truncate(fileprogress, position(fileprogress))
                end
                empty!(candidates)
                lastappend = nextappend
            end
        end
        if isnotifier[]
            notifier[] = 1
        end
        if isnothing(filestuff)
            lock(cond)
            try
                append!(allcandidates, candidates)
            finally
                unlock(cond)
            end
        else
            write(fileout, candidates)
            seekstart(fileprogress)
            write(fileprogress, progress[], acceptance[])
            truncate(fileprogress, position(fileprogress))
        end
    finally
        finalize(task)
        if !isnothing(filestuff)
            close(fileout)
            close(fileprogress)
        end
    end
end

function monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, batchsize)
    # to calculate the part-range-sizes, we have to duplicate some code from length(::MonomialIterator), but our particular
    # application here is too special to integrate it: We start calculating the length of the iterator starting from the left
    # and as soon as we hit the batchsize boundary, we know that all the parts to the right should be done by individual
    # tasks.
    cutat = 1
    occurrences = zeros(Int, maxdeg +1)
    @inbounds for deg₁ in minmultideg[1]:min(maxmultideg[1], maxdeg)
        occurrences[deg₁+1] = 1
    end
    nextround = similar(occurrences)
    restmax = sum(@view(maxmultideg[2:end]), init=0)
    for (minᵢ, maxᵢ) in Iterators.drop(zip(minmultideg, maxmultideg), 1)
        restmax -= min(maxᵢ, maxdeg)
        fill!(nextround, 0)
        for degᵢ in minᵢ:min(maxᵢ, maxdeg)
            for (degⱼ, occⱼ) in zip(Iterators.countfrom(0), occurrences)
                newdeg = degᵢ + degⱼ
                newdeg > maxdeg && break
                @inbounds nextround[newdeg+1] += occⱼ
            end
        end
        sum(@view(nextround[max(mindeg - restmax +1, 1):end])) > batchsize && break
        occurrences, nextround = nextround, occurrences
        cutat += 1
    end
    return cutat
end

function newton_halfpolytope_analyze(coeffs)
    # This is some quick preprocessing to further restrict the potential degrees (actually, this is what SumOfSquares.jl calls
    # Newton polytope)
    nv = size(coeffs, 1)

    maxdeg, mindeg = zero(UInt), typemax(UInt)
    maxmultideg, minmultideg = fill(zero(UInt), nv), fill(typemax(UInt), nv)
    for coeff in eachcol(coeffs)
        deg = 0
        @inbounds for (i, coeffᵢ) in enumerate(coeff)
            deg += coeffᵢ
            if coeffᵢ > maxmultideg[i]
                maxmultideg[i] = coeffᵢ
            end
            if coeffᵢ < minmultideg[i]
                minmultideg[i] = coeffᵢ
            end
        end
        if deg > maxdeg
            maxdeg = deg
        end
        if deg < mindeg
            mindeg = deg
        end
    end
    maxmultideg .= div.(maxmultideg, 2, RoundDown)
    minmultideg .= div.(minmultideg, 2, RoundUp)
    maxdeg = div(maxdeg, 2, RoundDown)
    mindeg = div(mindeg, 2, RoundUp)

    # Now that we know the ranges of all the types, we can make them as tight as possible to reduce the memory footprint
    maxval = max(maxdeg, maximum(maxmultideg, init=0))
    local T
    for outer T in (UInt8, UInt16, UInt32, UInt64)
        typemax(T) ≥ maxval && break
    end
    return convert(T, mindeg), convert(T, maxdeg), convert(Vector{T}, minmultideg), convert(Vector{T}, maxmultideg)
end

"""
    newton_halfpolytope_do_prepare(V, coeffs::AbstractMatrix{<:Integer}, num::Int)

This function is responsible for creating an optimization task that can be used to check membership in the Newton halfpolytope.
The vertices of the polytope are given by the column of `coeffs`. The total number of monomials that have to be checked is
given by `num`.
The function must return the number of threads that will be used to carry out the optimization (which is not the number of
threads that the optimizer uses internally, but determines how `PolynomialOptimization` will distribute the jobs), an internal
optimization task that is passed on to [`newton_polytope_do_worker`](@ref), and a copy of this task for use in a second thread
if the number of threads is greater than one, else `nothing`. More copies will be created as required by
[`newton_halfpolytope_clonetask`](@ref); however, assuming that setting up the task will potentially require more resources
than cloning a task allows the function to estimate the required memory (and therefore a sensible number of threads) better by
already performing one clone.

See also [`@allocdiff`](@ref).
"""
function newton_halfpolytope_do_prepare end

struct InitialStateIterator{I,S,L}
    iter::I
    initial_state::S
    skip_length::L

    InitialStateIterator(iter, initial_state, skip_length::Union{<:Integer,Missing}=missing) =
        new{typeof(iter),typeof(initial_state),typeof(skip_length)}(iter, initial_state, skip_length)
end

Base.iterate(iter::InitialStateIterator, state=iter.initial_state) = iterate(iter.iter, state)
Base.IteratorSize(::Type{InitialStateIterator{I,S,Missing}}) where {I,S} = Base.SizeUnknown()
Base.IteratorSize(::Type{<:InitialStateIterator{I}}) where {I} = Base.drop_iteratorsize(Base.IteratorSize(I))
Base.IteratorEltype(::Type{<:InitialStateIterator{I}}) where {I} = Base.IteratorEltype(I)
Base.eltype(::Type{<:InitialStateIterator{I}}) where {I} = eltype(I)
Base.length(iter::InitialStateIterator{I,S,<:Integer} where {I,S}) = length(iter.iter) - iter.skip_length
Base.isdone(iter::InitialStateIterator, state=iter.initial_state) = Base.isdone(iter, state)

function newton_halfpolytope_restore_status!(fileprogress, mindeg::I, maxdeg::I, minmultideg::AbstractVector{I},
    maxmultideg::AbstractVector{I}, powers=true) where {I<:Integer}
    lastprogress = UInt8[]
    seekstart(fileprogress)
    nb = readbytes!(fileprogress, lastprogress, 2sizeof(Int) + sizeof(minmultideg))
    GC.@preserve lastprogress begin
        lpp = Ptr{Int}(pointer(lastprogress))
        if iszero(nb)
            return 0, 0, MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, powers)
        elseif nb == 2sizeof(Int)
            return unsafe_load(lpp), unsafe_load(lpp, 2), nothing
        elseif nb == 2sizeof(Int) + sizeof(minmultideg)
            progress = unsafe_load(lpp)
            if powers === true
                powers = similar(minmultideg)
            end
            unsafe_copyto!(pointer(powers), Ptr{eltype(powers)}(lpp + 2sizeof(Int)), length(minmultideg))
            return progress, unsafe_load(lpp, 2), InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg,
                minmultideg, maxmultideg, powers), moniter_state(powers), progress)
        else
            error("Unknown progress file format - please delete existing files.")
        end
    end
end

function newton_halfpolytope_restore_status!(fileprogress, powers::Vector{<:Integer}, fixedsize::Integer)
    lastprogress = UInt8[]
    seekstart(fileprogress)
    T, len = eltype(powers), length(powers)
    s = 2sizeof(Int) + sizeof(powers)
    nb = readbytes!(fileprogress, lastprogress, s)
    GC.@preserve lastprogress begin
        lpp = Ptr{Int}(pointer(lastprogress))
        if iszero(nb)
            return nothing
        elseif nb == 2sizeof(Int)
            return unsafe_load(lpp), unsafe_load(lpp, 2), nothing, nothing
        elseif nb == 2sizeof(Int) + sizeof(T) * fixedsize
            fixed = similar(powers, fixedsize)
            unsafe_copyto!(pointer(fixed), Ptr{T}(lpp + 2sizeof(Int)), fixedsize)
            return unsafe_load(lpp), unsafe_load(lpp, 2), fixed, nothing
        elseif nb == s
            unsafe_copyto!(pointer(powers), Ptr{T}(lpp + 2sizeof(Int)), len)
            return unsafe_load(lpp), unsafe_load(lpp, 2), powers[end-fixedsize+1:end], powers
        else
            error("Unknown progress file format - please delete existing files.")
        end
    end
end

toSigned(x::UInt8) = Core.bitcast(Int8, x)
toSigned(x::UInt16) = Core.bitcast(Int16, x)
toSigned(x::UInt32) = Core.bitcast(Int32, x)
toSigned(x::UInt64) = Core.bitcast(Int64, x)
toSigned(x::Signed) = x

"""
    newton_halfpolytope_alloc_global(V, nv)

This function is called once in the main thread before [`newton_polytope_do_worker`](@ref) is executed (which, due to
multithreading, might occur more than once). It can create some shared data that is used in a read-only manner by all workers
at the same time.
The default implementation of this function does nothing.
"""
newton_halfpolytope_alloc_global(_, _) = nothing
"""
    newton_halfpolytope_alloc_local(V, nv)

This function is called once in every computation thread before [`newton_polytope_do_worker`](@ref) is executed (which, due to
task splitting, might occur more than once). It can create some shared data that is available for reading and writing by every
worker.
The default implementation of this function does nothing.
"""
newton_halfpolytope_alloc_local(_, _) = nothing

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath)
    @verbose_info("Preparing to determine Newton polytope (single-threaded)")

    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{typeof(maxdeg)}()

    progress = Ref(0)
    acceptance = Ref(0)
    if isnothing(filepath)
        fileprogress = nothing
        fileout = nothing
        iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg, maxmultideg, true)
    else
        fileprogress = open("$filepath.prog", read=true, write=true, create=true, lock=false)
        local iter
        try
            progress[], acceptance[], iter = newton_halfpolytope_restore_status!(fileprogress, mindeg, maxdeg, minmultideg,
                maxmultideg)
        catch
            close(fileprogress)
            rethrow()
        end
        if isnothing(iter)
            close(fileprogress)
            return progress[], acceptance[]
        end
        fileout = open("$filepath.out", append=true, lock=false)
    end
    try
        init_time = time_ns()
        init_progress = progress[]
        lastinfo = Ref(init_time)
        newton_polytope_do_worker(V, task, newton_halfpolytope_alloc_global(V, nv), newton_halfpolytope_alloc_local(V, nv),
            iter, progress, acceptance, @capture(p -> append!($candidates, p)),
            !verbose && isnothing(filepath) ? nothing : @capture(p -> let
                nextinfo = time_ns()
                if nextinfo - $lastinfo[] > 1_000_000_000
                    if !isnothing($filepath)
                        write($fileout, $candidates)
                        flush(fileout)
                        seekstart($fileprogress)
                        write(fileprogress, $progress[], $acceptance[], p)
                        flush(fileprogress)
                        empty!(candidates)
                    end
                    if $verbose
                        Δt = progress[] == $init_progress ? 1 : progress[] - init_progress
                        # ^ if a finished job is started, this might happen
                        rem_sec = round(Int, ((nextinfo - $init_time) / 1_000_000_000Δt) * (num - progress[]))
                        @printf("\33[2KStatus update: %.2f%%, acceptance: %.2f%%, remaining time: %02d:%02dmin\r",
                            100progress[] / num, 100 * acceptance[] / progress[], rem_sec ÷ 60, rem_sec % 60)
                        flush(stdout)
                    end
                    lastinfo[] = nextinfo
                end
            end)
        )
        verbose && print("\33[2K")
        # How about the order of monomials? Currently, the monomial order can be defined to be arbitrary, but monomials
        # in DynamicPolynomials always outputs Graded{LexOrder}. Therefore, it is currently not possible to use
        # different ordering in DP, and this translates to PolynomialOptimization, unless the user chooses to generate
        # the monomials manually.
        # Here, we can relatively easily make the basis compliant with the specified monomial ordering just by doing a
        # sorting as a postprocessing. Of course, this is not efficient and it would be much better to create functions
        # that directly generate the appropriate order, but let's defer this at least until #138 in DP is solved, for
        # other monomial orderings won't be in widespread use before this anyway.

        # sort!(candidates, lt=(a, b) -> compare(a, b, monomial_ordering(P)) < 0)
        # TODO: There is no monomial_ordering, so we cannot even do this
        if !isnothing(fileout)
            write(fileout, candidates)
            seekstart(fileprogress)
            write(fileprogress, progress[], acceptance[])
            truncate(fileprogress, position(fileprogress))
        end
    finally
        finalize(task)
        if isnothing(fileout)
            return SimpleMonomialVector{nv,0}(reshape(finish!(candidates), nv, length(candidates)÷nv))
        else
            close(fileout)
            close(fileprogress)
            return progress[], acceptance[]
        end
    end
end

"""
    newton_halfpolytope_clonetask(t)

This function must create a copy of the optimization task `t` that can run in parallel to `t` in a different thread.
"""
function newton_halfpolytope_clonetask end

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num,
    nthreads::Integer, task, secondtask, filepath)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath)
    end

    threadsize = div(num, nthreads, RoundUp)
    @verbose_info("Preparing to determine Newton polytope using ", nthreads, " threads, each checking about ", threadsize,
        " candidates")
    cutat = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
    cutlen = nv - cutat
    cutat += 1 # cutat is now the first entry to be fixed
    data_global = newton_halfpolytope_alloc_global(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    candidates = FastVec{typeof(maxdeg)}()

    ranges = Base.Channel{NTuple{2,Vector{typeof(maxdeg)}}}(typemax(Int))
    threadprogress = zeros(Int, nthreads)
    threadacceptance = zeros(Int, nthreads)
    if isnothing(filepath)
        iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end], maxmultideg[cutat:end],
            @view(minmultideg[cutat:end]))
    else
        fileprogresses = Vector{IOStream}(undef, nthreads)
        fileouts = Vector{IOStream}(undef, nthreads)
        restores = Vector{Union{NTuple{3,Vector{typeof(maxdeg)}},Missing,Nothing}}(undef, nthreads)
        maxitr = missing
        @inbounds try
            for i in 1:nthreads
                fileprogresses[i] = fileprogress = open("$filepath-$i.prog", read=true, write=true, create=true, lock=false)
                fileouts[i] = open("$filepath-$i.out", append=true, lock=false)
                curpower = Vector{typeof(maxdeg)}(undef, nv)
                currestore = newton_halfpolytope_restore_status!(fileprogress, curpower, nv - cutat +1)
                if isnothing(currestore)
                    restores[i] = missing
                else
                    threadprogress[i], threadacceptance[i], curitr, curpower_ = currestore
                    if isnothing(curitr)
                        restores[i] = nothing
                    else
                        if ismissing(maxitr) || toSigned(compare(maxitr, curitr, Graded{LexOrder})) < 0
                            maxitr = curitr
                        end
                        if isnothing(curpower_)
                            restores[i] = missing
                        else
                            curminrange, curmaxrange = copy(minmultideg), copy(maxmultideg)
                            copyto!(curminrange, cutat, curitr, 1, cutlen)
                            copyto!(curmaxrange, cutat, curitr, 1, cutlen)
                            restores[i] = (curminrange, curmaxrange, curpower)
                        end
                    end
                end
            end
        catch
            for i in 1:nthreads
                isassigned(fileprogresses, i) || break
                close(fileprogresses[i])
                isassigned(fileouts, i) && close(fileouts[i])
            end
            rethrow()
        end
        if ismissing(maxitr)
            iter = MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end], maxmultideg[cutat:end],
                @view(minmultideg[cutat:end]))
        else
            iter = InitialStateIterator(MonomialIterator{Graded{LexOrder}}(mindeg, maxdeg, minmultideg[cutat:end],
                maxmultideg[cutat:end], @view(minmultideg[cutat:end])),
                moniter_state(copyto!(@view(minmultideg[cutat:end]), maxitr)))
        end
    end
    cond = Threads.SpinLock()

    ccall(:jl_enter_threaded_region, Cvoid, ())
    try
        # To avoid naming confusion with Mosek's Task, we call the parallel Julia tasks threads.
        threads = Vector{Union{Task,Nothing}}(undef, nthreads)
        # We can already start all the tasks; this main task that must still feed the data will continue running until we
        # yield to the scheduler.
        init_time = time_ns()
        init_progress = sum(threadprogress, init=0)
        notifier = Ref(verbose ? 1 : 0)
        @inbounds for (tid, taskₜ) in Iterators.flatten((zip(nthreads:-1:3, Iterators.map(newton_halfpolytope_clonetask,
                                                                                         Iterators.repeated(task))),
                                                        ((2, secondtask), (1, task))))
            if isnothing(filepath)
                filestuff = nothing
            elseif isnothing(restores[tid])
                threads[tid] = nothing
                close(fileouts[tid])
                close(fileprogresses[tid])
                continue
            else
                filestuff = (fileprogresses[tid], fileouts[tid], cutat, restores[tid])
            end
            # secondtask has a solution, so we just use task (better than deletesolution).
            # We must create the copy in the main thread; Mosek will crash occasionally if the copies are created in
            # parallel, even if we make sure not to modify the base task until all copies are done.
            threads[tid] = Threads.@spawn newton_polytope_do_taskfun($V, $tid, $taskₜ, $ranges, $nv, $mindeg, $maxdeg,
                $data_global, $cond, $threadprogress, $threadacceptance, $candidates, $notifier, $init_time, $init_progress,
                $num, $filestuff)
        end
        # All tasks are created and waiting for stuff to do. So let's now feed them with their jobs.

        for subtask in iter
            copyto!(maxmultideg, cutat, subtask, 1, cutlen) # minmultideg is already set appropriately due to the
                                                            # @view trickery
            put!(ranges, (copy(minmultideg), copy(maxmultideg)))
        end
        close(ranges)
        for thread in threads
            isnothing(thread) || wait(thread)
        end
    finally
        ccall(:jl_exit_threaded_region, Cvoid, ())
    end
    @verbose_info("\33[2KAll tasks have finished, sorting the output")
    # We need to return the deglex monomial order, but due to the partitioning and multithreading, our output
    # is unordered (not completely, we have ordered of varying length, but does this help?).
    if isnothing(filepath)
        return SimpleMonomialVector{nv,0}(sortslices(reshape(finish!(candidates), nv, length(candidates)÷nv), dims=2,
            lt=isless_degree))
    else
        return sum(threadprogress, init=0), sum(threadacceptance, init=0)
    end
end

function newton_halfpolytope(V, objective::P, ::Val{false}; verbose::Bool=false,
    filepath::Union{<:AbstractString,Nothing}=nothing,
    zero::AbstractVector{P}=P[], nonneg::AbstractVector{P}=P[], psd::AbstractVector{<:AbstractMatrix{P}}=Matrix{P}[],
    degree::Int=maxhalfdegree(objective), kwargs...) where {P<:SimpleRealPolynomial}
    parameters, coeffs = newton_polytope_preproc(V, objective; verbose, zero, nonneg, psd, degree, kwargs...)
    newton_time = @elapsed candidates = let
        analysis = newton_halfpolytope_analyze(coeffs)
        # We don't construct the monomials using monomials(). First, it's not the most efficient implementation underlying,
        # and we also don't want to create a huge list that is then filtered (what if there's no space for the huge list?).
        # However, since we implement the monomial iteration by ourselves, we must make some assumptions about the
        # variables - this is commuting only.
        num = length(MonomialIterator{Graded{LexOrder}}(analysis..., true))
        @verbose_info("Starting point selection among ", num, " possible monomials")
        nthreads, task, secondtask = newton_halfpolytope_do_prepare(V, coeffs, num, verbose; parameters...)
        newton_halfpolytope_do_execute(V, size(coeffs, 1), analysis..., verbose, num, nthreads, task, secondtask, filepath)
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

function newton_halfpolytope(::Val{:complex}, objective::P, ::Any; verbose::Bool=false, zero::AbstractVector{P}=P[],
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
            exps = newton_polytope_preproc_merge_constraints(
                degree, SimplePolynomials.smallest_unsigned(monomial_count(2degree, nv)), objective, zero, nonneg, psd,
                monomials(objective).exponents_real isa DenseMatrix
            )
        end
    end
    @verbose_info("Found ", size(exps, 2), " elements in the complex-valued \"Newton halfpolytope\" in ", newton_time,
        " seconds")
    return SimpleMonomialVector{0,nv}(exps, convert(typeof(exps), spzeros(eltype(exps), size(exps)...)))
end

const newton_methods = Symbol[]

function default_newton_method()
    isempty(newton_methods) && error("No Newton method is available. Load a solver package that provides such a method (e.g., Mosek)")
    return first(newton_methods)
end

"""
    newton_halfpolytope_from_file(filepath, objective; estimate=false, verbose=false)

Constructs the Newton polytope for the sum of squares optimization of a given objective, which is half the Newton polytope of
the objective itself. This function does not do any calculation, but instead loads the data that has been generated using
[`newton_halfpolytope`](@ref) with the given `filepath` on the given `objective`.

!!! info
    This function will not take into account the current Julia configuration, but instead lists all files that are compatible
    with the given filepath. This allows you to, e.g., create the data in a multi-node context with moderate memory
    requirements per CPU, but load it later in a single process with lots of memory available. However, this requires you not
    to have multiple files from different configurations running.

    The function ignores the `.prog` files and just assembles the output of the `.out` files, so it does not check whether the
    calculation actually finished.

!!! tip "Memory requirements"
    If the parameter `estimate` is set to `true`, the function will only analyze the size of the files and from this return an
    estimation of how many monomials the output will contain. This is an overestimation, as it might happen that the files
    contain a small number of duplicates if the calculation was interrupted and subsequently resumed (although this is not very
    likely and the result should be pretty accurate).

See also [`newton_halfpolytope`](@ref).
"""
newton_halfpolytope_from_file(filepath::AbstractString, objective::SimpleRealPolynomial; estimate::Bool=false,
    verbose::Bool=false) =
    return newton_halfpolytope_from_file(filepath, objective, estimate, verbose, T)

function newton_halfpolytope_from_file(filepath::AbstractString, objective::AbstractPolynomialLike; verbose::Bool=false,
    kwargs...)
    out = newton_halfpolytope_from_file(filepath, SimplePolynomial(objective); verbose, kwargs...)
    if out isa SimpleMonomialVector
        conv_time = @elapsed begin
            real_vars = variable_union_type(objective)[]
            complex_vars = similar(real_vars)
            for v in variables(objective)
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
        @verbose_info("Converted monomials back to a $(typeof(mv)) in $conv_time seconds")
        return mv
    else
        return out
    end
end

function newton_halfpolytope_from_file(filepath::AbstractString, objective, estimate::Bool, verbose::Bool, ::Type{T}) where {T<:Integer}
    if isfile("$filepath.out")
        matches = r"^" * basename(filepath) * r"\.out$"
        dir = dirname(realpath("$filepath.out"))
        @verbose_info("Underlying data comes from single-node, single-thread calculation")
    elseif isfile("$filepath-1.out")
        matches = r"^" * basename(filepath) * r"-\d+\.out$"
        dir = dirname(realpath("$filepath-1.out"))
        @verbose_info("Underlying data comes from single-node, multi-thread calculation")
    elseif isfile("$filepath-n0.out")
        matches = r"^" + basename(filepath) * r"-n\d+\.out$"
        dir = dirname(realpath("$filepath-n0.out"))
        @verbose_info("Underlying data comes from multi-node, single-thread calculation")
    elseif isfile("$filepath-n0-1.out")
        matches = r"^" * basename(filepath) * r"-n\d+-\d+\.out$"
        dir = dirname(realpath("$filepath-n0-1.out"))
        @verbose_info("Underlying data comes from multi-node, multi-thread calculation")
    else
        error("Could not find data corresponding to the given filepath")
    end
    # the estimation process is always first
    len = 0
    lens = Int[]
    nv = nvariables(objective)
    for fn in readdir(dir)
        if !isnothing(match(matches, fn))
            fs = filesize("$dir/$fn")
            if !iszero(mod(fs, nv * sizeof(T)))
                error("Invalid file: $fn")
            else
                len += fs ÷ (nv * sizeof(T))
                push!(lens, fs ÷ (nv * sizeof(T)))
            end
        end
    end
    estimate && return len
    @verbose_info("Upper bound to number of monomials: ", len)
    candidates = resizable_array(T, nv, len)
    readbytebuffer = unsafe_wrap(Array, Ptr{UInt8}(pointer(candidates)), len * nv * sizeof(T))
    i = 1
    load_time = @elapsed begin
        for fn in readdir(dir)
            if !isnothing(match(matches, fn))
                @verbose_info("Opening file $fn")
                local fs
                while true
                    try
                        fs = Base.Filesystem.open("$dir/$fn", Base.JL_O_RDONLY | Base.JL_O_EXCL, 0o444)
                        break
                    catch ex
                        (isa(ex, Base.IOError) && ex.code == Base.UV_EEXIST) || rethrow(ex)
                        Base.prompt("Could not open file $fn exclusively. Release the file, then hit [Enter] to retry.")
                    end
                end
                try
                    stream = BufferedStreams.BufferedInputStream(fs)
                    i += BufferedStreams.readbytes!(stream, readbytebuffer, i, length(readbytebuffer))
                    isone(i % (nv * sizeof(T))) || error("Unexpected end of file: $fn")
                finally
                    Base.Filesystem.close(fs)
                end
            end
        end
    end
    i ÷ (nv * sizeof(T)) == len || error("Calculated length does not match number of loaded monomials")
    @verbose_info("Loaded $len monomials into memory in $load_time seconds. Now sorting and removing duplicates.")
    sort_time = @elapsed begin
        finalcandidates = SimpleMonomialVector{nv,0}(Base.unique(sortslices(candidates, dims=2, lt=isless_degree); dims=2))
    end
    @verbose_info("Sorted all monomials and removed $(len - length(finalcandidates)) duplicates in $sort_time seconds")
    return finalcandidates
end

function isless_degree(x::AbstractVector, y::AbstractVector)
    dx = sum(x)
    dy = sum(y)
    if dx == dy
        return isless(x, y)
    else
        return isless(dx, dy)
    end
end

struct FakeMonomialVector{S<:SimpleMonomialVector,V,M} <: AbstractVector{M}
    data::S
    real_vars::Vector{V}
    complex_vars::Vector{V}

    function FakeMonomialVector(data::S, real_vars::Vector{V}, complex_vars::Vector{V}) where {S<:SimpleMonomialVector,V<:AbstractVariable}
        length(real_vars) + length(complex_vars) == nvariables(data) || error("Invalid monomial vector construction")
        new{S,V,monomial_type(V)}(data, real_vars, complex_vars)
    end
end

Base.length(fmv::FakeMonomialVector) = length(fmv.data)
Base.size(fmv::FakeMonomialVector) = (length(fmv.data),)
function Base.getindex(fmv::FakeMonomialVector{S,V,M} where {V,S}, x) where {M}
    mon = fmv.data[x]
    isconstant(mon) && return constant_monomial(M)
    exps = exponents(mon)
    expit = iterate(exps)
    i = 1
    havemon = false
    while !isnothing(expit)
        i > length(fmv.real_vars) && break
        expᵢ, expitdata = expit
        if !iszero(expᵢ)
            if !havemon
                @inbounds mon = fmv.real_vars[i] ^ expᵢ
                havemon = true
            else
                @inbounds mon *= fmv.real_vars[i] ^ expᵢ
            end
        end
        i += 1
        expit = iterate(exps, expitdata)
    end
    i = 1
    while !isnothing(expit)
        i > length(fmv.complex_vars) && break
        expᵢ, expitdata = expit
        if !iszero(expᵢ)
            if !havemon
                @inbounds mon = fmv.complex_vars[i] ^ expᵢ
                havemon = true
            else
                @inbounds mon *= fmv.complex_vars[i] ^ expᵢ
            end
        end
        i += 1
        expit = iterate(exps, expitdata)
    end
    i = 1
    while !isnothing(expit)
        i > length(fmv.complex_vars) && break
        expᵢ, expitdata = expit
        if !iszero(expᵢ)
            if !havemon
                @inbounds mon = conj(fmv.complex_vars[i]) ^ expᵢ
                havemon = true
            else
                @inbounds mon *= conj(fmv.complex_vars[i]) ^ expᵢ
            end
        end
        i += 1
        expit = iterate(exps, expitdata)
    end
    @assert(isnothing(expit))
    return mon
end