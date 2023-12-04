export MonomialIterator, newton_halfpolytope, newton_halfpolytope_from_file

"""
    newton_halfpolytope(method, objective; verbose=false, preprocess_quick=true, preprocess_randomized=false,
        preprocess_fine=false, preprocess=nothing, filepath=nothing, parameters...)

Calculates the Newton polytope for the sum of squares optimization of a given objective, which is half the Newton polytope of
the objective itself. This requires the availability of a linear solver. Currently, `:Mosek` is the only supported method
(which is also the default).
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
newton_halfpolytope(method::Symbol, objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_halfpolytope(Val(method), objective, Val(haveMPI[]); kwargs...)

newton_halfpolytope(objective::P; kwargs...) where {P<:AbstractPolynomialLike} =
    newton_halfpolytope(Val(:Mosek), objective, Val(haveMPI[]); kwargs...)

function newton_polytope_preproc_quick end
function newton_polytope_preproc_remove end
#region Preprocessing for the full Newton polytope convex hull
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

# Delete "columns" from a vector that is interpreted as a matrix of height inc
function keepcol!(A::Vector, m::Vector{Bool}, inc)
    d = length(A) ÷ inc
    length(m) == d || throw(BoundsError(A, m))
    i = 1
    from = 1
    @inbounds while from ≤ d && !m[from]
        from += 1
    end
    to = from
    @inbounds while to ≤ d
        if m[to]
            to += 1
        else
            len = to - from # [from, to)
            isone(from) || copyto!(A, (i -1) * inc +1, A, (from -1) * inc +1, len * inc)
            i += len
            from = to +1
            while from ≤ d && !m[from]
                from += 1
            end
            to = from
        end
    end
    if from ≤ d
        len = to - from # [from, to), as now to = d +1
        isone(from) || copyto!(A, (i -1) * inc +1, A, (from -1) * inc +1, len * inc)
        i += len
    end
    Base._deleteend!(A, (d - i +1) * inc)
    return A
end

if VERSION < v"1.11"
    # mutable struct jl_array_t
    #     data::Ptr{Cvoid}
    #     length::UInt
    #     flags::UInt16 # how:2, ndims:9, pooled:1, ptrarray:1, hasptr:1, isshared:1, isaligned:1
    #     elsize::UInt16
    #     offset::UInt32
    #     nrows::UInt
    #     maxsize::UInt
    # end

    unsafe_array_get_flags(a::AbstractArray) = unsafe_load(Ptr{UInt16}(pointer_from_objref(a)) + sizeof(Ptr) + sizeof(UInt))
    unsafe_array_set_flags!(a::AbstractArray, flags::UInt16) = unsafe_store!(Ptr{UInt16}(pointer_from_objref(a)) +
        sizeof(Ptr) + sizeof(UInt), flags)

    # Reshape a vector according to the reshaping expression, then execute expr, and make sure that it is unshared again
    # afterwards
    # Two forms:
    # - @reshape_temp(myreshaped = reshape(myvec, dims...), expr that uses myreshaped)
    # - @reshape_temp(reshape(myarr, dim...), expr that uses myarr as if it were now reshaped)
    macro reshape_temp(reshaping, expr)
        if reshaping.head === :call && length(reshaping.args) ≥ 3 && reshaping.args[1] === :reshape &&
            reshaping.args[2] isa Symbol
            original = reshaping.args[2]
            target = original
            reshaping_expr = reshaping
        else
            (reshaping.head === :(=) && length(reshaping.args) === 2 && reshaping.args[1] isa Symbol &&
                reshaping.args[2] isa Expr && reshaping.args[2].head === :call && length(reshaping.args[2].args) ≥ 3 &&
                reshaping.args[2].args[1] === :reshape && reshaping.args[2].args[2] isa Symbol) ||
                error("@reshape_temp expect a reshape or reshape-assignment as first parameter")
            original = reshaping.args[2].args[2]
            target = reshaping.args[1]
            reshaping_expr = reshaping.args[2]
        end
        quote
            iszero(unsafe_array_get_flags($(esc(original))) & 0b0_1_0_0_0_000000000_00) ||
                error("@reshape_temp requires unshared vector in the beginning")
            let
                local result
                let $(esc(target))=$(esc(reshaping_expr))
                    result=$(esc(expr))
                end
                unsafe_array_set_flags!($(esc(original)), unsafe_array_get_flags($(esc(original))) & 0b1_0_1_1_1_111111111_11)
                result
            end
        end
    end
else
    macro reshape_temp(reshaping, expr)
        if reshaping.head === :call && length(reshaping.args) ≥ 3 && reshaping.args[1] === :reshape &&
            reshaping.args[2] isa Symbol
            target = reshaping.args[2]
            reshaping_expr = reshaping
        else
            (reshaping.head === :(=) && length(reshaping.args) === 2 && reshaping.args[1] isa Symbol &&
                reshaping.args[2] isa Expr && reshaping.args[2].head === :call && length(reshaping.args[2].args) ≥ 3 &&
                reshaping.args[2].args[1] === :reshape && reshaping.args[2].args[2] isa Symbol) ||
                error("@reshape_temp expect a reshape or reshape-assignment as first parameter")
            target = reshaping.args[1]
            reshaping_expr = reshaping.args[2]
        end
        quote
            let $(esc(target))=$(esc(reshaping_expr))
                esc($expr)
            end
        end
    end
end

function newton_polytope_preproc(V, objective::P; verbose::Bool=false, preprocess_quick::Bool=true,
    preprocess_randomized::Bool=false, preprocess_fine::Bool=false, preprocess::Union{Nothing,Bool}=nothing,
    warn_disable_randomization::Bool=true, parameters...) where {P<:AbstractPolynomialLike}
    if !isnothing(preprocess)
        preprocess_quick = preprocess_randomized = preprocess_fine = preprocess
    end
    @verbose_info("Determining Newton polytope (quick preprocessing: ", preprocess_quick, ", randomized preprocessing: ",
        preprocess_randomized, ", fine preprocessing: ", preprocess_fine, ")")
    nv = nvariables(objective)
    nc = length(objective)
    coeffs = Vector{Int}(undef, nv * nc)
    i = 1
    for mon in monomials(objective)
        @inbounds copyto!(coeffs, i, exponents(mon), 1, nv)
        i += nv
    end
    if preprocess_quick
        @verbose_info("Removing redundancies from the convex hull - quick heuristic, ", nc, " initial candidates")
        preproc_time = @elapsed keepcol!(coeffs,
            @reshape_temp(reshape(coeffs, nv, nc), newton_polytope_preproc_quick(V, coeffs, verbose; parameters...)), nv)
        nc = length(coeffs) ÷ nv
        @verbose_info("Found ", nc, " potential extremal points of the convex hull in ", preproc_time, " seconds")
    end
    if preprocess_randomized
        if nc ≥ 100
            @verbose_info("Removing redundancies from the convex hull - randomized, ", nc, " initial candidates")
            preproc_time = @elapsed keepcol!(coeffs,
                @reshape_temp(reshape(coeffs, nv, nc), newton_polytope_preproc_randomized(V, coeffs, verbose; parameters...)),
                nv)
            nc = length(coeffs) ÷ nv
            @verbose_info("Found ", nc, " extremal points of the convex hull via randomization in ", preproc_time, " seconds")
        else
            warn_disable_randomization &&
                @info("Removing redundancies from the convex hull via randomization was requested, but skipped due to the small size of the problem")
        end
    end
    if preprocess_fine
        # eliminate all the coefficients that are redundant themselves to make the linear system smaller
        @verbose_info("Removing redundancies from the convex hull - fine, ", nc, " initial candidates")
        preproc_time = @elapsed keepcol!(coeffs,
            @reshape_temp(reshape(coeffs, nv, nc), newton_polytope_preproc_remove(V, nv, nc,
                i -> @inbounds(@view coeffs[:, i]), verbose, false; parameters...)), nv)
        nc = length(coeffs) ÷ nv
        @verbose_info("Found ", nc, " extremal points of the convex hull in ", preproc_time, " seconds")
    end
    return parameters, reshape(coeffs, nv, nc)
end
#endregion

#region Monomial iterator
"""
    MonomialIterator{O}(mindeg, maxdeg, minmultideg, maxmultideg, ownpowers=false)

This is an advanced iterator that is able to iterate through all monomials with constraints specified not only by a minimum and
maximum total degree, but also by individual variable degrees. `ownpowers` can be set to `true` (or be passed a
`Vector{<:Integer}` of appropriate length), which will make the iterator use the same vector of powers whenever it is used, so
it must not be used multiple times simultaneously. Additionally, during iteration, no copy is created, so the vector must not
be modified and accumulation e.g. by `collect` won't work.
Note that the powers that this iterator returns will be of the common integer type of `mindeg`, `maxdeg`, and the element types
of `minmultideg`, `maxmultideg` (and potentially `ownpowers`).
"""
struct MonomialIterator{O<:AbstractMonomialOrdering,P,DI<:Integer}
    n::Int
    mindeg::DI
    maxdeg::DI
    minmultideg::Vector{DI}
    maxmultideg::Vector{DI}
    powers::P
    Σminmultideg::DI
    Σmaxmultideg::DI

    function MonomialIterator{O}(mindeg::DI, maxdeg::DI, minmultideg::Vector{DI}, maxmultideg::Vector{DI},
        ownpowers::Union{Bool,<:AbstractVector{DI}}=false) where {O,DI<:Integer}
        (mindeg < 0 || mindeg > maxdeg) && error("Invalid degree specification")
        n = length(minmultideg)
        (n != length(maxmultideg) ||
            any(minmax -> minmax[1] < 0 || minmax[1] > minmax[2], zip(minmultideg, maxmultideg))) &&
            error("Invalid multidegree specification")
        Σminmultideg = sum(minmultideg, init=zero(DI))
        Σmaxmultideg = sum(maxmultideg, init=zero(DI))
        if ownpowers === true
            return new{O,Vector{DI},DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, Vector{DI}(undef, n), Σminmultideg,
                Σmaxmultideg)
        elseif ownpowers === false
            return new{O,Nothing,DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, nothing, Σminmultideg, Σmaxmultideg)
        elseif length(ownpowers) != n
            error("Invalid length of ownpowers")
        else
            return new{O,typeof(ownpowers),DI}(n, mindeg, maxdeg, minmultideg, maxmultideg, ownpowers, Σminmultideg,
                Σmaxmultideg)
        end
    end
end

function Base.iterate(iter::MonomialIterator{Graded{LexOrder},P}) where {P}
    minmultideg, Σminmultideg, Σmaxmultideg = iter.minmultideg, iter.Σminmultideg, iter.Σmaxmultideg
    Σminmultideg > iter.maxdeg && return nothing
    Σmaxmultideg < iter.mindeg && return nothing
    powers = P === Nothing ? copy(minmultideg) : copyto!(iter.powers, minmultideg)
    if iter.mindeg > Σminmultideg
        powers_increment_right(iter, powers, iter.mindeg - Σminmultideg, 1) || @assert(false)
        deg = iter.mindeg
    else
        deg = Σminmultideg
    end
    return P === Nothing ? copy(powers) : powers, (deg, powers)
end

function Base.iterate(iter::MonomialIterator{Graded{LexOrder},P}, state::Tuple{DI,<:AbstractVector{DI}}) where {P,DI}
    deg, powers = state
    deg ≤ iter.maxdeg || return nothing
    minmultideg, maxmultideg = iter.minmultideg, iter.maxmultideg
    @inbounds while true
        # This is not a loop at all, we only go through it once, but we need to be able to leave the block at multiple
        # positions. If we do it with @goto, as would be proper, Julia begins to box all our arrays.

        # find the next power that can be decreased
        found = false
        local i
        for outer i in iter.n:-1:1
            if powers[i] > minmultideg[i]
                found = true
                break
            end
        end
        found || break
        # we must increment the powers to the left by 1 in total
        found = false
        local j
        for outer j in i-1:-1:1
            if powers[j] < maxmultideg[j]
                found = true
                break
            end
        end
        found || break

        powers[j] += 1
        # this implies that we reset everything to the right of the increment to its minimum and then compensate for all
        # the reductions by increasing the powers again where possible
        δ = sum(k -> powers[k] - minmultideg[k], j+1:i, init=zero(DI)) -1
        copyto!(powers, j +1, minmultideg, j +1, i - j)
        if powers_increment_right(iter, powers, δ, j +1)
            return P === Nothing ? copy(powers) : powers, (deg, powers)
        end
        break
    end
    # there's still hope: we can perhaps go to the next degree
    deg += one(DI)
    deg > iter.maxdeg && return nothing
    copyto!(powers, minmultideg)
    if powers_increment_right(iter, powers, deg - iter.Σminmultideg, 1)
        return P === Nothing ? copy(powers) : powers, (deg, powers)
    else
        return nothing
    end
end

Base.IteratorSize(::Type{<:MonomialIterator}) = Base.HasLength()
Base.IteratorEltype(::Type{<:MonomialIterator}) = Base.HasEltype()
Base.eltype(::Type{MonomialIterator{O,P,DI}}) where {O,P,DI} = Vector{DI}
function Base.length(iter::MonomialIterator, ::Val{:detailed})
    # internal function without checks or quick path
    # ~ O(n*d^2)
    maxdeg = iter.maxdeg
    occurrences = zeros(Int, maxdeg +1)
    @inbounds for deg₁ in iter.minmultideg[1]:min(iter.maxmultideg[1], maxdeg)
        occurrences[deg₁+1] = 1
    end
    nextround = similar(occurrences)
    for (minᵢ, maxᵢ) in Iterators.drop(zip(iter.minmultideg, iter.maxmultideg), 1)
        fill!(nextround, 0)
        for degᵢ in minᵢ:min(maxᵢ, maxdeg)
            for (degⱼ, occⱼ) in zip(Iterators.countfrom(0), occurrences)
                newdeg = degᵢ + degⱼ
                newdeg > maxdeg && break
                @inbounds nextround[newdeg+1] += occⱼ
            end
        end
        occurrences, nextround = nextround, occurrences
    end
    return occurrences
end
Base.@assume_effects :foldable :nothrow :notaskstate function Base.length(iter::MonomialIterator)
    maxdeg = iter.maxdeg
    iter.Σminmultideg > maxdeg && return 0
    iter.Σmaxmultideg < iter.mindeg && return 0
    @inbounds isone(iter.n) && return min(maxdeg, iter.maxmultideg[1]) - max(iter.mindeg, iter.minmultideg[1])
    @inbounds return sum(@view(@inline(length(iter, Val(:detailed)))[iter.mindeg+1:end]), init=0)
end
moniter_state(powers::AbstractVector{DI}) where {DI} = (DI(sum(powers, init=zero(DI))), powers)

function powers_increment_right(iter::MonomialIterator{Graded{LexOrder}}, powers, δ, from)
    @assert(δ ≥ 0 && from ≥ 0)
    maxmultideg = iter.maxmultideg
    i = iter.n
    @inbounds while δ > 0 && i ≥ from
        δᵢ = maxmultideg[i] - powers[i]
        if δᵢ ≥ δ
            powers[i] += δ
            return true
        else
            powers[i] = maxmultideg[i]
            δ -= δᵢ
        end
        i -= 1
    end
    return iszero(δ)
end
#endregion

function newton_polytope_do_worker end

function newton_polytope_do_taskfun(V, tid, task, ranges, nv, mindeg, maxdeg, bk, cond, progresses, acceptances,
    allcandidates, notifier, init_time, init_progress, num, filestuff)
    # notifier: 0 - no notification; 1 - the next to get becomes the notifier; 2 - notifier is taken
    verbose = notifier[] != 0
    lastappend = time_ns()
    isnotifier = Ref(false) # necessary due to the capturing/boxing bug
    lastinfo = Ref{Int}(lastappend)
    tmp = Vector{Float64}(undef, nv)
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
            newton_polytope_do_worker(V, task, bk, iter, tmp, progress, acceptance, @capture(p -> append!($candidates, p)),
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
                        prepare_push!(allcandidates, length(candidates) ÷ nv)
                        for i in 1:nv:length(candidates)
                            @inbounds unsafe_push!(allcandidates, convert(Vector{Int}, @view(candidates[i:i+nv-1])))
                        end
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
                prepare_push!(allcandidates, length(candidates) ÷ nv)
                for i in 1:nv:length(candidates)
                    @inbounds unsafe_push!(allcandidates, convert(Vector{Int}, @view(candidates[i:i+nv-1])))
                end
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

    maxdeg, mindeg = 0, typemax(Int)
    maxmultideg, minmultideg = fill(0, nv), fill(typemax(Int), nv)
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

    return mindeg, maxdeg, minmultideg, maxmultideg
end

function newton_halfpolytope_tighten(mindeg, maxdeg, minmultideg, maxmultideg)
    # In the multithreading case, we need maintain multiple copies of portions of the powers, which might use more space than
    # necessary. So instead of using copies, we take alternative smaller representations.
    maxval = max(maxdeg, maximum(maxmultideg, init=0))
    local T
    for outer T in (UInt8, UInt16, UInt32, UInt64)
        typemax(T) ≥ maxval && break
    end
    return convert(T, mindeg), convert(T, maxdeg), convert(Vector{T}, minmultideg), convert(Vector{T}, maxmultideg)
end

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

function newton_halfpolytope_alloc(V, nv) end

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task,
    filepath)
    @verbose_info("Preparing to determine Newton polytope (single-threaded)")

    bk = newton_halfpolytope_alloc(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    if isnothing(filepath)
        candidates = FastVec{Vector{Int}}() # don't try to save on the data type, DynamicPolynomials requires Vector{Int}
    else
        candidates = FastVec{typeof(maxdeg)}()
    end

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
        newton_polytope_do_worker(V, task, bk, iter, Vector{Float64}(undef, nv), progress, acceptance,
            isnothing(filepath) ? @capture(p -> push!($candidates, copy(p))) : @capture(p -> append!($candidates, p)),
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
                        Δt = progress[] == $init_progress ? progress[] - init_progress : 1
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
            return finish!(candidates)
        else
            close(fileout)
            close(fileprogress)
            return progress[], acceptance[]
        end
    end
end

function newton_halfpolytope_clonetask end

function newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num,
    nthreads::Integer, task, secondtask, filepath)
    if isone(nthreads)
        @assert(isnothing(secondtask))
        return newton_halfpolytope_do_execute(V, nv, mindeg, maxdeg, minmultideg, maxmultideg, verbose, num, task, filepath)
    end

    threadsize = div(num, nthreads, RoundUp)
    @verbose_info("Preparing to determine Newton polytope using ", nthreads, " threads, each checking about ",
        threadsize, " candidates")
    cutat = monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, threadsize)
    cutlen = nv - cutat
    cutat += 1 # cutat is now the first entry to be fixed
    bk = newton_halfpolytope_alloc(V, nv)
    # While we precalculate the size of the list exactly, we don't pre-allocate the output candidates - we hope to eliminate a
    # lot of powers by the polytope containment, so we might overallocate so much memory that we hit a resource constraint
    # here. If instead, we grow the list dynamically, we pay the price in speed, but the impossible might become feasible.
    if isnothing(filepath)
        candidates = FastVec{Vector{Int}}() # don't try to save on the data type, DynamicPolynomials requires Vector{Int}
    else
        candidates = FastVec{typeof(maxdeg)}()
    end

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
            threads[tid] = Threads.@spawn newton_polytope_do_taskfun($V, $tid, $taskₜ, $ranges, $nv, $mindeg, $maxdeg, $bk,
                $cond, $threadprogress, $threadacceptance, $candidates, $notifier, $init_time, $init_progress, $num,
                $filestuff)
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
    # We need to return the appropriate monomial order, but due to the partitioning and multithreading, our output
    # is unordered (not completely, we have ordered of varying length, but does this help?).
    # sort!(candidates, lt=(a, b) -> compare(a, b, monomial_ordering(P)) < 0)
    if isnothing(filepath)
        return sort!(finish!(candidates), lt=(a, b) -> compare(a, b, Graded{LexOrder}) < 0)
        # TODO: This is not appropriate, but as long as we cannot query the actual ordering, let's go for the default.
    else
        return sum(threadprogress, init=0), sum(threadacceptance, init=0)
    end
end

function newton_halfpolytope(V, objective::P, ::Val{false}; verbose::Bool=false,
    filepath::Union{<:AbstractString,Nothing}=nothing, kwargs...) where {P<:AbstractPolynomialLike}
    parameters, coeffs = newton_polytope_preproc(V, objective; verbose, kwargs...)
    newton_time = @elapsed candidates = let
        analysis = newton_halfpolytope_analyze(coeffs)
        num, nthreads, task, secondtask = newton_halfpolytope_do_prepare(V, coeffs, analysis..., verbose; parameters...)
        if isone(nthreads) && isnothing(filepath)
            newton_halfpolytope_do_execute(V, size(coeffs, 1), analysis..., verbose, num, nthreads, task, secondtask, filepath)
        else
            newton_halfpolytope_do_execute(V, size(coeffs, 1), newton_halfpolytope_tighten(analysis...)..., verbose, num,
                nthreads, task, secondtask, filepath)
        end
    end

    if isnothing(filepath)
        @verbose_info("Found ", length(candidates), " elements in the Newton polytope in ", newton_time, " seconds")
        return makemonovec(variables(objective), candidates)
    else
        @verbose_info("Found ", candidates[2], " elements in the Newton polytope in ", newton_time,
            " seconds and stored the results to the given file")
        return true
    end
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
function newton_halfpolytope_from_file(filepath::AbstractString, objective::P; estimate::Bool=false,
    verbose::Bool=false) where {P<:AbstractPolynomialLike}
    maxval = div(maxdegree(objective), 2, RoundDown)
    local T
    for outer T in (UInt8, UInt16, UInt32, UInt64)
        typemax(T) ≥ maxval && break
    end
    return newton_halfpolytope_from_file(filepath, objective, estimate, verbose, T)
end

function newton_halfpolytope_from_file(filepath::AbstractString, objective, estimate::Bool, verbose::Bool, ::Type{T}) where {T<:Integer}
    if isfile("$filepath.out")
        matches = Regex("^$filepath\\.out\$")
        dir = dirname(realpath("$filepath.out"))
        @verbose_info("Underlying data comes from single-node, single-thread calculation")
    elseif isfile("$filepath-1.out")
        matches = Regex("^$filepath-\\d+\\.out\$")
        dir = dirname(realpath("$filepath-1.out"))
        @verbose_info("Underlying data comes from single-node, multi-thread calculation")
    elseif isfile("$filepath-n0.out")
        matches = Regex("^$filepath-n\\d+\\.out\$")
        dir = dirname(realpath("$filepath-n0.out"))
        @verbose_info("Underlying data comes from multi-node, single-thread calculation")
    elseif isfile("$filepath-n0-1.out")
        matches = Regex("^$filepath-n\\d+-\\d+\\.out\$")
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
    candidates = FastVec{Vector{Int}}(buffer=len)
    readbuffer = Vector{T}(undef, nv)
    readbytebuffer = unsafe_wrap(Array, Ptr{UInt8}(pointer(readbuffer)), nv * sizeof(T))
    bufferlen = length(readbytebuffer)
    load_time = @elapsed begin
        for fn in readdir(dir)
            if !isnothing(match(matches, fn))
                @verbose_info("Opening file $fn")
                local fs
                while true
                    try
                        fs = Base.Filesystem.open(fn, Base.JL_O_RDONLY | Base.JL_O_EXCL, 0o444)
                        break
                    catch ex
                        (isa(ex, Base.IOError) && ex.code == Base.UV_EEXIST) || rethrow(ex)
                        Base.prompt("Could not open file $fn exclusively. Release the file, then hit [Enter] to retry.")
                    end
                end
                try
                    stream = BufferedStreams.BufferedInputStream(fs)
                    while !eof(stream)
                        readbytes!(stream, readbytebuffer, bufferlen) == bufferlen || error("Unexpected end of file: $fn")
                        cto = convert(Vector{Int}, readbuffer)
                        unsafe_push!(candidates, cto)
                    end
                finally
                    Base.Filesystem.close(fs)
                end
            end
        end
    end
    length(candidates) == len || error("Calculated length does not match number of loaded monomials")
    @verbose_info("Loaded $len monomials into memory in $load_time seconds. Now sorting and removing duplicates.")
    sort_time = @elapsed begin
        finalcandidates = Base._groupedunique!(sort!(finish!(candidates), lt=(a, b) -> compare(a, b, Graded{LexOrder}) < 0))
    end
    @verbose_info("Sorted all monomials and removed $(len - length(finalcandidates)) duplicates in $sort_time seconds")
    return makemonovec(variables(objective), finalcandidates)
end

makemonovec(vars::Vector{<:DynamicPolynomials.Variable}, exps::Vector{Vector{Int}}) =
    DynamicPolynomials.MonomialVector(vars, exps)
# for a relatively effecient construction of monomials from their powers if the backend is not DynamicPolynomials... although
# we don't actively support anything else.
makemonovec(vars::Vector, exps::Vector{Vector{Int}}) = monomial_vector(FakeMonomialVector(vars, exps))

struct FakeMonomialVector{V,M} <: AbstractVector{M}
    vars::Vector{V}
    exps::Vector{Vector{Int}}

    function FakeMonomialVector(vars::Vector{V}, exps::Vector{Vector{Int}}) where {V}
        length(vars) == length(exps) || error("Invalid monomial vector construction")
        new{V,monomial_type(V)}(vars, exps)
    end
end

Base.length(fmv::FakeMonomialVector) = length(fmv.vars)
function Base.getindex(fmv::FakeMonomialVector{V}, x) where {V}
    exps = fmv.exps[x]
    all(iszero, exps) && return constant_monomial(V)
    i = findfirst(Base.:! ∘ iszero, exps)
    @inbounds mon = monomial_type(V)(fmv.vars[i])
    for _ in 2:exps[i]
        @inbounds mon = MutableArithmetics.mul!!(mon, fmv.vars[i])
    end
    for i in i+1:length(exps)
        for _ in 1:exps[i]
            MutableArithmetics.mul!!(mon, fmv.vars[i])
        end
    end
    return mon
end