function restore_status!(fileprogress, mindeg::I, maxdeg::I, minmultideg::AbstractVector{I},
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

function restore_status!(fileprogress, powers::Vector{<:Integer}, fixedsize::Integer)
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

"""
    halfpolytope_from_file(filepath, objective; estimate=false, verbose=false)

Constructs the Newton polytope for the sum of squares optimization of a given objective, which is half the Newton polytope of
the objective itself. This function does not do any calculation, but instead loads the data that has been generated using
[`halfpolytope`](@ref) with the given `filepath` on the given `objective`.

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

See also [`halfpolytope`](@ref).
"""
halfpolytope_from_file(filepath::AbstractString, objective::SimpleRealPolynomial; estimate::Bool=false,
    verbose::Bool=false) =
    return halfpolytope_from_file(filepath, objective, estimate, verbose, T)

function halfpolytope_from_file(filepath::AbstractString, objective::AbstractPolynomialLike; verbose::Bool=false,
    kwargs...)
    out = halfpolytope_from_file(filepath, SimplePolynomial(objective); verbose, kwargs...)
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

function halfpolytope_from_file(filepath::AbstractString, objective, estimate::Bool, verbose::Bool, ::Type{T}) where {T<:Integer}
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