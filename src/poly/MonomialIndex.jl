export monomial_count, monomial_index, exponents_from_index!, exponents_from_indices

# everything here is marked as consistent - as we demand that our monomials be immutable (regardless of whether they are so in
# their strictest sense), this allows speedups in the real-valued case when the monomial index of both the monomials and their
# conjugates is computed: the compiler can directly infer that they are the same, as conj is an identity.

# calculates the index of a given monomial in a deglex ordering (lex according to vars - this is the same order as the one that
# isless uses, which is very important)
@doc raw"""
    monomial_count(degree::Int, nvars::Int)

Returns the size of the space of monomials in `nvars` variables with degree at most `degree`, i.e.
``\binom{\mathtt{degree} + \mathtt{nvars}}{\mathtt{nvars}}``.
"""
Base.@assume_effects :consistent monomial_count(degree::Int, nvars::Int) =
    binomial(degree + nvars, nvars)


function monomial_index(exponents::AbstractVector{<:Integer})
    nvars = length(exponents)
    mondeg::Int = sum(exponents, init=0)
    # how many monomials are there with a lower total degree?
    mindex = monomial_count(mondeg, nvars)
    for vardeg in Iterators.take(exponents, nvars -1)
        #=for i in vardeg+1:mondeg # check for all possible higher degrees that the current variable may have had
            mindex -= binomial((nvars -1) + (mondeg - i) -1, (nvars -1) -1) # and remove the number of possible monomials
                                                                            # that remain in this subspace
        end=#
        # This loop is equivalent to:
        # mindex -= (nvars + mondeg - vardeg -2) * binomial(nvars + mondeg - vardeg -3, nvars -2) ÷ (nvars -1)
        # And let's move it after necessary subtractions, so that we then get
        mondeg -= vardeg
        nvars -= one(nvars)
        # mindex -= (nvars + mondeg -1) * binomial(nvars + mondeg -2, nvars -1) ÷ nvars
        mindex -= ((nvars + mondeg -1) * monomial_count(mondeg -1, nvars -1)) ÷ nvars
    end
    return mindex
end
"""
    monomial_index(m::Union{<:SimpleMonomial{Nr,Nc,P},<:SimpleVariable}...) where {Nr,Nc,P<:Unsigned}

Returns the unique index of the monomial (or variable) `m` (if multiple monomials are specified, of the product of all those
`m`) with respect to the deglex ordering. No allocations are performed during the calculation, and the product of the monomials
is never explictly constructed.
"""
Base.@assume_effects :consistent monomial_index(m::SimpleMonomial) = monomial_index(exponents(m))
monomial_index(::Number) = 1 # just a lazy way to avoid constant_monomial constructs, but for type stability, probably don't
                             # use this function
monomial_index(m::SimpleVariable{Nr,Nc}) where {Nr,Nc} = 2 + Nr + 2Nc - m.index # constant has index 1

Base.@assume_effects :consistent @generated function monomial_index(m::Union{<:SimpleMonomial{Nr,Nc},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc}
    items = length(m)
    quote
        nvars = $(Nr + 2Nc)
        mondeg::Int = +($((:(degree(m[$i])) for i in 1:items)...))
        mindex = monomial_count(mondeg, nvars)
        # how many monomials are there with a lower degree?
        @inbounds for exps in Iterators.take(zip($((:(exponents(m[$i])) for i in 1:items)...)), nvars -1)
            mondeg -= +($((:(exps[$i]) for i in 1:items)...))
            nvars -= one(nvars)
            mindex -= ((nvars + mondeg -1) * monomial_count(mondeg -1, nvars -1)) ÷ nvars
        end
        return mindex
    end
end

"""
    exponents_from_index!(powers::AbstractVector{<:Integer}, index::Integer)

Constructs the vector of powers that is associated with the monomial index `index` and stores it in `powers`. This can be
thought of as an inverse function of [`monomial_index`](@ref), although for non-allocating purposes it works on a vector of
powers instead of a monomial.
"""
Base.@assume_effects :consistent function exponents_from_index!(powers::AbstractVector{<:Integer}, index::Integer)
    n = length(powers)
    degree = 0
    while true
        next = binomial(n + degree -1, n -1)
        if next ≥ index
            break
        else
            index -= next
            degree += 1
        end
    end
    for i in 1:n-1
        total = 0
        for degᵢ in 0:degree
            next = binomial(n - i -1 + degree - degᵢ, n - i -1)
            if total + next ≥ index
                degree -= degᵢ
                index -= total
                @inbounds powers[i] = degᵢ
                break
            else
                total += next
            end
        end
    end
    # special case i = n, as binomial(-1, -1) = 0 instead of 1
    @assert(1 ≥ index)
    @inbounds powers[n] = degree
    return powers
end

for (splitvars, params) in ((false, (:Nr,)), (true, (:Nr, :Nc)))
    eval(quote
        function exponents_from_indices(T, $(params...), mons_idx::Vector, ::Val{dense}, ::Val{:iterate}) where {dense}
            $(splitvars ? :(nv = Nr + 2Nc) : :(nv = Nr))
            next_col = 1
            max_col = length(mons_idx)
            @assert(max_col > 0)
            iter = MonomialIterator(Base.zero(T), typemax(T), zeros(T, nv), fill(typemax(T), nv), ownpowers)
            if dense
                coeffs_real = resizable_array(T, Nr, max_col)
                powers_real = @view(iter.powers[1:Nr])
                $(splitvars ? quote
                    coeffs_complex = resizable_array(T, Nc, max_col)
                    coeffs_conj = resizable_array(T, Nc, max_col)
                    powers_complex = @view(iter.powers[Nr+1:Nr+Nc])
                    powers_conj = @view(iter.powers[Nr+Nc+1:end])
                end : :(nothing))
                @inbounds for (idx, _) in enumerate(iter)
                    if idx == convert(Int, mons_idx[next_col])
                        copyto!(@view(coeffs_real[:, next_col]), powers_real)
                        $(splitvars ? quote
                            copyto!(@view(coeffs_complex[:, next_col]), powers_complex)
                            copyto!(@view(coeffs_conj[:, next_col]), powers_conj)
                        end : :(nothing))
                        next_col += 1
                        next_col > max_col && break
                    end
                end
                @assert(next_col == max_col +1)
                return $(splitvars ? :(coeffs_real, coeffs_complex, coeffs_conj) : :(coeffs_real))
            else
                colptr_real = resizable_array(UInt, max_col +1)
                rowval_real = FastVec{UInt}()
                nzval_real = FastVec{T}()
                powers_real = @view(iter.powers[1:Nr])
                $(splitvars ? quote
                    colptr_complex = resizable_array(UInt, max_col +1)
                    rowval_complex = FastVec{UInt}()
                    nzval_complex = FastVec{T}()
                    colptr_conj = resizable_array(UInt, max_col +1)
                    rowval_conj = FastVec{UInt}()
                    nzval_conj = FastVec{T}()
                    powers_complex = @view(iter.powers[Nr+1:Nr+Nc])
                    powers_conj = @view(iter.powers[Nr+Nc+1:end])
                end : :(nothing))
                @inbounds for (idx, _) in enumerate(iter)
                    if idx == convert(Int, mons_idx[next_col])
                        colptr_real[next_col] = length(rowval_real) +1
                        for (row, val) in enumerate(powers_real)
                            if !iszero(val)
                                push!(rowval_real, row)
                                push!(nzval_real, val)
                            end
                        end
                        $(splitvars ? quote
                            colptr_complex[next_col] = length(rowval_complex) +1
                            for (row, val) in enumerate(powers_complex)
                                if !iszero(val)
                                    push!(rowval_complex, row)
                                    push!(nzval_complex, val)
                                end
                            end
                            colptr_conj[next_col] = length(rowval_conj) +1
                            for (row, val) in enumerate(powers_conj)
                                if !iszero(val)
                                    push!(rowval_conj, row)
                                    push!(nzval_conj, val)
                                end
                            end
                        end : :(nothing))
                        next_col += 1
                        next_col > max_col && break
                    end
                end
                @assert(next_col == max_col +1)
                colptr_real[next_col] = length(rowval_real) +1
                $(splitvars ? quote
                    colptr_complex[next_col] = length(rowval_complex) +1
                    colptr_conj[next_col] = length(rowval_conj) +1
                    return SparseMatrixCSC{T,UInt}(nv, max_col, colptr_real, finish!(rowval_real), finish!(nzval_real)),
                        SparseMatrixCSC{T,UInt}(nv, max_col, colptr_complex, finish!(rowval_complex), finish!(nzval_complex)),
                        SparseMatrixCSC{T,UInt}(nv, max_col, colptr_conj, finish!(rowval_conj), finish!(nzval_conj))
                end : :(return SparseMatrixCSC{T,UInt}(nv, max_col, colptr_real, finish!(rowval_real), finish!(nzval_real))))
            end
        end

        function exponents_from_indices(T, $(params...), mons_idx::Vector, ::Val{dense}, ::Val{:index}) where {dense}
            $(splitvars ? :(nv = Nr + 2Nc) : :(nv = Nr))
            max_col = length(mons_idx)
            iter = MonomialIterator(Base.zero(T), typemax(T), zeros(T, nv), fill(typemax(T), nv), ownpowers)
            prepare = exponents_from_index_prepare(iter)
            if dense
                coeffs_real = resizable_array(T, Nr, max_col)
                $(splitvars ? quote
                    coeffs_complex = resizable_array(T, Nc, max_col)
                    coeffs_conj = resizable_array(T, Nc, max_col)
                    powers = iter.powers
                    powers_real = @view(powers[1:Nr])
                    powers_complex = @view(powers[Nr+1:Nr+Nc])
                    powers_conj = @view(powers[Nr+Nc+1:Nr+2Nc])
                    @inbounds for (i, c_real, c_complex, c_conj) in zip(mons_idx, eachcol(coeffs_real),
                                                                        eachcol(coeffs_complex), eachcol(coeffs_conj))
                        exponents_from_index!(powers, iter, prepare, convert(Int, idx))
                        copyto!(c_real, powers_real)
                        copyto!(c_complex, powers_complex)
                        copyto!(c_conj, powers_conj)
                    end
                    return coeffs_real, coeffs_complex, coeffs_conj
                end : quote
                    @inbounds for (idx, c) in zip(mons_idx, eachcol(coeffs_real))
                        exponents_from_index!(c, iter, prepare, convert(Int, idx))
                    end
                    return coeffs_real
                end)
            else
                colptr_real = resizable_array(UInt, max_col +1)
                rowval_real = FastVec{UInt}()
                nzval_real = FastVec{T}()
                powers = iter.powers
                $(splitvars ? quote
                    colptr_complex = resizable_array(UInt, max_col +1)
                    rowval_complex = FastVec{UInt}()
                    nzval_complex = FastVec{T}()
                    colptr_conj = resizable_array(UInt, max_col +1)
                    rowval_conj = FastVec{UInt}()
                    nzval_conj = FastVec{T}()
                    powers_real = @view(powers[1:Nr])
                    powers_complex = @view(powers[Nr+1:Nr+Nc])
                    powers_conj = @view(powers[Nr+Nc+1:end])
                end : :(powers_real = powers))
                @inbounds for (i, idx) in enumerate(mons_idx)
                    exponents_from_index!(powers, iter, prepare, convert(Int, idx))
                    colptr_real[i] = length(rowval_real) +1
                    for (row, val) in enumerate(powers_real)
                        if !iszero(val)
                            push!(rowval_real, row)
                            push!(nzval_real, val)
                        end
                    end
                    $(splitvars ? quote
                        colptr_complex[i] = length(rowval_complex) +1
                        for (row, val) in enumerate(powers_complex)
                            if !iszero(val)
                                push!(rowval_complex, row)
                                push!(nzval_complex, val)
                            end
                        end
                        colptr_conj[i] = length(rowval_conj) +1
                        for (row, val) in enumerate(powers_conj)
                            if !iszero(val)
                                push!(rowval_conj, row)
                                push!(nzval_conj, val)
                            end
                        end
                    end : :(nothing))
                end
                colptr_real[end] = length(rowval_real) +1
                $(splitvars ? quote
                    colptr_complex[next_col] = length(rowval_complex) +1
                    colptr_conj[next_col] = length(rowval_conj) +1
                    return SparseMatrixCSC{T,UInt}(nv, max_col, colptr_real, finish!(rowval_real), finish!(nzval_real)),
                        SparseMatrixCSC{T,UInt}(nv, max_col, colptr_complex, finish!(rowval_complex), finish!(nzval_complex)),
                        SparseMatrixCSC{T,UInt}(nv, max_col, colptr_conj, finish!(rowval_conj), finish!(nzval_conj))
                end : :(return SparseMatrixCSC{T,UInt}(nv, max_col, colptr_real, finish!(rowval_real), finish!(nzval_real))))
            end
        end

        function exponents_from_indices(T, $(params...), mons_idx_set::Set, dense::Val)
            isempty(mons_idx_set) && return exponents_from_indices(T, $(params...), mons_idx, dense, Val(:index))
            $(splitvars ? :(nv = Nr + 2Nc) : :(nv = Nr))
            mons_idx = sort!(collect(mons_idx_set))
            tmp = Vector{T}(undef, nv)
            exponents_from_index!(tmp, convert(Int, last(mons_idx)))
            maxdeg = Int(sum(tmp))
            return exponents_from_indices(T, $(params...), mons_idx, dense,
                2length(mons_idx) < monomial_count(maxdeg, nv) ? Val(:index) : Val(:iterate))
        end
    end)
end

"""
    exponents_from_indices(P, nv, mons::Set{FastKey{Int}}, dense::Bool)
    exponents_from_indices(P, Nr, Nc, mons::Set{FastKey{Int}}, dense::Bool)

Constructs the matrices of powers associated with the monomial indices listed in `mons`, where `P` is the eltype of the
matrices. Depending on `dense`, the result will be a `Matrix{P}` or a `SparseMatrixCSC{P}`; the `nv` form will generate just
one matrix for `nv` variables in total; the `Nr, Nc` form will generate three matrices for the exponents of `Nr` real and `Nc`
complex/conjugate variables.
"""
exponents_from_indices

# apart from this, also define a function for the multiplication of monomials, which gives a lazy iterator
struct SimpleMonomialProduct{Nr,Nc,P<:Unsigned,Ms<:NTuple{<:Any,SimpleMonomial{Nr,Nc,P}}}
    monomials::Ms
end

Base.IteratorSize(::Type{<:SimpleMonomialProduct}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialProduct}) = Base.HasEltype()
Base.eltype(::Type{<:SimpleMonomialProduct{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned} =
    Tuple{SimpleVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)},P}
Base.@assume_effects :consistent @generated function Base.iterate(p::SimpleMonomialProduct{Nr,Nc,P,Ms},
    state=ntuple(i -> 1, Val(items))) where {Nr,Nc,P<:Unsigned,items,Ms<:NTuple{items,SimpleMonomial{Nr,Nc,P}}}
    states = Expr(:local)
    for i in 1:fieldcount(Ms)
        push!(states.args, Symbol(:state, i))
    end
    result = Expr(:block, :(pow = zero(P)), :(minidx = typemax(Int)), states)
    for (fᵢ, field) in enumerate((:exponents_real, :exponents_complex, :exponents_conj))
        fᵢ === 1 && iszero(Nr) && continue
        fᵢ > 1 && iszero(Nc) && break
        iffoundbody = Expr(:block)
        iffound = Expr(:if, :(minidx < typemax(Int)), iffoundbody)
        δ = Vector{Any}(undef, fieldcount(Ms))
        for (i, Mᵢ) in enumerate(fieldtypes(Ms))
            stateᵢ = Symbol(:state, i)
            if Mᵢ <: SimpleSparseMonomialOrView
                if fᵢ === 1
                    δ[i] = 0
                elseif fᵢ === 2
                    δ[i] = :(nnz(p.monomials[$i].exponents_real))
                else
                    δ[i] = :(nnz(p.monomials[$i].exponents_real) + nnz(p.monomials[$i].exponents_complex))
                end
                # "for outer x in r" will not assign a value to x if isempty(r), but we need it in any case!
                push!(result.args, :(
                    let idx=rowvals(p.monomials[$i].$field), vals=nonzeros(p.monomials[$i].$field)
                        $stateᵢ = state[$i]-$(δ[i])
                        for outer $stateᵢ in $stateᵢ:length(idx)
                            if !iszero(vals[$stateᵢ])
                                if idx[$stateᵢ] < minidx
                                    minidx = idx[$stateᵢ]
                                end
                                break
                            end
                        end
                    end
                ))
                push!(iffoundbody.args, :(
                    let idx=rowvals(p.monomials[$i].$field)
                        if lastindex(idx) ≥ $stateᵢ && idx[$stateᵢ] == minidx
                            pow += nonzeros(p.monomials[$i].$field)[$stateᵢ]
                            $stateᵢ += 1
                        end
                    end
                ))
            else
                if fᵢ === 1
                    δ[i] = 0
                elseif fᵢ === 2
                    δ[i] = :Nr
                else
                    δ[i] = :(Nr + Nc)
                end
                push!(result.args, :(
                    $stateᵢ = state[$i] - $(δ[i]);
                    for outer $stateᵢ in $stateᵢ:$(fᵢ === 1 ? :Nr : :Nc)
                        if !iszero(p.monomials[$i].$field[$stateᵢ])
                            if $stateᵢ < minidx
                                minidx = $stateᵢ
                            end
                            break
                        end
                    end
                ))
                push!(iffoundbody.args, :(
                    if $stateᵢ == minidx
                        pow += p.monomials[$i].$field[$stateᵢ]
                        $stateᵢ += 1
                    end
                ))
            end
        end
        newstate = Expr(:tuple)
        for i in 1:fieldcount(Ms)
            push!(newstate.args, :($(Symbol(:state, i)) + $(δ[i])))
        end
        if fᵢ === 1
            push!(iffoundbody.args, :(return (SimpleVariable{Nr,Nc}(minidx), pow), $newstate))
        elseif fᵢ === 2
            push!(iffoundbody.args, :(return (SimpleVariable{Nr,Nc}(minidx + Nr), pow), $newstate))
        else
            push!(iffoundbody.args, :(return (SimpleVariable{Nr,Nc}(minidx + Nr + Nc), pow), $newstate))
        end
        push!(result.args, iffound)
    end
    push!(result.args, :(return nothing))
    return :(@inbounds $result)
end
function Base.length(p::SimpleMonomialProduct{Nr,Nc,P,Ms}) where {Nr,Nc,P<:Unsigned,items,Ms<:NTuple{items,SimpleMonomial{Nr,Nc,P}}}
    len = 0
    for exps in zip(exponents.(p.monomials)...)
        all(iszero, exps) || (len += 1)
    end
    return len
end

Base.:*(m::SimpleMonomial{Nr,Nc,P}...) where {Nr,Nc,P} = SimpleMonomialProduct(m)