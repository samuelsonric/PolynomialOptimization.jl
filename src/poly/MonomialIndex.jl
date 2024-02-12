export monomial_count, monomial_index, exponents_from_index!

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

"""
    monomial_index(m::Union{<:SimpleMonomial{Nr,Nc,P},<:SimpleVariable}...) where {Nr,Nc,P<:Unsigned}

Returns the unique index of the monomial (or variable) `m` (if multiple monomials are specified, of the product of all those
`m`) with respect to the deglex ordering. No allocations are performed during the calculation, and the product of the monomials
is never explictly constructed.
"""
Base.@assume_effects :consistent function monomial_index(m::SimpleMonomial)
    nvars = nvariables(m)
    mondeg::Int = degree(m)
    # how many monomials are there with a lower total degree?
    mindex = monomial_count(mondeg, nvars)
    for vardeg in Iterators.take(exponents(m), nvars -1)
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
monomial_index(::Number) = 1 # just a lazy way to avoid constant_monomial constructs, but for type stability, probably don't
                             # use this function
monomial_index(m::SimpleRealVariable{Nr,Nc}) where {Nr,Nc} = 2 + Nr + 2Nc - m.index # constant has index 1
monomial_index(m::SimpleComplexVariable{Nr,Nc}) where {Nr,Nc} = 2 + (m.isconj ? Nc : 2Nc) - m.index

Base.@assume_effects :consistent @generated function monomial_index(m::Union{<:SimpleMonomial{Nr,Nc,P},<:SimpleVariable{Nr,Nc}}...) where {Nr,Nc,P<:Unsigned}
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

# apart from this, also define a function for the multiplication of monomials, which gives a lazy iterator
struct SimpleMonomialProduct{Nr,Nc,P<:Unsigned,Ms<:NTuple{<:Any,SimpleMonomial{Nr,Nc,P}}}
    monomials::Ms
end

Base.IteratorSize(::Type{<:SimpleMonomialProduct}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialProduct}) = Base.HasEltype()
Base.eltype(::Type{<:SimpleMonomialProduct{Nr,0,P}}) where {Nr,P<:Unsigned} =
    Tuple{SimpleRealVariable{Nr,0,smallest_unsigned(Nr)},P}
Base.eltype(::Type{<:SimpleMonomialProduct{Nc,P}}) where {Nc,P<:Unsigned} =
    Tuple{SimpleComplexVariable{0,Nc,smallest_unsigned(Nc)},P}
Base.eltype(::Type{<:SimpleMonomialProduct{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned} =
    Tuple{Union{SimpleRealVariable{Nr,Nc,smallest_unsigned(Nr)},SimpleComplexVariable{Nr,Nc,smallest_unsigned(Nc)}},P}
Base.@assume_effects :consistent @generated function Base.iterate(p::SimpleMonomialProduct{Nr,Nc,P,Ms},
    state=ntuple(i -> 1, Val(items))) where {Nr,Nc,P<:Unsigned,items,Ms<:NTuple{items,SimpleMonomial{Nr,Nc,P}}}
    states = Expr(:local)
    for i in 1:fieldcount(Ms)
        push!(states.args, Symbol(:state, i))
    end
    result = Expr(:block, :(pow = zero(P)), :(minidx = typemax(Int)), states)
    for (fᵢ, field) in enumerate((:exponents_real, :exponents_complex, :exponents_conj))
        fᵢ === 1 && iszero(Nr) && continue
        fᵢ > 1 && iszero(Nc) && continue
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
            push!(iffoundbody.args, :(return (SimpleRealVariable{Nr,Nc}(minidx), pow), $newstate))
        elseif fᵢ === 2
            push!(iffoundbody.args, :(return (SimpleComplexVariable{Nr,Nc}(minidx), pow), $newstate))
        else
            push!(iffoundbody.args, :(return (SimpleComplexVariable{Nr,Nc}(minidx, true), pow), $newstate))
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