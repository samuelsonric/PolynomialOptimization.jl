export monomial_count, trisize

"""
    monomial_count(n, d)

Short helper function that allows to determine the number of monomials in `n` variables up to degree `d`.
"""
monomial_count(n, d) = length(ExponentsDegree{n,UInt}(0:d))

trisize(n) = (n * (n +1)) >> 1
realtype(::Type{<:Union{R,Complex{R}}}) where {R<:Real} = R

macro twice(symb::Symbol, condition, body)
    esc(quote
        let $symb=false
            $body
        end
        $condition && let $symb=true
            $body
        end
    end)
end

# For complex-valued problems, there are m and conj(m); all solvers require real-valued inputs.
# Therefore, we first define the canonicalized monomial m̃ as the one of m, conj(m) that has the smaller index.
iscanonical(::SimpleMonomial{<:Any,0}) = true
Base.@assume_effects :consistent iscanonical(m::SimpleMonomialOrConj) = m ≤ SimpleConjMonomial(m)
canonicalize(m::SimpleMonomialOrConj) = iscanonical(m) ? m : SimpleConjMonomial(m)

@inline function getreim(state, args::SimpleMonomialOrConj{Nr,Nc}...) where {Nr,Nc}
    idx₁ = mindex(state, args...)
    idx₂ = mindex(state, SimpleConjMonomial.(reverse(args))...) # reverse is rather unnecessary (commuting)
    if idx₁ ≤ idx₂
        return idx₁, idx₂, true # make sure to test for equality whenever the imaginary part is involved
    else
        return idx₂, idx₁, false
    end
end

@inline @generated function findin(haystacks::NTuple{N,Any}, needles::NTuple{N,Any}, startfrom::Integer=firstindex(haystacks[1])) where {N}
    quote
        # should we assert that all lengths be equal?
        @inbounds for i in startfrom:length(haystacks[1])
            # Assume haystacks and needles are short and the comparison itself cheap, but jumping after the comparison not. So
            # to minimize the number of jumps, we compare everything always.
            if +($((:(haystacks[$i][i] == needles[$i]) for i in 1:N)...),) == $N
                return i
            end
        end
        return nothing
    end
end

macro pushorupdate!(part::Symbol, icoeff...)
    iseven(length(icoeff)) || throw(MethodError(var"@pushorupdate!", (__source__, __module__, part, icoeff...)))
    n = length(icoeff) ÷ 2
    esc(quote
        let $((Expr(:(=), Symbol(:coeff, i), icoeff[2i]) for i in 1:n)...),
            partidx=@isdefined(rows) ? findin((indices, rows), ($part, $(icoeff[1])), length(indices) - curlen +1) :
                                       findin((indices,), ($part,), length(indices) - curlen +1)
            if isnothing(partidx)
                @isdefined(rows) && unsafe_push!(rows, $((icoeff[2i-1] for i in 1:n)...))
                unsafe_push!(indices, $((part for _ in 1:n)...))
                unsafe_push!(values, $((Symbol(:coeff, i) for i in 1:n)...))
                curlen += $n
            else
                $((:(values[partidx+$(i-1)] += $(Symbol(:coeff, i))) for i in 1:n)...)
            end
        end
    end)
end

@inline function pushorupdate!(idxvec::FastVec, index, valuevec::FastVec, value)
    i = findfirst(isequal(index), idxvec)
    if isnothing(i)
        unsafe_push!(idxvec, index)
        unsafe_push!(valuevec, value)
    else
        @inbounds valuevec[i] += value
    end
end

struct ScalarMatrix{X} <: AbstractMatrix{X} # we could make it mutable and fully implement the AbstractArray interface, but we
                                            # don't need it
    x::X
end

Base.size(::ScalarMatrix) = (1, 1)
Base.length(::ScalarMatrix) = 1
@inline function Base.getindex(m::ScalarMatrix, args...)
    @boundscheck checkbounds(m, args...)
    return m.x
end
Base.iterate(m::ScalarMatrix) = m.x, nothing
Base.iterate(::ScalarMatrix, ::Nothing) = nothing

collect_grouping(g::AbstractVector{M} where M<:SimpleMonomial) = g
collect_grouping(g) = collect(g)