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