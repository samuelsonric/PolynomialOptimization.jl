# These are simplified versions, relying on default fill-ins. A much more intricate way of dealing with incomplete type
# specifications was present previously (see commit 04ab29d91c42045d3d10bf54344ec34eeb1c0c68).
MultivariatePolynomials.variable_union_type(::XorTX{<:Union{<:IntVariable{Nr,Nc},
                                                            <:IntMonomialOrConj{Nr,Nc},
                                                            <:Term{<:Any,<:IntMonomial{Nr,Nc}},
                                                            <:IntPolynomial{<:Any,Nr,Nc}}}) where {Nr,Nc} =
    IntVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)}

MultivariatePolynomials.monomial_type(::XorTX{<:IntVariable{Nr,Nc}}) where {Nr,Nc} =
    IntMonomial{Nr,Nc,UInt,ExponentsAll{Nr+2Nc,UInt}}

MultivariatePolynomials.term_type(::XorTX{<:Union{IntMonomial{Nr,Nc,I,E},<:IntMonomial{Nr,Nc,I},<:IntMonomial{Nr,Nc,<:Integer,E}}},
    ::Type{C}) where {Nr,Nc,I<:Integer,E<:AbstractExponents{<:Any,I},C} =
    Term{C,IntMonomial{Nr,Nc,I,@isdefined(E) ? E : ExponentsAll{Nr+2Nc,I}}}
MultivariatePolynomials.term_type(::XorTX{<:IntPolynomial{Cold,Nr,Nc,<:IntMonomialVector{Nr,Nc,I,<:Any,M}}},
    ::Type{C}=Cold) where {Cold,C,Nr,Nc,I<:Integer,M<:IntMonomial{Nr,Nc,I}} =
    Term{C,monomial_type(M)}

MultivariatePolynomials.polynomial_type(::XorTX{<:Term{C,<:IntMonomial{Nr,Nc,I}}}) where {C,Nr,Nc,I<:Integer} =
    IntPolynomial{C,Nr,Nc,IntMonomialVector{Nr,Nc,I,Tuple{ExponentsAll{Nr+2Nc,I},Vector{I}},IntMonomial{Nr,Nc,I,ExponentsAll{Nr+2Nc,I}}}}
MultivariatePolynomials.polynomial_type(::XorTX{<:IntPolynomial{<:Any,Nr,Nc,M}}, ::Type{C}) where {C,Nr,Nc,M<:IntMonomialVector{Nr,Nc}} =
    IntPolynomial{C,Nr,Nc,M}
MultivariatePolynomials.polynomial_type(::XorTX{M}, ::Type{C}) where {C,Nr,Nc,M<:IntMonomialVector{Nr,Nc}} =
    IntPolynomial{C,Nr,Nc,M}

MultivariatePolynomials.coefficient_type(::XorTX{IntPolynomial{C}}) where {C} = C

Base.convert(::Type{T}, x::Q) where {T<:Union{<:IntVariable,<:IntMonomial,<:IntMonomialVector,<:IntPolynomial},Q<:T} = x

MultivariatePolynomials.promote_rule_constant(::Type{C}, M::Type{<:IntMonomial}) where {C} =
    term_type(M, promote_type(C, Int))
MultivariatePolynomials.promote_rule_constant(::Type{C}, T::Type{<:Term{Cold,<:IntMonomial}}) where {C,Cold} =
    term_type(T, promote_type(C, Cold))
MultivariatePolynomials.promote_rule_constant(::Type{C}, P::Type{<:IntPolynomial{Cold}}) where {C,Cold} =
    polynomial_type(P, promote_type(T, Cold))
MultivariatePolynomials.promote_rule_constant(::Type, ::Type{<:Union{<:Term{<:Any,<:IntMonomial},<:IntPolynomial}}) = Any

function Base.promote_rule(::Type{<:Term{C1,<:IntMonomial{Nr1,Nc1,I1,E1}}}, ::Type{<:Term{C2,<:IntMonomial{Nr2,Nc2,I2,E2}}}) where
    {C1,Nr1,Nc1,I1<:Integer,E1<:AbstractExponents,C2,Nr2,Nc2,I2<:Integer,E2<:AbstractExponents}
    # make sure this will always be an error and not recurse infinitely
    (Nr1 === Nr2 && Nc1 === Nc2) || throw(MethodError(promote, (IntMonomial{Nr1,Nc1}, IntMonomial{Nr2,Nc2})))
    I = promote_type(I1, I2)
    return Term{promote_type(C1, C2),
                <:(E1 isa ExponentsAll && E2 isa ExponentsAll ? IntMonomial{Nr1,Nc1,I,ExponentsAll{Nr1+2Nc1,I}} :
                                                                IntMonomial{Nr1,Nc1,I,<:AbstractExponents{Nr1+2Nc1,I}})}
end
function Base.promote_rule(::Type{<:Term{C,<:IntMonomial{Nr1,Nc1,I1,E1}}}, ::Type{IntMonomial{Nr2,Nc2,I2,E2}}) where
    {C,Nr1,Nc1,I1<:Integer,E1<:AbstractExponents,Nr2,Nc2,I2<:Integer,E2<:AbstractExponents}
    # make sure this will always be an error and not recurse infinitely
    (Nr1 === Nr2 && Nc1 === Nc2) || throw(MethodError(promote, (IntMonomial{Nr1,Nc1}, IntMonomial{Nr2,Nc2})))
    I = promote_type(I1, I2)
    return Term{C,<:(E1 isa ExponentsAll && E2 isa ExponentsAll ? IntMonomial{Nr1,Nc1,I,ExponentsAll{Nr1+2Nc1,I}} :
                                                                  IntMonomial{Nr1,Nc1,I,<:AbstractExponents{Nr1+2Nc1,I}})}
end
function Base.promote_rule(::Type{<:Term{C,<:IntMonomial{Nr1,Nc1,I,E}}}, ::Type{<:IntVariable{Nr2,Nc2,<:Unsigned}}) where
    {C,Nr1,Nc1,I<:Integer,E<:AbstractExponents,Nr2,Nc2}
    # make sure this will always be an error and not recurse infinitely
    (Nr1 === Nr2 && Nc1 === Nc2) || throw(MethodError(promote, (IntMonomial{Nr1,Nc1}, IntVariable{Nr2,Nc2})))
    return Term{C,IntMonomial{Nr1,Nc1,I,E}}
end
function Base.promote_rule(::Type{<:IntMonomial{Nr1,Nc1,I,E}}, ::Type{<:IntVariable{Nr2,Nc2,<:Unsigned}}) where
    {Nr1,Nc1,I<:Integer,E<:AbstractExponents,Nr2,Nc2}
    # make sure this will always be an error and not recurse infinitely
    (Nr1 === Nr2 && Nc1 === Nc2) || throw(MethodError(promote, (IntMonomial{Nr1,Nc1}, IntVariable{Nr2,Nc2})))
    return IntMonomial{Nr1,Nc1,I,E}
end
function Base.promote_rule(::Type{IntVariable{Nr1,Nc1,I1}}, ::Type{IntVariable{Nr2,Nc2,I2}}) where
    {Nr1,Nc1,I1<:Unsigned,Nr2,Nc2,I2<:Unsigned}
    # make sure this will always be an error and not recurse infinitely
    (Nr1 === Nr2 && Nc1 === Nc2) || throw(MethodError(promote, (IntVariable{Nr1,Nc1}, IntVariable{Nr2,Nc2})))
    @assert I1 === I2
    return IntVariable{Nr1,Nc1,I1}
end