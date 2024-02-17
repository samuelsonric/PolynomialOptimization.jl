# The implementations here are tricky, as we also want to capture all partially specified types (UnionAll) as accurately as
# possible. For this reason, the _get_... functions are also designed to work on these types. The most "logical" way to do this
# would be for them to return TypeVar, which are then appropriately converted to UnionAll. However, anything that explicitly
# involves calling UnionAll is not inferrable. We therefore must really construct the types explicitly using curly expressions,
# which is quite cumbersome. All macros in turn create generated functions, which must know the full types at compile time, so
# the _get_... functions return a Val(upper bound) if the type is unspecified. smallest_unsigned also works with this Val form.
function find_interps(expr::Expr, items, args)
    if expr.head === :$
        symb = gensym()
        push!(args, expr.args[1])
        push!(items, symb)
        return symb
    end
    for i in eachindex(expr.args)
        if expr.args[i] isa Expr
            expr.args[i] = find_interps(expr.args[i], items, args)
        end
    end
    return expr
end

# This is really tricky. Given a type in T, we want to construct the expression that will give us this type. Unfortunately, we
# cannot simply use the expressions instead of the types, as those cannot be made into static type parameters.
# The function will really run into trouble if some of the subtypes are not loaded in the SimplePolynomials context. This
# should not happen with the intended use, but it might be possible if other weird types are used. There's no easy way around
# this; while we could certainly prefix the module name in front of the expressions, nothing tells us that the module is
# available. And getting the module from Base.loaded_modules would work, but this would no longer be statically inferrable.
function _reconstruct_expr(T, replacements=Dict{TypeVar,Symbol}())
    if T isa Union
        return Expr(:curly, :Union, _reconstruct_expr.(getproperty.((T,), propertynames(T)), (replacements,))...)
    elseif hasproperty(T, :parameters) && !isempty(T.parameters)
        return Expr(:curly, T.name.name, _reconstruct_expr.(T.parameters, (replacements,))...)
    elseif T isa UnionAll
        newname = gensym(T.var.name) # TypeVars are compared by instance, so when translating into a where expression, preserve
                                     # uniqueness
        replacements[T.var] = newname
        return Expr(:where, _reconstruct_expr(T.body, replacements),
            Expr(:comparison, _reconstruct_expr(T.var.lb, replacements), :<:, newname,
                :<:, _reconstruct_expr(T.var.ub, replacements)))
    elseif T isa TypeVar
        haskey(replacements, T) || error("Unknown type var")
        return replacements[T]
    elseif T === Union{}
        return :(Union{})
    elseif T isa DataType
        return T.name.name
    else
        return T
    end
end
_get_val(::Type{Val{T}}) where {T} = _reconstruct_expr(T)
_extract_val(::Union{Val{T},Type{Val{T}}}) where {T} = T

maketype_curly(arg, _, replacements) = arg
function maketype_curly(arg::Symbol, items, replacements)
    idx = findfirst(isequal(arg), items)
    if isnothing(idx)
        return arg
    elseif ismissing(replacements[idx])
        return :(
            $arg <: Val ? Expr(:(<:), _get_val($arg)) : $(QuoteNode(arg))
        )
    else
        return replacements[idx]
    end
end
function maketype_curly(curly::Expr, items, replacements)
    if curly.head !== :curly
        for i in 1:length(curly.args)
            curly.args[i] = maketype_curly(curly.args[i], items, replacements)
        end
        return curly
    end
    output = Expr(:let, Expr(:block,
        :(resultBody = Expr(:curly, $(QuoteNode(curly.args[1])))),
        :(result = Expr(:where, resultBody))
    ), Expr(:block))
    output_items = output.args[2].args
    # step 1: gather information about all the items that occur at this level and define replacements
    replacements = copy(replacements)
    for type in Iterators.drop(curly.args, 1)
        if type isa Symbol
            idx = findfirst(isequal(type), items)
            if !isnothing(idx)
                if ismissing(replacements[idx])
                    replacements[idx] = typevar = gensym()
                    push!(output_items, :(
                        if $type <: Val
                            push!(result.args, Expr(:(<:), $(QuoteNode(typevar)), _get_val($type)))
                            $typevar = $(QuoteNode(typevar))
                        else
                            $typevar = $(QuoteNode(type))
                        end
                    ))
                end
            end
        end
    end
    for type in Iterators.drop(curly.args, 1)
        if type isa Symbol
            if type ∈ items
                typevar = replacements[findfirst(isequal(type), items)]
            else
                typevar = QuoteNode(type)
            end
            push!(output_items, :(
                push!(resultBody.args, $typevar)
            ))
        elseif type isa Expr
            if type.head === :curly
                push!(output_items, :(
                    push!(resultBody.args, $(maketype_curly(type, items, replacements)))
                ))
            else
                push!(output_items, :(
                    push!(resultBody.args, Expr(
                        $(QuoteNode(type.head)),
                        $(maketype_curly.(type.args, (items,), (replacements,))...)
                    ))
                ))
            end
        else
            @assert(isbits(type))
            push!(output_items, :(
                push!(resultBody.args, $type)
            ))
        end
    end
    push!(output_items, :(
        length(result.args) == 1 ? resultBody : result
    ))
    return output
end

function maketype_impl(expr::Expr)
    # lift all interpolations (here, we don't use Base._lift_one_interp!, as it escapes its variables, and also not
    # PolynomialOptimization._lift_interps!, as this is meant for variables, not expressions)
    items = Symbol[]
    args = Any[]
    expr.args[2] = find_interps(expr.args[2], items, args)
    return maketype_impl(expr, items, args)
end

function maketype_impl(expr::Expr, items::Vector{Symbol}, args::Vector)
    expr.head ∈ (:(=), :function) || error("Invalid use of maketype: must be a function definition")
    # part 1: generate two functions, one for the (incomplete) type, one for the instance.
    expr2 = Expr(expr.head, deepcopy(expr.args[1])) # we don't need to copy the body yet, it might be changed
    local function_name
    # For this, we just need to deal with the function signature
    parameters = Symbol[]
    let header=expr.args[1], header2=expr2.args[1]
        header_expr = header
        header2_expr = header2
        while header_expr.head === :where
            header_expr = header_expr.args[1]
            header2_expr = header2_expr.args[1]
        end
        @assert(header_expr.head === :call)
        function_name = header_expr.args[1]
        while function_name isa Expr
            @assert(function_name.head === :(.))
            function_name = function_name.args[end]
        end
        if function_name isa QuoteNode
            function_name = function_name.value
        end
        search_for = Tuple{Int,Symbol}[]
        for i in 2:length(header_expr.args)
            header_param = header_expr.args[i]
            if header_param isa Symbol
                push!(parameters, header_param.args[1])
                continue
            end
            @assert(header_param isa Expr && header_param.head === :(::))
            if length(header_param.args) != 1
                push!(parameters, header_param.args[1])
                continue
            end
            param_type = header_param.args[1]
            param_type.head === :curly || continue
            param_type.args[1] === :XorTX || continue
            push!(search_for, (i, param_type.args[2]))
            header2_expr.args[i].args[1] = header2_expr.args[i].args[1].args[2]
        end
        if isempty(search_for)
            # Sometimes, we might just define the function in the Type{}-form and don't want it for the instance case.
            # This happens when the instance case - where the type must be fully specified - is already defined and the more
            # general overloading with unspecified types would never be called
            expr2 = nothing
        else
            parent = expr
            while header.head === :where
                for j in length(header.args):-1:2
                    where_expr = header.args[j]
                    where_expr isa Symbol && push!(parameters, where_expr)
                    where_expr isa Expr || continue
                    if where_expr.head === :comparison
                        push!(parameters, where_expr.args[3]) # a <: T <: b
                    else
                        push!(parameters, where_expr.args[1]) # T <: b or T >: b
                    end
                    where_expr.head === :(<:) || continue
                    for sf in search_for
                        if sf[2] === where_expr.args[1]
                            header_param = header_expr.args[sf[1]]
                            deleteat!(header.args, j)
                            pushfirst!(header_param.args, sf[2])
                            header_param.args[2] = :(Type{<:$(where_expr.args[2])})
                            break
                        end
                    end
                end
                if length(header.args) == 1 # empty where
                    parent.args[1] = header.args[1]
                else
                    parent = header
                end
                header = header.args[1]
            end
        end
    end
    # do we have anything to interpolate?
    body = expr.args[2]
    if isempty(items)
        if isnothing(expr2)
            return expr
        else
            push!(expr2.args, body)
            return Expr(:block, expr, expr2)
        end
    else
        # part 2: we found interpolations and replaced them by symbols. The function body now becomes trivial, calling a
        # generated function.
        impl_name = gensym(function_name)
        expr.args[2] = Expr(:block)
        for (item, arg) in zip(items, args)
            push!(expr.args[2].args, Expr(:(=), item, arg))
        end
        push!(expr.args[2].args, Expr(:call, impl_name, parameters..., items...))
        # part 4: implement the generated function
        output = Expr(:block,
            expr,
            :(@generated function $impl_name($(parameters...), $(items...))
                return $(maketype_curly(body, items, fill!(Vector{Union{Symbol,Missing}}(undef, length(items)), missing)))
            end)
        )
        if !isnothing(expr2)
            push!(expr2.args, expr.args[2])
            insert!(output.args, 1, expr2)
        end
        return output
    end
end

@doc raw"""
    @maketype <function definition>
    @maketype substitutions... <function definition>

Rewrites a function definition in a most spectacular way.
1. Assume that the function contains an unnamed parameter of the form `::XorTX{T}` with a `where {T<:...}`.
   This will be split in two functions:
   - one in which the `where` clause is removed and the parameter is changed into `T::Type{<:...}`. This form will allow calls
     with the type, but it will also allow calls with incomplete (`UnionAll`) types!
   - one that only allow for the instances to be passed; the `where` clause remains and the parameter now looks like `::T`.
   Note that this will not generate exponentially many functions: if multiple `XorTX` appear, there will be one function that
   wants all of them to be types, and one that wants all of them to be instances.
2. First form only: scans the function body for $(...) interpolations and accumulates them into `tempvar=...`, where `tempvar`
   is a unique identifier for every interpolation. Then, the second form is called, where all the accumulated substitions go
   into the `substitutions` parameters.
3. Second form, only if `substitutions` is non-empty.
   The function body must consist in a single parametric type that is implicitly returned. No other statements, no explicit
   `return`. This single type will have the form `Q{...}`.
   The function body will be completely rewritten: first, all the `substitutions` are inserted one after the other in their
   order of appearance (so later ones might depend on previous ones). Note that if an element in `substitutions` is not a
   variable assignment, it will be copied verbatim without any further effect. Then, a new function with a unique name is
   called that is created by the macro, and the return value of this function is returned.
   The newly created function will be a @generated function, its parameters are:
   - all named parameters of the function as it was originally declared (before potentially moving `where` into the params).
   - all `where` type variables from the original definition.
   - all variables that were assigned in the `substitutions`.
   It will construct the type mostly at compile time. For this, the parameters in `Q{...}` are parsed (recursing into
   subexpressions to some degree). Everything that is a `Val{R}` type will instead be replaced by `<:R`. Note that this is much
   more tricky than it seems, because `R` will anything that can go in a type parameter (isbits, concrete/abstract type,
   incomplete type (UnionAll), type union, bottom) and the macro has to extract the way to represent this thing as an Expr.
   Hopefully all cases are covered.
   The reason for this contrived procedure is that it appears to be the only way to make Julia able to actually infer the
   output type of the functions statically.

!!! example
    The function definition
    ```julia
@maketype MultivariatePolynomials.monomial_type(::XorTX{V}) where {V<:SimpleVariable} =
    SimpleMonomial{$(_get_nr(V)),$(_get_nc(V))}
    ```
    will be expanded to
    ```julia
function MultivariatePolynomials.monomial_type(::V) where {V<:SimpleVariable}
    var"##1" = _get_nr(V)
    var"##2" = _get_nc(V)
    var"##monomial_type#3"(V, var"##1", var"##2")
end
function MultivariatePolynomials.monomial_type(V::Type{<:SimpleVariable})
    var"##1" = _get_nr(V)
    var"##2" = _get_nc(V)
    var"##monomial_type#3"(V, var"##1", var"##2")
end
@generated function var"##monomial_type#3"(V, var"##1", var"##2")
    return begin
        let resultBody = Expr(:curly, :SimpleMonomial), result = Expr(:where, resultBody)
            if var"##1" <: Val
                push!(result.args, Expr(:<:, Symbol("##4"), _get_val(var"##1")))
                var"##4" = Symbol("##4")
            else
                var"##4" = Symbol("##1")
            end
            if var"##2" <: Val
                push!(result.args, Expr(:<:, Symbol("##5"), _get_val(var"##2")))
                var"##5" = Symbol("##5")
            else
                var"##5" = Symbol("##2")
            end
            push!(resultBody.args, var"##4")
            push!(resultBody.args, var"##5")
            length(result.args) == 1 ? resultBody : result
        end
    end
end
    ```
"""
macro maketype(expr...)
    if length(expr) === 1
        esc(maketype_impl(expr[1]))
    else
        items = Symbol[]
        args = Any[]
        for i in 1:length(expr) -1
            asgn = expr[i]
            if asgn isa Expr && asgn.head === :(=) && asgn.args[1] isa Symbol
                push!(items, asgn.args[1])
                push!(args, asgn.args[2])
            else
                push!(args, asgn)
            end
        end
        esc(maketype_impl(expr[end], items, args))
    end
end


@maketype MultivariatePolynomials.variable_union_type(::XorTX{V}) where {V<:SimpleVariable} = V
@maketype MultivariatePolynomials.variable_union_type(::XorTX{MTP}) where {Nr,Nc,MTP<:Union{<:SimpleMonomial{Nr,Nc},<:Term{<:Any,<:SimpleMonomial{Nr,Nc}},<:SimplePolynomial{<:Any,Nr,Nc}}} =
    SimpleVariable{Nr,Nc,$(smallest_unsigned(Nr + 2Nc))}
@maketype(
    Nr=_get_nr(MTP), Nc=_get_nc(MTP), I=smallest_unsigned(Nr isa Integer && Nc isa Integer ? Nr + 2Nc : Val(Any)),
    MultivariatePolynomials.variable_union_type(MTP::Type{<:Union{<:SimpleMonomial,<:Term{<:Any,<:SimpleMonomial},<:SimplePolynomial}}) =
        SimpleVariable{Nr,Nc,I}
)

@maketype MultivariatePolynomials.monomial_type(::XorTX{V}) where {V<:SimpleVariable} =
    SimpleMonomial{$(_get_nr(V)),$(_get_nc(V))}
@maketype MultivariatePolynomials.monomial_type(::XorTX{M}) where {M<:SimpleMonomial} = M
@maketype MultivariatePolynomials.monomial_type(T::Type{<:Term{<:Any,<:SimpleMonomial}}) =
    SimpleMonomial{$(_get_nr(T)),$(_get_nc(T)),$(_get_p(T)),$(_get_v(T))}

@maketype MultivariatePolynomials.term_type(::XorTX{V}, ::Type{C}) where {V<:SimpleVariable,C} =
    Term{C,<:SimpleMonomial{$(_get_nr(V)),$(_get_nc(V))}}
@maketype MultivariatePolynomials.term_type(::XorTX{M}, ::Type{C}) where {M<:SimpleMonomial,C} = Term{C,M}
@maketype MultivariatePolynomials.term_type(::XorTX{P}) where {P<:SimpleRealPolynomial} =
    Term{$(_get_c(P)),SimpleRealMonomial{$(_get_nr(P)),$(_get_p(P)),$(_monvectype(P))}}
@maketype MultivariatePolynomials.term_type(::XorTX{P}) where {P<:SimpleComplexPolynomial} =
    Term{$(_get_c(P)),SimpleComplexMonomial{$(_get_nc(P)),$(_get_p(P)),$(_monvectype(P))}}
@maketype MultivariatePolynomials.term_type(::XorTX{P}) where {Nr,Nc,P<:SimplePolynomial{<:Any,Nr,Nc}} =
    Term{$(_get_c(P)),SimpleMixedMonomial{Nr,Nc,$(_get_p(P)),$(_monvectype(P))}}
@maketype MultivariatePolynomials.term_type(P::Type{<:SimplePolynomial}) =
    Term{$(_get_c(P)),SimpleMonomial{$(_get_nr(P)),$(_get_nc(P)),$(_get_p(P)),$(_monvectype(P))}}

@maketype MultivariatePolynomials.polynomial_type(::XorTX{V}, ::Type{C}) where {C,V<:SimpleVariable} =
    SimplePolynomial{C,$(_get_nr(V)),$(_get_nc(V))}
@maketype MultivariatePolynomials.polynomial_type(::XorTX{M}, ::Type{C}) where {C,M<:SimpleMonomial} =
    SimplePolynomial{C,$(_get_nr(M)),$(_get_nc(M)),$(_get_p(M))}
# We must make the polynomial_type of terms stuff very explicit; putting it all into a single definition with a couple of case
# distinctions will also break inferrability.
_sindex(i::Integer) = smallest_unsigned(i)
_sindex(::TypeVar) = UInt
_sindex(i1, i2) = promote_type(_sindex(i1), _sindex(i2))
const SimpleDefaultDenseRealPolynomial{C,Nr,P<:Unsigned} = SimpleRealPolynomial{C,Nr,P,<:SimpleRealMonomialVector{Nr,P,<:Matrix{P}}}
const SimpleDefaultDenseComplexPolynomial{C,Nc,P<:Unsigned} = SimpleComplexPolynomial{C,Nc,P,<:SimpleComplexMonomialVector{Nc,P,<:Matrix{P}}}
const SimpleDefaultDenseMixedPolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleMixedMonomialVector{Nr,Nc,P,<:Matrix{P}}}
const SimpleDefaultDensePolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleMonomialVector{Nr,Nc,P,<:Matrix{P}}}
const SimpleDefaultSparseRealPolynomial{C,Nr,P<:Unsigned} = SimpleRealPolynomial{C,Nr,P,<:SimpleRealMonomialVector{Nr,P,<:AbstractSparseMatrixCSC{P,<:_sindex(Nr)}}}
const SimpleDefaultSparseComplexPolynomial{C,Nc,P<:Unsigned} = SimpleComplexPolynomial{C,Nc,P,<:SimpleComplexMonomialVector{Nc,P,<:AbstractSparseMatrixCSC{P,<:_sindex(Nc)}}}
const SimpleDefaultSparseMixedPolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleMixedMonomialVector{Nr,Nc,P,<:AbstractSparseMatrixCSC{P,<:_sindex(Nr,Nc)}}}
const SimpleDefaultSparsePolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleMonomialVector{Nr,Nc,P,<:AbstractSparseMatrixCSC{P,<:_sindex(Nr,Nc)}}}
const SimpleDefaultRealPolynomial = SimpleRealPolynomial
const SimpleDefaultComplexPolynomial = SimpleComplexPolynomial
const SimpleDefaultMixedPolynomial{C,Nr,Nc,P<:Unsigned} = SimplePolynomial{C,Nr,Nc,P,<:SimpleMixedMonomialVector{Nr,Nc,P,<:AbstractMatrix{P}}}
const SimpleDefaultPolynomial = SimplePolynomial
for id in (:Dense, :Sparse, Symbol())
    @eval begin
        @maketype(
            C=_get_c(T), Nr=_get_nr(T), P=_get_p(T),
            MultivariatePolynomials.polynomial_type(::XorTX{T}) where {T<:Term{<:Any,<:$(Symbol(:SimpleReal, id, :Monomial))}} =
                $(Symbol(:SimpleDefault, id, :RealPolynomial)){C,Nr,P}
        )
        @maketype(
            C=_get_c(T), Nc=_get_nc(T), P=_get_p(T),
            MultivariatePolynomials.polynomial_type(::XorTX{T}) where {T<:Term{<:Any,<:$(Symbol(:SimpleComplex, id, :Monomial))}} =
                $(Symbol(:SimpleDefault, id, :ComplexPolynomial)){C,Nc,P}
        )
        @maketype(
            C=_get_c(T), P=_get_p(T),
            MultivariatePolynomials.polynomial_type(::XorTX{T}) where {Nr,Nc,T<:Term{<:Any,<:$(Symbol(:Simple, id, :Monomial)){Nr,Nc}}} =
                $(Symbol(:SimpleDefault, id, :MixedPolynomial)){C,Nr,Nc,P}
        )
        @maketype(
            C=_get_c(T), P=_get_p(T),
            MultivariatePolynomials.polynomial_type(T::Type{<:Term{<:Any,<:$(Symbol(:Simple, id, :Monomial))}}) =
                $(Symbol(:SimpleDefault, id, :Polynomial)){C,<:Any,<:Any,P}
        ) # define this for the Type{} form only, as the instance form is already fully specified
    end
end
@maketype MultivariatePolynomials.polynomial_type(::XorTX{P}) where {P<:SimplePolynomial} = P
# very complicated, so let's split this off. First the fully specified case.
_get_m(::Poly) where {Nr,Nc,P<:Unsigned,M<:SimpleMonomialVector{Nr,Nc,P},Poly<:SimplePolynomial{<:Any,Nr,Nc,P,M}} = M
MultivariatePolynomials.polynomial_type(::P, ::Type{C}) where {P<:SimplePolynomial,C} =
    SimplePolynomial{C,_get_nr(P),_get_nc(P),_get_p(P),_get_m(P)}
function _replace_cvar(P::UnionAll, subs)
    sub = _replace_cvar(P.body, subs)
    sub[1] && return true, UnionAll(P.var, sub[2])
    sub = _replace_cvar(P.var.ub, subs)
    P.var.ub = sub[2]
    return sub[1], P
end
_replace_cvar(d::DataType, subs) = d.name === Base.typename(SimplePolynomial) ?
                                    (true, SimplePolynomial{subs,d.parameters[2:end]...}) : (false, d)
_replace_cvar(x::Any, _) = (false, x)
@generated function MultivariatePolynomials.polynomial_type(P::Type{<:SimplePolynomial}, ::Type{C}) where {C}
    P = P.parameters[1]
    if P isa DataType
        # still fully specified
        return :(SimplePolynomial{C,$(P.parameters[2:end]...)})
    else
        P isa UnionAll || error("Invalid parameter")
        return _reconstruct_expr(_replace_cvar(P, Symbol(C))[2])
    end
end

MultivariatePolynomials.coefficient_type(::XorTX{SimplePolynomial{C}}) where {C} = C

Base.convert(::Type{T}, x::T) where {T<:Union{<:SimpleVariable,<:SimpleMonomial,<:SimpleMonomialVector,<:SimplePolynomial}} = x

MultivariatePolynomials.promote_rule_constant(::Type{C}, M::Type{<:SimpleMonomial}) where {C} =
    term_type(M, promote_type(C, Int))
MultivariatePolynomials.promote_rule_constant(::Type{C}, T::Type{<:Term{Cold,<:SimpleMonomial}}) where {C,Cold} =
    term_type(T, promote_type(C, Cold))
MultivariatePolynomials.promote_rule_constant(::Type{C}, P::Type{<:SimplePolynomial{Cold}}) where {C,Cold} =
    polynomial_type(P, promote_type(T, Cold))
MultivariatePolynomials.promote_rule_constant(::Type, ::Type{<:Union{<:Term{<:Any,<:SimpleMonomial},<:SimplePolynomial}}) = Any

promote_known(::Val{a}, ::Val{b}) where {a,b} = promote_known(a, b)
promote_known(::Val{a}, b) where {a} = promote_known(a, b)
promote_known(a, ::Val{b}) where {b} = promote_known(a, b)
function promote_known(a, b)
    common = Base.promote_typejoin(a, b)
    return common isa DataType && !isabstracttype(common) ? common : Val(common)
end
validate_same(::Val{a}, ::Val{a}) where {a} = Val(a)
validate_same(::Val{a}, b) where {a} = b isa a ? b : error("Incompatible domains $b and <:$a")
validate_same(a, ::Val{b}) where {b} = a isa b ? a : error("Incompatible domains $a and <:$b")
validate_same(a, b) = a == b ? promote_type(typeof(a), typeof(b))(a) : error("Incompatible domains $a and $b")

# TODO (maybe): make these better, distinguish real/complex/mixed cases
@maketype(
    C1=_get_c(T1), C2=_get_c(T2), C=promote_known(C1, C2),
    Nr1=_get_nr(T1), Nr2=_get_nr(T2), Nr=validate_same(Nr1, Nr2),
    Nc1=_get_nc(T1), Nc2=_get_nc(T2), Nc=validate_same(Nc1, Nc2),
    P1=_get_p(T1), P2=_get_p(T2), P=promote_known(P1, P2),
    V1=_get_v(T1), V2=_get_v(T2), V=promote_known(V1, V1),
    Base.promote_rule(T1::Type{<:Term{<:Any,<:SimpleMonomial}}, T2::Type{<:Term{<:Any,<:SimpleMonomial}}) =
        Term{C,<:SimpleMonomial{Nr,Nc,P,V}}
)
@maketype(
    C=_get_c(T),
    Nr1=_get_nr(T), Nr2=_get_nr(M), Nr=validate_same(Nr1, Nr2),
    Nc1=_get_nc(T), Nc2=_get_nc(M), Nc=validate_same(Nc1, Nc2),
    P1=_get_p(T), P2=_get_p(M), P=promote_known(P1, P2),
    V1=_get_v(T), V2=_get_v(M), V=promote_known(V1, V2),
    Base.promote_rule(T::Type{<:Term{<:Any,<:SimpleMonomial}}, M::Type{<:SimpleMonomial}) =
        Term{C,<:SimpleMonomial{Nr,Nc,P,V}}
)
@maketype(
    Nr1=_get_nr(M), Nr2=_get_nr(V), Nr=validate_same(Nr1, Nr2),
    Nc1=_get_nc(M), Nc2=_get_nc(V), Nc=validate_same(Nc1, Nc2),
    P=_get_p(M), V_=_get_v(M),
    Base.promote_rule(M::Type{<:SimpleMonomial}, V::Type{<:SimpleVariable}) =
        SimpleMonomial{Nr,Nc,P,V_}
)
@maketype(
    C=_get_c(T),
    Nr1=_get_nr(T), Nr2=_get_nr(V), Nr=validate_same(Nr1, Nr2),
    Nc1=_get_nc(T), Nc2=_get_nc(V), Nc=validate_same(Nc1, Nc2),
    P=_get_p(T), V_=_get_v(T),
    Base.promote_rule(T::Type{<:Term{<:Any,<:SimpleMonomial}}, V::Type{<:SimpleVariable}) =
        Term{C,<:SimpleMonomial{Nr,Nc,P,V_}}
)
@maketype(
    Nr1=_get_nr(V1), Nr2=_get_nr(V2), Nr=validate_same(Nr1, Nr2),
    Nc1=_get_nc(V1), Nc2=_get_nc(V2), Nc=validate_same(Nc1, Nc2),
    Base.promote_rule(V1::Type{<:SimpleVariable}, V2::Type{<:SimpleVariable}) =
        SimpleVariable{Nr,Nc}
)