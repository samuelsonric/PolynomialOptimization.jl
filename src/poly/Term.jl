# we can just use the default implementation
_get_c(::XorTX{Term{C,<:SimpleMonomial}}) where {C} = C
_get_c(::XorTX{Term{<:Any,<:SimpleMonomial}}) = Val(Any)
_get_nr(::XorTX{Term{<:Any,<:SimpleMonomial{Nr}}}) where {Nr} = Nr
_get_nr(::XorTX{Term{<:Any,<:SimpleMonomial}}) = Val(Any)
_get_nc(::XorTX{Term{<:Any,<:SimpleMonomial{<:Any,Nc}}}) where {Nc} = Nc
_get_nc(::XorTX{Term{<:Any,<:SimpleMonomial}}) = Val(Any)
_get_p(::XorTX{Term{<:Any,<:SimpleMonomial{<:Any,<:Any,P}}}) where {P<:Unsigned} = P
_get_p(::XorTX{Term{<:Any,<:SimpleMonomial}}) = Val(Unsigned)
_get_v(::XorTX{Term{<:Any,<:SimpleMonomial{<:Any,<:Any,P,V}}}) where {P<:Unsigned,V<:AbstractVector{P}} = V
_get_v(::XorTX{Term{<:Any,<:SimpleMonomial{<:Any,<:Any,P}}}) where {P<:Unsigned} = Val(AbstractVector{P})
_get_v(::XorTX{Term{<:Any,<:SimpleMonomial}}) = Val(AbstractVector)