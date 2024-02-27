export SimpleMonomialVector, effective_nvariables, LazyMonomials, lazy_unalias

# in the matrix, the rows correspond to the exponents and the cols to the monomials
struct SimpleMonomialVector{Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P},Mr<:XorA{M},Mc<:XorA{M}} <: AbstractVector{SimpleMonomial{Nr,Nc,P}}
    exponents_real::Mr
    exponents_complex::Mc
    exponents_conj::Mc

    # internal functions, don't use
    SimpleMonomialVector{Nr,0,P,M}(exponents_real::M, ::Absent, ::Absent) where {Nr,P<:Unsigned,M<:AbstractMatrix{P}} =
        new{Nr,0,P,M,M,Absent}(exponents_real, absent, absent)

    SimpleMonomialVector{0,Nc,P,M}(::Absent, exponents_complex::M, exponents_conj::M) where
        {Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
        new{0,Nc,P,M,Absent,M}(absent, exponents_complex, exponents_conj)

    SimpleMonomialVector{Nr,Nc,P,M}(exponents_real::M, exponents_complex::M, exponents_conj::M) where
        {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
        new{Nr,Nc,P,M,M,M}(exponents_real, exponents_complex, exponents_conj)
end

function _sortedallunique(v::AbstractVector)
    for (x, y) in zip(v, Iterators.drop(v, 1))
        x == y && return false
    end
    return true
end

"""
    SimpleMonomialVector{Nr,0}(exponents_real::AbstractMatrix{<:Integer}, along...)
    SimpleMonomialVector{0,Nc}(exponents_complex::AbstractMatrix{<:Integer},
        exponents_conj::AbstractMatrix{<:Integer}, along...)
    SimpleMonomialVector{Nr,Nc}(exponents_real::AbstractMatrix{<:Integer},
        exponents_complex::AbstractMatrix{<:Integer}, exponents_conj::AbstractMatrix{<:Integer}, along...)

Creates a monomial vector, where each column corresponds to one monomial and each row is contains its exponents. The
element types of the matrices will be promoted to a common unsigned integer type.
All matrices must have the same number of columns; complex and conjugate matrices must have the same number of rows.
All matrices will be converted a common matrix type; dense matrices (or views) are possible as well as sparse matrices
(or views).
Taking views of a `SimpleMonomialVector` will return another `SimpleMonomialVector` whose exponents are the corresponding
views.
The input will be sorted; if `along` are present, those vectors will be put in the same order as the inputs.
The input must not contain duplicates.
"""
function SimpleMonomialVector{Nr,0}(exponents_real::AbstractMatrix{<:Integer}, along...) where {Nr}
    size(exponents_real, 1) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(size(exponents_real, 1))"))
    P = Unsigned(eltype(exponents_real))
    exps = convert(AbstractMatrix{P}, exponents_real)
    smv = sort_along!(SimpleMonomialVector{Nr,0,P,typeof(exps)}(exps, absent, absent), along...)[1]
    _sortedallunique(smv) || throw(ArgumentError("Monomial vector must not contain duplicates"))
    return smv
end

function SimpleMonomialVector{0,Nc}(exponents_complex::AbstractMatrix{<:Integer}, exponents_conj::AbstractMatrix{<:Integer},
    along...) where {Nc}
    size(exponents_complex, 1) == size(exponents_conj, 1) ||
        throw(ArgumentError("Complex and conjugate exponents lengths are different"))
    size(exponents_complex, 1) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(size(exponents_complex, 1))"))
    size(exponents_complex, 2) == size(exponents_conj, 2) ||
        throw(ArgumentError("Number of monomials is different"))
    P = promote_type(Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    M1 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_complex))
    M2 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_conj))
    M = promote_type(M1, M2)
    smv = sort_along!(SimpleMonomialVector{0,Nc,P,M}(
        absent, convert(M, exponents_complex), convert(M, exponents_conj)
    ), along...)[1]
    _sortedallunique(smv) || throw(ArgumentError("Monomial vector must not contain duplicates"))
    return smv
end

function SimpleMonomialVector{Nr,Nc}(exponents_real::AbstractMatrix{<:Integer}, exponents_complex::AbstractMatrix{<:Integer},
        exponents_conj::AbstractMatrix{<:Integer}, along...) where {Nr,Nc}
    size(exponents_real, 1) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(size(exponents_real, 1))"))
    size(exponents_complex, 1) == size(exponents_conj, 1) ||
        throw(ArgumentError("Complex and conjugate exponents lengths are different"))
    size(exponents_complex, 1) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(size(exponents_complex, 1))"))
    size(exponents_real, 2) == size(exponents_complex, 2) == size(exponents_conj, 2) ||
        throw(ArgumentError("Number of monomials is different"))
    P = promote_type(Unsigned(eltype(exponents_real)), Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    M1 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_real))
    M2 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_complex))
    M3 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_conj))
    M = promote_type(M1, M2, M3)
    smv = sort_along!(SimpleMonomialVector{Nr,Nc,P,M}(
        convert(M, exponents_real), convert(M, exponents_complex), convert(M, exponents_conj)
    ), along...)[1]
    _sortedallunique(smv) || throw(ArgumentError("Monomial vector must not contain duplicates"))
    return smv
end

"""
    SimpleMonomialVector(mv::AbstractVector{<:AbstractMonomialLike}, along...;
        max_exponent::Integer=maxdegree(mv), representation::Symbol=:auto, vars=variables(mv))

Creates a `SimpleMonomialVector` from a generic monomial vector that supports `MultivariatePolynomials`'s interface.
It is possible to specify the maximal exponent that the monomial vector should be able to hold explicitly (which will determine
the element type of the internal matrices).
The keyword argument `representation` determines whether a `Matrix` (for `:dense`) or a `SparseMatrixCSC` (for `:sparse`) is
chosen as the underlying representation. The default, `:auto`, will take a small sample of the monomials in the vector and from
this determine which representation is more efficient.
The keyword argument `vars` must contain all real-valued and original complex-valued (so not the conjugates) variables that
occur in the monomial vector. However, the order of this iterable (which must have a length) controls how the MP variables are
mapped to [`SimpleVariable`](@ref)s.
The input must not contain duplicates. It will be sorted; if `along` are present, those vectors will be put in the same order
as the inputs.
"""
function SimpleMonomialVector(mv::AbstractVector{<:AbstractMonomialLike}, along::AbstractVector...;
    max_exponent::Integer=maxdegree(mv), representation::Symbol=:auto,
    vars=unique!((x -> isconj(x) ? conj(x) : x).(variables(mv))))
    if representation === :auto
        sample_size = 10
        nvar_threshold_factor = 3
        # choose a sample of several terms in the polynomial: if they are sparse, i.e., if they have less than 1/3 of all
        # variables involved, we choose a sparse representation.
        sample = length(mv) ≤ sample_size ? mv : StatsBase.sample(mv, sample_size, replace=false)
        sampled_vars = 0
        for t in sample
            sampled_vars += length(effective_variables(t))
        end
        if sampled_vars * nvar_threshold_factor ≤ length(sample) * sample_size * (length(vars) + count(∘(!, isreal), vars))
            representation = :sparse
        else
            representation = :dense
        end
    end
    return SimpleMonomialVector(mv, max_exponent, representation, vars, along...)
end

# mv must be an iterable with length
"""
    SimpleMonomialVector(mv, max_exponent::Integer, representation::Symbol, vars)

Creates a `SimpleMonomialVector` from a generic iterable that gives `AbstractMonomialLike` elements. In contrast to when `mv`
is a vector, now all arguments must be provided. `representation` must be either `:dense` or `:sparse`.
"""
function SimpleMonomialVector(mv, max_exponent::Integer, representation::Symbol, vars, along::AbstractVector...)
    isempty(vars) && throw(ArgumentError("Variables must be present"))
    any(isconj, vars) && throw(ArgumentError("The variables must not contain conjuates"))
    allunique(vars) || throw(ArgumentError("Variables must not contain duplicates"))
    representation ∈ (:dense, :sparse) || throw(ArgumentError("The representation must be :dense or :sparse"))
    P = smallest_unsigned(max_exponent)

    vars_real = count(isreal, vars)
    vars_complex = count(∘(!, isreal), vars)
    n = length(mv)
    if representation === :dense
        exponents_real = iszero(vars_real) ? absent : Matrix{P}(undef, vars_real, n)
        if iszero(vars_complex)
            exponents_complex = absent
            exponents_conj = absent
        else
            exponents_complex = Matrix{P}(undef, vars_complex, n)
            exponents_conj = Matrix{P}(undef, vars_complex, n)
        end
        for (j, m) in enumerate(mv)
            i_real, i_complex = 1, 1
            for v in vars
                if isreal(v)
                    exponents_real[i_real, j] = degree(m, v)
                    i_real += 1
                else
                    exponents_complex[i_complex, j] = degree(m, v)
                    exponents_conj[i_complex, j] = degree(m, conj(v))
                    i_complex += 1
                end
            end
        end
        smv = sort_along!(
            SimpleMonomialVector{vars_real,vars_complex,P,Matrix{P}}(exponents_real, exponents_complex, exponents_conj),
            along...
        )[1]
    else
        nz_real = 0
        nz_complex = 0
        nz_conj = 0
        for m in mv
            for v in vars
                if isreal(v)
                    iszero(degree(m, v)) || (nz_real += 1)
                else
                    iszero(degree(m, v)) || (nz_complex += 1)
                    iszero(degree(m, conj(v))) || (nz_conj += 1)
                end
            end
        end
        Ti = UInt # We don't use the smallest_unsigned of what is actually required as the index type. This would imply that
                  # polynomials with different numbers of terms (or nonzero exponents in their monomials) might have different
                  # types due to different sparse matrix index types. While this would be space-optimal, it is inconvenient,
                  # as this property has no algebraic meaning.
        if !iszero(vars_real)
            colptr_real = FastVec{Ti}(buffer=n +1)
            rowval_real = FastVec{Ti}(buffer=nz_real)
            nzval_real = FastVec{P}(buffer=nz_real)
        end
        if !iszero(vars_complex)
            colptr_complex = FastVec{Ti}(buffer=n +1)
            rowval_complex = FastVec{Ti}(buffer=nz_complex)
            nzval_complex = FastVec{P}(buffer=nz_complex)
            colptr_conj = FastVec{Ti}(buffer=n +1)
            rowval_conj = FastVec{Ti}(buffer=nz_conj)
            nzval_conj = FastVec{P}(buffer=nz_conj)
        end
        for m in mv
            if !iszero(vars_real)
                unsafe_push!(colptr_real, Ti(length(rowval_real)) + one(Ti))
                i_real = one(Ti)
            end
            if !iszero(vars_complex)
                unsafe_push!(colptr_complex, Ti(length(rowval_complex)) + one(Ti))
                unsafe_push!(colptr_conj, Ti(length(rowval_conj)) + one(Ti))
                i_complex = one(Ti)
            end
            for v in vars
                d = degree(m, v)
                if isreal(v)
                    if !iszero(d)
                        unsafe_push!(rowval_real, i_real)
                        unsafe_push!(nzval_real, P(d))
                    end
                    i_real += one(Ti)
                else
                    if !iszero(d)
                        unsafe_push!(rowval_complex, i_complex)
                        unsafe_push!(nzval_complex, P(d))
                    end
                    dc = degree(m, conj(v))
                    if !iszero(dc)
                        unsafe_push!(rowval_conj, i_complex)
                        unsafe_push!(nzval_conj, P(dc))
                    end
                    i_complex += one(Ti)
                end
            end
        end
        if iszero(vars_real)
            spexponents_real = absent
        else
            @assert(nz_real == length(rowval_real))
            unsafe_push!(colptr_real, Ti(nz_real +1))
            spexponents_real = SparseMatrixCSC{P,Ti}(vars_real, n, finish!(colptr_real), finish!(rowval_real),
                finish!(nzval_real))
        end
        if iszero(vars_complex)
            spexponents_complex = absent
            spexponents_conj = absent
        else
            @assert(nz_complex == length(rowval_complex))
            @assert(nz_conj == length(rowval_conj))
            unsafe_push!(colptr_complex, Ti(nz_complex +1))
            unsafe_push!(colptr_conj, Ti(nz_conj +1))
            spexponents_complex = SparseMatrixCSC{P,Ti}(vars_complex, n, finish!(colptr_complex), finish!(rowval_complex),
                finish!(nzval_complex))
            spexponents_conj = SparseMatrixCSC{P,Ti}(vars_complex, n, finish!(colptr_conj), finish!(rowval_conj),
                finish!(nzval_conj))
        end
        smv = sort_along!(
            SimpleMonomialVector{vars_real,vars_complex,P,SparseMatrixCSC{P,Ti}}(
                spexponents_real, spexponents_complex, spexponents_conj
            ), along...
        )[1]
    end
    _sortedallunique(smv) || throw(ArgumentError("Monomial vector must not contain duplicates"))
    return smv
end

const SimpleRealMonomialVector{Nr,P<:Unsigned,M<:AbstractMatrix{P}} = SimpleMonomialVector{Nr,0,P,M,M,Absent}
const SimpleComplexMonomialVector{Nc,P<:Unsigned,M<:AbstractMatrix{P}} = SimpleMonomialVector{0,Nc,P,M,Absent,M}
const SimpleMixedMonomialVector{Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}} = SimpleMonomialVector{Nr,Nc,P,M,M,M}
const SimpleDenseMonomialVector{Nr,Nc,P<:Unsigned} = SimpleMonomialVector{Nr,Nc,P,<:DenseMatrix}
const SimpleSparseMonomialVector{Nr,Nc,P<:Unsigned} = SimpleMonomialVector{Nr,Nc,P,<:AbstractSparseMatrixCSC}
const SimpleRealDenseMonomialVector{Nr,P<:Unsigned} = SimpleRealMonomialVector{Nr,P,<:DenseMatrix}
const SimpleComplexDenseMonomialVector{Nc,P<:Unsigned} = SimpleComplexMonomialVector{Nc,P,<:DenseMatrix}
const SimpleRealSparseMonomialVector{Nr,P<:Unsigned} = SimpleRealMonomialVector{Nr,P,<:AbstractSparseMatrixCSC}
const SimpleComplexSparseMonomialVector{Nc,P<:Unsigned} = SimpleComplexMonomialVector{Nc,P,<:AbstractSparseMatrixCSC}
const SimpleDenseMonomialVectorOrView{Nr,Nc,P<:Unsigned} = SimpleMonomialVector{Nr,Nc,P,<:Union{<:DenseMatrix{P},<:(SubArray{P,2,<:DenseMatrix{P}})}}
const SimpleSparseMonomialVectorOrView{Nr,Nc,P<:Unsigned} = SimpleMonomialVector{Nr,Nc,P,<:Union{<:AbstractSparseMatrixCSC{P},<:(SubArray{P,2,<:AbstractSparseMatrixCSC{P}})}}
const SimpleRealDenseMonomialVectorOrView{Nr,P<:Unsigned} = SimpleRealMonomialVector{Nr,P,<:Union{<:DenseMatrix{P},<:(SubArray{P,2,<:DenseMatrix{P}})}}
const SimpleComplexDenseMonomialVectorOrView{Nc,P<:Unsigned} = SimpleComplexMonomialVector{Nc,P,<:Union{<:DenseMatrix{P},<:(SubArray{P,2,<:DenseMatrix{P}})}}
const SimpleRealSparseMonomialVectorOrView{Nr,P<:Unsigned} = SimpleRealMonomialVector{Nr,P,<:Union{<:AbstractSparseMatrixCSC{P},<:(SubArray{P,2,<:AbstractSparseMatrixCSC{P}})}}
const SimpleComplexSparseMonomialVectorOrView{Nc,P<:Unsigned} = SimpleComplexMonomialVector{Nc,P,<:Union{<:AbstractSparseMatrixCSC{P},<:(SubArray{P,2,<:AbstractSparseMatrixCSC{P}})}}

_get_nr(::XorTX{AbstractVector{<:SimpleMonomial{Nr}}}) where {Nr} = Nr
_get_nr(::XorTX{AbstractVector{<:SimpleMonomial}}) = Val(Any)
_get_nc(::XorTX{AbstractVector{<:SimpleMonomial{<:Any,Nc}}}) where {Nc} = Nc
_get_nc(::XorTX{AbstractVector{<:SimpleMonomial}}) = Val(Any)
_get_p(::XorTX{AbstractVector{<:SimpleMonomial{<:Any,<:Any,P}}}) where {P<:Unsigned} = P
_get_p(::XorTX{AbstractVector{<:SimpleMonomial}}) = Val(Unsigned)
_monvectype(::Type{M}) where {M<:AbstractMatrix} = Base.promote_op(view, M, Colon, Int)
_monvectype(::XorTX{SimpleMonomialVector{<:Any,<:Any,P,M}}) where {P<:Unsigned,M<:AbstractMatrix{P}} = _monvectype(M)

SortAlong.swap_items!(::Absent, _, _) = nothing
Base.@propagate_inbounds function SortAlong.swap_items!(v::SimpleMonomialVector, i, j)
    @assert(i < j)
    SortAlong.swap_items!(v.exponents_real, i, j)
    SortAlong.swap_items!(v.exponents_complex, i, j)
    SortAlong.swap_items!(v.exponents_conj, i, j)
    return
end
SortAlong.rotate_items_left!(::Absent, _, _, _) = nothing
Base.@propagate_inbounds function SortAlong.rotate_items_left!(v::SimpleMonomialVector, i, j, k)
    @assert(i < j < k)
    SortAlong.rotate_items_left!(v.exponents_real, i, j, k)
    SortAlong.rotate_items_left!(v.exponents_complex, i, j, k)
    SortAlong.rotate_items_left!(v.exponents_conj, i, j, k)
    return
end
SortAlong.can_extract(::Type{<:SimpleMonomialVector}) = false

@inline function Base.getindex(x::SimpleMonomialVector{Nr,Nc,P}, i::Integer) where {Nr,Nc,P<:Unsigned}
    @boundscheck checkbounds(x, i)
    @inbounds return SimpleMonomial{Nr,Nc,P,_monvectype(x)}(
        view(x.exponents_real, :, i), view(x.exponents_complex, :, i), view(x.exponents_conj, :, i)
     )
end
@inline function Base.getindex(x::SimpleMonomialVector{Nr,Nc,P,M}, idx) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}}
    @boundscheck checkbounds(x, idx)
    @inbounds return SimpleMonomialVector{Nr,Nc,P,Base.promote_op(getindex, M, Colon, typeof(idx))}(
        x.exponents_real[:, idx], x.exponents_complex[:, idx], x.exponents_conj[:, idx]
    )
end
@inline function Base.view(x::SimpleMonomialVector{Nr,Nc,P,M}, range) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}}
    @boundscheck checkbounds(x, range)
    @inbounds return SimpleMonomialVector{Nr,Nc,P,Base.promote_op(view, M, Colon, typeof(range))}(
        view(x.exponents_real, :, range), view(x.exponents_complex, :, range), view(x.exponents_conj, :, range)
    )
end

Base.firstindex(x::SimpleMonomialVector) = 1
Base.lastindex(x::SimpleMonomialVector) = size(x.exponents_real, 2)
Base.lastindex(x::SimpleMonomialVector{0}) = size(x.exponents_complex, 2)
Base.size(x::SimpleMonomialVector) = (size(x.exponents_real, 2),)
Base.size(x::SimpleMonomialVector{0}) = (size(x.exponents_complex, 2),)

Base.iterate(x::SimpleMonomialVector) = isempty(x) ? nothing : (@inbounds(x[1]), 1)
Base.iterate(x::SimpleMonomialVector, state::Int) = state < length(x) ? (@inbounds(x[state+1]), state +1) : nothing
Base.IteratorSize(::Type{<:SimpleMonomialVector}) = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialVector}) = Base.HasEltype()
Base.eltype(::Type{SimpleRealMonomialVector{Nr,P,M}}) where {Nr,P<:Unsigned,M<:AbstractMatrix{P}} =
    SimpleRealMonomial{Nr,P,_monvectype(M)}
Base.eltype(::Type{SimpleComplexMonomialVector{Nc,P,M}}) where {Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
    SimpleComplexMonomial{Nc,P,_monvectype(M)}
Base.eltype(::Type{SimpleMixedMonomialVector{Nr,Nc,P,M}}) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
    SimpleMixedMonomial{Nr,Nc,P,_monvectype(M)}
Base.length(x::SimpleMonomialVector) = size(x.exponents_real, 2)
Base.length(x::SimpleMonomialVector{0}) = size(x.exponents_complex, 2)

for (fun, call, def) in [
    (:extdegree, :extrema, (0, 0)),
    (:mindegree, :minimum, 0),
    (:maxdegree, :maximum, 0)
]
    eval(quote
        function MultivariatePolynomials.$fun(x::SimpleRealMonomialVector)
            isempty(x) && return $def
            return $call(arg -> Int(sum(arg, init=0)), eachcol(x.exponents_real))
        end
        function MultivariatePolynomials.$fun(x::SimpleComplexMonomialVector)
            isempty(x) && return $def
            return $call(args -> Int(sum(args[1], init=0) + sum(args[2], init=0)),
                zip(eachcol(x.exponents_complex), eachcol(x.exponents_conj)))
        end
        function MultivariatePolynomials.$fun(x::SimpleMonomialVector)
            isempty(x) && return $def
            return $call(args -> Int(sum(args[1], init=0) + sum(args[2], init=0) + sum(args[3], init=0)),
                zip(eachcol(x.exponents_real), eachcol(x.exponents_complex), eachcol(x.exponents_conj)))
        end
    end)
end
MultivariatePolynomials.extdegree_complex(x::SimpleRealMonomialVector) = extdegree(x)
MultivariatePolynomials.mindegree_complex(x::SimpleRealMonomialVector) = mindegree(x)
MultivariatePolynomials.maxdegree_complex(x::SimpleRealMonomialVector) = maxdegree(x)
MultivariatePolynomials.exthalfdegree(x::SimpleRealMonomialVector) = div.(extdegree(x), 2, RoundUp)
MultivariatePolynomials.minhalfdegree(x::SimpleRealMonomialVector) = div(mindegree(x), 2, RoundUp)
MultivariatePolynomials.maxhalfdegree(x::SimpleRealMonomialVector) = div(maxdegree(x), 2, RoundUp)
for (fun, call, def, realfn) in [
    (:extdegree_complex, :extrema, (0, 0), identity),
    (:mindegree_complex, :minimum, 0, identity),
    (:maxdegree_complex, :maximum, 0, identity),
    (:exthalfdegree, :extrema, (0, 0), expr -> :(div($expr, 2, RoundUp))),
    (:minhalfdegree, :minimum, 0, expr -> :(div($expr, 2, RoundUp))),
    (:maxhalfdegree, :maximum, 0, expr -> :(div($expr, 2, RoundUp)))
]
    eval(quote
        function MultivariatePolynomials.$fun(x::SimpleComplexMonomialVector)
            isempty(x) && return $def
            return $call(
                args -> Int(max(sum(args[1], init=0), sum(args[2], init=0))),
                zip(eachcol(x.exponents_complex), eachcol(x.exponents_conj))
            )
        end
        function MultivariatePolynomials.$fun(x::SimpleMonomialVector)
            isempty(x) && return $def
            return $call(
                args -> Int($(realfn(:(sum(args[1], init=0)))) + max(sum(args[2], init=0), sum(args[3], init=0))),
                zip(eachcol(x.exponents_real), eachcol(x.exponents_complex), eachcol(x.exponents_conj))
            )
        end
    end)
end

Base.isreal(::SimpleRealMonomialVector) = true
Base.isreal(x::SimpleMonomialVector) = all(isreal, x)

Base.in(x::SimpleMonomial{Nr,Nc,P}, v::SimpleMonomialVector{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} = searchsortedlast(v, x) > 0
function Base.in(x::SimpleMonomialProduct{Nr,Nc,P}, v::SimpleMonomialVector{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned}
    isempty(v) && return false
    # the product can only be accessed via iteration, so we'll have to implement the binary search manually
    xiterate = iterate(x)
    isnothing(xiterate) && return isnothing(iterate(first(v)))
    xdegree = degree(x)
    # first step: find the correct degree range
    degrange = searchsorted(v, _DummyMonomial(xdegree), by=degree)
    isempty(degrange) && return false
    l, u = extrema(degrange)
    # within the degree range, we can do lexicographic comparison
    # second step: find monomials within the degree range. For this, bisect until the first exponent matches. Continue
    # comparing until the first mismatch. In the next bisection, first compare all the exponents that previously matched, then
    # go on with the iteration.
    xvariable, xexponent = xiterate
    local item
    m = (l + u) >> 1
    @inbounds item = x[m]
    varidx = 1
    @inbounds itemexponents = @view(exponents(item)[1:0]) # empty comparison at the beginning
    @inbounds while l ≤ u
        # compare with previous choice in the range where we know the previous choice to be valid
        nextitem = x[m]
        nextitemexponents = @view(exponents(nextitem)[1:varidx-1])
        if nextitemexponents < itemexponents
            l = m +1
        elseif nextitemexponents > itemexponents
            u = m -1
        else
            # It is identical in this range. First, set this item to our last choice.
            item = nextitem
            # Then go on comparing the variables.
            itemiter = Iterators.drop(iterate(item), varidx -1)
            itemiterate = iterate(itemiter)
            if isnothing(itemiterate)
                l = m +1 # we don't have any exponents left at the current position, but we still need ones: we are too low
            else
                itemvariable, itemexponent = itemiterate
                if itemvariable == xvariable
                    itemexponent == xexponent || return false
                    varidx += 1
                    itemiterate = iterate(itemiter, itemiterate[2])
                    xiterate = iterate(x, xiterate[2])
                    if isnothing(xiterate)
                        isnothing(itemiterate) && return true
                        u = m -1 # we still have exponents left at the  position, but we don't need them any more: we are
                                 # too high
                    else
                        xvariable, xexponent = xiterate
                        continue
                    end
                elseif itemvariable < xvariable
                    l = m +1
                else
                    u = m -1
                end
            end
        end
    end
    return false
end

Base.conj(x::SimpleMonomialVector{Nr,Nc,P,M}) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
    SimpleMonomialVector{Nr,Nc,P,M}(x.exponents_real, x.exponents_conj, x.exponents_complex)

# zero-allocation variable vector
struct SimpleMonomialVariables{Nr,Nc,V<:SimpleVariable{Nr,Nc}} <: AbstractVector{V}
    SimpleMonomialVariables{Nr,Nc}() where {Nr,Nc} = new{Nr,Nc,SimpleVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)}}()
end

Base.IteratorSize(::Type{<:SimpleMonomialVariables{Nr,Nc}}) where {Nr,Nc} = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialVariables{<:Any,<:Any,V}}) where {V} = Base.HasEltype()
Base.length(::SimpleMonomialVariables{Nr,Nc}) where {Nr,Nc} = Nr + 2Nc
Base.size(::SimpleMonomialVariables{Nr,Nc}) where {Nr,Nc} = (Nr + 2Nc,)

@inline function Base.getindex(smv::SimpleMonomialVariables{Nr,Nc}, idx::Integer) where {Nr,Nc}
    @boundscheck checkbounds(smv, idx)
    return SimpleVariable{Nr,Nc}(idx)
end

Base.collect(::SimpleMonomialVariables{Nr,Nc}) where {Nr,Nc} = map(SimpleVariable{Nr,Nc}, 1:Nr+2Nc)

function Base.iterate(smv::SimpleMonomialVariables{Nr,Nc}, state::Int=1) where {Nr,Nc}
    state > Nr + 2Nc && return nothing
    @inbounds return (smv[state], state +1)
end

MultivariatePolynomials.variables(::XorTX{Union{<:SimpleVariable{Nr,Nc},<:SimpleMonomial{Nr,Nc},
                                                <:AbstractVector{<:SimpleMonomial{Nr,Nc}}}}) where {Nr,Nc} =
    SimpleMonomialVariables{Nr,Nc}()

MultivariatePolynomials.nvariables(::XorTX{AbstractVector{<:SimpleMonomial{Nr,Nc}}}) where {Nr,Nc} = Nr + 2Nc

struct SimpleMonomialVectorEffectiveVariables{Nr,Nc,MV<:SimpleMonomialVector{Nr,Nc}}
    mv::MV
end

Base.IteratorSize(::Type{<:SimpleMonomialVectorEffectiveVariables{Nr,Nc}}) where {Nr,Nc} = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:SimpleMonomialVectorEffectiveVariables{Nr,Nc}}) where {Nr,Nc} = Base.HasEltype()
Base.eltype(::Type{<:SimpleMonomialVectorEffectiveVariables{Nr,Nc}}) where {Nr,Nc} =
    SimpleVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)}
function Base.iterate(smvev::SimpleMonomialVectorEffectiveVariables{Nr,Nc,<:SimpleMonomialVector}, state=(1, 1)) where {Nr,Nc}
    type, idx = state
    if type ≤ 1
        @inbounds while idx ≤ Nr
            all(iszero, @view(smvev.mv.exponents_real[idx, :])) || return SimpleVariable{Nr,Nc}(idx), (1, idx +1)
            idx += 1
        end
        idx = 1
    end
    if type ≤ 2
        @inbounds while idx ≤ Nc
            all(iszero, @view(smvev.mv.exponents_copmlex[idx, :])) || return SimpleVariable{Nr,Nc}(idx + Nr), (2, idx +1)
            idx += 1
        end
        idx = 1
    end
    @inbounds while idx ≤ Nc
        all(iszero, @view(smvev.mv.exponents_conj[idx, :])) || return SimpleVariable{Nr,Nc}(idx + Nr + Nc), (3, idx +1)
    end
    return nothing
end
function Base.iterate(smvev::SimpleMonomialVectorEffectiveVariables{Nr,Nc,<:SimpleSparseMonomialVectorOrView}, state=(1, 1)) where {Nr,Nc}
    type, idx = state
    if type ≤ 1
        @inbounds while idx ≤ Nr
            for (j, r) in enumerate(rowvals(smvev.mv.exponents_real))
                # assume that explicit zeros happen so rarely that it is better not to iterate through both vectors at the same
                # time - we'll only need the nonzeros once except in rare circumstances.
                r == idx && !iszero(nonzeros(smvev.mv.exponents_real)[j]) && return SimpleVariable{Nr,Nc}(idx), (1, idx +1)
            end
            idx += 1
        end
        idx = 1
    end
    if type ≤ 2
        while idx ≤ Nc
            for (j, r) in enumerate(rowvals(smvev.mv.exponents_complex))
                r == idx && !iszero(nonzeros(smvev.mv.exponents_complex)[j]) &&
                    return SimpleVariable{Nr,Nc}(idx + Nr), (2, idx +1)
            end
            idx += 1
        end
        idx = 1
    end
    while idx ≤ Nc
        for (j, r) in enumerate(rowvals(smvev.mv.exponents_conj))
            r == idx && !iszero(nonzeros(smvev.mv.exponents_conj)[j]) &&
                return SimpleVariable{Nr,Nc}(idx + Nr + Nc), (3, idx +1)
        end
    end
    return nothing
end

MultivariatePolynomials.effective_variables(x::SimpleMonomialVector) = SimpleMonomialVectorEffectiveVariables(x)

function nzlength(iter, nreal::Integer, ncomplex::Integer, quick_exit::Union{Nothing,<:Integer})
    @specialize quick_exit
    nz_real = 0
    nz_complex = 0
    nz_conj = 0
    real_range = 1:nreal
    complex_range = nreal+1:nreal+ncomplex
    conj_range = nreal+ncomplex+1:nreal+2ncomplex
    @inbounds for m in iter
        # we add these checks to help the compiler with loop unswitching, although they really are redundant.
        if !iszero(nreal)
            for i in real_range
                iszero(m[i]) || (nz_real += 1)
            end
        end
        if !iszero(ncomplex)
            for i in complex_range
                iszero(m[i]) || (nz_complex += 1)
            end
            for i in conj_range
                iszero(m[i]) || (nz_conj += 1)
            end
        end
        if !isnothing(quick_exit) && nz_real + nz_complex + nz_conj ≥ quick_exit
            return
        end
    end
    return nz_real, nz_complex, nz_conj
end

"""
    monomials(nreal::Int, ncomplex::Int, degree::AbstractUnitRange{P};
        minmultideg=nothing, maxmultideg=nothing, representation=:auto,
        filter=exponents -> true) where {P}

Returns a [`SimpleMonomialVector`](@ref) with `nreal` real and `ncomplex` complex variables, total degrees contained in
`degree`, ordered according to `Graded{LexOrder}` and individual variable degrees varying between `minmultideg` and
`maxmultideg` (where real variables come first, then complex variables, then their conjugates).
The representation is either `:dense` or `:sparse`; if `:auto` is selected, the method will estimate (rather accurately) which
representation requires more memory and choose an appropriate one. To make this funtion type stable, use `Val(:dnese)` or
`Val(:sparse)` instead of the symbols.
The maximal exponent of the return type is chosen as the smallest unsigned integer that can still hold the largest degree
according to `degree` (ignoring `maxmultideg`).
An additional `filter` may be employed to drop monomials during the construction. Note that size estimation cannot take the
filter into account.

This method internally relies on [`MonomialIterator`](@ref). The `minmultideg` and `maxmultideg` parameters will automatically
be converted to `Vector{P}` instances.
"""
function MultivariatePolynomials.monomials(nreal::Int, ncomplex::Int, degree::AbstractUnitRange{P_};
    minmultideg::Union{Nothing,<:AbstractVector{P_},Tuple{Vararg{P_}}}=nothing,
    maxmultideg::Union{Nothing,<:AbstractVector{P_},Tuple{Vararg{P_}}}=nothing,
    representation::Union{Symbol,Val{:dense},Val{:sparse}}=:auto, filter=exponents -> true) where {P_<:Integer}
    if representation isa Symbol
        representation ∈ (:auto, :dense, :sparse) ||
            throw(ArgumentError("The representation must be :dense, :sparse, or :auto"))
    end
    (first(degree) < 0 || first(degree) > last(degree)) && throw(ArgumentError("Invalid degree specification"))
    P = Unsigned(P_)

    n = nreal + 2ncomplex
    if isnothing(minmultideg)
        minmultideg = fill(zero(P), n)
    elseif length(minmultideg) != n
        throw(DimensionMismatch("minmultideg has length $(length(minmultideg)), expected $n"))
    elseif !(minmultideg isa Vector{P})
        minmultideg = P.(collect(minmultideg))
    end
    if isnothing(maxmultideg)
        maxmultideg = fill(P(last(degree)), n)
    elseif length(maxmultideg) != n
        throw(DimensionMismatch("maxmultideg has length $(length(maxmultideg)), expected $n"))
    elseif !(maxmultideg isa Vector{P})
        maxmultideg = P.(collect(maxmultideg))
    end
    iter = MonomialIterator(P(first(degree)), P(last(degree)), minmultideg, maxmultideg, ownexponents)
    len = length(iter)
    # How to decide whether dense or sparse representation is better?
    # We take the laborious approach of counting the nonzeros in every monomial beforehand. This is costly - we need to iterate
    # twice - but we enter a shortcut: if the threshold (1/3 the total number of variables * total number of monomials) is met,
    # we immediately quit the counting and go for the dense approach. If it is not, we need to know the number of nonzeros
    # anyway; if we don't determine it beforehand, we need to grow vectors (plus we don't know the best element type for the
    # indices). So doing this the hard way isn't so bad.
    if representation === :auto
        nzl = nzlength(iter, nreal, ncomplex, len * n ÷ 3)
        if isnothing(nzl)
            dense = true
        else
            dense = false
            nz_real, nz_complex, nz_conj = nzl
        end
    elseif representation === :sparse || representation isa Val{:sparse}
        dense = false
        nz_real, nz_complex, nz_conj = nzlength(iter, nreal, ncomplex, nothing)
    else
        dense = true
    end
    if dense
        offset_complex = nreal +1
        offset_conj = nreal + ncomplex +1
        if !iszero(nreal)
            exponents_real = resizable_array(P, nreal, len)
        end
        j = 0
        if !iszero(ncomplex)
            exponents_complex = resizable_array(P, ncomplex, len)
            exponents_conj = resizable_array(P, ncomplex, len)
            if iszero(nreal)
                @inbounds for exps in iter
                    filter(exps) || continue
                    j += 1
                    copyto!(@view(exponents_complex[:, j]), 1, exps, 1, ncomplex)
                    copyto!(@view(exponents_conj[:, j]), 1, exps, offset_conj, ncomplex)
                end
                return SimpleMonomialVector{nreal,ncomplex}(matrix_delete_end!(exponents_complex, len - j),
                    matrix_delete_end!(exponents_conj, len - j))
            else
                @inbounds for exps in iter
                    filter(exps) || continue
                    j += 1
                    copyto!(@view(exponents_real[:, j]), 1, exps, 1, nreal)
                    copyto!(@view(exponents_complex[:, j]), 1, exps, offset_complex, ncomplex)
                    copyto!(@view(exponents_conj[:, j]), 1, exps, offset_conj, ncomplex)
                end
                return SimpleMonomialVector{nreal,ncomplex}(matrix_delete_end!(exponents_real, len - j),
                    matrix_delete_end!(exponents_complex, len - j), matrix_delete_end!(exponents_conj, len - j))
            end
        else
            @inbounds for exps in iter
                filter(exps) || continue
                j += 1
                copyto!(@view(exponents_real[:, j]), exps)
            end
            return SimpleMonomialVector{nreal,ncomplex}(matrix_delete_end!(exponents_real, len - j))
        end
    else
        Ti = smallest_unsigned(max(nreal, ncomplex, nz_real +1, nz_complex +1, nz_conj +1))
        if !iszero(nreal)
            colptr_real = FastVec{Ti}(buffer=len +1)
            rowval_real = FastVec{Ti}(buffer=nz_real)
            nzval_real = FastVec{P}(buffer=nz_real)
        end
        if !iszero(ncomplex)
            colptr_complex = FastVec{Ti}(buffer=len +1)
            rowval_complex = FastVec{Ti}(buffer=nz_complex)
            nzval_complex = FastVec{P}(buffer=nz_complex)
            colptr_conj = FastVec{Ti}(buffer=len +1)
            rowval_conj = FastVec{Ti}(buffer=nz_conj)
            nzval_conj = FastVec{P}(buffer=nz_conj)
        end
        range_real = 1:nreal
        range_complex = nreal+1:nreal+ncomplex
        j = 0
        for m in iter
            filter(m) || continue
            j += 1
            if !iszero(nreal)
                unsafe_push!(colptr_real, Ti(length(rowval_real)) + one(Ti))
                i_real = one(Ti)
                @inbounds for i in range_real
                    d = m[i]
                    if !iszero(d)
                        unsafe_push!(rowval_real, i_real)
                        unsafe_push!(nzval_real, P(d))
                    end
                    i_real += one(Ti)
                end
            end
            if !iszero(ncomplex)
                unsafe_push!(colptr_complex, Ti(length(rowval_complex)) + one(Ti))
                unsafe_push!(colptr_conj, Ti(length(rowval_conj)) + one(Ti))
                i_complex = one(Ti)
                @inbounds for i in range_complex
                    d = m[i]
                    if !iszero(d)
                        unsafe_push!(rowval_complex, i_complex)
                        unsafe_push!(nzval_complex, P(d))
                    end
                    dc = m[i+ncomplex]
                    if !iszero(dc)
                        unsafe_push!(rowval_conj, i_complex)
                        unsafe_push!(nzval_conj, P(dc))
                    end
                    i_complex += one(Ti)
                end
            end
        end
        if iszero(ncomplex)
            unsafe_push!(colptr_real, Ti(length(rowval_real) +1))
            return SimpleMonomialVector{nreal,ncomplex}(
                SparseMatrixCSC(nreal, j, finish!(colptr_real), finish!(rowval_real), finish!(nzval_real))
            )
        elseif iszero(nreal)
            unsafe_push!(colptr_complex, Ti(length(rowval_complex) +1))
            unsafe_push!(colptr_conj, Ti(length(rowval_conj) +1))
            return SimpleMonomialVector{nreal,ncomplex}(
                SparseMatrixCSC(ncomplex, j, finish!(colptr_complex), finish!(rowval_complex), finish!(nzval_complex)),
                SparseMatrixCSC(ncomplex, j, finish!(colptr_conj), finish!(rowval_conj), finish!(nzval_conj))
            )
        else
            unsafe_push!(colptr_real, Ti(length(rowval_real) +1))
            unsafe_push!(colptr_complex, Ti(length(rowval_complex) +1))
            unsafe_push!(colptr_conj, Ti(length(rowval_conj) +1))
            return SimpleMonomialVector{nreal,ncomplex}(
                SparseMatrixCSC(nreal, j, finish!(colptr_real), finish!(rowval_real), finish!(nzval_real)),
                SparseMatrixCSC(ncomplex, j, finish!(colptr_complex), finish!(rowval_complex), finish!(nzval_complex)),
                SparseMatrixCSC(ncomplex, j, finish!(colptr_conj), finish!(rowval_conj), finish!(nzval_conj))
            )
        end
    end
end

const _LazyMonomialsView{P} = SubArray{P,1,Vector{P},Tuple{UnitRange{Int}},true}
struct LazyMonomials{Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,P}} <: AbstractVector{SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}}
    iter::MI
    index_data::Matrix{Int}

    @doc """
        LazyMonomials{Nr,Nc}(degree::AbstractUnitRange{P}; minmultideg=nothing,
            maxmultideg=nothing[, exponents])

Constructs a memory-efficient vector of monomials that contains the same data as given by [`monomials`](@ref) (with dense
representation and no filter allowed). The monomials will be constructed on-demand and only a small precomputation is done to
be able to quickly perform the indexing operation.
If the monomials are only accessed one-at-a-time and never referenced when another one is requested, `exponents` may be set to
[`ownexponents`](@ref). Then, obtaining a monomial will not allocate any memory; instead, only the memory that represents the
monomial is changed.
    """
    function LazyMonomials{Nr,Nc}(degree::AbstractUnitRange{P};
        minmultideg::Union{Nothing,<:AbstractVector{P},Tuple{Vararg{P}}}=nothing,
        maxmultideg::Union{Nothing,<:AbstractVector{P},Tuple{Vararg{P}}}=nothing,
        exponents::Union{Nothing,OwnExponents}=nothing) where {Nr,Nc,P<:Integer}
        Pu = Unsigned(P)
        mindeg = Pu(first(degree))
        maxdeg = Pu(last(degree))
        n = Nr + 2Nc
        if isnothing(minmultideg)
            minmultideg = fill(zero(Pu), n)
        elseif length(minmultideg) != n
            throw(DimensionMismatch("minmultideg has length $(length(minmultideg)), expected $n"))
        elseif !(minmultideg isa Vector{Pu})
            minmultideg = Pu.(min.(collect(minmultideg), maxdeg))
        end
        if isnothing(maxmultideg)
            maxmultideg = fill(maxdeg, n)
        elseif length(maxmultideg) != n
            throw(DimensionMismatch("maxmultideg has length $(length(maxmultideg)), expected $n"))
        elseif !(maxmultideg isa Vector{Pu})
            maxmultideg = Pu.(min.(collect(maxmultideg), maxdeg))
        end
        iter = MonomialIterator(mindeg, maxdeg, minmultideg, maxmultideg, exponents)
        index_data = exponents_from_index_prepare(iter)
        new{Nr,Nc,Pu,typeof(iter)}(iter, index_data)
    end

    LazyMonomials{Nr,Nc,P,MI}(iter::MI, index_data::Matrix{Int}) where {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,P}} =
        new{Nr,Nc,P,MI}(iter, index_data)
end

# while we could always call length(lm.iter), index_data already contains our relevant pre-calculations
function Base.length(lm::LazyMonomials{<:Any,<:Any,P,<:MonomialIterator{<:Any,P}}) where {P<:Unsigned}
    iszero(size(lm.index_data, 2)) && return 0
    isone(size(lm.index_data, 2)) && @inbounds return lm.index_data[2] - lm.index_data[1]
    @inbounds return sum(@view(lm.index_data[lm.iter.mindeg+1:end, 1]), init=0)
end
Base.length(lm::LazyMonomials) = length(lm.iter) # fallback for RangedMonomialIterator, which already has it precomputed
Base.size(lm::LazyMonomials) = (length(lm),)
Base.IteratorSize(::Type{<:LazyMonomials}) = Base.HasLength()
Base.IteratorEltype(::Type{<:LazyMonomials}) = Base.HasEltype()
function Base.eltype(::Type{<:LazyMonomials{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned}
    M = _LazyMonomialsView{P}
    return SimpleMonomial{Nr,Nc,P,M,iszero(Nr) ? Absent : M,iszero(Nc) ? Absent : M}
end
function Base.iterate(lm::LazyMonomials{Nr,Nc,P}, args...) where {Nr,Nc,P<:Unsigned}
    result = iterate(lm.iter, args...)
    isnothing(result) && return nothing
    return SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}(
        iszero(Nr) ? absent : @view(result[1][1:Nr]),
        iszero(Nc) ? absent : @view(result[1][Nr+1:Nr+Nc]),
        iszero(Nc) ? absent : @view(result[1][Nr+Nc+1:end])
        ), result[2]
end
@inline function Base.getindex(lm::LazyMonomials{Nr,Nc,P,<:AbstractMonomialIterator{V}}, i::Integer) where {Nr,Nc,P<:Unsigned,V}
    @boundscheck checkbounds(lm, i)
    exponents = V === Nothing ? Vector{P}(undef, Nr + 2Nc) : lm.iter.exponents
    result = @inbounds exponents_from_index!(exponents, lm.iter, lm.index_data, i)
    @boundscheck @assert(result)
    return SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}(
        iszero(Nr) ? absent : @view(exponents[1:Nr]),
        iszero(Nc) ? absent : @view(exponents[Nr+1:Nr+Nc]),
        iszero(Nc) ? absent : @view(exponents[Nr+Nc+1:end])
    )
end
@inline function Base.getindex(lm::LazyMonomials{Nr,Nc,P,MI}, range::AbstractUnitRange) where
    {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,P}}
    @boundscheck checkbounds(lm, range)
    iter = RangedMonomialIterator(lm.iter, first(range), length(range), copy=true)
    return LazyMonomials{Nr,Nc,P,typeof(iter)}(iter, lm.index_data)
end
Base.@propagate_inbounds Base.getindex(lm::LazyMonomials, indices::AbstractVector) = LazySubMonomials(lm, indices, copy=true)
@inline function Base.view(lm::LazyMonomials{Nr,Nc,P,MI}, range::AbstractUnitRange) where
    {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,P}}
    @boundscheck checkbounds(lm, range)
    iter = RangedMonomialIterator(lm.iter, first(range), length(range), copy=false)
    return LazyMonomials{Nr,Nc,P,typeof(iter)}(iter, lm.index_data)
end
Base.@propagate_inbounds Base.view(lm::LazyMonomials, indices::AbstractVector) = LazySubMonomials(lm, indices, copy=false)
MultivariatePolynomials.mindegree(lm::LazyMonomials) = mindegree(lm.iter)
MultivariatePolynomials.maxdegree(lm::LazyMonomials) = maxdegree(lm.iter)
MultivariatePolynomials.extdegree(lm::LazyMonomials) = extdegree(lm.iter)
_get_iter(lm::LazyMonomials) = lm.iter

struct LazySubMonomials{Nr,Nc,P<:Unsigned,LM<:LazyMonomials{Nr,Nc,P,<:MonomialIterator{<:Any,P}},I<:(AbstractVector{IT} where IT<:Integer)} <: AbstractVector{SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}}
    parent::LM
    indices::I
    iteration::Bool

    @doc """
        LazySubMonomials(lm::LazyMonomials, indices::AbstractVector; copy::Bool)

Constructs a memory-efficient vector of monomials that corresponds to a scattered view into `lm`. The monomials will be
constructed on-demand.
If `lm` was constructed using [`ownexponents`](@ref), `copy` determines whether this view shares the same memory or allocates
new space.
Note that the object will take possession of `indices`. It is not allowed to use the vector afterwards, as it might be
modified.
Elements of this type will be automatically constructed when indexing a [`LazyMonomials`](@ref) vector with an array (where the
`copy` parameter will be set if the indexing is done by `view`).
    """
    @inline function LazySubMonomials(lm::LM, indices::AbstractVector{IT}; copy::Bool) where {Nr,Nc,P<:Unsigned,LM<:LazyMonomials{Nr,Nc,P},IT<:Integer}
        Base.require_one_based_indexing(indices)
        @boundscheck issorted(indices) || !_sortedallunique(indices) ||
            throw(ArgumentError("indices must be sorted and not contain duplicates"))
        @boundscheck @inbounds if !isempty(indices)
            first(indices) ≤ 0 && throw(BoundsError(lm, first(indices)))
            last(indices) > length(lm) && throw(BoundsError(lm, last(indices)))
        end
        if lm.iter isa RangedMonomialIterator
            δ = lm.iter.start -1
            @inbounds @simd for i in eachindex(indices)
                indices[i] += δ
            end
        end
        # Check how to go through this. In general, it takes about twice as long to calculate the exponents from the position
        # compared to doing an iteration step. So if indices is pretty dense, use the iterative method, else use direct
        # calculation (see also exponents_from_indices). This should not introduce type instabilities, so let's make it a
        # variable.
        new{Nr,Nc,P,LM,typeof(indices)}(LM(copy ? MonomialIterator(parent(lm.iter)) : parent(lm.iter), lm.index_data), indices,
            2length(indices) ≥ length(lm))
    end

    LazySubMonomials{Nr,Nc,P,LM,I}(parent::LM, indices::I, iteration::Bool) where
        {Nr,Nc,P<:Unsigned,LM<:LazyMonomials{Nr,Nc,P,<:MonomialIterator{<:Any,P}},I<:(AbstractVector{IT} where IT<:Integer)} =
        new{Nr,Nc,P,LM,I}(parent, indices, iteration)
end

Base.@propagate_inbounds LazySubMonomials(lsm::LazySubMonomials, indices::AbstractVector{IT};
    copy::Bool) where {IT<:Integer} = LazySubMonomials(lsm.parent, @view(lsm.indices[indices]); copy)

Base.length(lsm::LazySubMonomials) = length(lsm.indices)
Base.size(lsm::LazySubMonomials) = (length(lsm),)
Base.IteratorSize(::Type{<:LazySubMonomials}) = Base.HasLength()
Base.IteratorEltype(::Type{<:LazySubMonomials}) = Base.HasEltype()
Base.eltype(::Type{<:LazySubMonomials{Nr,Nc,P,LM}}) where {Nr,Nc,P<:Unsigned,LM<:LazyMonomials{Nr,Nc,P}} = eltype(LM)
function Base.iterate(lsm::LazySubMonomials)
    isempty(lsm.indices) && return nothing
    if lsm.iteration
        result, subi = iterate(lsm.parent)
        for _ in 2:@inbounds(first(lsm.indices))
            result, subi = iterate(lsm.parent, subi)
        end
        return result, (1, subi)
    else
        @inbounds return lsm.parent[first(lsm.indices)], 1
    end
end
function Base.iterate(lsm::LazySubMonomials, state::Tuple{Integer,<:Any})
    @assert(lsm.iteration)
    oldpos, iterstate = state
    oldpos ≥ length(lsm.indices) && return nothing
    result, iterstate = iterate(lsm.parent, iterstate)
    @inbounds for _ in lsm.indices[oldpos]+1:lsm.indices[oldpos+1]-1
        result, iterstate = iterate(lsm.parent, iterstate)
    end
    return result, (oldpos +1, iterstate)
end
function Base.iterate(lsm::LazySubMonomials, state::Int)
    @assert(!lsm.iteration)
    state ≥ length(lsm.indices) && return nothing
    @inbounds return lsm.parent[lsm.indices[state+1]], state +1
end
Base.@propagate_inbounds Base.getindex(lsm::LazySubMonomials, i::Integer) = lsm.parent[lsm.indices[i]]
Base.@propagate_inbounds Base.getindex(lsm::LazySubMonomials, range::Union{<:AbstractRange,<:AbstractVector}) =
    LazySubMonomials(lsm.parent, lsm.indices[range], copy=true)
Base.@propagate_inbounds Base.view(lsm::LazySubMonomials, range::Union{<:AbstractRange,<:AbstractVector}) =
    LazySubMonomials(lsm.parent, lsm.indices[range], copy=false)
MultivariatePolynomials.mindegree(lsm::LazySubMonomials) = isempty(lsm.indices) ? 0 : degree(first(lsm))
MultivariatePolynomials.maxdegree(lsm::LazySubMonomials) = isempty(lsm.indices) ? 0 : degree(lsm[length(lsm)])
MultivariatePolynomials.extdegree(lsm::LazySubMonomials) =
    isempty(lsm.indices) ? (0, 0) : (degree(first(iter)), degree(lsm[length(lsm)]))
_get_iter(lsm::LazySubMonomials) = _get_iter(lsm.parent)

const LazyMonomialsUnion{Nr,Nc,P<:Unsigned} =
    Union{<:LazyMonomials{Nr,Nc,P},<:LazySubMonomials{Nr,Nc,P,<:LazyMonomials{Nr,Nc,P,<:MonomialIterator{<:Any,P}}}}

struct LazyMonomialsEffectiveVariables{Nr,Nc,LM}
    lm::LM
end

Base.IteratorSize(::Type{<:LazyMonomialsEffectiveVariables{Nr,Nc}}) where {Nr,Nc} = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:LazyMonomialsEffectiveVariables{Nr,Nc}}) where {Nr,Nc} = Base.HasEltype()
Base.eltype(::Type{<:LazyMonomialsEffectiveVariables{Nr,Nc}}) where {Nr,Nc} =
    SimpleVariable{Nr,Nc,smallest_unsigned(Nr + 2Nc)}
function Base.iterate(lmev::LazyMonomialsEffectiveVariables{Nr,Nc,<:(LazyMonomials{Nr,Nc,P,<:MonomialIterator{<:Any,P}} where {P<:Unsigned})}) where {Nr,Nc}
    isempty(lmev.lm) && return nothing
    return iterate(lmev, (0,))
end
function Base.iterate(lmev::LazyMonomialsEffectiveVariables{Nr,Nc,<:Union{<:(LazyMonomials{Nr,Nc,P,<:RangedMonomialIterator{<:Any,P}} where {P<:Unsigned}),
                                                                          <:LazySubMonomials{Nr,Nc}}}) where {Nr,Nc}
    isempty(lmev.lm) && return nothing
    iter = parent(_get_iter(lmev.lm))
    return iterate(lmev, (0, all(∘(!, iszero), iter.minmultideg) ? iter.minmultideg : copy(iter.minmultideg),
                          Matrix{Int}(undef, iter.maxdeg +1, 2)))
end
function Base.iterate(lmev::LazyMonomialsEffectiveVariables{Nr,Nc}, state) where {Nr,Nc}
    iter = parent(_get_iter(lmev.lm))
    idx = state[1]
    rest = state[2:end]
    @inbounds while true
        idx = findnext(∘(!, iszero), iter.maxmultideg, idx +1)
        isnothing(idx) && return nothing
        if _effective_variables_has(lmev.lm, idx, rest..., Val(:unsafe))
            return SimpleVariable{Nr,Nc}(idx), (idx, rest...)
        end
    end
end

function _effective_variables_has(lm::LazyMonomials{<:Any,<:Any,<:Any,<:MonomialIterator},
    idx::Integer, ::Check=Val(:check)) where {Check<:Union{Val{:check},Val{:bounds},Val{:empty},Val{:unsafe}}}
    iter = _get_iter(lm)
    Check <: Union{Val{:check},Val{:bounds}} && checkbounds(iter.minmultideg, idx)
    if Check <: Union{Val{:check},Val{:empty}}
        isempty(lm) && return false
        @inbounds iszero(iter.maxmultideg[idx]) && return false
    end
    @inbounds return iter.Σminmultideg - iter.minmultideg[idx] < iter.maxdeg # only != should be necessary, as lm is nonempty
end
function _effective_variables_has(lm::LazyMonomials{<:Any,<:Any,<:Any,<:RangedMonomialIterator},
    idx::Integer, tmpminmultideg::AbstractVector, lengthcache::AbstractMatrix{Int},
    ::Check=Val(:check)) where {Check<:Union{Val{:check},Val{:bounds},Val{:empty},Val{:unsafe}}}
    riter = _get_iter(lm)
    iter = parent(riter)
    if Check <: Union{Val{:check},Val{:bounds}}
        checkbounds(iter.minmultideg, idx)
        tmpminmultideg == iter.minmultideg || throw(ArgumentError("tmpminmultideg must be a copy of minmultideg"))
        Base.require_one_based_indexing(tmpminmultideg)
        size(lengthcache) == (iter.maxdeg +1, 2) || throw(ArgumentError("Wrong dimensions of lengthcache"))
    end
    if Check <: Union{Val{:check},Val{:empty}}
        isempty(lm) && return false
        iszero(iter.maxmultideg[idx]) && return false
    end
    @inbounds begin
        iszero(tmpminmultideg[idx]) || return true
        index_data = lm.index_data
        tmpminmultideg[idx] = 1
        tmpiter = typeof(iter)(iter.mindeg, iter.maxdeg, tmpminmultideg, iter.maxmultideg, iter.exponents,
            iter.Σminmultideg + one(iter.Σminmultideg), iter.Σmaxmultideg)
        # tmpiter only gives the items that we actually need to certify existence of the variable; so we'll do a binary
        # search within this iterator.
        minindex = riter.start
        maxindex = riter.start + riter.length -1
        lo = 1
        hi = length(tmpiter, lengthcache)
        while lo ≤ hi
            mid = Base.midpoint(lo, hi)
            miditem = tmpiter[mid]
            miditemindex = monomial_index(miditem, iter, index_data)
            @assert(!iszero(miditemindex)) # tmpiter is a refinement of iter, so it must be there
            if miditemindex < minindex
                lo = mid +1
            elseif miditemindex > maxindex
                hi = mid -1
            else
                break
            end
        end
        tmpminmultideg[idx] = 0
        return lo ≤ hi
    end
end
function _effective_variables_has(lsm::LazySubMonomials, idx::Integer, tmpminmultideg::AbstractVector,
    lengthcache::AbstractMatrix{Int}, ::Check=Val(:check)) where {Check<:Union{Val{:check},Val{:bounds},Val{:empty},Val{:unsafe}}}
    iter = _get_iter(lsm)
    if Check <: Union{Val{:check},Val{:bounds}}
        checkbounds(iter.minmultideg, idx)
        tmpminmultideg == iter.minmultideg || throw(ArgumentError("tmpminmultideg must be a copy of minmultideg"))
        Base.require_one_based_indexing(tmpminmultideg)
        size(lengthcache) == (iter.maxdeg +1, 2) || throw(ArgumentError("Wrong dimensions of lengthcache"))
    end
    if Check <: Union{Val{:check},Val{:empty}}
        isempty(lsm) && return false
        iszero(iter.maxmultideg[idx]) && return false
    end
    @inbounds begin
        iszero(tmpminmultideg[idx]) || return true
        indices = lsm.indices
        index_data = lsm.parent.index_data
        # The submonomials contain any scattered list of items, but it is sorted. So it is efficient to check membership of an
        # index in the submonomials.
        # Idea: we use tmpiter (which only gives items that have the required variable present) to get our first candidate for
        # such a monomial; if its index with respect to iter is in the indices, we are done.
        # If it is not, we do an exponential search in order to find the most distant monomial in tmpiter whose distance in
        # iter is the same - this gives us a range of indices in which all monomials will have the required variable set. We
        # can efficiently check for intersection of this range with indices (this also allows to reduce the part of indices
        # that have to be looked at later).
        # If the intersection was empty, we continue on with the next item.
        # This is quite efficient unless the iterator is chosen such that we can expect those ranges to be approximately of
        # size 1.
        # And if the list of indices is very small compared to the number of items in tmpiter, we can instead just travel
        # through the indices and check whether they have the variable.
        tmpminmultideg[idx] = 1
        tmpiter = typeof(iter)(iter.mindeg, iter.maxdeg, tmpminmultideg, iter.maxmultideg, iter.exponents,
            iter.Σminmultideg + one(iter.Σminmultideg), iter.Σmaxmultideg)
        lotmpiter = 1
        maxtmpiter = length(tmpiter, lengthcache)
        loindices = 1
        hiindices = length(indices)
        # Iteration requires hiindices-1 operations
        # Check requires at best log(maxtmpiter) iterations, at worst maxtmpiter through tmpiter with (smaller and smaller)
        # binary searches through no more than log(hiindices-1) items in indices.
        # hiindices < log(maxtmpiter) * log(hiindices -1) ⇒ 2^hiindices < maxtmpiter + hiindices -1
        # hiindices < maxtmpiter * log(hiindices -1) ⇒ 2^(hiindices - maxtmpiter) < hiindices -1
        # Let's take a crude heuristic between log(maxtmpiter) and maxtmpiter: 2log(maxtmpiter)
        if 1 << (hiindices -1) < maxtmpiter^2 + hiindices -1
            for mon in lsm
                if !iszero(exponents(mon)[idx])
                    tmpminmultideg[idx] = 0
                    return true
                end
            end
        else
            while lotmpiter ≤ maxtmpiter
                startitem = tmpiter[lotmpiter]
                startitemiterindex = monomial_index(startitem, iter, index_data)
                @assert(!iszero(startitemiterindex)) # tmpiter is a refinement of iter
                loindices = searchsortedfirst(indices, startitemiterindex, loindices, hiindices, Base.Forward)
                # loindices: first position in indices ≥ startitemiterindex
                loindices > hiindices && break
                if indices[loindices] == startitemiterindex
                    tmpminmultideg[idx] = 0
                    return true
                end
                # not found yet -> exponential search (assume that the relevant item is closer to the starting point than the
                # end of the list). Exponential part first.
                local stopitemiterindex
                hitmpiter = lotmpiter +1
                while hitmpiter ≤ maxtmpiter
                    stopitem = tmpiter[hitmpiter]
                    stopitemiterindex = monomial_index(stopitem, iter, index_data)
                    @assert(!iszero(stopitemiterindex))
                    δ = (stopitemiterindex - startitemiterindex) - (hitmpiter - lotmpiter)
                    if iszero(δ)
                        # all elements so far have the variable present, let's double the search space.
                        hitmpiter = lotmpiter + 2(hitmpiter - lotmpiter)
                    else
                        @assert(δ > 0)
                        hitmpiter -= 1
                        break
                    end
                end
                if lotmpiter == hitmpiter
                    # the next item already doesn't have the variable any more, short-circuit
                    lotmpiter += 1
                    continue
                elseif lotmpiter == hitmpiter -1
                    # in the +1 case, there was only a single item found. In this case, the exponential step only included
                    # a single item, which we already know to be valid. So there's no need to bisect the remaining interval
                    # again to check where we are.
                    # However, steopitemiterindex was overwritten with an invalid item, we need to restore it. Fortunately,
                    # this is much easier now, as we know that the start and stop item are adjacent in iter.
                    stopitemiterindex = startitemiterindex + 1
                else
                    lotmpiter, hitmpiter = Base.midpoint(lotmpiter, hitmpiter), min(hitmpiter, maxtmpiter)
                    # Binary part: find last item in the range for which this still holds
                    hitmpiter = let lo=lotmpiter-1, hi=hitmpiter+1
                        while lo < hi -1
                            m = Base.midpoint(lo, hi)
                            stopitem = tmpiter[m]
                            stopitemiterindex = monomial_index(stopitem, iter, index_data)
                            @assert(!iszero(stopitemiterindex))
                            δ = (stopitemiterindex - startitemiterindex) - (m - lotmpiter)
                            if iszero(δ)
                                lo = m
                            else
                                @assert(δ > 0)
                                hi = m
                            end
                        end
                        lo
                    end
                end
                if hitmpiter < lotmpiter
                    break
                else
                    stopinindices = searchsortedlast(indices, stopitemiterindex, loindices, hiindices, Base.Forward)
                    # stopinindices: last position in indices ≤ stopitemiterindex
                    if stopinindices > loindices
                        tmpminmultideg[idx] = 0
                        return true
                    else
                        # indices does not contain anything in startitemindex:stopitemindex. Go to the next range of
                        # monomials.
                        lotmpiter = hitmpiter +1
                    end
                end
            end
        end
        tmpminmultideg[idx] = 0
        return false
    end
end

MultivariatePolynomials.effective_variables(x::LazyMonomialsUnion{Nr,Nc}) where {Nr,Nc} =
    LazyMonomialsEffectiveVariables{Nr,Nc,typeof(x)}(x)

"""
    lazy_unalias(lm::AbstractVector)

Makes sure that for a vector of SimpleMonomials, the results of lm[i] and unalias(lm)[j] are distinct whenever the elements
are. This is intended to be used for [`LazyMonomials`](@ref), where by setting `exponents=ownexponents` extracting any monomial
will always write to the same memory location. `unalias` will then produce a second iterator, identical in all respects except
for the memory location (the second iteration will still have `ownexponents` set, but to a different temporary vector). For all
other types of vectors, `unalias` is an identity.
"""
lazy_unalias(lm::LazyMonomials{Nr,Nc,P,MI}) where {Nr,Nc,P<:Unsigned,MI<:MonomialIterator{<:Any,P}} =
    LazyMonomials{Nr,Nc,P,MI}(MonomialIterator(lm.iter), lm.index_data)
lazy_unalias(lm::LazyMonomials{Nr,Nc,P,MI}) where {Nr,Nc,P<:Unsigned,MI<:RangedMonomialIterator{<:Any,P}} =
    LazyMonomials{Nr,Nc,P,MI}(RangedMonomialIterator(lm.iter), lm.index_data)
lazy_unalias(lsm::LazySubMonomials{Nr,Nc,P,LM,I}) where
    {Nr,Nc,P<:Unsigned,LM<:LazyMonomials{Nr,Nc,P,<:MonomialIterator{<:Any,P}},I<:(AbstractVector{IT} where IT<:Integer)} =
    LazySubMonomials{Nr,Nc,P,LM,I}(lazy_unalias(lsm.parent), lsm.indices, lsm.iteration)
lazy_unalias(v::AbstractVector) = v

_effective_nvariables_has(name, field, k, ::Type{<:SimpleDenseMonomialVectorOrView}) = quote
    !all(iszero, view($name.$field, $k, :))
end
_effective_nvariables_has(name, field, k, ::Type{<:SimpleSparseMonomialVectorOrView}) = quote
    let rvs=rowvals($name.$field), nzs=nonzeros($name.$field), ret=false, find=isequal($k)
        idx = findfirst(find, rvs)
        while !isnothing(idx)
            @inbounds if !iszero(nzs[idx])
                ret = true
                break
            end
            idx = findnext(find, rvs, idx +1)
        end
        ret
    end
end
_effective_nvariables_has(name, field, k, ::Type{<:AbstractArray{E}}) where {E} = quote
    let found=false
        for xₖ in $name
            $(_effective_nvariables_has(:xₖ, field, k, E)) && (found = true; break)
        end
        found
    end
end
_effective_nvariables_has(name, field, k, ::Type{<:LazyMonomialsUnion{Nr,Nc}}) where {Nr,Nc} = quote
    _effective_variables_has($name,
        $(field === :exponents_real ? :k :
            (field === :exponents_complex ? :(k + Nr) : :(k + Nr + Nc))
        ), $(Symbol(:prep, name))..., Val(:empty))
end
_effective_nvariables_prepare(_, _) = :(nothing)
_effective_nvariables_prepare(name, ::Type{<:LazyMonomials{<:Any,<:Any,<:Any,<:MonomialIterator}}) =
    :($(Symbol(:prep, name)) = ())
_effective_nvariables_prepare(name, ::Type{<:Union{<:LazyMonomials{<:Any,<:Any,<:Any,<:RangedMonomialIterator},
                                                                   <:LazySubMonomials}}) =
    :($(Symbol(:prep, name)) = let iter=parent(_get_iter($name))
        (all(∘(!, iszero), iter.minmultideg) ? iter.minmultideg : copy(iter.minmultideg),
         Matrix{Int}(undef, iter.maxdeg +1, 2))
    end)
"""
    effective_nvariables(x::Union{<:SimpleMonomialVector{Nr,Nc},
                                  <:AbstractArray{<:SimpleMonomialVector{Nr,Nc}}}...) where {Nr,Nc}

Calculates the number of effective variable of its arguments: there are at most `Nr + 2Nc` variables that may occur in any
of the monomial vectors or arrays of monomial vectors in the arguments. This function calculates efficiently the number of
variables that actually occur at least once anywhere in any argument.
"""
@generated function effective_nvariables(x::Union{<:SimpleMonomialVector{Nr,Nc},<:AbstractArray{<:SimpleMonomialVector{Nr,Nc}},
                                                  <:LazyMonomialsUnion{Nr,Nc},
                                                  <:(AbstractArray{LM} where LM<:LazyMonomialsUnion{Nr,Nc})}...) where {Nr,Nc}
    n = length(x)
    quote
        $((_effective_nvariables_prepare(:(x[$i]), x[i]) for i in 1:n)...)
        i = 0
        for k in 1:$Nr
            $(Expr(:||, (_effective_nvariables_has(:(x[$i]), :exponents_real, :k, x[i]) for i in 1:n)...)) && (i += 1)
        end
        for k in 1:$Nc
            $(Expr(:||, (_effective_nvariables_has(:(x[$i]), :exponents_complex, :k, x[i]) for i in 1:n)...)) && (i += 1)
        end
        # two loops should be more efficient, as the memory regions are closer
        for k in 1:$Nc
            $(Expr(:||, (_effective_nvariables_has(:(x[$i]), :exponents_conj, :k, x[i]) for i in 1:n)...)) && (i += 1)
        end
        return i
    end
end

function Base.intersect(a::MV, b::MV) where {Nr,Nc,P<:Unsigned,MV<:SimpleDenseMonomialVectorOrView{Nr,Nc,P}}
    minlen = min(length(a), length(b))
    exponents_real = iszero(Nr) ? absent : resizable_array(P, Nr, minlen)
    exponents_complex = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    exponents_conj = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    iszero(minlen) && return SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(exponents_real, exponents_complex, exponents_conj)
    i = 1
    rem_i = length(a)
    j = 1
    rem_j = length(b)
    k = 1
    @inbounds while true
        if a[i] == b[j]
            iszero(Nr) || copyto!(@view(exponents_real[:, k]), @view(a.exponents_real[:, i]))
            if !iszero(Nc)
                copyto!(@view(exponents_complex[:, k]), @view(a.exponents_complex[:, i]))
                copyto!(@view(exponents_conj[:, k]), @view(a.exponents_conj[:, i]))
            end
            i += 1
            iszero(rem_i -= 1) && break
            j += 1
            iszero(rem_j -= 1) && break
            k += 1
        elseif a[i] < b[j]
            i += 1
            iszero(rem_i -= 1) && break
        else
            j += 1
            iszero(rem_j -= 1) && break
        end
    end
    del = minlen - k +1
    return SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(
        matrix_delete_end!(exponents_real, del),
        matrix_delete_end!(exponents_complex, del),
        matrix_delete_end!(exponents_conj, del)
    )
end

function Base.intersect(a::MV, b::MV) where {Nr,Nc,P<:Unsigned,MV<:SimpleSparseMonomialVectorOrView{Nr,Nc,P}}
    minlen = min(length(a), length(b))
    if !iszero(Nr)
        Ti = SparseArrays.indtype(a.exponents_real)
        nz_real = min(length(rowvals(a.exponents_real)), length(rowvals(b.exponents_real)))
        colptr_real = FastVec{Ti}(buffer=minlen +1)
        rowval_real = FastVec{Ti}(buffer=nz_real)
        nzval_real = FastVec{P}(buffer=nz_real)
    end
    if !iszero(Nc)
        Ti = SparseArrays.indtype(a.exponents_complex) # won't change anything if !iszero(nreal)
        nz_complex = min(length(rowvals(a.exponents_complex)), length(rowvals(b.exponents_complex)))
        nz_conj = min(length(rowvals(a.exponents_conj)), length(rowvals(b.exponents_conj)))
        colptr_complex = FastVec{Ti}(buffer=minlen +1)
        rowval_complex = FastVec{Ti}(buffer=nz_complex)
        nzval_complex = FastVec{P}(buffer=nz_complex)
        colptr_conj = FastVec{Ti}(buffer=len +1)
        rowval_conj = FastVec{Ti}(buffer=nz_conj)
        nzval_conj = FastVec{P}(buffer=nz_conj)
    end
    i = 1
    rem_i = length(a)
    j = 1
    rem_j = length(b)
    iszero(rem_i) || iszero(rem_j) || @inbounds while true
        if a[i] == b[i]
            if !iszero(Nr)
                unsafe_push!(colptr_real, length(rowval_real) +1)
                unsafe_append!(rowval_real, rowvals(@view(a.exponents_real[:, i])))
                unsafe_append!(nzval_real, nonzeros(@view(a.exponents_real[:, i])))
            end
            if !iszero(Nc)
                unsafe_push!(colptr_complex, length(rowval_complex) +1)
                unsafe_append!(rowval_complex, rowvals(@view(a.exponents_complex[:, i])))
                unsafe_append!(nzval_complex, nonzeros(@view(a.exponents_complex[:, i])))
                unsafe_push!(colptr_conj, length(rowval_conj) +1)
                unsafe_append!(rowval_conj, rowvals(@view(a.exponents_conj[:, i])))
                unsafe_append!(nzval_conj, nonzeros(@view(a.exponents_conj[:, i])))
            end
            i += 1
            iszero(rem_i -= 1) && break
            j += 1
            iszero(rem_j -= 1) && break
        elseif a[i] < b[j]
            i += 1
            iszero(rem_i -= 1) && break
        else
            j += 1
            iszero(rem_j -= 1) && break
        end
    end
    iszero(Nr) || unsafe_push!(colptr_real, length(rowval_real) +1)
    if !iszero(Nc)
        unsafe_push!(colptr_complex, length(rowval_complex) +1)
        unsafe_push!(colptr_conj, length(rowval_conj) +1)
        @assert(length(colptr_complex) == length(colptr_conj))
    end
    iszero(Nr) || iszero(Nc) || @assert(length(colptr_real) == length(colptr_complex))
    return SimpleMonomialVector{Nr,Nc,P,SparseMatrixCSC{P,Ti}}(
        iszero(Nr) ? absent : SparseMatrixCSC{P,Ti}(Nr, length(colptr_real) -1, finish!(colptr_real), finish!(rowval_real),
                                                    finish!(nzval_real)),
        iszero(Nc) ? absent : SparseMatrixCSC{P,Ti}(Nc, length(colptr_complex) -1, finish!(colptr_complex),
                                                    finish!(rowval_complex), finish!(nzval_complex)),
        iszero(Nc) ? absent : SparseMatrixCSC{P,Ti}(Nc, length(colptr_conj) -1, finish!(colptr_conj), finish!(rowval_conj),
                                                    finish!(nzval_conj))
    )
end

mutable struct SortedIteratorIntersection{I1,I2,X}
    const a::I1
    const b::I2
    data::Union{Missing,X}

    SortedIteratorIntersection{X}(a::I1, b::I2) where {I1,I2,X} = new{I1,I2,X}(a, b, missing)
end

Base.IteratorSize(::Type{<:SortedIteratorIntersection}) = Base.HasLength()
function Base.IteratorEltype(::Type{<:SortedIteratorIntersection{I1,I2}}) where {I1,I2}
    e1 = Base.IteratorEltype(I1)
    e2 = Base.IteratorEltype(I2)
    return e1 === e2 ? e1 : Base.EltypeUnknown()
    error("Iterators are incompatible")
end
function Base.eltype(::Type{<:SortedIteratorIntersection{I1,I2}}) where {I1,I2}
    t1 = eltype(I1)
    t2 = eltype(I2)
    return Base.promote_typejoin(t1, t2)
end
function Base.iterate(iter::SortedIteratorIntersection, state=nothing)
    i1 = isnothing(state) ? iterate(iter.a) : iterate(iter.a, state[1])
    isnothing(i1) && return nothing
    i2 = isnothing(state) ? iterate(iter.b) : iterate(iter.b, state[2])
    isnothing(i2) && return nothing
    while true
        if i1[1] == i2[1]
            return i1[1], (i1[2], i2[2])
        elseif i1[1] < i2[1]
            i1 = iterate(iter.a, i1[2])
            isnothing(i1) && return nothing
        else
            i2 = iterate(iter.b, i2[2])
            isnothing(i2) && return nothing
        end
    end
end
Base.length(iter::SortedIteratorIntersection) = length(collect(iter))
function Base.collect(iter::SortedIteratorIntersection)
    if ismissing(iter.data)
        iter.data = @invoke collect(iter::Any)
    end
    return iter.data
end
function Base.collect(iter::SortedIteratorIntersection{<:LazyMonomialsUnion{Nr,Nc,P},
                                                       <:SimpleDenseMonomialVectorOrView{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned}
    ismissing(iter.data) || return iter.data
    a, b = iter.a, iter.b
    astate = iterate(a)
    minlen = isnothing(astate) ? 0 : min(length(a), length(b))
    exponents_real = iszero(Nr) ? absent : resizable_array(P, Nr, minlen)
    exponents_complex = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    exponents_conj = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    if iszero(minlen)
        iter.data = SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(exponents_real, exponents_complex, exponents_conj)
        return iter.data
    end
    k = 1
    @inbounds for mon in b
        while astate[1] < mon
            astate = iterate(a, astate[2])
            isnothing(astate) && @goto done
        end
        if astate[1] == mon
            copyto!(@view(exponents_real[:, k]), mon.exponents_real)
            copyto!(@view(exponents_complex[:, k]), mon.exponents_complex)
            copyto!(@view(exponents_conj[:, k]), mon.exponents_conj)
            k += 1
        end
    end
    @label done
    del = minlen - k +1
    iter.data = SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(
        matrix_delete_end!(exponents_real, del),
        matrix_delete_end!(exponents_complex, del),
        matrix_delete_end!(exponents_conj, del)
    )
    return iter.data
end
function Base.collect(iter::SortedIteratorIntersection{<:LazyMonomialsUnion{Nr,Nc,P},
                                                       <:SimpleSparseMonomialVectorOrView{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned}
    ismissing(iter.data) || return iter.data
    a, b = iter.a, iter.b
    astate = iterate(a)
    minlen = isnothing(astate) ? 0 : min(length(a), length(b))
    if !iszero(Nr)
        Ti = SparseArrays.indtype(b.exponents_real)
        nz_real = length(rowvals(b.exponents_real))
        colptr_real = FastVec{Ti}(buffer=minlen +1)
        rowval_real = FastVec{Ti}(buffer=nz_real)
        nzval_real = FastVec{P}(buffer=nz_real)
    end
    if !iszero(Nc)
        Ti = SparseArrays.indtype(b.exponents_complex) # won't change anything if !iszero(nreal)
        nz_complex = length(rowvals(b.exponents_complex))
        nz_conj = length(rowvals(b.exponents_conj))
        colptr_complex = FastVec{Ti}(buffer=minlen +1)
        rowval_complex = FastVec{Ti}(buffer=nz_complex)
        nzval_complex = FastVec{P}(buffer=nz_complex)
        colptr_conj = FastVec{Ti}(buffer=len +1)
        rowval_conj = FastVec{Ti}(buffer=nz_conj)
        nzval_conj = FastVec{P}(buffer=nz_conj)
    end
    iszero(minlen) || @inbounds for mon in b
        while astate[1] < mon
            astate = iterate(a, astate[2])
            isnothing(astate) && @goto done
        end
        astate[1] == mon || continue
        if !iszero(Nr)
            unsafe_push!(colptr_real, length(rowval_real) +1)
            unsafe_append!(rowval_real, rowvals(mon.exponents_real))
            unsafe_append!(nzval_real, nonzeros(mon.exponents_real))
        end
        if !iszero(Nc)
            unsafe_push!(colptr_complex, length(rowval_complex) +1)
            unsafe_append!(rowval_complex, rowvals(mon.exponents_complex))
            unsafe_append!(nzval_complex, nonzeros(mon.exponents_complex))
            unsafe_push!(colptr_conj, length(rowval_conj) +1)
            unsafe_append!(rowval_conj, rowvals(mon.exponents_conj))
            unsafe_append!(nzval_conj, nonzeros(mon.exponents_conj))
        end
    end
    @label done
    iszero(Nr) || unsafe_push!(colptr_real, length(rowval_real) +1)
    if !iszero(Nc)
        unsafe_push!(colptr_complex, length(rowval_complex) +1)
        unsafe_push!(colptr_conj, length(rowval_conj) +1)
        @assert(length(colptr_complex) == length(colptr_conj))
    end
    iszero(Nr) || iszero(Nc) || @assert(length(colptr_real) == length(colptr_complex))
    iter.data = SimpleMonomialVector{Nr,Nc,P,SparseMatrixCSC{P,Ti}}(
        iszero(Nr) ? absent : SparseMatrixCSC{P,Ti}(Nr, length(colptr_real) -1, finish!(colptr_real), finish!(rowval_real),
                                                    finish!(nzval_real)),
        iszero(Nc) ? absent : SparseMatrixCSC{P,Ti}(Nc, length(colptr_complex) -1, finish!(colptr_complex),
                                                    finish!(rowval_complex), finish!(nzval_complex)),
        iszero(Nc) ? absent : SparseMatrixCSC{P,Ti}(Nc, length(colptr_conj) -1, finish!(colptr_conj), finish!(rowval_conj),
                                                    finish!(nzval_conj))
    )
    return iter.data
end
function Base.collect(iter::SortedIteratorIntersection{<:LazyMonomialsUnion{Nr,Nc,P},
                                                       <:LazyMonomialsUnion{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned}
    ismissing(iter.data) || return iter.data
    a, b = iter.a, iter.b
    astate = iterate(a)
    bstate = iterate(b)
    minlen = isnothing(astate) || isnothing(bstate) ? 0 : min(length(a), length(b))
    exponents_real = iszero(Nr) ? absent : resizable_array(P, Nr, minlen)
    exponents_complex = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    exponents_conj = iszero(Nc) ? absent : resizable_array(P, Nc, minlen)
    if iszero(minlen)
        iter.data = SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(exponents_real, exponents_complex, exponents_conj)
        return iter.data
    end
    k = 1
    @inbounds while true
        while astate[1] < bstate[1]
            astate = iterate(a, astate[2])
            isnothing(astate) && @goto done
        end
        while bstate[1] < astate[1]
            bstate = iterate(b, bstate[2])
            isnothing(bstate) && @goto done
        end
        if astate[1] == bstate[1]
            copyto!(@view(exponents_real[:, k]), astate[1].exponents_real)
            copyto!(@view(exponents_complex[:, k]), astate[1].exponents_complex)
            copyto!(@view(exponents_conj[:, k]), astate[1].exponents_conj)
            k += 1
        end
    end
    @label done
    del = minlen - k +1
    iter.data = SimpleMonomialVector{Nr,Nc,P,Matrix{P}}(
        matrix_delete_end!(exponents_real, del),
        matrix_delete_end!(exponents_complex, del),
        matrix_delete_end!(exponents_conj, del)
    )
    return iter.data
end
Base.@propagate_inbounds Base.getindex(iter::SortedIteratorIntersection, i) = collect(iter)[i]

Base.intersect(a::LazyMonomialsUnion{Nr,Nc,P}, b::SimpleDenseMonomialVectorOrView{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} =
    SortedIteratorIntersection{SimpleMonomialVector{Nr,Nc,P,Matrix{P},iszero(Nr) ? Absent : Matrix{P},
                                                    iszero(Nc) ? Absent : Matrix{P}}}(a, b)
function Base.intersect(a::LazyMonomialsUnion{Nr,Nc,P}, b::SimpleSparseMonomialVectorOrView{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned}
    M = SparseMatrixCSC{P,SparseArrays.indtype(iszero(Nr) ? b.exponents_complex : b.exponents_real)}
    return SortedIteratorIntersection{SimpleMonomialVector{Nr,Nc,P,M,iszero(Nr) ? Absent : M,iszero(Nc) ? Absent : M}}(a, b)
end
Base.intersect(a::SimpleMonomialVector{Nr,Nc,P}, b::LazyMonomialsUnion{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} =
    intersect(b, a) # just so that less compilation is necessary
Base.intersect(a::LazyMonomials{Nr,Nc,P,<:MonomialIterator{E1}},
    b::LazyMonomials{Nr,Nc,P,<:MonomialIterator{E2}}) where {Nr,Nc,P<:Unsigned,E1,E2} =
    LazyMonomials{Nr,Nc}(max(a.iter.mindeg, b.iter.mindeg):min(a.iter.maxdeg, b.iter.maxdeg),
        minmultideg=max.(a.iter.minmultideg, b.iter.minmultideg), maxmultideg=min.(a.iter.maxmultideg, b.iter.maxmultideg),
        exponents=E1 === Nothing || E2 === Nothing ? nothing : ownexponents)
Base.intersect(a::LazyMonomialsUnion{Nr,Nc,P}, b::LazyMonomialsUnion{Nr,Nc,P}) where {Nr,Nc,P<:Unsigned} =
    SortedIteratorIntersection{SimpleMonomialVector{Nr,Nc,P,Matrix{P},iszero(Nr) ? Absent : Matrix{P},
                                                    iszero(Nc) ? Absent : Matrix{P}}}(a, b)
function Base.intersect(a::LSM, b::LSM) where {Nr,Nc,P<:Unsigned,LSM<:LazySubMonomials{Nr,Nc,P}}
    if a.lm === b.lm
        return LazySubMonomials(a.lm, intersect(a.indices, b.indices), copy=!isnothing(a.lm.iter.exponents))
    else
        return SortedIteratorIntersection{SimpleMonomialVector{Nr,Nc,P,Matrix{P},iszero(Nr) ? Absent : Matrix{P},
                                                               iszero(Nc) ? Absent : Matrix{P}}}(a, b)
    end
end

"""
    merge_monomial_vectors(::Type{<:SimpleMonomialVector{Nr,Nc,P}},
        X::AbstractVector) where {Nr,Nc,P<:Unsigned}

Returns the vector of monomials in the entries of X in increasing order and without any duplicates. The output format is
specified in the first parameter; it may be a dense or sparse `SimpleMonomialVector`. If the sparsity is not specified, a
primitive heuristic will be chosen (note that this introduces a type instability). The individual elements in `X` must be
sorted iterables with a length and return `SimpleMonomial`s compatible with the output format.
"""
function MultivariatePolynomials.merge_monomial_vectors(::Type{<:SimpleMonomialVector{Nr,Nc,P,M}},
    @nospecialize(X::AbstractVector)) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}}
    # We must be very careful here. The function must be fast, there's no way to allow for dynamic dispatch within the loop.
    # However, we might get a collection of different types in the the vector - dense and sparse SimpleMonomialVector and
    # LazyMonomials with all kinds of iterators. It is certainly not viable to compile a new function for every possible
    # combination of inputs. Therefore, here we first sort them all by type, then pass them to a generated function.
    grouped = Dict{DataType,Vector{<:AbstractVector}}()
    for Xᵢ in X
        t = typeof(Xᵢ)
        eltype(t) <: SimpleMonomial{Nr,Nc,P} || error("An iterator does not have a compatible element type")
        Base.IteratorSize(eltype(t)) isa Base.SizeUnknown && error("All iterators must have a length")
        v = get!(@capture(() -> $t[]), grouped, t)
        push!(v, Xᵢ)
    end
    # We need to sort the types in an arbitrary, but consistent manner. The function is commutative, so no need to generate two
    # different functions just because the order differs. Assuming hash probably never collides on the few couple of types that
    # are possible, we'll use this as a comparison between Type objects.
    # Make sure outtype is not the fully specified type, but the one that has the constructor defined, regardless of what the
    # input parameter actually was.
    return merge_monomial_vectors_impl(SimpleMonomialVector{Nr,Nc,P,M}, values(sort(grouped, by=hash))...)::
        SimpleMonomialVector{Nr,Nc,P,M,iszero(Nr) ? Absent : M,iszero(Nc) ? Absent : M}
end

MultivariatePolynomials.merge_monomial_vectors(::Type{<:SimpleMonomialVector{Nr,Nc,P}},
    @nospecialize(X::AbstractVector)) where {Nr,Nc,P<:Unsigned} =
    # stupid heuristic: #elements in dense vectors > #elements in sparse vectors -> dense.
    # TODO: maybe consider the actual storage. sparse size = 2length(nzvals) + size(, 2); dense size = *(size()...)
    # sparse size of dense: ≤ 2 * *(size()...) + size(, 2)
    merge_monomial_vectors(
        SimpleMonomialVector{Nr,Nc,P,
            sum(length, X) > 2sum(mv -> eltype(mv) isa SimpleSparseMonomialOrView ? length(mv) : 0, X) ?
                Matrix{P} : SparseMatrixCSC{P,UInt}
        }, X)

@generated function merge_monomial_vectors_impl(outtype::Type{<:SimpleMonomialVector{Nr,Nc,P,M}},
    Xs::Vector...) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}}
    # Here, every vector contained in a single Xs[i] is of the same type and we can specialize
    dense = M <: Matrix{P}
    if !dense
        sparse_idxtype = SparseArrays.indtype(M)
    end
    N = length(Xs)
    result = Expr(:block, :(maxitems::Int = 0), :(remaining = 0))
    iters = [Symbol(:iters, i) for i in 1:N]
    idxs = [Symbol(:idxs, i) for i in 1:N]
    mins = [Symbol(:min, i) for i in 1:N]
    for i in 1:N
        push!(result.args, quote
            maxitems += sum(length, Xs[$i], init=0)
            # we cannot just do iterate.(Xs[i]) - if none are Nothing, the Vector won't allow it.
            $(iters[i]) = Vector{$(Base.promote_op(iterate, eltype(Xs[i])))}(undef, length(Xs[$i]))
            $(idxs[i]) = similar($(iters[i]), Int)
            $(mins[i]) = 1
            curminidx = typemax(Int)
            for (j, Xⱼ) in enumerate(Xs[$i])
                $(iters[i])[j] = iterval = iterate(Xⱼ)
                if isnothing(iterval)
                    $(idxs[i])[j] = typemax(Int)
                else
                    $(idxs[i])[j] = mi = monomial_index(iterval[1])
                    if mi < curminidx
                        $(mins[i]) = j
                        curminidx = mi
                    end
                end
            end
        end)
    end
    if dense
        push!(result.args, iszero(Nr) ? :(exponents_real = absent) :
            :(exponents_real = resizable_array(P, Nr, maxitems)))
        if iszero(Nc)
            push!(result.args, :(exponents_complex = absent), :(exponents_conj = absent))
        else
            push!(result.args, :(exponents_complex = resizable_array(P, Nc, maxitems)),
                :(exponents_conj = resizable_array(P, Nc, maxitems)))
        end
    else
        push!(result.args, :(Ti = $sparse_idxtype))
        iszero(Nr) || push!(result.args, quote
            colptr_real = FastVec{Ti}(buffer=maxitems +1)
            rowval_real = FastVec{Ti}()
            nzval_real = FastVec{P}()
        end)
        iszero(Nc) || push!(result.args, quote
            colptr_complex = FastVec{Ti}(buffer=maxitems +1)
            rowval_complex = FastVec{Ti}()
            nzval_complex = FastVec{P}()
            colptr_conj = FastVec{Ti}(buffer=maxitems +1)
            rowval_conj = FastVec{Ti}()
            nzval_conj = FastVec{P}()
        end)
    end
    process_min = Expr(:if)
    process_min_cur = process_min
    for i in 1:N
        if !isone(i)
            process_min_cur = let process_min_next=Expr(:elseif)
                push!(process_min_cur.args, process_min_next)
                process_min_next
            end
        end
        process_min_i = Expr(:block)
        # Probably Julia won't be able to figure out that !isnothing($(iters[i])) actually holds true. This would be a good
        # case for the @ensure/@assume... macro proposal to complement @assert (Julia issue #51729). But we don't have it
        # (yet). On the other hand, Cthulhu seems to suggest that while $(iters[i])[...] is indeed always a Union (regardless
        # of whether we wrap it in isnothing or not), accessing an index will automatically give the correct type again, even
        # here.
        if dense
            iszero(Nr) ||
                push!(process_min_i.args, :(copyto!(@view(exponents_real[:, col]), $(iters[i])[$(mins[i])][1].exponents_real)))
            if !iszero(Nc)
                push!(process_min_i.args,
                :(copyto!(@view(exponents_complex[:, col]), $(iters[i])[$(mins[i])][1].exponents_complex)),
                :(copyto!(@view(exponents_conj[:, col]), $(iters[i])[$(mins[i])][1].exponents_conj)))
            end
        else
            process_block = Expr(:let, Expr(:block, :(mon=$(iters[i])[$(mins[i])][1])), Expr(:block))
            if !iszero(Nr)
                push!(process_block.args[1].args, :(mon_real = mon.exponents_real))
                push!(process_block.args[2].args, :(unsafe_push!(colptr_real, length(rowval_real) +1)))
            end
            if !iszero(Nc)
                push!(process_block.args[1].args, :(mon_complex = mon.exponents_complex), :(mon_conj = mon.exponents_conj))
                push!(process_block.args[2].args, :(unsafe_push!(colptr_complex, length(rowval_complex) +1)),
                    :(unsafe_push!(colptr_conj, length(rowval_conj) +1)))
            end
            if eltype(eltype(Xs[i])) <: SimpleSparseMonomialOrView
                iszero(Nr) || push!(process_block.args[2].args,
                    :(append!(rowval_real, rowvals(mon_real))),
                    :(append!(nzval_real, nonzeros(mon_real)))
                )
                iszero(Nc) || push!(process_block.args[2].args,
                    :(append!(rowval_complex, rowvals(mon_complex))),
                    :(append!(rowval_conj, rowvals(mon_conj))),
                    :(append!(nzval_complex, nonzeros(mon_complex))),
                    :(append!(nzval_conj, nonzeros(mon_conj)))
                )
            else
                if !iszero(Nr)
                    push!(process_block.args[1].args, :(nz_real = count(∘(!, iszero), mon_real)))
                    push!(process_block.args[2].args,
                        :(prepare_push!(rowval_real, nz_real)),
                        :(prepare_push!(nzval_real, nz_real)),
                        :(for (j, v) in enumerate(mon_real)
                            if !iszero(v)
                                unsafe_push!(rowval_real, j)
                                unsafe_push!(nzval_real, v)
                            end
                        end)
                    )
                end
                if !iszero(Nc)
                    push!(process_block.args[1].args, :(nz_complex = count(∘(!, iszero), mon_complex)),
                        :(nz_conj = count(∘(!, iszero), mon_conj)))
                    push!(process_block.args[2].args,
                        :(prepare_push!(rowval_complex, nz_complex)),
                        :(prepare_push!(nzval_complex, nz_complex)),
                        :(prepare_push!(rowval_conj, nz_conj)),
                        :(prepare_push!(nzval_conj, nz_conj)),
                        :(for (j, (v₁, v₂)) in enumerate(zip(mon_complex, mon_conj))
                            if !iszero(v₁)
                                unsafe_push!(rowval_complex, j)
                                unsafe_push!(nzval_complex, v₁)
                            end
                            if !iszero(v₂)
                                unsafe_push!(rowval_conj, j)
                                unsafe_push!(nzval_conj, v₂)
                            end
                        end)
                    )
                end
            end
            push!(process_min_i.args, process_block)
        end
        push!(process_min_i.args, quote
            lastidx = $(idxs[i])[$(mins[i])]
            while true
                $(iters[i])[$(mins[i])] = nextiter = iterate(Xs[$i][$(mins[i])], $(iters[i])[$(mins[i])][2])
                if isnothing(nextiter)
                    $(idxs[i])[$(mins[i])] = typemax(Int)
                else
                    $(idxs[i])[$(mins[i])] = monomial_index(nextiter[1])
                end
                $(mins[i]) = argmin($(idxs[i]))
                $(idxs[i])[$(mins[i])] == lastidx || break
            end
        end)
        push!(process_min_cur.args, :(curmin == $i), process_min_i)
    end
    push!(process_min_cur.args, Expr(:break))
    push!(result.args, quote
        col = 1
        while true
            curmin = 0
            curminidx = typemax(Int) -1 # Let's assume we don't ever encounter the two largest values, then we can safe some
                                        # additional comparison logic
            $((
                :(if $(idxs[i])[$(mins[i])] < curminidx
                    curmin = $i
                    curminidx = $(idxs[i])[$(mins[i])]
                else
                    while $(idxs[i])[$(mins[i])] == curminidx
                        # duplicate, skip it (doesn't matter whether it is the global minimum, if not it will be a duplicate
                        # later)
                        $(iters[i])[$(mins[i])] = nextiter = iterate(Xs[$i][$(mins[i])], $(iters[i])[$(mins[i])][2])
                        if isnothing(nextiter)
                            $(idxs[i])[$(mins[i])] = typemax(Int)
                        else
                            $(idxs[i])[$(mins[i])] = monomial_index(nextiter[1])
                        end
                        $(mins[i]) = argmin($(idxs[i]))
                    end
                end)
                for i in 1:N
            )...)
            $process_min
            col += 1
        end
    end)
    if dense
        push!(result.args,
            :(del = maxitems - col +1),
            :(return outtype(
                matrix_delete_end!(exponents_real, del),
                matrix_delete_end!(exponents_complex, del),
                matrix_delete_end!(exponents_conj, del)
            ))
        )
    else
        iszero(Nr) || push!(result.args,
            :(unsafe_push!(colptr_real, length(rowval_real) +1)),
            :(@assert(length(colptr_real) == col))
        )
        iszero(Nc) || push!(result.args,
            :(unsafe_push!(colptr_complex, length(rowval_complex) +1)),
            :(unsafe_push!(colptr_conj, length(rowval_conj) +1)),
            :(@assert(length(colptr_complex) == length(colptr_conj) == col))
        )
        push!(result.args,
            :(return outtype(
                $(iszero(Nr) ? :(absent) : :(SparseMatrixCSC{P,$sparse_idxtype}(Nr, col -1, finish!(colptr_real),
                    finish!(rowval_real), finish!(nzval_real)))),
                $(iszero(Nc) ? :(absent) : :(SparseMatrixCSC{P,$sparse_idxtype}(Nc, col -1, finish!(colptr_complex),
                    finish!(rowval_complex), finish!(nzval_complex)))),
                $(iszero(Nc) ? :(absent) : :(SparseMatrixCSC{P,$sparse_idxtype}(Nc, col -1, finish!(colptr_conj),
                    finish!(rowval_conj), finish!(nzval_conj))))
            ))
        )
    end
    return :(@inbounds($result))
end