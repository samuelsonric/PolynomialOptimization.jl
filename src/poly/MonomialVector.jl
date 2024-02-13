export SimpleMonomialVector, effective_nvariables, LazyMonomials

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

"""
    SimpleMonomialVector{Nr,0}(exponents_real::AbstractMatrix{<:Integer})
    SimpleMonomialVector{0,Nc}(exponents_complex::AbstractMatrix{<:Integer},
        exponents_conj::AbstractMatrix{<:Integer})
    SimpleMonomialVector{Nr,Nc}(exponents_real::AbstractMatrix{<:Integer},
        exponents_complex::AbstractMatrix{<:Integer}, exponents_conj::AbstractMatrix{<:Integer})

Creates a monomial vector, where each column corresponds to one monomial and each row is contains its exponents. The
element types of the matrices will be promoted to a common unsigned integer type.
All matrices must have the same number of columns; complex and conjugate matrices must have the same number of rows.
All matrices will be converted a common matrix type; dense matrices (or views) are possible as well as sparse matrices
(or views).
Taking views of a `SimpleMonomialVector` will return another `SimpleMonomialVector` whose exponents are the corresponding
views.

No particular monomial order is enforced on a `SimpleMonomialVector`. It must not contain duplicates.
"""
function SimpleMonomialVector{Nr,0}(exponents_real::AbstractMatrix{<:Integer}) where {Nr}
    size(exponents_real, 1) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(size(exponents_real, 1))"))
    allunique(eachcol(exponents_real)) || throw(ArgumentError("Monomial vector must not contain duplicates"))
    P = Unsigned(eltype(exponents_real))
    exps = convert(AbstractMatrix{P}, exponents_real)
    return SimpleMonomialVector{Nr,0,P,typeof(exps)}(exps, absent, absent)
end

function SimpleMonomialVector{0,Nc}(exponents_complex::AbstractMatrix{<:Integer}, exponents_conj::AbstractMatrix{<:Integer}) where {Nc}
    size(exponents_complex, 1) == size(exponents_conj, 1) ||
        throw(ArgumentError("Complex and conjugate exponents lengths are different"))
    size(exponents_complex, 1) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(size(exponents_complex, 1))"))
    size(exponents_complex, 2) == size(exponents_conj, 2) ||
        throw(ArgumentError("Number of monomials is different"))
    allunique(zip(eachcol(exponents_complex), eachcol(exponents_conj))) ||
        throw(ArgumentError("Monomial vector must not contain duplicates"))
    P = promote_type(Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    M1 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_complex))
    M2 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_conj))
    M = promote_type(M1, M2)
    return SimpleMonomialVector{0,Nc,P,M}(absent, convert(M, exponents_complex), convert(M, exponents_conj))
end

function SimpleMonomialVector{Nr,Nc}(exponents_real::AbstractMatrix{<:Integer}, exponents_complex::AbstractMatrix{<:Integer},
        exponents_conj::AbstractMatrix{<:Integer}) where {Nr,Nc}
    size(exponents_real, 1) == Nr || throw(ArgumentError("Requested $Nr real variables, but got $(size(exponents_real, 1))"))
    size(exponents_complex, 1) == size(exponents_conj, 1) ||
        throw(ArgumentError("Complex and conjugate exponents lengths are different"))
    size(exponents_complex, 1) == Nc ||
        throw(ArgumentError("Requested $Nc complex variables, but got $(size(exponents_complex, 1))"))
    size(exponents_real, 2) == size(exponents_complex, 2) == size(exponents_conj, 2) ||
        throw(ArgumentError("Number of monomials is different"))
    allunique(zip(eachcol(exponents_real), eachcol(exponents_complex), eachcol(exponents_conj))) ||
        throw(ArgumentError("Monomial vector must not contain duplicates"))
    P = promote_type(Unsigned(eltype(exponents_real)), Unsigned(eltype(exponents_complex)), Unsigned(eltype(exponents_conj)))
    M1 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_real))
    M2 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_complex))
    M3 = Base.promote_op(convert, Type{AbstractMatrix{P}}, typeof(exponents_conj))
    M = promote_type(M1, M2, M3)
    return SimpleMonomialVector{Nr,Nc,P,M}(
        convert(M, exponents_real), convert(M, exponents_complex), convert(M, exponents_conj)
    )
end

"""
    SimpleMonomialVector(mv::AbstractVector{<:AbstractMonomialLike};
        max_power::Integer=maxdegree(mv), representation::Symbol=:auto, vars=variables(mv))

Creates a `SimpleMonomialVector` from a generic monomial vector that supports `MultivariatePolynomials`'s interface.
It is possible to specify the maximal power that the monomial vector should be able to hold explicitly (which will determine
the element type of the internal matrices).
The keyword argument `representation` determines whether a `Matrix` (for `:dense`) or a `SparseMatrixCSC` (for `:sparse`) is
chosen as the underlying representation. The default, `:auto`, will take a small sample of the monomials in the vector and from
this determine which representation is more efficient.
The keyword argument `vars` must contain all real-valued and original complex-valued (so not the conjugates) variables that
occur in the monomial vector. However, the order of this iterable (which must have a length) controls how the MP variables are
mapped to [`SimpleVariable`](@ref)s.
"""
function SimpleMonomialVector(mv::AbstractVector{<:AbstractMonomialLike}; max_power::Integer=maxdegree(mv),
    representation::Symbol=:auto, vars=unique!((x -> isconj(x) ? conj(x) : x).(variables(mv))))
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
    return SimpleMonomialVector(mv, max_power, representation, vars)
end

# mv must be an iterable with length
"""
    SimpleMonomialVector(mv, max_power::Integer, representation::Symbol, vars)

Creates a `SimpleMonomialVector` from a generic iterable that gives `AbstractMonomialLike` elements. In contrast to when `mv`
is a vector, now all arguments must be provided. `representation` must be either `:dense` or `:sparse`.
"""
function SimpleMonomialVector(mv, max_power::Integer, representation::Symbol, vars)
    isempty(vars) && throw(ArgumentError("Variables must be present"))
    any(isconj, vars) && throw(ArgumentError("The variables must not contain conjuates"))
    allunique(vars) || throw(ArgumentError("Variables must not contain duplicates"))
    representation ∈ (:dense, :sparse) || throw(ArgumentError("The representation must be :dense or :sparse"))
    P = smallest_unsigned(max_power)

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
        return SimpleMonomialVector{vars_real,vars_complex,P,Matrix{P}}(exponents_real, exponents_complex, exponents_conj)
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
        return SimpleMonomialVector{vars_real,vars_complex,P,SparseMatrixCSC{P,Ti}}(
            spexponents_real, spexponents_complex, spexponents_conj
        )
    end
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
                args -> max(sum(args[1], init=0), sum(args[2], init=0)),
                zip(eachcol(x.exponents_complex), eachcol(x.exponents_conj))
            )
        end
        function MultivariatePolynomials.$fun(x::SimpleMonomialVector)
            isempty(x) && return $def
            return $call(
                args -> $(realfn(:(sum(args[1], init=0)))) + max(sum(args[2], init=0), sum(args[3], init=0)),
                zip(eachcol(x.exponents_real), eachcol(x.exponents_complex), eachcol(x.exponents_conj))
            )
        end
    end)
end

Base.isreal(::SimpleRealMonomialVector) = true
Base.isreal(x::SimpleMonomialVector) = all(isreal, x)

Base.conj(x::SimpleMonomialVector{Nr,Nc,P,M}) where {Nr,Nc,P<:Unsigned,M<:AbstractMatrix{P}} =
    SimpleMonomialVector{Nr,Nc,P,M}(x.exponents_real, x.exponents_conj, x.exponents_complex)

# zero-allocation variable vector
struct SimpleMonomialVariables{Nr,Nc,V<:SimpleVariable{Nr,Nc}} <: AbstractVector{V}
     SimpleMonomialVariables{Nr,0}() where {Nr} = new{Nr,0,SimpleRealVariable{Nr,0,smallest_unsigned(Nr)}}()
     SimpleMonomialVariables{0,Nc}() where {Nc} = new{0,Nc,SimpleComplexVariable{0,Nc,smallest_unsigned(Nc)}}()
     SimpleMonomialVariables{Nr,Nc}() where {Nr,Nc} = new{Nr,Nc,SimpleVariable{Nr,Nc}}()
end

Base.IteratorSize(::Type{<:SimpleMonomialVariables{Nr,Nc}}) where {Nr,Nc} = Base.HasLength()
Base.IteratorEltype(::Type{<:SimpleMonomialVariables{Nr,Nc,V}}) where {Nr,Nc,V<:SimpleVariable{Nr,Nc}} = Base.HasEltype()
Base.eltype(::Type{SimpleMonomialVariables{Nr,Nc,V}}) where {Nr,Nc,V<:SimpleVariable{Nr,Nc}} = V
Base.length(::SimpleMonomialVariables{Nr,Nc}) where {Nr,Nc} = Nr + 2Nc
Base.size(::SimpleMonomialVariables{Nr,Nc}) where {Nr,Nc} = (Nr + 2Nc,)

@inline function Base.getindex(smv::SimpleMonomialVariables{Nr,Nc}, idx::Integer) where {Nr,Nc}
    @boundscheck checkbounds(smv, idx)
    idx ≤ Nr && return SimpleRealVariable{Nr,Nc}(idx)
    idx -= Nr
    idx ≤ Nc && return SimpleComplexVariable{Nr,Nc}(idx)
    idx -= Nc
    @boundscheck @assert(idx ≤ Nc)
    return SimpleComplexVariable{Nr,Nc}(idx, true)
end

Base.collect(::SimpleMonomialVariables{Nr,0}) where {Nr} = map(SimpleRealVariable{Nr,0}, 1:Nr)
function Base.collect(::SimpleMonomialVariables{Nr,Nc,V}) where {Nr,Nc,V<:SimpleVariable{Nr,Nc}}
    result = Vector{V}(undef, Nr + 2Nc)
    for i in 1:Nr
        @inbounds result[i] = SimpleRealVariable{Nr,Nc}(i)
    end
    for i in 1:Nc
        @inbounds result[i+Nr] = SimpleComplexVariable{Nr,Nc}(i)
    end
    for i in 1:Nc
        @inbounds result[i+Nr+Nc] = SimpleComplexVariable{Nr,Nc}(i, true)
    end
    return result
end

function Base.iterate(smv::SimpleMonomialVariables{Nr,Nc}, state::Int=1) where {Nr,Nc}
    state > Nr + 2Nc && return nothing
    @inbounds return (smv[state], state +1)
end

MultivariatePolynomials.variables(::XorTX{Union{<:SimpleVariable{Nr,Nc},<:SimpleMonomial{Nr,Nc},
                                                <:AbstractVector{<:SimpleMonomial{Nr,Nc}}}}) where {Nr,Nc} =
    SimpleMonomialVariables{Nr,Nc}()

MultivariatePolynomials.nvariables(::XorTX{AbstractVector{<:SimpleMonomial{Nr,Nc}}}) where {Nr,Nc} = Nr + 2Nc

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
"""
    effective_nvariables(x::Union{<:SimpleMonomialVector{Nr,Nc},
                                  <:AbstractArray{<:SimpleMonomialVector{Nr,Nc}}}...) where {Nr,Nc}

Calculates the number of effective variable of its arguments: there are at most `Nr + 2Nc` variables that may occur in any
of the monomial vectors or arrays of monomial vectors in the arguments. This function calculates efficiently the number of
variables that actually occur at least once anywhere in any argument.
"""
@generated function effective_nvariables(x::Union{<:SimpleMonomialVector{Nr,Nc},<:AbstractArray{<:SimpleMonomialVector{Nr,Nc}}}...) where {Nr,Nc}
    n = length(x)
    quote
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
    monomials(nreal::Integer, ncomplex::Integer, degree::AbstractUnitRange{DI},
        order=Graded{LexOrder}; minmultideg=nothing, maxmultideg=nothing,
        representation=:auto, filter=powers -> true) where {DI}

Returns a [`SimpleMonomialVector`](@ref) with `nreal` real and `ncomplex` complex variables, total degrees contained in
`degree`, ordered by `order` (which currently is a dummy argument, as only `Graded{LexOrder}` is supported) and individual
variable degrees varying between `minmultideg` and `maxmultideg` (where real variables come first, then complex variables, then
their conjugates).
The representation is either `:dense` or `:sparse`; if `:auto` is selected, the method will estimate (rather accurately) which
representation requires more memory and choose an appropriate one.
The maximal exponent of the return type is chosen as the smallest unsigned integer that can still hold the largest degree
according to `degree` (ignoring `maxmultideg`).
An additional `filter` may be employed to drop monomials during the construction. Note that size estimation cannot take the
filter into account.

This method internally relies on [`MonomialIterator`](@ref). The `minmultideg` and `maxmultideg` parameters will automatically
be converted to `Vector{DI}` instances.
"""
function MultivariatePolynomials.monomials(nreal::Integer, ncomplex::Integer,
    degree::AbstractUnitRange{DI}, order=Graded{LexOrder};
    minmultideg::Union{Nothing,<:AbstractVector{DI},Tuple{Vararg{DI}}}=nothing,
    maxmultideg::Union{Nothing,<:AbstractVector{DI},Tuple{Vararg{DI}}}=nothing,
    representation::Symbol=:auto, filter=powers -> true) where {DI<:Integer}
    representation ∈ (:auto, :dense, :sparse) || throw(ArgumentError("The representation must be :dense, :sparse, or :auto"))

    n = nreal + 2ncomplex
    if isnothing(minmultideg)
        minmultideg = fill(zero(first(degree)), n)
    elseif length(minmultideg) != n
        throw(DimensionMismatch("minmultideg has length $(length(minmultideg)), expected $n"))
    elseif !(minmultideg isa Vector{DI})
        minmultideg = DI.(collect(minmultideg))
    end
    if isnothing(maxmultideg)
        maxmultideg = fill(last(degree), n)
    elseif length(maxmultideg) != n
        throw(DimensionMismatch("maxmultideg has length $(length(maxmultideg)), expected $n"))
    elseif !(maxmultideg isa Vector{DI})
        maxmultideg = DI.(collect(maxmultideg))
    end
    iter = MonomialIterator{order}(first(degree), last(degree), minmultideg, maxmultideg, true)
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
            representation = :dense
        else
            representation = :sparse
            nz_real, nz_complex, nz_conj = nzl
        end
    else
        nz_real, nz_complex, nz_conj = nzlength(iter, nreal, ncomplex, nothing)
    end
    P = smallest_unsigned(last(degree))
    if representation === :dense
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
struct LazyMonomials{Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,<:Any,P}} <: AbstractVector{SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}}
    iter::MI

    @doc """
        LazyMonomials(nreal::Integer, ncomplex::Integer, degree::AbstractUnitRange{DI},
            order=Graded{LexOrder}; minmultideg=nothing, maxmultideg=nothing, copy=true)

Constructs a memory-efficient vector of monomials that contains the same data as given by [`monomials`](@ref) (with dense
representation and no filter allowed). The monomials will be constructed on-demand. They can be accessed via indexing or
iteration (which is much more efficient if multiple monomials in a row are required).
If the monomials are only accessed one-at-a-time and never referenced when another one is requested, `copy` may be set to
`false`. Then, obtaining a monomial will not allocate any memory; instead, only the memory that represents the monomial is
changed.
    """
    function LazyMonomials(nreal::Integer, ncomplex::Integer, degree::AbstractUnitRange{DI}, order=Graded{LexOrder};
        minmultideg::Union{Nothing,<:AbstractVector{DI},Tuple{Vararg{DI}}}=nothing,
        maxmultideg::Union{Nothing,<:AbstractVector{DI},Tuple{Vararg{DI}}}=nothing, copy::Bool=true) where {DI<:Integer}
        P = smallest_unsigned(last(degree))
        mindeg = P(first(degree))
        maxdeg = P(last(degree))
        n = nreal + 2ncomplex
        if isnothing(minmultideg)
            minmultideg = fill(zero(P), n)
        elseif length(minmultideg) != n
            throw(DimensionMismatch("minmultideg has length $(length(minmultideg)), expected $n"))
        elseif !(minmultideg isa Vector{P})
            minmultideg = P.(min.(collect(minmultideg), maxdeg))
        end
        if isnothing(maxmultideg)
            maxmultideg = fill(maxdeg, n)
        elseif length(maxmultideg) != n
            throw(DimensionMismatch("maxmultideg has length $(length(maxmultideg)), expected $n"))
        elseif !(maxmultideg isa Vector{P})
            maxmultideg = P.(min.(collect(maxmultideg), maxdeg))
        end
        iter = MonomialIterator{order}(mindeg, maxdeg, minmultideg, maxmultideg, !copy)
        new{nreal,ncomplex,P,typeof(iter)}(iter)
    end

    LazyMonomials{Nr,Nc,P,MI}(iter::MI) where {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,<:Any,P}} =
        new{Nr,Nc,P,MI}(iter)
end

Base.length(lm::LazyMonomials) = length(lm.iter)
Base.size(lm::LazyMonomials) = (length(lm.iter),)
@inline function Base.getindex(lm::LazyMonomials{Nr,Nc,P,<:AbstractMonomialIterator{<:Any,V}}, i::Integer) where {Nr,Nc,P<:Unsigned,V}
    @boundscheck checkbounds(lm, i)
    powers = V === Nothing ? Vector{P}(undef, Nr + 2Nc) : lm.iter.powers
    result = exponents_from_index!(powers, lm.iter, i)
    @boundscheck @assert(result)
    return SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}(
        iszero(Nr) ? absent : @view(powers[1:Nr]),
        iszero(Nc) ? absent : @view(powers[Nr+1:Nr+Nc]),
        iszero(Nc) ? absent : @view(powers[Nr+Nc+1:end])
    )
end
Base.IteratorSize(::Type{<:LazyMonomials}) = Base.HasLength()
Base.IteratorEltype(::Type{<:LazyMonomials}) = Base.HasEltype()
Base.eltype(::Type{<:LazyMonomials{Nr,Nc,P}}) where {Nr,Nc,P<:Unsigned} = SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}
function Base.iterate(lm::LazyMonomials{Nr,Nc,P}, args...) where {Nr,Nc,P<:Unsigned}
    result = iterate(lm.iter, args...)
    isnothing(result) && return nothing
    return SimpleMonomial{Nr,Nc,P,_LazyMonomialsView{P}}(
        iszero(Nr) ? absent : @view(result[1][1:Nr]),
        iszero(Nc) ? absent : @view(result[1][Nr+1:Nr+Nc]),
        iszero(Nc) ? absent : @view(result[1][Nr+Nc+1:end])
    ), result[2]
end
@inline function Base.getindex(lm::LazyMonomials{Nr,Nc,P,MI}, range::AbstractUnitRange) where {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,<:Any,P}}
    @boundscheck checkbounds(lm, range)
    iter = RangedMonomialIterator(lm.iter, first(range), length(range), copy=true)
    return LazyMonomials{Nr,Nc,P,typeof(iter)}(iter)
end
@inline function Base.view(lm::LazyMonomials{Nr,Nc,P,MI}, range::AbstractUnitRange) where {Nr,Nc,P<:Unsigned,MI<:AbstractMonomialIterator{<:Any,<:Any,P}}
    @boundscheck checkbounds(lm, range)
    iter = RangedMonomialIterator(lm.iter, first(range), length(range), copy=false)
    return LazyMonomials{Nr,Nc,P,typeof(iter)}(iter)
end