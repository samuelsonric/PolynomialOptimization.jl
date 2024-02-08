export poly_problem, poly_optimize

abstract type AbstractPOProblem{P<:SimplePolynomial} end

const AbstractRealPOProblem = AbstractPOProblem{<:SimpleRealPolynomial}
const AbstractComplexPOProblem = AbstractPOProblem{<:SimpleComplexPolynomial}

"""
    variables(problem::AbstractPOProblem)

Returns the original variables (not their internal rewrites) associated to a given polynomial optimization problem. This
defines the order in which solutions are returned. In the complex case, they do not contain conjugates.

See also [`poly_optimize`](@ref), [`poly_solutions`](@ref), [`poly_all_solutions`](@ref).
"""
MultivariatePolynomials.variables(problem::AbstractPOProblem) = dense_problem(problem).original_variables
"""
    nvariables(problem::AbstractPOProblem)

Returns the number of variables associated to a given polynomial optimization problem. This defines the order in which
solutions are returned. In the complex case, conjugates are not counted.

See also [`poly_optimize`](@ref), [`poly_solutions`](@ref), [`poly_all_solutions`](@ref).
"""
MultivariatePolynomials.nvariables(::AbstractPOProblem{<:SimplePolynomial{<:Any,Nr,Nc}}) where {Nr,Nc} = Nr + Nc
"""
    degree(problem::AbstractPOProblem)

Returns the degree associated with a polynomial optimization problem. This is -1 if the problem was constructed without using
the default dense basis.

See also [`poly_problem`](@ref).
"""
MultivariatePolynomials.degree(problem::AbstractPOProblem) = dense_problem(problem).degree
"""
    length(problem::AbstractPOProblem)

Return the length of the full basis associated with a polynomial optimization problem.
"""
Base.length(problem::AbstractPOProblem) = length(dense_problem(problem).basis)
"""
    isreal(problem::AbstractPOProblem)

Returns whether a given polynomial optimization problem contains only real-valued variables or also complex ones.
"""
Base.isreal(::AbstractRealPOProblem) = true
Base.isreal(::AbstractComplexPOProblem) = false

"""
    POProblem <: AbstractPOProblem

The basic structure for a polynomial optimization problem.

Generate this type using [`poly_problem`](@ref); perform optimizations using [`poly_optimize`] or wrap it in an appropriate
[`AbstractSPOProblem`](@ref) first.
Note that the variables in a `POProblem` are rewritten to internal data types, i.e., they will probably not display in the same
way as the original variables (they are simply numbered consecutively).

See also [`poly_problem`](@ref), [`poly_optimize`](@ref), [`SparsityCorrelative`](@ref), [`SparsityTermBlock`](@ref),
[`SparsityTermCliques`](@ref), [`SparsityCorrelativeTerm`](@ref).
"""
struct POProblem{P<:SimplePolynomial,MV<:SimpleMonomialVector,OV} <: AbstractPOProblem{P}
    objective::P
    prefactor::P
    degree::Int
    basis::MV
    constr_zero::Vector{P}
    constr_nonneg::Vector{P}
    constr_psd::Vector{Matrix{P}}
    original_variables::Vector{OV}
end

const RealPOProblem = POProblem{<:SimpleRealPolynomial}
const ComplexPOProblem = POProblem{<:SimpleComplexPolynomial}

function Base.show(io::IO, m::MIME"text/plain", p::POProblem)
    nv = nvariables(p)
    print(io, isreal(p) ? "Real" : "Complex", "-valued polynomial optimization hierarchy of degree ", p.degree, " in ", nv,
        " variable", isone(nv) ? "" : "s", "\nObjective: ")
    show(io, m, p.objective)
    if !isone(p.prefactor)
        print(io, "\nObjective was scaled by the prefactor ")
        show(io, m, p.prefactor)
    end
    ∑length = length(p.constr_zero) + length(p.constr_nonneg) + length(p.constr_psd)
    if !iszero(∑length)
        len = ceil(Int, log10(∑length))
        i = 1
        for (c, t, s) in ((p.constr_zero, "equality", "0 = "),
                          (p.constr_nonneg, "nonnegative", "0 ≤ "),
                          (p.constr_psd, "semidefinite", "0 ⪯ "))
            if !isempty(c)
                print(io, "\n", length(c), " ", t, " constraint", isone(length(c)) ? "" : "s")
                for constr in c
                    print(io, "\n", lpad(i, len, "0"), ": ", s)
                    show(io, m, constr)
                    i += 1
                end
            end
        end
    end
    print(io, "\nSize of full basis: ", length(p.basis))
end

function request_degree(mindeg)
    while true
        action = readline()
        if action == "Y" || action == "y"
            return mindeg
        elseif action == "N" || action == "n"
            error("The problem construction failed due to a too small degree.")
        else
            degree = tryparse(Int, action)
            if isnothing(degree)
                @warn("The specified value was neither Y, N, nor any number. Please retry.")
            elseif degree < mindeg
                @warn("The specified degree must be at least $mindeg. Please retry.")
            else
                return degree
            end
        end
    end
end

@doc raw"""
    poly_problem(objective, degree=0; zero=[], nonneg=[], psd=[], basis=:auto,
        perturbation=0., factor_coercive=1, perturbation_coefficient=0., perturbation_form=0,
        noncompact=(0., 0), tighter=false, verbose=false, representation=:auto)

Analyze a polynomial optimization problem and return a [`POProblem`](@ref) that can be used for sparse analysis and
optimization.

# Arguments
## Problem formulation
- `objective::AbstractPolynomial`: the objective that is to be minimized
- `degree::Int`: the degree of the Lasserre relaxation, which must be larger or equal to half of the (complex) degree of all
  polynomials that are involved. The default value `0` will automatically determine the minimum required degree (this may fail
  when `tighter` is set to `true`). The value is only relevant when no user-defined `basis` is provided.
  A nonzero value only makes sense if there are inequality or PSD constraints present, else it needlessly complicates
  calculations without any benefit.
- `zero::AbstractVector{<:AbstractPolynomialLike}`: a vector with all polynomials that should be constrained to be zero.
- `nonneg::AbstractVector{<:AbstractPolynomialLike}`: a vector with all polynomials that should be constrained to be
  nonnegative. The values of the polynomials must always be effectively real-valued (in the sense that their imaginary parts
  evaluate to zero even if no values are plugged in).
- `psd::AbstractVector{<:AbstractMatrix{<:AbstractPolynomialLike}}`: a vector of matrices that should be constrainted to be
  positive semidefinite. The matrices must be symmetric/hermitian.
## Problem representation
- `basis::Union{<:SimpleMonomialVector,<:AbstractVector{<:AbstractMonomialLike},Symbol}`: specifies how the basis of the dense
  moment matrix is constructed:
  - `:dense` (default) will construct a dense moment matrix with degrees from zero to `degree`. This is wasteful and never
    produces better results than `:newton`; however, the latter requires the availability of a suitable Newton solver, and it
    might not be possible to extract a solution from a Newton basis.
  - `:newton` will construct a basis based on the Newton halfpolytope; this is a far smaller, but still exact basis choice -
    while it might not work, it will never introduce a failure when there isn't one with `:dense` also.
    When constraints are present, the polytope is determined based on the Putinar reformulation, where all constraints and the
    objective are moved to one side (comprising a new virtual objective). In this case, the specified `degree` is relevant to
    make clear how large the constraint multipliers are.
    Note that for the complex-valued hierarchy, there is no "Newton polytope"; as the representation of complex-valued
    polynomials is unique, the process is much simpler there. Still, this corresponds to a reduction in the size of the basis,
    and it will be activated using the `:newton` symbol; no solver is required.
  - if a valid basis is already known for this problem, then it can be passed directly as a degree-ordered monomial vector.
    Note that this vector has to be converted to a [`SimpleMonomialVector`](@ref) internally, so if you can already create the
    basis data in this format, it is recommended to do so and pass the `SimpleMonomialVector` instead.
- `representation::Symbol`: internally, whatever interface of `MultivariatePolynomials` is used, the data is converted to the
  efficient [`SimplePolynomial`](@ref) representation. There is a dense or a sparse version of this representation, and by
  default, a heuristic will choose the optimal one for the objective, which will then be the one taken for all data, also
  including the constraints. Setting this parameter to `:dense` or `:sparse` allows to overwrite the choice.
  Note that this only affects the _polynomials_ involved in the problem; whether the _monomials_ in the basis use a dense or
  sparse storage is determined automatically (and can be overwritten by using `basis` appropriately).
## Problem modification
### For unique solution extraction
- `perturbation::Union{Float64, <:AbstractVector{Float64}}`: adds a random linear perturbation with an absolute value not
  greater than this value to the objective for every variable (or, in the vector case, with different magnitudes for each
  variable). This will ensure that the result is unique and hence solution extraction will always work, at the cost of
  potentially reducing sparsity and slightly changing the actual result.
### For noncompact sets
The following four parameters allow to automatically modify the problem according to a strategy by
[Mai, Lasserre, and Magron](https://doi.org/10.1007/s10107-021-01634-1) that was mainly developed for noncompact semialgebraic
sets. It will modify the objective to

```math
\mathrm{factor\_coercive} \bigl(
    \mathrm{objective} + \mathrm{perturbation\_coefficient} \cdot \mathrm{perturbation\_form}
  \bigr)\text.
```

Usually, `perturbation_form` and `factor_coercive` are both given by ``1 + \lVert\mathrm{variables}\rVert^2``.
If `perturbation_coefficient` is strictly positive, then for almost all degrees, the optimal value of the modified problem is
then in ``\bigl[f_{\mathrm{opt}}, f_{\mathrm{opt}} + \mathrm{perturbation\_coefficient} \cdot \mathrm{perturbation\_form}^{d_\mathrm o}(x_{\mathrm{opt}})\bigr]``.
Often, this even works for a strictly zero coefficient (relatively generic conditions were found by
[Huang, Nie, and Yuan](https://doi.org/10.1007/s10107-022-01878-5)).
Note that when modifying a problem in such a way, all sparsity methods provided by this package will be useless.
- `factor_coercive::AbstractPolynomial`: Let ``k`` be divisible by ``2r``, then this must be the dehomogenization of a coercive
  positive form in ``n+1`` variables of degree ``2r`` to the power ``k/(2r)``.
  Be aware that using this parameter will multiply the objective by another polynomial and therefore require a higher total
  relaxation degree to model the problem!
- `perturbation_coefficient::Float64`: a nonnegative prefactor that determines the strength of the perturbation and whose
  inverse dictates the scaling of a sufficient lower degree bound that guarantees optimality.
- `perturbation_form::AbstractPolynomial`: must be the dehomogenization of a positive form in `n+1` variables of degree
  `2(1+degree(objective)÷2)`.
- `noncompact::Tuple{Real,Int}`, now called ``(\epsilon, k)``: this is a shorthand that will set the previous three
  parameters to their standard values, `factor_coercive=(1 + sum(variables.^2))^k`, `perturbation_coefficient` to the value
  passed to this parameter, and `perturbation_form=(1 + sum(variables.^2))^maxhalfdegree(objective)`.
  Be aware that using this parameter will multiply the objective by another polynomial and therefore require a higher total
  relaxation degree to model the problem!
### For convergence at earlier levels
[Nie](https://doi.org/10.1007/s10107-018-1276-2) provides a way to add additional constraints based on optimality conditions to
the problems. This can speed up or make possible convergence at all. However, not every problem can be tightened in such a way,
and sometimes, tightening might also increase the minimal degree required to optimize the problem.
Note that the problem will end up with more equality and inequality constraints than originally entered. The augmentation not
change the solution of the original problem in case the minimum is attained at a critical point; if it is not, tightening will
lead to missing this minimum.
- `tighter::Union{Bool,Symbol}`: if set to a valid solver or `true` (= choose default), tries to automatically construct
  constraints using Nie's method. Note that the algorithm internally needs to create lots of dense polynomials of appropriate
  degrees before solving for the coefficients. It is therefore possible that for larger problems, this can take a very long
  time.
  For a list of supported solvers, see [the solver reference](@ref solvers_tighten).
  The automatic detection of the minimal degree may not work together with this procedure. It will only consider the degrees
  required before tightning, not the increase in degree that may occur during the process.
  This parameter can also be called `tighten`; if any of those two is `true`, it is assumed that this was the intended value.
## Progress monitoring
- `verbose::Bool`: if set to true, information about the current state of the method is printed; this may be useful for large
  and complicated problems whose construction can take some time.


    poly_problem(objective; perturbation=0., verbose=false, method=default_newton_method(),
        kwargs...)

Analyze an unconstrained polynomial optimization problem and return a [`POProblem`](@ref). This differs from using the full
version of this method in that the Newton polytope is calculated in order to automatically determine a suitable basis.
The keyword arguments are passed through to [`newton_halfpolytope`](@ref).

See also [`POProblem`](@ref), [`poly_optimize`](@ref), [`SparsityNone`](@ref), [`SparsityCorrelative`](@ref),
[`SparsityTermBlock`](@ref), [`SparsityTermCliques`](@ref), [`SparsityCorrelativeTerm`](@ref), [`newton_halfpolytope`](@ref).
"""
function poly_problem(objective::P, degree::Int=0;
    zero::AbstractVector{<:AbstractPolynomialLike}=P[],
    nonneg::AbstractVector{<:AbstractPolynomialLike}=P[],
    psd::AbstractVector{<:AbstractMatrix{<:AbstractPolynomialLike}}=AbstractMatrix{P}[],
    basis::Union{<:SimpleMonomialVector,<:AbstractVector{<:AbstractMonomialLike},Symbol}=:dense,
    perturbation::Union{Float64,<:AbstractVector{Float64}}=0.,
    factor_coercive::AbstractPolynomialLike=one(P), perturbation_coefficient::Float64=0.,
    perturbation_form::AbstractPolynomialLike=Base.zero(P), noncompact::Tuple{Real,Integer}=(0.,0),
    tighter::Union{Bool,Symbol}=false, tighten::Union{Bool,Symbol}=false, verbose::Bool=false,
    representation::Symbol=:auto, newton_args::Tuple=()) where {P<:AbstractPolynomialLike}
    if tighten !== false
        tighter = tighten
    end

    #region List of variables
    vars = sort!(collect(let vars = Set(variables(objective))
        for c in zero
            union!(vars, variables(c))
        end
        for c in nonneg
            union!(vars, variables(c))
        end
        for c in psd
            union!(vars, variables(c))
        end
        # Simple polynomial does not allow for conjugates, but we need to make sure all the base variables are included
        filter!(∘(!, isconj), union!(vars, conj.(vars)))
    end), rev=true) # for better compatibility with DynamicPolynomials, we reverse-sort the variables.
    #endregion

    #region Type of the problem
    nreal = count(isreal, vars)
    ncomplex = length(vars) - nreal
    complex = !iszero(ncomplex) # this only captures the presence of complex-valued variables, not of complex-valued
                                # coefficients

    T = let checkcomplex=p -> coefficient_type(p)<:Complex
        # our solvers and other operations will require machine float. For now, this is hard-coded here.
        if complex || checkcomplex(objective) || any(checkcomplex, zero) || any(checkcomplex, nonneg) ||
            any(m -> any(checkcomplex, m), psd)
            Complex{Float64}
        else
            Float64
        end
    end
    PT = polynomial_type(objective, T)
    #endregion

    #region Validation: no imaginary parts, correct form of inputs
    if T <: Complex
        tighter !== false && error("Complex-valued problems cannot be tightened")
        imag(objective) == 0 || error("Nonvanishing imaginary part in objective")
        imag(factor_coercive) == 0 || error("Nonvanishing imaginary party in coercive factor")
        all(∘(iszero, imag), nonneg) || error("Nonvanishing imaginary in nonnegative constraint")
        all(constr -> constr' == constr, psd) || error("Nonhermitian PSD constraint")
    else
        all(constr -> transpose(constr) == constr, psd) || error("Nonsymmetric PSD constraint")
    end
    if basis isa Symbol && !(basis in (:dense, :newton))
        error("If no basis is specified, the value must be one of :dense or :newton")
    end
    #endregion

    #region Preprocessing: perturbation, coercive factor
    objective = convert(PT, objective)
    zero = AbstractVector{PT}(convert.((PT,), zero))
    nonneg = AbstractVector{PT}(convert.((PT,), nonneg))
    psd = AbstractVector{Matrix{PT}}(convert.((Matrix{PT},), psd))
    factor_coercive = convert(PT, factor_coercive)
    if !iszero(noncompact[1])
        perturbation_coefficient = noncompact[1]
        # complex case? which kind of degree?
        if isempty(zero) && isempty(nonneg) && tighter === false
            perturbation_form = (1 + vars' * vars)^ceil(Int, maxdegree(objective)/2)
        else
            perturbation_form = (1 + vars' * vars)^(maxdegree(objective)÷2 +1)
        end
    end
    if !iszero(noncompact[2])
        factor_coercive = (1 + vars' * vars)^noncompact[2]
    end
    if !iszero(perturbation_coefficient)
        objective += perturbation_coefficient * perturbation_form
    end
    if !isone(factor_coercive)
        objective *= factor_coercive
    end
    if perturbation isa AbstractVector || !iszero(perturbation)
        @verbose_info("Constructing perturbation")
        # To extract the solution, we add small random linear terms to the objective. This lifts potential degeneracies and
        # allows for an easy extraction of the solution.
        if complex
            # the objective must not be complex-valued. We also don't want to "just" introduce an absolute-value
            # dependency, which would not lift phase degeneracies. So we introduce real and imaginary part
            # perturbations, but in the language of the conjugates.
            objective += sum(((rand(Float64, length(vars)) .- 0.5) .* perturbation) .* (vars .+ conj(vars))) +
                sum((im .* (rand(Float64, length(vars)) .- 0.5) .* perturbation) .* (vars .- conj(vars)))
        else
            objective += sum(((rand(Float64, length(vars)) .- 0.5) .* perturbation) .* vars)
        end
    end
    if tighter === true
        tighter = default_tightening_method()
    end
    #endregion

    #region Obtaining and setting degree information
    degrees_eqs = maxhalfdegree.(zero)
    degrees_nonnegs = maxhalfdegree.(nonneg)
    degrees_psds = convert.(Int, maxhalfdegree.(psd)) # somehow, this is the only expression that gives Any[] when empty
    mindeg = max(maxhalfdegree(objective), maximum(degrees_eqs, init=0), maximum(degrees_nonnegs, init=0),
            maximum(degrees_psds, init=0))
    if basis isa Symbol
        if iszero(degree)
            degree = mindeg
            @info("Automatically selecting minimal degree $degree for the relaxation")
        elseif degree < mindeg
            @warn("The minimum required degree for the relaxation hierarchy is $mindeg, but $degree was given. Should we use the minimum degree [Y], abort [N], or use different degree altogether [number]?")
            degree = request_degree(mindeg)
        end
    else
        degree = maxdegree(basis)
    end
    #endregion

    #region Tightening and degree adjustment
    if tighter !== false
        @verbose_info("Beginning tightening process: Constructing the matrix C out of the constraints")
        zero_len = length(zero)
        nonneg_len = length(nonneg)
        tighten!(tighter, objective, vars, degree, zero, nonneg; verbose)
        # tightening will lead to new zero constraints (of type simple) and new inequality constraints.
        @verbose_info("Tightening completed")
        @inbounds append!(degrees_eqs, maxhalfdegree.(@view(zero[zero_len+1:end])))
        @inbounds append!(degrees_nonnegs, maxhalfdegree.(@view(nonneg[nonneg_len+1:end])))
        if basis === :dense
            mindeg = max(mindeg, maximum(@view(degrees_eqs[zero_len+1:end]), init=0),
                maximum(@view(degrees_nonnegs[nonneg_len+1:end]), init=0))
            if degree < mindeg
                @warn("The mimimum required degree for the relaxation hierarchy has increased due to tightening and is now $mindeg, but $degree was given. Should we use the minimum degree [Y], abort [N], or use different degree altogether [number]?")
                degree = request_degree(mindeg)
            end
        end
    end
    #endregion

    #region SimplePolynomial conversion
    @verbose_info("Converting data to simple polynomials")
    max_power = 2degree
    sobj = SimplePolynomial(objective, T; max_power, vars, representation)
    if sobj isa SimplePolynomials.SimpleDensePolynomial
        representation = :dense
    else
        representation = :sparse
    end
    sprefactor = SimplePolynomial(factor_coercive, T; max_power, vars, representation)
    szero = FastVec{typeof(sobj)}(buffer=length(zero))
    for zeroᵢ in zero
        unsafe_push!(szero, SimplePolynomial(zeroᵢ, T; max_power, representation, vars))
    end
    snonneg = FastVec{typeof(sobj)}(buffer=length(nonneg))
    for nonnegᵢ in nonneg
        unsafe_push!(snonneg, SimplePolynomial(nonnegᵢ, T; max_power, representation, vars))
    end
    spsd = FastVec{Matrix{typeof(sobj)}}(buffer=length(psd))
    for psdᵢ in psd
        m = Matrix{typeof(sobj)}(undef, size(psdᵢ)...)
        for j in eachindex(psdᵢ, m)
            @inbounds m[j] = SimplePolynomial(psdᵢ[j], T; max_power, representation, vars)
        end
        unsafe_push!(spsd, m)
    end
    #endregion

    #region Basis
    @verbose_info("Constructing basis")
    local sbasis
    if basis === :dense
        maxpower_T = SimplePolynomials.smallest_unsigned(max_power)
        sbasis = monomials(nreal, ncomplex, Base.zero(maxpower_T):maxpower_T(degree);
                           maxmultideg=[fill(maxpower_T(degree), nreal + ncomplex); zeros(maxpower_T, ncomplex)])
    elseif basis === :newton
        if degree > mindeg && isempty(snonneg) && isempty(spsd)
            @info("Requested degree $degree is ignored, as no constraints are present.")
        end
        if !iszero(ncomplex) && !iszero(nreal)
            # Well, we could do this. For a polynomial ∑ᵢⱼₖ αᵢⱼₖ xⁱ zʲ z̄ᵏ, we could factor the complex valued part and then
            # apply the Newton polytope to the real factor: ∑ⱼₖ NP(∑ᵢ αᵢⱼₖ xⁱ) zʲ z̄ᵏ; and then simplify the complex valued part.
            # Not implemented at the moment.
            error("Mixing real- and complex-valued variables prevents Newton polytope methods")
        end
        let
            if !iszero(ncomplex)
                newton_method = :complex
            elseif isempty(newton_args)
                newton_method = default_newton_method()
            else
                newton_method = newton_args[1]
            end
            if length(newton_args) > 1
                newton_kwargs = newton_args[2]
            else
                newton_kwargs = ()
            end
            sbasis = newton_halfpolytope(newton_method, sobj; zero=szero, nonneg=snonneg, psd=spsd, degree, verbose,
                newton_kwargs...)
            sbasis isa SimpleMonomialVector ||
                error("Newton polytope calculation did not give results. Were the results written to a file?")
        end
    else
        issorted(basis, by=degree) || error("Any custom basis must be sorted by degree")
        if !(basis isa SimpleMonomialVector)
            # If we already have a custom basis, then the largest exponent in this basis will be doubled in the moment matrix -
            # this defines the data type.
            max_power = 0
            for m in basis
                max_power = max(max_power, maximum(exponents(m), init=0))
            end
            max_power *= 2
            sbasis = SimpleMonomialVector(basis; max_power, vars)
            all(iszero, sbasis.exponents_conj) || error("Any custom basis must not contain explicit conjugates")
        end
    end
    #endregion

    return POProblem{typeof(sobj),typeof(sbasis),eltype(vars)}(sobj, sprefactor, degree, sbasis, finish!(szero),
        finish!(snonneg), finish!(spsd), vars)
end

dense_problem(problem::POProblem) = problem

function poly_structure_indices(problem::POProblem, objective::Bool, zero::Union{Bool,<:AbstractSet{<:Integer}},
    nonneg::Union{Bool,<:AbstractSet{<:Integer}}, psd::Union{Bool,<:AbstractSet{<:Integer}})
    len = objective ? 1 : 0
    if zero === true
        len += count(c -> c.type ∈ (pctEqualitySimple, pctEqualityGröbner, pctEqualityNonneg), problem.constraints)
    elseif zero !== false
        len += length(zero)
    end
    if nonneg === true
        len += count(c -> c.type == pctNonneg, problem.constraints)
    elseif nonneg !== false
        len += length(nonneg)
    end
    if psd === true
        len += count(c -> c.type == pctPSD, problem.constraints)
    elseif psd !== false
        len += length(psd)
    end

    indices = FastVec{Int}(buffer=len)
    objective && unsafe_push!(indices, 1)
    eq_idx = 1
    nonneg_idx = 1
    psd_idx = 1
    for (i, constr) in enumerate(problem.constraints)
        if constr.type == pctNonneg
            (nonneg === true || nonneg_idx ∈ nonneg) && unsafe_push!(indices, i +1)
            nonneg_idx += 1
        elseif constr.type == pctPSD
            (psd === true || psd_idx ∈ psd) && unsafe_push!(indices, i +1)
            psd_idx += 1
        else
            (zero === true || eq_idx ∈ zero) && unsafe_push!(indices, i +1)
            eq_idx += 1
        end
    end
    @assert(len == length(indices))
    return finish!(indices)
end