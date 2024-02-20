struct CustomRelaxation{P<:POProblem,MV<:SimpleMonomialVector} <: AbstractBasisRelaxation{P}
    problem::P
    degree::Int
    basis::MV
    zero_basis::MV

    @doc """
        CustomRelaxation(problem, basis)

    Constructs a relaxation out of a polynomial optimization problem for the case in which a suitable basis is already known.
    """
    function CustomRelaxation(problem::P,
        basis::Union{<:SimpleMonomialVector{Nr,Nc},<:AbstractVector{<:AbstractMonomialLike}}) where {Nr,Nc,Poly<:SimplePolynomial{<:Any,Nr,Nc},P<:POProblem{Poly}}
        issorted(basis, by=degree) || throw(ArgumentError("Any custom basis must be sorted by degree"))
        if !(basis isa SimpleMonomialVector)
            # If we already have a custom basis, then the largest exponent in this basis will be doubled in the moment matrix -
            # this defines the data type.
            max_power = 0
            for m in basis
                max_power = max(max_power, maximum(exponents(m), init=0))
            end
            max_power *= 2
            basis = SimpleMonomialVector(basis; max_power, vars)
            all(iszero, basis.exponents_conj) || throw(ArgumentError("Any custom basis must not contain explicit conjugates"))
        end
        degree = Int(maxdegree(basis)) # ≡ maxdegree_complex, but maxdegree cannot break branch prediction
        # We still need to construct the basis for the zero constraints by squaring custom basis and directly eliminating the
        # duplicates.
        maxzerodeg, maxmultizerodeg = zero_maxdegs(problem.constr_zero, degree)
        zero_halfbasis = truncate_basis(basis, maxzerodeg)
        mmz_real = @view(maxmultizerodeg[1:Nr])
        mmz_complex = @view(maxmultizerodeg[Nr+1:Nr+Nc])
        mmz_conj = @view(maxmultizerodeg[Nr+Nc+1:Nr+2Nc])
        zero_basis = sizehint!(Set{UInt}(), length(zero_halfbasis)^2) # could also be l(l +1)÷2 if everything was real
        for b₁ in zero_halfbasis
            deg_r = sum(b₁.exponents_real, init=0)
            deg_c = sum(b₁.exponents_complex, init=0)
            all(t -> t[2] ≤ t[1], zip(mmz_complex, b₁.exponents_complex)) || continue
            for b₂ in zero_halfbasis
                b₁.exponents_complex ≤ b₂.exponents_complex || continue # iscanonical
                deg_r₂ = sum(b₂.exponents_real, init=0)
                deg_c₂ = sum(b₂.exponents_complex, init=0)
                deg_r + deg_r₂ + max(deg_c, deg_c₂) ≤ maxzerodeg || continue
                all(t -> t[2] + t[3] ≤ t[1], zip(mmz_real, b₁.exponents_real, b₂.exponents_real)) || continue
                all(t -> t[2] ≤ t[1], zip(mmz_conj, b₂.exponents_complex)) || continue
                push!(zero_basis, monomial_index(b₁, conj(b₂)))
            end
        end
        zero_basis_real = similar(basis.exponents_real, Nr, length(zero_basis))
        zero_basis_complex = similar(basis.exponents_complex, Nc, length(zero_basis))
        zero_basis_conj = similar(zero_basis_complex)
        powers = Vector{SimplePolynomials._get_p(basis)}(undef, Nr + 2Nc)
        powers_real = @view(powers[1:Nr])
        powers_complex = @view(powers[Nr+1:Nr+Nc])
        powers_conj = @view(powers[Nr+Nc+1:Nr+2Nc])
        @inbounds for (i, idx) in enumeratea(sort(zero_basis))
            exponents_from_index!(powers, idx)
            SimplePolynomials.isabsent(zero_basis_real) || copyto!(@view(zero_basis_real[:, i]), powers_real)
            if !SimplePolynomials.isabsent(zero_basis_complex)
                copyto!(@view(zero_basis_complex[:, i]), powers_complex)
                copyto!(@view(zero_basis_conj[:, i]), powers_conj)
            end
        end
        new{P,typeof(basis)}(problem, Int(degree), basis, SimpleMonomialVector{Nr,Nc}(zero_basis_real, zero_basis_complex,
            zero_basis_conj))
    end
end

default_solution_method(::CustomRelaxation) = :heuristic