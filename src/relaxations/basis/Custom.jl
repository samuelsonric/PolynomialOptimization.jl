struct CustomRelaxation{P<:POProblem,MV<:SimpleMonomialVector} <: AbstractBasisRelaxation{P}
    problem::P
    degree::Int
    basis::MV

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
        new{P,typeof(basis)}(problem, degree, basis)
    end
end

default_solution_method(::CustomRelaxation) = :heuristic