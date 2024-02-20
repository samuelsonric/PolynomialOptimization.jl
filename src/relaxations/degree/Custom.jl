struct RelaxationCustom{P<:POProblem,MV<:SimpleMonomialVector,G<:RelaxationGroupings} <: AbstractRelaxationDegree{P}
    problem::P
    degree::Int
    basis::MV
    groupings::G

    @doc """
        RelaxationCustom(problem::POProblem, basis)

    Constructs a relaxation out of a polynomial optimization problem for the case in which a suitable basis is already known.
    """
    function RelaxationCustom(problem::P,
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
            basis = SimpleMonomialVector(basis; max_power, problem.original_variables)
            all(iszero, basis.exponents_conj) || throw(ArgumentError("Any custom basis must not contain explicit conjugates"))
        end
        degree = maxdegree(basis) # ≡ maxdegree_complex, but maxdegree cannot break branch prediction
        gr = groupings(problem, basis, degree, nothing)
        new{P,typeof(basis),typeof(gr)}(problem, Int(degree), basis, gr)
    end
end

default_solution_method(::RelaxationCustom) = :heuristic