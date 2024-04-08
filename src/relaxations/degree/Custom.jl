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
        basis::Union{<:SimpleMonomialVector{Nr,Nc},<:AbstractVector{<:AbstractMonomialLike}}) where {Nr,Nc,I<:Integer,Poly<:SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}},P<:POProblem{Poly}}
        issorted(basis, by=degree) || throw(ArgumentError("Any custom basis must be sorted by degree"))
        if !(basis isa SimpleMonomialVector)
            basis = SimpleMonomialVector{I}(basis; vars=problem.original_variables)
            if !iszero(Nc)
                for v in effective_variables(basis)
                    isconj(v) && throw(ArgumentError("Any custom basis must not contain explicit conjugates"))
                end
            end
        end
        degree = maxdegree(basis) # ≡ maxdegree_complex, but maxdegree cannot break branch prediction
        gr = groupings(problem, basis, degree, nothing)
        new{P,typeof(basis),typeof(gr)}(problem, Int(degree), basis, gr)
    end
end

default_solution_method(::RelaxationCustom) = :heuristic