export RelaxationDense

struct RelaxationDense{P<:POProblem,MV<:SimpleMonomialVector,G<:RelaxationGroupings} <: AbstractRelaxationDegree{P}
    problem::P
    degree::Int
    basis::MV
    groupings::G

    @doc """
        RelaxationDense(problem::POProblem[, degree])

    Constructs a full dense relaxation out of a polynomial optimization problem. This is the largest possible representation
    for a given degree bound, giving the best bounds. It is wastful at the same time, as a Newton relaxation gives equally good
    bounds; but contrary to the Newton one, solution reconstruction works much better with a dense basis.
    `degree` is the degree of the Lasserre relaxation, which must be larger or equal to the halfdegree of all polynomials that
    are involved. If `degree` is omitted, the minimum required degree will be used.
    Specifying a degree larger than the minimal only makes sense if there are inequality or PSD constraints present, else it
    needlessly complicates calculations without any benefit.
    """
    function RelaxationDense(problem::P,
        degree::Integer=(@info("Automatically selecting minimal degree cutoff $(problem.mindegree)"); problem.mindegree)) where
        {Nr,Nc,Poly<:SimplePolynomial{<:Any,Nr,Nc},P<:POProblem{Poly}}
        degree < problem.mindegree && throw(ArgumentError("The minimally required degree is $(problem.mindegree)"))
        basis = monomials(Val(Nr), Val(Nc), 0:degree, maxmultideg=[SimplePolynomials.ConstantVector(degree, Nr + Nc);
                                                                   SimplePolynomials.ConstantVector(0, Nc)])
        gr = groupings(problem, basis, degree, nothing)
        new{P,typeof(basis),typeof(gr)}(problem, Int(degree), basis, gr)
    end
end

default_solution_method(::RelaxationDense) = :mvhankel