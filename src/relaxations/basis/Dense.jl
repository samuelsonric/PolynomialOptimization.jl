export DenseRelaxation

struct DenseRelaxation{P<:POProblem,MV<:SimpleMonomialVector} <: AbstractBasisRelaxation{P}
    problem::P
    degree::Int
    basis::MV
    zero_basis::MV

    @doc """
        DenseRelaxation(problem[, degree])

    Constructs a full dense relaxation out of a polynomial optimization problem. This is the largest possible representation
    for a given degree bound, giving the best bounds. It is wastful at the same time, as a Newton relaxation gives equally good
    bounds; but contrary to the Newton one, solution reconstruction works much better with a dense basis.
    `degree` is the degree of the Lasserre relaxation, which must be larger or equal to half of the (complex) degree of all
    polynomials that are involved. If `degree` is omitted, the minimum required degree will be used.
    A nonzero value only makes sense if there are inequality or PSD constraints present, else it needlessly complicates
    calculations without any benefit.
    """
    function DenseRelaxation(problem::P, degree::Integer=problem.mindegree) where {Nr,Nc,Poly<:SimplePolynomial{<:Any,Nr,Nc},P<:POProblem{Poly}}
        degree < problem.mindegree && throw(ArgumentError("The minimally required degree is $(problem.mindegree)"))
        maxpower_T = SimplePolynomials.smallest_unsigned(2degree)
        basis = monomials(Nr, Nc, Base.zero(maxpower_T):maxpower_T(degree);
                          maxmultideg=[fill(maxpower_T(degree), Nr + Nc); zeros(maxpower_T, Nc)])
        maxzerodeg, maxmultizerodeg = zero_maxdegs(problem.constr_zero, degree)
        zero_basis = monomials(Nr, Nc, Base.zero(maxpower_T):maxpower_T(maxzerodeg); maxmultideg=maxmultizerodeg,
            representation=basis isa SimplePolynomials.SimpleDenseMonomialVector ? :dense : :sparse,
            filter=powers -> @inbounds(@view(powers[Nr+1:Nr+Nc]) ≤ @view(powers[Nr+Nc+1:end])))
        new{P,typeof(basis)}(problem, Int(degree), basis, zero_basis)
    end
end

default_solution_method(::DenseRelaxation) = :mvhankel