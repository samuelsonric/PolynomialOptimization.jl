using Test
using PolynomialOptimization, PolynomialOptimization.SimplePolynomials
# ^ we don't require SimplePolynomials in the namespace, but for printing we want the guarantee that it is there
using MultivariatePolynomials
import DynamicPolynomials
import COPT, #= COSMO, =# Hypatia, Mosek, SCS
import StatsBase

if !@isdefined(solvers)
    optimize = true

    solvers = copy(PolynomialOptimization.Solver.solver_methods)
    # Mosek requires a license to work at all. COPT will work without a license for small problems.
    try
        Mosek.maketask() do t
            Mosek.optimize(t)
        end
    catch e
        if e isa Mosek.MosekError && Mosek.MSK_RES_ERR_LICENSE.value ≤ e.rcode ≤ Mosek.MSK_RES_ERR_LICENSE_NO_SERVER_LINE.value
            filter!(s -> !occursin("Mosek", string(s)), solvers)
        else
            rethrow(e)
        end
    end

    function strRep(x)
        io = IOBuffer()
        show(IOContext(io, :limit => false, :displaysize => (10, 10)), "text/plain", x)
        return String(take!(io))
    end
end