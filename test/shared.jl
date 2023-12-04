using Test
using PolynomialOptimization
using MultivariatePolynomials
import DynamicPolynomials
import Mosek, COSMO, Hypatia, COPT

if !@isdefined(all_solvers)
    optimize = true

    if try
        Mosek.maketask() do t
            Mosek.optimize(t)
        end
        true
    catch e
        if e isa Mosek.MosekError && Mosek.MSK_RES_ERR_LICENSE.value ≤ e.rcode ≤ Mosek.MSK_RES_ERR_LICENSE_NO_SERVER_LINE.value
            false
        else
            rethrow(e)
        end
    end
        all_solvers = [:MosekMoment, :MosekSOS, :COSMOMoment, :HypatiaMoment, :COPTSOS]
        complex_solvers = [:MosekMoment, :HypatiaMoment]
    else
        all_solvers = [:COSMOMoment, :HypatiaMoment]
        complex_solvers = [:HypatiaMoment]
    end

    function strRep(x)
        io = IOBuffer()
        show(IOContext(io, :limit => false, :displaysize => (10, 10)), "text/plain", x)
        return String(take!(io))
    end
end