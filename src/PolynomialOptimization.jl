__precompile__()
module PolynomialOptimization

using MultivariatePolynomials
using SparseArrays
using LinearAlgebra
using Printf
import Graphs
import Combinatorics
import MutableArithmetics
import StatsBase

export Newton

const sqrt2 = sqrt(2.0)
const haveMPI = Ref{Bool}(false)
const debugging = true

if debugging
    macro assert(args...)
        :(Base.@assert($(esc.(args)...)))
    end
else
    macro assert(args...)
    end
end

macro myinbounds(expr)
    esc(expr)
end

macro verbose_info(str...)
    quote
        if $(esc(:verbose))
            println($(esc.(str)...))
            flush(stdout)
        end
    end
end

# Simpler version of FastClosure's @closure: We just require that the variable be marked at least once with an interpolation
# sign. In this way, there's precise control of what will be captured. The code here is almost identical to
# Base._lift_one_interp!, but we descend into macros (allowing for defining an escape list), and we also take care of duplicate
# interpolations. Once a variable was marked with an interpolation sign at any position, all occurrences are captured.
function _lift_interps!(e, escape_macros)
    letargs = Set{Any}()  # store the new gensymed arguments
    _lift_interps_helper(e, false, letargs, escape_macros) # Start out _not_ in a quote context (false)
    return letargs
end
_lift_interps_helper(v, _, _, escape_macros) = v
function _lift_interps_helper(expr::Expr, in_quote_context, letargs, escape_macros)
    if expr.head === :$
        if in_quote_context  # This $ is simply interpolating out of the quote
            # Now, we're out of the quote, so any _further_ $ is ours.
            in_quote_context = false
        else
            push!(letargs, :($(esc(expr.args[1])) = $(esc(expr.args[1]))))
            return expr.args[1] # Don't recurse into the lifted $() exprs
        end
    elseif expr.head === :quote
        in_quote_context = true   # Don't try to lift $ directly out of quotes
    elseif expr.head === :macrocall && expr.args[1] ∈ escape_macros
        return expr # Don't recur into escaped macro calls, since some other macros use $
    end
    for (i,e) in enumerate(expr.args)
        expr.args[i] = _lift_interps_helper(e, in_quote_context, letargs, escape_macros)
    end
    return expr
end

macro capture(firstarg, secondarg=nothing)
    if isnothing(secondarg)
        expr = firstarg
        escape_macros = Set{Symbol}()
    else
        (firstarg.head === :(=) && length(firstarg.args) === 2) || throw(ArgumentError("Invalid use of @capture"))
        firstarg.args[1] === :escape || throw(ArgumentError("Invalid keyword for @capture: $(firstarg.args[1])"))
        firstarg.args[2] isa Symbol || firstargs.args[2] ∈ (:vect, :tuple) ||
            throw(ArgumentError("Invalid value for keyword escape"))
        expr = secondarg
        escape_macros = Set{Symbol}(firstarg.args[2])
    end
    letargs = _lift_interps!(expr, escape_macros)
    quote
        let $(letargs...)
            $(esc(expr))
        end
    end
end

include("./helpers/Mutation.jl")
include("./helpers/FastVector.jl")
using .FastVector
include("./poly/SimplePolynomials.jl")
using .SimplePolynomials
using .SimplePolynomials: SimpleRealPolynomial, SimpleComplexPolynomial, SimpleRealMonomial
include("./helpers/StackVector.jl")
include("./helpers/FastKey.jl")
include("./helpers/SortAlong.jl")
include("./helpers/MatrixPolynomials.jl")
include("./helpers/Allocations.jl")

include("./Problem.jl")
include("./relaxations/Relaxation.jl")
include("./optimization/Optimization.jl")
include("./Tightening.jl")
include("./newton/Newton.jl")
import .Newton

end