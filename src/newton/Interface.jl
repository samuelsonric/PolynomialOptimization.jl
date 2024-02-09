"""
    preproc_quick(V, coeffs::AbstractMatrix{<:Integer}, vertexindices::Vector{Int},
        verbose::Bool; parameters...)

Eliminate all the coefficients that by the Akl-Toussaint heuristic cannot be part of the convex hull anyway. The implementation
has to return a `Vector{Bool}` that for every column in `coeffs` indicates whether this particular column can be obtained as
the convex combinations of the columns indexed by `vertexindices` (`true` if the answer is yes).
"""
function preproc_quick end

"""
    preproc_remove(V, nv::Int, nc::Int, getvarcon::Function, verbose::Bool,
        singlethread::Bool; parameters...)

Removes all convex dependencies from the list of coefficients. There are `nc` coefficients, each with `nv` entries of Integer
type. The `AbstractVector` representing the `i`th coefficient (`1 ≤ i ≤ nc`) can be obtained by calling `getvarcon(i)`.
The implementation has to return a `Vector{Bool}` of length `nc` that for every coefficient contains `false` if and only if the
convex hull spanned by all coefficients is invariant under removal of this coefficient.
"""
function preproc_remove end

"""
    prepare(V, coeffs::AbstractMatrix{<:Integer}, num::Int, verbose::Bool; parameters...)

This function is responsible for creating an optimization task that can be used to check membership in the Newton halfpolytope.
The vertices of the polytope are given by the column of `coeffs`. The total number of monomials that have to be checked is
given by `num`.
The function must return the number of threads that will be used to carry out the optimization (which is not the number of
threads that the optimizer uses internally, but determines how `PolynomialOptimization` will distribute the jobs), an internal
optimization task that is passed on to [`work`](@ref), and a copy of this task for use in a second thread if the number of
threads is greater than one, else `nothing`. More copies will be created as required by [`clonetask`](@ref); however, assuming
that setting up the task will potentially require more resources than cloning a task allows the function to estimate the
required memory (and therefore a sensible number of threads) better by already performing one clone.

See also [`@allocdiff`](@ref).
"""
function prepare end

"""
    alloc_global(V, nv)

This function is called once in the main thread before [`work`](@ref) is executed (which, due to multithreading, might occur
more than once). It can create some shared data that is used in a read-only manner by all workers at the same time.
The default implementation of this function does nothing.
"""
alloc_global(_, _) = nothing

"""
    alloc_local(V, nv)

This function is called once in every computation thread before [`work`](@ref) is executed (which, due to task splitting,
might occur more than once). It can create some shared data that is available for reading and writing by every worker.
The default implementation of this function does nothing.
"""
alloc_local(_, _) = nothing

"""
    clonetask(t)

This function must create a copy of the optimization task `t` that can run in parallel to `t` in a different thread.
"""
function clonetask end

"""
    work(V, task, data_global, data_local, moniter::MonomialIterator,
        Δprogress::Ref{Int}, Δacceptance::Ref{Int}, add_callback::Function,
        iteration_callback::Union{Nothing,Function})

Iterates through `moniter` and for every monomial checks whether this set of exponents can be reached by a convex combination
of the coefficients as they are set up in `task`. If yes, `add_callback` must be called with the exponents as a parameter, and
`Δacceptance` should be incremented. In any case, `Δprogress` should be incremented. Additionally, after every check,
`iteration_callback` should be called with the exponents of the iteration as a parameter, if it is a function.
The `data` parameters contain the custom data that was previously generated using [`alloc_global`](@ref) and
[`alloc_local`](@ref). Only `data_local` may be mutated.
"""
function work end
