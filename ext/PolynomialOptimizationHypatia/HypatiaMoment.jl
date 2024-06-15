struct StateMoment{K<:Integer,V<:Real}
    Acoo::Tuple{FastVec{Int},FastVec{K},FastVec{V}}
    b::Tuple{FastVec{Int},FastVec{V}}
    minusGcoo::Tuple{FastVec{Int},FastVec{K},FastVec{V}}
    c::Ref{Tuple{Vector{K},Vector{V}}}
    cones::FastVec{Cones.Cone{V}}
end

Solver.mindex(::StateMoment{K}, monomials::SimpleMonomialOrConj{Nr,Nc}...) where {K,Nr,Nc} = monomial_index(monomials...)::K

Solver.supports_quadratic(::StateMoment) = true

Solver.supports_complex_psd(::StateMoment) = true

Solver.psd_indextype(::StateMoment) = PSDIndextypeVector(:U)

function Solver.add_constr_nonnegative!(state::StateMoment{K,V}, indices::AbstractVector{K}, values::AbstractVector{V}) where {K,V}
    @assert(length(indices) == length(values))
    prepare_push!(state.minusGcoo[1], length(indices))
    v = isempty(state.minusGcoo[1]) ? 1 : last(state.minusGcoo[1]) +1
    for _ in 1:length(indices)
        unsafe_push!(state.minusGcoo[1], v)
    end
    append!(state.minusGcoo[2], indices)
    append!(state.minusGcoo[3], values)
    push!(state.cones, Cones.Nonnegative{V}(1))
end

function Solver.add_constr_quadratic!(state::StateMoment{K,V}, indvals::Tuple{AbstractVector{K},AbstractVector{V}}...) where {K,V}
    @assert(all(t -> length(t[1]) == length(t[2]), indvals))
    prepare_push!(state.minusGcoo[1], sum(∘(length, first), indvals, init=0))
    v = isempty(state.minusGcoo[1]) ? 0 : last(state.minusGcoo[1])
    for (r, _) in indvals
        v += 1
        for _ in 1:length(r)
            unsafe_push!(state.minusGcoo[1], v)
        end
    end
    append!(state.minusGcoo[2], first.(indvals)...)
    append!(state.minusGcoo[3], last.(indvals)...)
    push!(state.cones, Cones.EpiPerSquare{V}(length(indvals)))
end

for fn in (:add_constr_psd!, :add_constr_psd_complex!)
    @eval begin
        function Solver.$fn(state::StateMoment{K,V}, dim::Int, data::PSDVector{K,V}) where {K,V}
            prepare_push!(state.minusGcoo[1], length(rowvals(data)))
            v = isempty(state.minusGcoo[1]) ? 0 : last(state.minusGcoo[1])
            for l in Base.index_lengths(data)
                v += 1
                for _ in 1:l
                    unsafe_push!(state.minusGcoo[1], v)
                end
            end
            append!(state.minusGcoo[2], rowvals(data))
            append!(state.minusGcoo[3], nonzeros(data))
            push!(state.cones, $(fn === :add_constr_psd! ? :(Cones.PosSemidefTri{V,V}(trisize(dim))) :
                                                           :(Cones.PosSemidefTri{V,Complex{V}}(dim^2))))
        end
    end
end

function Solver.add_constr_fix_prepare!(state::StateMoment, num::Int)
    # Those are all just lower bounds, as the number of constraints is multiplied by the individual index count. But better
    # than nothing.
    prepare_push!(state.Acoo[1], num)
    prepare_push!(state.Acoo[2], num)
    prepare_push!(state.Acoo[3], num)
    return
end

function Solver.add_constr_fix!(state::StateMoment{K,V}, ::Nothing, indices::AbstractVector{K}, values::AbstractVector{V},
    rhs::V) where {K,V}
    v = isempty(state.Acoo[1]) ? 1 : last(state.Acoo[1]) +1
    prepare_push!(state.Acoo[1], length(indices))
    for _ in 1:length(indices)
        unsafe_push!(state.Acoo[1], v)
    end
    append!(state.Acoo[2], indices)
    append!(state.Acoo[3], values)
    if !iszero(rhs)
        push!(state.b[1], v)
        push!(state.b[2], rhs)
    end
    return
end

Solver.fix_objective!(state::StateMoment{K,V}, indices::AbstractVector{K}, values::AbstractVector{V}) where {K,V} =
    state.c[] = (indices, values)

function consolidate_vecs(vec₁::AbstractVector{I}, vec₂::AbstractVector{I}, callback) where {I}
    # vec₁ and vec₂ are sorted vectors with possible duplicates. Iterate through all of them, count the unique ones, call the
    # callback for every unique entry with the respective last indices that correspond to this element.
    i₁ = 1
    i₂ = 1
    remaining₁ = length(vec₁)
    remaining₂ = length(vec₂)
    index = 1
    @inbounds while !iszero(remaining₁) && !iszero(remaining₂)
        cur₁ = vec₁[i₁]
        cur₂ = vec₂[i₂]
        # skip over duplicates
        while remaining₁ > 1 && vec₁[i₁+1] == cur₁
            i₁ += 1; remaining₁ -= 1
        end
        while remaining₂ > 1 && vec₂[i₂+1] == cur₂
            i₂ += 1; remaining₂ -= 1
        end
        # and work with the smaller one until it is no longer the smaller one
        if cur₁ == cur₂
            @inline callback(index, i₁, i₂)
            i₁ += 1; remaining₁ -= 1
            i₂ += 1; remaining₂ -= 1
            index += 1
        elseif cur₁ < cur₂
            @inline callback(index, i₁, missing)
            index += 1
            while !iszero(remaining₁)
                i₁ += 1; remaining₁ -= 1
                cur₁ = vec₁[i₁]
                cur₁ ≥ cur₂ && break
                # skip over duplicates
                while remaining₁ > 1 && vec₁[i₁+1] == cur₁
                    i₁ += 1; remaining₁ -= 1
                end
                @inline callback(index, i₁, missing)
                index += 1
            end
        else
            @inline callback(index, missing, i₂)
            index += 1
            while !iszero(remaining₂)
                i₂ += 1; remaining₂ -= 1
                cur₂ = vec₂[i₂]
                cur₂ ≥ cur₁ && break
                # skip over duplicates
                while remaining₂ > 1 && vec₂[i₂+1] == cur₂
                    i₂ += 1; remaining₂ -= 1
                end
                @inline callback(index, missing, i₂)
                index += 1
            end
        end
    end
    # tail checks. At most one of the two still has elements.
    @inbounds while !iszero(remaining₁)
        cur₁ = vec₁[i₁]
        # skip over duplicates
        while remaining₁ > 1 && vec₁[i₁+1] == cur₁
            i₁ += 1; remaining₁ -= 1
        end
        @inline callback(index, i₁, missing)
        i₁ += 1; remaining₁ -= 1
        index += 1
    end
    @inbounds while !iszero(remaining₂)
        cur₂ = vec₂[i₂]
        # skip over duplicates
        while remaining₂ > 1 && vec₂[i₂+1] == cur₂
            i₂ += 1; remaining₂ -= 1
        end
        @inline callback(index, missing, i₂)
        i₂ += 1; remaining₂ -= 1
        index += 1
    end
    return index -1 # return the count
end

function Solver.poly_optimize(::Val{:HypatiaMoment}, relaxation::AbstractRelaxation,
    groupings::RelaxationGroupings; verbose::Bool=false, dense::Bool=!isone(poly_problem(relaxation).prefactor),
    customize::Function=_ -> nothing, parameters...)
    setup_time = @elapsed begin
        K = _get_I(eltype(monomials(poly_problem(relaxation).objective)))
        V = Solver.realtype(coefficient_type(poly_problem(relaxation).objective))
        state = StateMoment(
            (FastVec{Int}(), FastVec{K}(), FastVec{V}()),
            (FastVec{Int}(), FastVec{V}()),
            (FastVec{Int}(), FastVec{K}(), FastVec{V}()),
            Ref{Tuple{Vector{K},Vector{V}}}(),
            FastVec{Cones.Cone{V}}()
        )

        moment_setup!(state, relaxation, groupings)
        customize(state)

        # Now we have all the data in COO form. The reason for this choice is that we were able to assign arbitrary column
        # indices - i.e., we could just use the monomial index. However, now we have to modify the column indices to make them
        # consecutive, removing all monomials that do not occur. We already know that no entry will ever occur twice, so we can
        # make our own optimized COO -> CSC function.
        Acoo = finish!.(state.Acoo)
        minusGcoo = finish!.(state.minusGcoo)
        Ccoo = state.c[]
        b = zeros(V, isempty(Acoo[1]) ? 0 : last(Acoo[1]))
        copy!(@view(b[state.b[1]]), state.b[2])
        h = zeros(V, isempty(minusGcoo[1]) ? 0 : last(minusGcoo[1]))

        sort_along!(Acoo[2], Acoo[1], Acoo[3], relevant=2) # sort according to col indices, but keep row indices in order
                                                           # (sort_along is in general not order-preserving, so use row indices
                                                           # as secondary option)
        sort_along!(minusGcoo[2], minusGcoo[1], minusGcoo[3], relevant=2)
        sort_along!(Ccoo[1], Ccoo[2])
        # how many distinct monomials do we have?
        moncount = consolidate_vecs(Acoo[2], minusGcoo[2], (_, _, _) -> nothing)
        # We only need to construct the colptrs - Acoo[1] is already the correct rowval and Acoo[3] the correct nonzeros.
        Acolptr = Vector{Int}(undef, moncount +1)
        Gcolptr = similar(Acolptr)
        @inbounds Acolptr[1] = 1
        @inbounds Gcolptr[1] = 1
        c = zeros(V, moncount)
        ic = Ref(1)
        consolidate_vecs(Acoo[2], minusGcoo[2], @capture((index, ia, ig) -> @inbounds begin
            colidx = ismissing(ia) ? $minusGcoo[2][ig] : $Acoo[2][ia]
            if $ic[] ≤ length($Ccoo[1]) && Ccoo[1][ic[]] == colidx
                $c[index] = Ccoo[2][ic[]]
                ic[] += 1
            end
            $Acolptr[index+1] = ismissing(ia) ? $Acolptr[index] : ia +1
            $Gcolptr[index+1] = ismissing(ig) ? $Gcolptr[index] : ig +1
        end))

        model = Models.Model{V}(
            c, # c
            SparseMatrixCSC{V,Int}(length(b), moncount, Acolptr, Acoo[1], Acoo[3]), # A
            b, # b
            SparseMatrixCSC{V,Int}(length(h), moncount, Gcolptr, minusGcoo[1], rmul!(minusGcoo[3], -one(V))), # G
            h, # h
            finish!(state.cones) # cones
        )
        if !dense
            # for lots of smaller constraints, a sparse solver is much better. However, all non-QRCholDenseSystemSolver
            # types also require to turn off reduction (else, we just get completely wrong results), and performing a dense
            # preprocessing also defeats the purpose of a sparse solver.
            parameters = (syssolver=get(() -> Solvers.SymIndefSparseSystemSolver{Float64}(), parameters, :syssolver),
                preprocess=get(parameters, :preprocess, false), reduce=get(parameters, :reduce, false), parameters...)
        end
        solver = Solvers.Solver{V}(; verbose, parameters...)
        Solvers.load(solver, model)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    Solvers.solve(solver)
    status = Solvers.get_status(solver)
    value = Solvers.get_primal_obj(solver)
    @verbose_info("Optimization complete, retrieving moments")

    max_mons = monomial_count(nvariables(relaxation.objective), 2degree(relaxation))
    x = Solvers.get_x(solver)
    if length(x) == max_mons # dense case
        solution = x
    else
        # We need to build the vector of monomial indices.
        mon_pos = Vector{Int}(undef, moncount)
        consolidate_vecs(Acoo[2], minusGcoo[2], @capture((index, ia, ig) -> @inbounds begin
            $mon_pos[index] = ismissing(ia) ? $minusGcoo[2][ig] : $Acoo[2][ia]
        end))
        if 3length(x) < max_mons
            solution = SparseVector(max_mons, mon_pos, x)
        else
            solution = fill(NaN, max_mons)
            copy!(@view(solution[mon_pos]), x)
        end
    end
    return status, value, MomentVector(relaxation, solution)
end