mutable struct HypatiaStateMoment{M}
    variable_mappings::AbstractDict{M,Int}
    num_constr::Int
    constraints::Vector{Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Float64},Hypatia.Cones.Cone{Float64}}}
end

get_variable!(hsm::HypatiaStateMoment{M}, mon::M) where {M} =
    get!(hsm.variable_mappings, mon, length(hsm.variable_mappings) + 1)

# This is a bit tricky. All variables are real-valued, so we need two variables per monomial for the representation in the
# solver. At the same time, the monomial and its conjugate are considered independent variables. Of course, we now need to
# avoid having four variables in the solver, so we determine a "canonical" representation of the monomial to fix which is _the_
# monomial and which is "just" the conjugate. As a consequence, we cannot only give back the index of the monomial, but also
# need to incorporate a possible sign flip
function get_variable!(hsm::HypatiaStateMoment{MonomialComplexContainer{M}}, mon::M, coefficient::R, re::Bool) where {M,R<:Real}
    mon2 = conj(mon)
    if mon == mon2 && !re
        return missing
    elseif mon < mon2
        return get!(hsm.variable_mappings, MonomialComplexContainer(mon, re), length(hsm.variable_mappings) + 1) => coefficient
    else
        return get!(hsm.variable_mappings, MonomialComplexContainer(mon2, re),
            length(hsm.variable_mappings) + 1) => (re ? coefficient : -coefficient)
    end
end

function get_varrepr!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64}, hsm::HypatiaStateMoment{M}, gröbner_basis,
    row::M, col::M, constraint::P) where {P,M}
    constr_terms = mergewith(+, (get_variable!(hsm, monomial(term)) => -coefficient(term)
                                 for mon_constr in constraint
                                 for term in rem(row * conj(col) * mon_constr, gröbner_basis)),
        Int, Float64)
    hsm.num_constr += 1
    append!(rows, Iterators.repeated(hsm.num_constr, length(constr_terms)))
    append!(cols, keys(constr_terms))
    append!(vals, values(constr_terms))
    return
end

function get_varrepr!(rows::Vector{Int}, cols::Vector{Int}, vals::Vector{Float64},
    hsm::HypatiaStateMoment{MonomialComplexContainer{M}}, gröbner_basis, row::M, col::M, constraint::P, ensure_real::Bool) where {P,M}
    if ensure_real
        # This assert should in principle be valid and might be helpful in finding errors. However, we should check this
        # modulo the gröbner_basis; but this means that we'd have to resubstitute the real and imaginary parts (that are
        # introduced by imag) by sums/differences of the complex variables and their conjugates. This is a lot of work for a
        # simple assert. Well, complex gröbner_basis is disabled at the moment, but still...
        #@assert(imag(row*conj(col)*constraint) == 0)
        constr_terms = mergewith(+, (p for mon_constr in constraint
                                     for term in rem(row * conj(col) * mon_constr, gröbner_basis)
                                     for p in (get_variable!(hsm, monomial(term), -real(coefficient(term)), true),
                                               get_variable!(hsm, monomial(term), imag(coefficient(term)), false))
                                     if !ismissing(p)),
            Int, Float64)
        hsm.num_constr += 1
        append!(rows, Iterators.repeated(hsm.num_constr, length(constr_terms)))
        append!(cols, keys(constr_terms))
        append!(vals, values(constr_terms))
    else
        constr_terms_re = Dict{Int,Float64}()
        constr_terms_im = Dict{Int,Float64}()
        for mon_constr in constraint
            for term in rem(row * conj(col) * mon_constr, gröbner_basis)
                for p in (get_variable!(hsm, monomial(term), -real(coefficient(term)), true),
                    get_variable!(hsm, monomial(term), imag(coefficient(term)), false))
                    if !ismissing(p)
                        constr_terms_re[p.first] = haskey(constr_terms_re, p.first) ? constr_terms_re[p.first] + p.second : p.second
                    end
                end
                for p in (get_variable!(hsm, monomial(term), -real(coefficient(term)), false),
                    get_variable!(hsm, monomial(term), -imag(coefficient(term)), true))
                    if !ismissing(p)
                        constr_terms_im[p.first] = haskey(constr_terms_im, p.first) ? constr_terms_im[p.first] + p.second : p.second
                    end
                end
            end
        end
        hsm.num_constr += 2
        append!(rows, Iterators.repeated(hsm.num_constr - 1, length(constr_terms_re)),
            Iterators.repeated(hsm.num_constr, length(constr_terms_im)))
        append!(cols, keys(constr_terms_re), keys(constr_terms_im))
        append!(vals, values(constr_terms_re), values(constr_terms_im))
    end
    return
end

function moment_matrix!(hsm::HypatiaStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    lg = length(grouping)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if lg == 1
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint)
        push!(hsm.constraints, (rows, cols, vals, [0.0], Hypatia.Cones.Nonnegative{Float64}(1)))
    elseif lg == 2
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint / 2)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[2], grouping[2], constraint)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[2], grouping[1], constraint)
        push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1), Hypatia.Cones.EpiPerSquare{Float64}(3)))
    else
        for exp2 in 1:lg
            for exp1 in 1:exp2
                @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2],
                    exp1 == exp2 ? constraint : sqrt2 * constraint)
            end
        end
        @inbounds push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1),
            Hypatia.Cones.PosSemidefTri{Float64,Float64}((lg * (lg + 1)) >> 1)))
    end
    return
end

function moment_matrix!(hsm::HypatiaStateMoment{MonomialComplexContainer{M}}, gröbner_basis, grouping::AbstractVector{M},
    constraint::P) where {P,M}
    lg = length(grouping)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if lg == 1
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint, true)
        push!(hsm.constraints, (rows, cols, vals, [0.0], Hypatia.Cones.Nonnegative{Float64}(1)))
    elseif lg == 2
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint / 2, true)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[2], grouping[2], constraint, true)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[2], grouping[1], constraint, false)
        push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1), Hypatia.Cones.EpiPerSquare{Float64}(4)))
    else
        for exp2 in 1:lg
            for exp1 in 1:exp2
                @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2],
                    exp1 == exp2 ? constraint : sqrt2 * constraint, exp1 == exp2)
            end
        end
        @inbounds push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1),
            Hypatia.Cones.PosSemidefTri{Float64,ComplexF64}(lg * lg)))
    end
    return
end

function moment_matrix!(hsm::HypatiaStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return moment_matrix!(hsm, gröbner_basis, grouping, constraint[1, 1])
    end
    lg = length(grouping)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if lg == 1 && block_size == 2
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[1, 1] / 2)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[2, 2])
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[2, 1])
        push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1), Hypatia.Cones.EpiPerSquare{Float64}(3)))
    else
        dim = lg * block_size
        for exp2 in 1:lg
            for block_j in 1:block_size
                for exp1 in 1:exp2
                    for block_i in 1:(exp1 == exp2 ? block_j : block_size)
                        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2],
                            exp2 == exp1 && block_j == block_i ? constraint[block_i, block_i] : sqrt2 * constraint[block_i, block_j])
                    end
                end
            end
        end
        @inbounds push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1),
            Hypatia.Cones.PosSemidefTri{Float64,Float64}((dim * (dim + 1)) >> 1)))
    end
    return
end

function moment_matrix!(hsm::HypatiaStateMoment{MonomialComplexContainer{M}}, gröbner_basis, grouping::AbstractVector{M},
    constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return moment_matrix!(hsm, gröbner_basis, grouping, constraint[1, 1])
    end
    lg = length(grouping)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    if lg == 1 && block_size == 2
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[1, 1] / 2, true)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[2, 2], true)
        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[1], grouping[1], constraint[2, 1], false)
        push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1), Hypatia.Cones.EpiPerSquare{Float64}(4)))
    else
        dim = lg * block_size
        for exp2 in 1:lg
            for block_j in 1:block_size
                for exp1 in 1:exp2
                    for block_i in 1:(exp1 == exp2 ? block_j : block_size)
                        diag = exp1 == exp2 && block_i == block_j
                        @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2],
                            diag ? constraint[block_i, block_j] : sqrt2 * constraint[block_i, block_j], diag)
                    end
                end
            end
        end
        @inbounds push!(hsm.constraints, (rows, cols, vals, fill(0.0, rows[end] - rows[1] + 1),
            Hypatia.Cones.PosSemidefTri{Float64,ComplexF64}(dim * dim)))
    end
    return
end

function moment_matrix_eq!(hsm::HypatiaStateMoment{C}, A, b, gröbner_basis, grouping::AbstractVector, constraint) where {C}
    lg = length(grouping)
    rows, cols, vals = A
    before = length(rows)
    # to use get_varrepr!, we hijack num_constr
    old_numconstr = hsm.num_constr
    hsm.num_constr = rows[end] # maximum(rows) is more correct, but we always add in increasing order
    for exp2 in 1:lg
        for exp1 in 1:exp2
            if C <: MonomialComplexContainer
                @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2], constraint,
                    exp2 == exp1)
            else
                @inbounds get_varrepr!(rows, cols, vals, hsm, gröbner_basis, grouping[exp1], grouping[exp2], constraint)
            end
        end
    end
    hsm.num_constr = old_numconstr
    @inbounds append!(b, Iterators.repeated(zero(eltype(b)), rows[end] - rows[before+1] +1))
    return
end

function sparse_optimize(::Val{:HypatiaMoment}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, dense::Bool=!isone(problem.prefactor),
    customize::Function=(hsm, c, A, b) -> (c, A, b), parameters...) where {P,M,V}
    setup_time = @elapsed begin
        hsm = HypatiaStateMoment(Dict{problem.complex ? MonomialComplexContainer{M} : M,Int}(), 0,
            Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Float64},Hypatia.Cones.Cone{Float64}}[])

        # Riesz functional of objective mod ideal. Note that during problem setup, the objective (which includes prefactor) was
        # already taken modulo the ideal.
        # The objective can contain truely complex monomials; but we checked before that it will not have an imaginary part.
        if isone(problem.prefactor)
            # fix constant term to 1 - this is a linear constraint, which is entered separately
            A = ([1], [problem.complex ? get_variable!(hsm, constant_monomial(problem.objective), 1.0, true).first :
            get_variable!(hsm, constant_monomial(problem.objective))], [1.0])
        else
            if problem.complex
                pref_terms = mergewith(+, (p for term in problem.prefactor
                                             for p in (get_variable!(hsm, monomial(term), -real(coefficient(term)), true),
                                                       get_variable!(hsm, monomial(term), imag(coefficient(term)), false))
                                             if !ismissing(p)), Int, Float64)
                A = (fill(1, length(pref_terms)), collect(keys(pref_terms)), collect(values(pref_terms)))
            else
                A = (fill(1, nterms(problem.prefactor)), get_variable!.((hsm,), monomials(problem.prefactor)),
                    Float64.(coefficients(problem.prefactor)))
            end
        end
        b = [1.0]
        if problem.complex
            # This get_varrep! simplified; note that the objective is already mod Gröbner basis
            constr_terms = mergewith(+, (p for term in problem.objective
                                         for p in (get_variable!(hsm, monomial(term), real(coefficient(term)), true),
                                                   get_variable!(hsm, monomial(term), -imag(coefficient(term)), false))
                                         if !ismissing(p)), Int, Float64)
            objective = (collect(keys(constr_terms)), collect(values(constr_terms)))
        else
            objective = (get_variable!.((hsm,), monomials(problem.objective)), Float64.(coefficients(problem.objective)))
        end

        # moment matrix
        for grouping in groupings[1]
            moment_matrix!(hsm, problem.gröbner_basis, sort(grouping, by=degree),
                polynomial(constant_monomial(problem.objective)))
        end
        # localizing matrices
        for (groupings, constr) in zip(Iterators.drop(groupings, 1), problem.constraints)
            if constr.type == pctNonneg || constr.type == pctPSD
                for grouping in groupings
                    moment_matrix!(hsm, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                end
            elseif constr.type == pctEqualityNonneg
                for grouping in groupings
                    let sg = sort(grouping, by=degree)
                        moment_matrix!(hsm, problem.gröbner_basis, sg, constr.constraint)
                        moment_matrix!(hsm, problem.gröbner_basis, sg, -constr.constraint)
                    end
                end
            elseif constr.type == pctEqualityGröbner
                for grouping in groupings
                    moment_matrix_eq!(hsm, A, b, EmptyGröbnerBasis{P}(), sort(grouping, by=degree), constr.constraint)
                end
            elseif constr.type == pctEqualitySimple
                for grouping in groupings
                    moment_matrix_eq!(hsm, A, b, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                end
            else
                @assert(false)
            end
        end

        num_vars = length(hsm.variable_mappings)
        obj = zeros(num_vars)
        obj[objective[1]] = objective[2]

        obj, A, b = customize(hsm, obj, A, b)

        model = Hypatia.Models.Model{Float64}(
            obj, # c
            sparse(A[1], A[2], A[3], maximum(A[1]), num_vars), # A
            b, # b
            sparse(vcat((x[1] for x in hsm.constraints)...), vcat((x[2] for x in hsm.constraints)...),
                vcat((x[3] for x in hsm.constraints)...), hsm.num_constr, num_vars), # G
            vcat((x[4] for x in hsm.constraints)...), # h
            vcat((x[5] for x in hsm.constraints)...) # cones
        )
        if !dense
            # for lots of smaller constraints, a sparse solver is much better. However, all non-QRCholDenseSystemSolver
            # types also require to turn off reduction (else, we just get completely wrong results), and performing a dense
            # preprocessing also defeats the purpose of a sparse solver.
            parameters = (syssolver=get(() -> Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}(), parameters, :syssolver),
                preprocess=get(parameters, :preprocess, false), reduce=get(parameters, :reduce, false), parameters...)
        end
        solver = Hypatia.Solvers.Solver{Float64}(verbose=verbose; parameters...)
        Hypatia.Solvers.load(solver, model)
    end
    @verbose_info("Setup complete in ", setup_time, " seconds")
    Hypatia.Solvers.solve(solver)
    status = Hypatia.Solvers.get_status(solver)
    value = Hypatia.Solvers.get_primal_obj(solver)
    @verbose_info("Optimization complete")

    empty!(problem.last_moments)
    sizehint!(problem.last_moments, length(hsm.variable_mappings))
    x = Hypatia.Solvers.get_x(solver)
    for (mon, i) in hsm.variable_mappings
        push!(problem.last_moments, mon => x[i])
    end
    return status, value
end