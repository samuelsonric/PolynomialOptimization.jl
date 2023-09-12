mutable struct MosekStateMoment{M}
    task::Mosek.MSKtask
    num_vars::Int32
    num_afes::Int64
    variable_mappings::AbstractDict{M,Int32}
    accs::AbstractDict{Int,Int64}
end

get_variable_add!(msk::MosekStateMoment{M}) where {M} = () -> begin
    if msk.num_vars == length(msk.variable_mappings)
        Mosek.appendvars(msk.task, 20)
        Mosek.putvarboundsliceconst(msk.task, msk.num_vars + Int32(1), msk.num_vars + Int32(21), Mosek.MSK_BK_FR, -Inf, Inf)
        msk.num_vars += 20
    end
    return length(msk.variable_mappings) + Int32(1)
end

get_variable!(msk::MosekStateMoment{M}, mon::M) where {M} = get!(get_variable_add!(msk), msk.variable_mappings, mon)

function get_variable!(msk::MosekStateMoment{MonomialComplexContainer{M}}, mon::M, coefficient::R, re::Bool) where {M,R<:Real}
    mon2 = conj(mon)
    if mon == mon2 && !re
        return missing
    elseif mon < mon2
        return get!(get_variable_add!(msk), msk.variable_mappings, MonomialComplexContainer(mon, re)) => coefficient
    else
        return get!(get_variable_add!(msk), msk.variable_mappings,
            MonomialComplexContainer(mon2, re)) => (re ? coefficient : -coefficient)
    end
end

function get_varrepr!(msk::MosekStateMoment{M}, gröbner_basis, row::M, col::M, constraint::P) where {P,M}
    constr_terms = mergewith(+, (get_variable!(msk, monomial(term)) => coefficient(term)
                                 for mon_constr in constraint
                                 for term in rem(row * col * mon_constr, gröbner_basis)),
        Int32, Float64)
    msk.num_afes += 1
    Mosek.putafefrow(msk.task, msk.num_afes, collect(keys(constr_terms)), collect(values(constr_terms)))
end

function get_varrepr!(msk::MosekStateMoment{MonomialComplexContainer{M}}, gröbner_basis, row::M, col::M, constraint::P,
    ensure_real::Bool) where {P,M}
    if ensure_real
        # This assert should in principle be valid and might be helpful in finding errors. However, we should check this
        # modulo the gröbner_basis; but this means that we'd have to resubstitute the real and imaginary parts (that are
        # introduced by imag) by sums/differences of the complex variables and their conjugates. This is a lot of work for a
        # simple assert. Well, complex gröbner_basis is disabled at the moment, but still...
        #@assert(imag(row*conj(col)*constraint) == 0)
        constr_terms = mergewith(+, (p for mon_constr in constraint
                                     for term in rem(row * conj(col) * mon_constr, gröbner_basis)
                                     for p in (get_variable!(msk, monomial(term), real(coefficient(term)), true),
                get_variable!(msk, monomial(term), -imag(coefficient(term)), false))
                                     if !ismissing(p)),
            Int32, Float64)
        msk.num_afes += 1
        Mosek.putafefrow(msk.task, msk.num_afes, collect(keys(constr_terms)), collect(values(constr_terms)))
        return
    else
        # now we have a problem. Mosek does not support hermitian PSD constraints, so we have to rewrite everything in terms of
        # real matrices, which is a mess with lots of duplication and size doubling. We just output the afes with real part and
        # imaginary part directly following and fix the rest in the acc.
        constr_terms_re = Dict{Int32,Float64}()
        constr_terms_im = Dict{Int32,Float64}()
        for mon_constr in constraint
            for term in rem(row * conj(col) * mon_constr, gröbner_basis)
                for p in (get_variable!(msk, monomial(term), real(coefficient(term)), true),
                    get_variable!(msk, monomial(term), -imag(coefficient(term)), false))
                    if !ismissing(p)
                        constr_terms_re[p.first] = haskey(constr_terms_re, p.first) ? constr_terms_re[p.first] + p.second : p.second
                    end
                end
                for p in (get_variable!(msk, monomial(term), real(coefficient(term)), false),
                    get_variable!(msk, monomial(term), imag(coefficient(term)), true))
                    if !ismissing(p)
                        constr_terms_im[p.first] = haskey(constr_terms_im, p.first) ? constr_terms_im[p.first] + p.second : p.second
                    end
                end
            end
        end
        msk.num_afes += 2
        Mosek.putafefrow(msk.task, msk.num_afes - 1, collect(keys(constr_terms_re)), collect(values(constr_terms_re)))
        Mosek.putafefrow(msk.task, msk.num_afes, collect(keys(constr_terms_im)), collect(values(constr_terms_im)))
        return constr_terms_im # we need this
    end
end

function moment_matrix!(msk::MosekStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::P) where {P,M}
    task = msk.task
    lg = length(grouping)
    if lg == 1
        Mosek.appendafes(task, 1)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint)
        Mosek.appendacc(task, get!(() -> Mosek.appendrplusdomain(task, 1), msk.accs, 1), [msk.num_afes], [0.0])
    elseif lg == 2
        Mosek.appendafes(task, 3)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint / 2)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[2], grouping[2], constraint)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[2], grouping[1], constraint)
        Mosek.appendacc(task, get!(() -> Mosek.appendrquadraticconedomain(task, 3), msk.accs, 2),
            [msk.num_afes - 2, msk.num_afes - 1, msk.num_afes], zeros(3))
    else
        items = (lg * (lg + 1)) >> 1
        Mosek.appendafes(task, items)
        for exp2 in 1:lg
            for exp1 in exp2:lg
                @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2],
                    exp1 == exp2 ? constraint : sqrt2 * constraint)
            end
        end
        Mosek.appendacc(task, get!(() -> Mosek.appendsvecpsdconedomain(task, items), msk.accs, lg),
            collect(msk.num_afes-items+1:msk.num_afes), zeros(items))
    end
    return
end

function moment_matrix!(msk::MosekStateMoment{MonomialComplexContainer{M}}, gröbner_basis, grouping::AbstractVector{M},
    constraint::P) where {P,M}
    task = msk.task
    lg = length(grouping)
    if lg == 1
        Mosek.appendafes(task, 1)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint, true)
        Mosek.appendacc(task, get!(() -> Mosek.appendrplusdomain(task, 1), msk.accs, 1), [msk.num_afes], [0.0])
    elseif lg == 2
        Mosek.appendafes(task, 4)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint / 2, true)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[2], grouping[2], constraint, true)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[2], grouping[1], constraint, false)
        Mosek.appendacc(task, get!(() -> Mosek.appendrquadraticconedomain(task, 4), msk.accs, 2),
            [msk.num_afes - 3, msk.num_afes - 2, msk.num_afes - 1, msk.num_afes], zeros(4))
    else
        # we need to double the dimension
        items = ((2lg) * (2lg + 1)) >> 1
        # The accs are tricky. We must now arrange the afes in such a manner that the structure
        #   [re   -im]
        #   [im   re]
        # automatically follows.
        afes = fill(1, items) # all filled with empty afe
        i = 0
        second_re = (lg + 3lg * lg) >> 1 # start index of the second real part
        Mosek.appendafes(task, items)
        for exp2 in 1:lg
            # diagonal part
            @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp2], grouping[exp2], constraint, true)
            i += 1
            afes[i] = msk.num_afes # first real part
            second_re += 1
            afes[second_re] = msk.num_afes # second real part
            # off-diagonal parts
            for exp1 in exp2+1:lg
                @inbounds im = get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2], sqrt2 * constraint, false)
                i += 1
                afes[i] = msk.num_afes - 1 # first real part
                afes[i+lg] = msk.num_afes # imaginary part lower triangle
                second_re += 1
                afes[second_re] = msk.num_afes - 1 # second real part
                # we also have the upper triangle of the imaginary part, which resides in the lower triangle of
                # the full matrix - and which is negative, so we need a new afe
                msk.num_afes += 1
                Mosek.putafefrow(msk.task, msk.num_afes, collect(keys(im)), -collect(values(im)))
                afes[exp2+lg-((exp1-1)*(exp1-4lg))>>1] = msk.num_afes # imaginary part upper triangle
            end
            i += lg # skip over the imaginary part
        end
        Mosek.appendacc(task, get!(() -> Mosek.appendsvecpsdconedomain(task, items), msk.accs, lg),
            afes, zeros(items))
    end
    return
end

function moment_matrix!(msk::MosekStateMoment{M}, gröbner_basis, grouping::AbstractVector{M}, constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return moment_matrix!(msk, gröbner_basis, grouping, constraint[1, 1])
    end
    task = msk.task
    lg = length(grouping)
    if lg == 1 && block_size == 2
        Mosek.appendafes(task, 3)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[1, 1] / 2)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[2, 2])
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[2, 1])
        Mosek.appendacc(task, get!(() -> Mosek.appendrquadraticconedomain(task, 3), msk.accs, 2),
            [msk.num_afes - 2, msk.num_afes - 1, msk.num_afes], zeros(3))
    else
        dim = lg * block_size
        items = (dim * (dim + 1)) >> 1
        Mosek.appendafes(task, items)
        for exp2 in 1:lg
            for block_j in 1:block_size
                for exp1 in exp2:lg
                    for block_i in (exp1 == exp2 ? block_j : 1):block_size
                        @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2],
                            exp1 == exp2 && block_j == block_i ? constraint[block_i, block_j] : sqrt2 * constraint[block_i, block_j])
                    end
                end
            end
        end
        Mosek.appendacc(task, get!(() -> Mosek.appendsvecpsdconedomain(task, items), msk.accs, dim),
            collect(msk.num_afes-items+1:msk.num_afes), zeros(items))
    end
    return
end

function moment_matrix!(msk::MosekStateMoment{MonomialComplexContainer{M}}, gröbner_basis, grouping::AbstractVector{M},
    constraint::AbstractMatrix{P}) where {P,M}
    @assert size(constraint, 1) == size(constraint, 2)
    block_size = size(constraint, 1)
    if block_size == 1
        @inbounds return moment_matrix!(msk, gröbner_basis, grouping, constraint[1, 1])
    end
    task = msk.task
    lg = length(grouping)
    if lg == 1 && block_size == 2
        Mosek.appendafes(task, 4)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[1, 1] / 2, true)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[2, 2], true)
        @inbounds get_varrepr!(msk, gröbner_basis, grouping[1], grouping[1], constraint[2, 1], false)
        Mosek.appendacc(task, get!(() -> Mosek.appendrquadraticconedomain(task, 4), msk.accs, 2),
            [msk.num_afes - 3, msk.num_afes - 2, msk.num_afes - 1, msk.num_afes], zeros(4))
    else
        dim = lg * block_size
        # we need to double the dimension
        items = ((2dim) * (2dim + 1)) >> 1
        # The accs are tricky. We must now arrange the afes in such a manner that the structure
        #   [re   -im]
        #   [im   re]
        # automatically follows.
        afes = fill(1, items) # all filled with empty afe
        i = 0
        full_j = 1
        second_re = (dim + 3dim * dim) >> 1 # start index of the second real part
        Mosek.appendafes(task, items)
        for exp2 in 1:lg
            for block_j in 1:block_size
                # diagonal part
                @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp2], grouping[exp2], constraint[block_j, block_j], true)
                i += 1
                afes[i] = msk.num_afes # first real part
                second_re += 1
                afes[second_re] = msk.num_afes # second real part
                outer_i = (exp2 - 1) * block_size
                # off-diagonal parts
                for exp1 in exp2:lg
                    for block_i in (exp1 == exp2 ? block_j + 1 : 1):block_size
                        @inbounds im = get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2],
                            sqrt2 * constraint[block_i, block_j], false)
                        i += 1
                        afes[i] = msk.num_afes - 1 # first real part
                        afes[i+dim] = msk.num_afes # imaginary part lower triangle
                        second_re += 1
                        afes[second_re] = msk.num_afes - 1 # second real part
                        # we also have the upper triangle of the imaginary part, which resides in the lower triangle of
                        # the full matrix - and which is negative, so we need a new afe
                        msk.num_afes += 1
                        Mosek.putafefrow(msk.task, msk.num_afes, collect(keys(im)), -collect(values(im)))
                        full_i = outer_i + block_i
                        afes[full_j+dim-((full_i-1)*(full_i-4dim))>>1] = msk.num_afes # imaginary part upper triangle
                    end
                    outer_i += block_size
                end
                i += dim # skip over the imaginary part
                full_j += 1
            end
        end
        Mosek.appendacc(task, get!(() -> Mosek.appendsvecpsdconedomain(task, items), msk.accs, dim),
            afes, zeros(items))
    end
    return
end

function moment_matrix_eq!(msk::MosekStateMoment{C}, gröbner_basis, grouping::AbstractVector, constraint) where {C}
    task = msk.task
    lg = length(grouping)
    if C <: MonomialComplexContainer
        items = lg^2
    else
        items = (lg * (lg + 1)) >> 1
    end
    Mosek.appendafes(task, items)
    for exp2 in 1:lg
        for exp1 in exp2:lg
            if C <: MonomialComplexContainer
                @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2], constraint, exp2 == exp1)
            else
                @inbounds get_varrepr!(msk, gröbner_basis, grouping[exp1], grouping[exp2], constraint)
            end
        end
    end
    # to cache zero domains in the same dictionary, we use the negative lengths as keys
    Mosek.appendacc(task, get!(() -> Mosek.appendrzerodomain(task, items), msk.accs, -items),
        collect(msk.num_afes-items+1:msk.num_afes), zeros(items))
    return
end

function sparse_optimize(::Val{:MosekMoment}, problem::PolyOptProblem{P,M,V},
    groupings::Vector{<:Vector{<:AbstractVector{M}}}; verbose::Bool=false, customize::Function=(msk) -> nothing,
    parameters...) where {P,M,V}
    Mosek.maketask() do task
        setup_time = @elapsed begin
            verbose && Mosek.putstreamfunc(task, Mosek.MSK_STREAM_LOG, printstream)
            for (k, v) in parameters
                Mosek.putparam(task, string(k), v)
            end

            Mosek.putobjsense(task, Mosek.MSK_OBJECTIVE_SENSE_MINIMIZE)

            if !isone(problem.prefactor)
                Mosek.appendcons(task, 1)
                Mosek.putconbound(task, 1, Mosek.MSK_BK_FX, 1.0, 1.0)
            end
            if problem.complex
                msk = MosekStateMoment{MonomialComplexContainer{M}}(task, zero(Int32), 1,
                    Dict{MonomialComplexContainer{M},Int32}(), Dict{Int,Int64}())
                Mosek.appendafes(task, 1) # this is the empty afe

                # Riesz functional of objective mod ideal. Note that during problem setup, the objective (which includes
                # prefactor) was already taken modulo the ideal.
                # This is get_var_repr!, but without the afe part
                obj_terms = mergewith(+, (p for term in problem.objective
                                          for p in (get_variable!(msk, monomial(term), real(coefficient(term)), true),
                        get_variable!(msk, monomial(term), -imag(coefficient(term)), false))
                                          if !ismissing(p)),
                    Int32, Float64)
                Mosek.putclist(task, collect(keys(obj_terms)), collect(values(obj_terms)))

                if isone(problem.prefactor)
                    # fix constant term to 1
                    Mosek.putvarbound(task, get_variable!(msk, constant_monomial(problem.objective), 1.0, true).first,
                        Mosek.MSK_BK_FX, 1.0, 1.0)
                else
                    pref_terms = mergewith(+, (p for term in problem.prefactor
                                               for p in (get_variable!(msk, monomial(term), real(coefficient(term)), true),
                                                         get_variable!(msk, monomial(term), -imag(coefficient(term)), false))
                                               if !ismissing(p)),
                        Int32, Float64)
                    Mosek.putarow(task, 1, collect(keys(pref_terms)), collect(values(pref_terms)))
                end
            else
                msk = MosekStateMoment{M}(task, zero(Int32), 0, Dict{M,Int32}(), Dict{Int,Int64}())

                # Riesz functional of objective mod ideal. Note that during problem setup, the objective (which includes
                # prefactor) was already taken modulo the ideal.
                # Convert necessary for empty objectives (would give Vector{Any} in this case)
                Mosek.putclist(task, convert(Vector{Int32}, get_variable!.((msk,), monomials(problem.objective))),
                    Float64.(coefficients(problem.objective)))

                if isone(problem.prefactor)
                    # fix constant term to 1
                    Mosek.putvarbound(task, get_variable!(msk, constant_monomial(problem.objective)), Mosek.MSK_BK_FX, 1.0, 1.0)
                else
                    Mosek.putarow(task, 1, convert(Vector{Int32}, get_variable!.((msk,), monomials(problem.prefactor))),
                        Float64.(coefficients(problem.prefactor)))
                end
            end

            # moment matrix
            for grouping in groupings[1]
                moment_matrix!(msk, problem.gröbner_basis, sort(grouping, by=degree),
                    polynomial(constant_monomial(problem.objective)))
            end
            # localizing matrices
            for (groupings, constr) in zip(Iterators.drop(groupings, 1), problem.constraints)
                if constr.type == pctNonneg || constr.type == pctPSD
                    for grouping in groupings
                        moment_matrix!(msk, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualityNonneg
                    for grouping in groupings
                        let sg = sort(grouping, by=degree)
                            moment_matrix!(msk, problem.gröbner_basis, sg, constr.constraint)
                            moment_matrix!(msk, problem.gröbner_basis, sg, -constr.constraint)
                        end
                    end
                elseif constr.type == pctEqualityGröbner
                    for grouping in groupings
                        moment_matrix_eq!(msk, EmptyGröbnerBasis{P}(), sort(grouping, by=degree), constr.constraint)
                    end
                elseif constr.type == pctEqualitySimple
                    for grouping in groupings
                        moment_matrix_eq!(msk, problem.gröbner_basis, sort(grouping, by=degree), constr.constraint)
                    end
                else
                    @assert(false)
                end
            end
        end
        @verbose_info("Setup complete in ", setup_time, " seconds")

        customize(msk)

        Mosek.optimize(task)
        status = Mosek.getsolsta(task, Mosek.MSK_SOL_ITR)
        value = Mosek.getprimalobj(task, Mosek.MSK_SOL_ITR)
        @verbose_info("Optimization complete")

        empty!(problem.last_moments)
        sizehint!(problem.last_moments, length(msk.variable_mappings))
        mon_vals = Mosek.getxx(task, Mosek.MSK_SOL_ITR)
        for (mon, i) in msk.variable_mappings
            push!(problem.last_moments, mon => mon_vals[i])
        end
        return status, value
    end
end