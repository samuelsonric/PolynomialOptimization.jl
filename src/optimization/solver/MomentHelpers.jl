export moment_add_matrix!, moment_add_equality!, moment_setup!
# This file is for the commuting case; nevertheless, we already write the monomial index/multiplication calculation with a
# possible extension to the noncommuting case in mind.

function to_soc!(indices, values, lens, supports_rotated)
    @assert(length(lens) ≥ 2)
    @inbounds if supports_rotated
        for j in length(lens):-1:3
            iszero(lens[j]) && deleteat!(lens, j)
        end
    else
        range₁ = 1:lens[1]
        range₂ = lens[1]+1:lens[1]+lens[2]
        prevlen = last(range₂)
        inds = @views count_uniques(indices[range₁], indices[range₂])
        total = 2inds
        if total > prevlen
            Base._growat!(indices, prevlen +1, total - prevlen)
            Base._growat!(values, prevlen +1, total - prevlen)
        end
        # we must make a copy to avoid potential overwrites, doesn't matter in which direction we work
        oldinds = indices[1:prevlen]
        oldvals = values[1:prevlen]
        count_uniques(@view(oldinds[range₁]), @view(oldinds[range₂]), let oldinds=oldinds, oldvals=oldvals
            (o₁, i₁, i₂) -> @inbounds begin
                if ismissing(i₁)
                    indices[o₁+inds] = indices[o₁] = oldinds[lens[1]+i₂]
                    values[o₁+inds] = -(values[o₁] = oldvals[lens[1]+i₂])
                elseif ismissing(i₂)
                    indices[o₁+inds] = indices[o₁] = oldinds[i₁]
                    values[o₁+inds] = values[o₁] = oldvals[i₁]
                else
                    indices[o₁+inds] = indices[o₁] = oldinds[i₁]
                    values[o₁] = oldvals[i₁] + oldvals[lens[1]+i₂]
                    values[o₁+inds] = oldvals[i₁] - oldvals[lens[1]+i₂]
                end
            end
        end)
        lens[1] = inds
        lens[2] = inds
        for j in length(lens):-1:2
            iszero(lens[j]) && deleteat!(lens, j)
        end
    end
    return IndvalsIterator(indices, values, lens)
end

# This is an iterator that mimicks two nested loops:
# :U -> exp1 ∈ 1:exp2,  block_i ∈ 1:(exp1 == exp2 ? block_j : block_size)
# :L -> exp1 ∈ exp2:lg, block_i ∈ (exp1 == exp2 ? block_j : 1):block_size
# :F -> exp1 ∈ 1:lg,    block_i ∈ 1:block_size
# However, for the case :DF, we need to take the diagonal element first (exp1 = exp2, block_i = block_j), then iterate through
# the others (in an arbitrary order).
struct MatrixPartIter{Tri,complex,G}
    lg::Int
    block_size::Int
    exp2::Int
    block_j::Int
    grouping::G

    function MatrixPartIter{Tri,complex}(lg::Int, block_size::Int, exp2::Int, block_j::Int, grouping::G) where {Tri,complex,G}
        (Tri ∈ (:U, :L, :F, :DF) && complex isa Bool) ||
            throw(MethodError(MatrixPartIter{Tri,complex}, (lg, block_size, exp2, block_j, grouping)))
        new{Tri,complex,G}(lg, block_size, exp2, block_j, grouping)
    end
end

Base.IteratorEltype(::Type{<:MatrixPartIter}) = Base.HasEltype()
Base.IteratorSize(::Type{<:MatrixPartIter}) = Base.HasLength()
Base.eltype(::Type{<:MatrixPartIter{<:Any,G}}) where {G} = Tuple{Int,eltype(G),Bool,Int}

Base.length(m::MatrixPartIter{:U}) = (m.exp2 -1) * m.block_size + m.block_j
Base.length(m::MatrixPartIter{:L}) = (m.lg - m.exp2) * m.block_size + (m.block_size - m.block_j +1)
Base.length(m::Union{<:MatrixPartIter{:F},<:MatrixPartIter{:DF}}) = m.lg * m.block_size
@inline function Base.iterate(m::MatrixPartIter{Tri,complex}) where {Tri,complex}
    Tri ∈ (:U, :L) || throw(MethodError(iterate, (m,)))
    exp1_range = Tri === :U ? (1:m.exp2) : (m.exp2:m.lg)
    outer_iterator = @inbounds zip(exp1_range, @view(m.grouping[exp1_range]))
    outer_it = iterate(outer_iterator)
    while true
        isnothing(outer_it) && return nothing
        exp1, g₁ = outer_it[1]
        isreal_g₁ = !complex || isreal(g₁)
        inner_iterator = Tri === :U ? (1:(exp1 == m.exp2 ? m.block_j : m.block_size)) :
                                      ((exp1 == m.exp2 ? m.block_j : 1):m.block_size)
        inner_it = iterate(inner_iterator)
        isnothing(inner_it) ||
            return (exp1, g₁, isreal_g₁, inner_it[1]), (outer_iterator, outer_it, isreal_g₁, inner_iterator, inner_it[2])
        outer_it = iterate(outer_iterator, outer_it[2])
    end
end
@inline function Base.iterate(m::MatrixPartIter{Tri,complex}, (outer_iterator, outer_it, isreal_g₁, inner_iterator, inner_state)) where {Tri,complex}
    Tri ∈ (:U, :L) || throw(MethodError(iterate, (m,)))
    inner_it = iterate(inner_iterator, inner_state)
    while isnothing(inner_it)
        outer_it = iterate(outer_iterator, outer_it[2])
        isnothing(outer_it) && return nothing
        exp1, g₁ = outer_it[1]
        isreal_g₁ = !complex || isreal(g₁)
        inner_iterator = Tri === :U ? (1:(exp1 == m.exp2 ? m.block_j : m.block_size)) :
                                      ((exp1 == m.exp2 ? m.block_j : 1):m.block_size)
        inner_it = iterate(inner_iterator)
    end
    return (outer_it[1]..., isreal_g₁, inner_it[1]), (outer_iterator, outer_it, isreal_g₁, inner_iterator, inner_it[2])
end
@inline function Base.iterate(m::MatrixPartIter{:F,complex}) where {complex}
    outer_iterator = enumerate(m.grouping)
    outer_it = iterate(outer_iterator)
    while true
        isnothing(outer_it) && return nothing
        exp1, g₁ = outer_it[1]
        isreal_g₁ = !complex || isreal(g₁)
        iszero(m.block_size) || return (exp1, g₁, isreal_g₁, 1), (outer_iterator, outer_it, isreal_g₁, 1)
        outer_it = iterate(outer_iterator, outer_it[2])
    end
end
@inline function Base.iterate(m::MatrixPartIter{:F,complex}, (outer_iterator, outer_it, isreal_g₁, inner_pos)) where {complex}
    while inner_pos ≥ m.block_size
        outer_it = iterate(outer_iterator, outer_it[2])
        isnothing(outer_it) && return nothing
        exp1, g₁ = outer_it[1]
        isreal_g₁ = !complex || isreal(g₁)
        inner_pos = 0
    end
    return (outer_it[1]..., isreal_g₁, inner_pos +1), (outer_iterator, outer_it, isreal_g₁, inner_pos +1)
end
@inline function Base.iterate(m::MatrixPartIter{:DF,complex}) where {complex}
    # This is basically a concatenation of the :L with the :U iterator and the last element dropped. To avoid another burden on
    # dispatch, we'll make this work statically.
    iter_l = MatrixPartIter{:L,complex}(m.lg, m.block_size, m.exp2, m.block_j, m.grouping)
    it = iterate(iter_l)
    isnothing(it) && return nothing
    iter_u = MatrixPartIter{:U,complex}(m.lg, m.block_size, m.exp2, m.block_j, m.grouping)
    return it[1], (iter_l, iter_u, it[2], true, m.lg * m.block_size -1)
end
@inline function Base.iterate(m::MatrixPartIter{:DF}, (iter_l, iter_u, state, in_l, remaining))
    iszero(remaining) && return nothing
    if in_l
        it = iterate(iter_l, state)
        if isnothing(it)
            it = iterate(iter_u)
            in_l = false
        end
    else
        it = iterate(iter_u, state)
    end
    it::Tuple
    return it[1], (iter_l, iter_u, it[2], in_l, remaining -1)
end

# generic moment matrix constraint with
# - only real-valued monomials involved in the grouping, and only real-valued polynomials involved in the constraint (so if it
#   contains complex coefficients/monomials, imaginary parts cancel out)
# - or complex-valued monomials involved in the grouping, but the solver supports the complex-valued PSD cone explicitly
# - or DSOS/SDOS representations in the real case and in the complex case if the quadratic cone is requested
function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri},
    type::Union{Tuple{Val{true},Val},Tuple{Val{false},Val{true}}}, representation::RepresentationMethod) where {Tri}
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    complex = type isa Tuple{Val{false},Val{true}}
    maxlen = maximum(length, constraint, init=0)
    if dim == 1
        matrix_indexing = false
        representation = RepresentationPSD() # will be linear anyway
        tri = :U
    elseif representation isa RepresentationDD
        matrix_indexing = false
        if complex
            if supports_complex_lnorm(state)
                # complex ℓ₁ norm case: {cᵢᵢ, cᵢ₁ᵣ, cᵢ₁ᵢ, ... cᵢᵢ₋₁ᵣ, cᵢᵢ₋₁ᵢ, cᵢᵢ₊₁ᵣ, ...}
                # lens:         2dim -1
                # indices:      maxlen + 2maxlen * 2(dim -1))
                # each constraint is submitted per column, no accumulation in the same vector, all real-valued
                colcount = 2dim -1
                indlen = maxlen * (4dim -3)
                tri = :DF # full format required, but diagonals must come first
            else
                # quadratic case: sᵢⱼ² ≥ crᵢⱼ² + ciᵢⱼ², cᵢᵢ ≥ ∑ⱼ sᵢⱼ
                # lens:           3,                    dim
                # indices:        (1   + 4maxlen),      maxlen + dim -1
                # each constraint is submitted individually, no accumulation in the same vector
                colcount = max(3, dim)
                indlen = max(1 + 4maxlen, maxlen + dim -1)
                tri = :U
            end
        elseif supports_lnorm(state)
            # ℓ₁ norm case: {cᵢᵢ, cᵢ₁, ... cᵢᵢ₋₁, cᵢᵢ₊₁, ...}
            # lens:         dim
            # indices:      maxlen * dim
            # each constraint is submitted per column, no accumulation in the same vector, all real-valued
            colcount = dim
            indlen = maxlen * dim
            tri = :DF # full format required, but diagonals must come first
        else
            # linear case: sᵢⱼ ≥ cᵢⱼ,  sᵢⱼ ≥ -cᵢⱼ, cᵢᵢ ≥ ∑ⱼ sᵢⱼ (n linear constraints)
            # lens:        1,          1,          1
            # indices:     1 + maxlen, 1 + maxlen, maxlen + dim -1
            # each constraint is submitted individually, no accumulation in the same vector
            colcount = 1
            indlen = max(1 + maxlen, maxlen + dim -1)
            tri = :U
        end
    elseif dim == 2 && (supports_rotated_quadratic(state) || supports_quadratic(state))
        matrix_indexing = false
        representation = RepresentationPSD() # will be quadratic anyway
        tri = :U # we always create the data in U format; this ensures the scaling is already taken care of
    elseif representation isa RepresentationSDD
        matrix_indexing = false
        if complex
            #          cᵢᵢ = ∑ₓ sₓ,     sᵢⱼ₁sᵢⱼ₃ ≥ cᵢⱼᵣ² + cᵢⱼᵢ²
            # lens:    1,               4
            # indices: maxlen + dim -1, 2 + 4maxlen
            # each constraint is submitted individually, no accumulation in the same vector
            colcount = 4
            indlen = max(maxlen + dim -1, 2 + 4maxlen)
            tri = :U
        else
            #          cᵢᵢ = ∑ₓ sₓ,     sᵢⱼ₁sᵢⱼ₃ ≥ cᵢⱼ²
            # lens:    1,               3
            # indices: maxlen + dim -1, 2 + maxlen
            # each constraint is submitted individually, no accumulation in the same vector
            colcount = 3
            indlen = max(maxlen + dim -1, 2 + maxlen)
            tri = :U
        end
    elseif indextype isa PSDIndextypeMatrixCartesian
        Tri ∈ (:L, :U) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype, type)))
        matrix_indexing = true
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U. However, if Tri === :L, we will fill
                 # the values with conjugates.
    else
        Tri ∈ (:L, :U, :F) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            type)))
        matrix_indexing = false
        tri = Tri
    end
    if representation isa RepresentationPSD
        colcount = (tri === :F ? (complex ? 2dim^2 - dim : dim^2) : (complex ? dim^2 : trisize(dim)))
        indlen = 2maxlen * colcount
    end
    if matrix_indexing
        rows = FastVec{T}(buffer=indlen)
        indices = similar(rows)
        values = similar(rows, complex ? Complex{V} : V)
    else
        lens = FastVec{SimplePolynomials.smallest_unsigned(2maxlen)}(buffer=colcount)
        indices = FastVec{T}(buffer=indlen)
        values = similar(indices, V)
    end
    # introduce a method barrier to fix the potentially unknown eltype of lens make sure "dynamic" constants can be folded
    moment_add_matrix_helper!(state, T, V, grouping, constraint, indextype, Val(tri), Val(matrix_indexing), Val(complex), lg,
        block_size, dim, matrix_indexing ? (rows, indices, values) : (lens, indices, values), representation)
end

function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri}, ::Val{tri}, ::Val{matrix_indexing},
    ::Val{complex}, lg, block_size, dim, data, representation::RepresentationMethod) where {Tri,tri,matrix_indexing,complex}
    if matrix_indexing
        rows, indices, values = data
        row = zero(T)
    else
        lens, indices, values = data
        if representation isa RepresentationDD
            scaling = 1.
            if complex
                # We only end up here if the user requested the use of the complex rewrite in the representation.
                l1 = supports_complex_lnorm(state) # In principle, a solver could decide to just implement the complex-valued cone
                                                # and we would not automatically rewrite things to use this one.
                l1 || supports_quadratic(state) ||
                    error("Invalid configuration: the quadratic cone or the complex-valued ℓ₁ norm cone must be supported for this model of DSOS constraints")
            else
                l1 = supports_lnorm(state)
            end
            unchecked_iterator = IndvalsIterator(indices, values, lens)
            if !l1
                slack_indices = add_var_slack!(state, trisize(dim -1))
                slack_i = 1
            end
        elseif representation isa RepresentationSDD
            # An off-diagonal entry in the originally PSD matrix stays. However, diagonal entries are replaced by sums of dim-1
            # values that enter the rotated quadratic cone. We need those as slack variables.
            slack_indices = add_var_slack!(state, dim * (dim -1))
            slack_i = 1
            rquad = supports_rotated_quadratic(state)
            scaling = rquad ? sqrt(V(2)) : V(2)
            fixstate = add_constr_fix_prepare!(state, dim) # for every diagonal
            unchecked_iterator = IndvalsIterator(indices, values, lens)
        else
            # Off-diagonals are multiplied by √2 in order to put variables into the vectorized PSD cone. Even if state isa
            # SOSWrapper (then, the variables directly correspond to a vectorized PSD cone), the actual values in the PSD
            # matrix are still multiplied by 1/√2, so we must indeed always multiply the coefficients by √2 to undo this.
            # This is unless we are in the case of a rotated quadratic cone, for which SOS/moment doesn't make any difference.
            # In the case of a (normal) quadratic cone, we canonically take the rotated cone and transform it by multiplying
            # the left-hand side by 1/√2, giving (x₁/√2)² ≥ (x₂/√2)² + ∑ᵢ (√2 xᵢ)² ⇔ x₁² ≥ x₂² + ∑ᵢ (2 xᵢ)².
            if dim == 2
                rquad = supports_rotated_quadratic(state)
                quad = supports_quadratic(state)
                if !rquad && quad
                    scaling = V(2)
                else
                    scaling = sqrt(V(2))
                end
            else
                scaling = sqrt(V(2))
            end
        end
    end
    @inbounds for (exp2, g₂) in enumerate(grouping)
        isreal_g₂ = !complex || isreal(g₂)
        for block_j in 1:block_size
            for (exp1, g₁, isreal_g₁, block_i) in MatrixPartIter{tri,complex}(lg, block_size, exp2, block_j, grouping)
                # - !complex: all polynomials are real-valued in total. Only take into account canonical monomials, but
                #   double the prefactor if their conjugated must also occur.
                # - block_i == block_j && exp1 == exp2: we are on the total diagonal. Even if not all polynomials were
                #   real, these are for sure.
                # - block_i == block_j && isreal(g₁) && isreal(g₂): we are on the inner diagonal, but since the outer basis
                #   elements are real-valued, canonical and non-canonicals will also occur in pairs.
                total_real = !complex || (block_i == block_j && (exp1 == exp2 || (isreal_g₁ && isreal_g₂)))
                ondiag = exp1 == exp2 && block_i == block_j
                if representation isa RepresentationDD && !l1 && !ondiag
                    unsafe_push!(indices, slack_indices[slack_i])
                    unsafe_push!(values, one(V))
                    complex && unsafe_push!(lens, one(eltype(lens)))
                    slack_i += 1
                end
                if representation isa RepresentationSDD && !ondiag
                    unsafe_push!(indices, slack_indices[slack_i], slack_indices[slack_i+1])
                    unsafe_push!(values, one(V), one(V))
                    unsafe_push!(lens, one(eltype(lens)), one(eltype(lens)))
                    slack_i += 2
                end
                @twice secondtime (!matrix_indexing && !total_real) begin
                    curlen = 0
                    for term_constr in constraint[block_i, block_j]
                        mon_constr = monomial(term_constr)
                        coeff_constr = coefficient(term_constr)
                        if tri !== :F && tri !== :DF && !matrix_indexing && !ondiag
                            coeff_constr *= scaling
                        end
                        recoeff = real(coeff_constr)
                        imcoeff = imag(coeff_constr)
                        repart, impart, canonical = getreim(state, g₁, mon_constr, SimpleConjMonomial(g₂))
                        if total_real
                            # Even if `complex` - so we are on the diagonal and the values should be Complex{V} - we can
                            # just do the push!, the conversion is implicitly done; as the imaginary part should be zero,
                            # we don't need to discriminate against matrix_indexing.
                            if canonical # ≡ iscanonical(mon_constr)
                                if !iszero(recoeff)
                                    matrix_indexing && unsafe_push!(rows, row)
                                    unsafe_push!(indices, repart)
                                    unsafe_push!(values, repart == impart ? recoeff : V(2) * recoeff)
                                    curlen += 1
                                end
                                if repart == impart
                                    @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                    @assert(isreal(mon_constr))
                                elseif !iszero(imcoeff)
                                    matrix_indexing && unsafe_push!(rows, row)
                                    unsafe_push!(indices, impart)
                                    unsafe_push!(values, V(-2) * imcoeff)
                                    curlen += 1
                                end
                            end
                        else
                            # We are not on the diagonal, so we must work with every entry. Updating might become necessary
                            # as the conjugate can pop up later.
                            if matrix_indexing
                                # Interpretation: Complex(x, y) means that the real part of the entry is given by x and
                                #                 the imaginary part by y.
                                let coeff=Complex(recoeff, Tri === :L ? -imcoeff : imcoeff)
                                    if !iszero(coeff)
                                        @pushorupdate!(repart, row, coeff)
                                        repart == impart ||
                                            @pushorupdate!(impart, row,
                                                canonical ? Complex(-imcoeff, Tri === :L ? -recoeff : recoeff) :
                                                            Complex(imcoeff, Tri === :L ? recoeff : -recoeff))
                                    end
                                end
                            elseif !secondtime
                                iszero(recoeff) || @pushorupdate!(repart, row, recoeff)
                                repart == impart || iszero(imcoeff) ||
                                    @pushorupdate!(impart, row, canonical ? -imcoeff : imcoeff)
                            else
                                iszero(imcoeff) || @pushorupdate!(repart, row, imcoeff)
                                repart == impart || iszero(recoeff) ||
                                    @pushorupdate!(impart, row, canonical ? recoeff : -recoeff)
                            end
                        end
                    end
                    if matrix_indexing
                        row += one(row)
                    elseif representation isa RepresentationPSD ||
                        (representation isa RepresentationSDD && !iszero(curlen)) ||
                        (representation isa RepresentationDD && ((complex && (l1 || !iszero(curlen))) ||
                                                                 (l1 && (!iszero(curlen) || isempty(lens)))))
                        # For the real-valued ℓ₁ norm cone, we don't need empty entries - unless they are the first one (which
                        # can only happen if the user specified a zero diagonal element in a PSD constraint).
                        # For the quadratic cone, the first entry is already filled, so we don't need the zero in any case.
                        # For the complex-valued ℓ₁ norm cone, we must crucially maintain a new complex-valued number every
                        # other index. As we don't know here whether the other part is zero as well, we must add all zeros
                        # explicitly.
                        unsafe_push!(lens, Core.Intrinsics.trunc_int(eltype(lens), curlen)) # no check for overflows
                    end
                end
                if representation isa RepresentationDD && !l1
                    if ondiag
                        # We always traverse in :U order, so we already assigned slack variables to be the absolutes of all the
                        # items above the diagonal; we find them in slack_indices[slack_i-row+1:slack_i-1].
                        # However, we must also add the corresponding absolute values _below_ the diagonal to our constraint;
                        # these are values that have not been assigned yet, and their indices are more difficult to obtain.
                        δ = (exp1 -1) * block_size + block_i # current row index
                        for i in slack_i-δ+1:slack_i-1
                            unsafe_push!(indices, slack_indices[i])
                            unsafe_push!(values, -one(V))
                        end
                        let i=slack_i-1+δ
                            for _ in (exp2 -1)*block_size+block_j+1:dim # the absolutes below are the same as to the right
                                unsafe_push!(indices, slack_indices[i])
                                unsafe_push!(values, -one(V))
                                i += δ
                                δ += 1
                            end
                        end
                        #lens[end] += dim -1 # we don't need it as we only add a single Indvals
                        add_constr_nonnegative!(state, Indvals(indices, values))
                    elseif !complex || length(lens) ≤ 2
                        # The first entry is already set up to contain the slack variable; as we don't have an imaginary
                        # part here, we can impose two linear constraints.
                        add_constr_nonnegative!(state, Indvals(indices, values))
                        rmul!(@view(values[2:end]), -one(V))
                        add_constr_nonnegative!(state, Indvals(indices, values))
                    else
                        # The slack variable should be the absolute value of the complex variable, so we'll need a
                        # quadratic cone. And we know it is supported.
                        add_constr_quadratic!(state, unchecked_iterator)
                    end
                    empty!(indices)
                    empty!(values)
                    empty!(lens)
                elseif representation isa RepresentationSDD
                    if ondiag
                        # The quadratic cones are set up on the off-diagonals; here, we must just put an equality to the
                        # supposed diagonal and all the slack variables that make it up. We always traverse in :U order, but
                        # now we have two slack variables per entry. We take the second one for everything above and the first
                        # for everything below.
                        δ = (exp1 -1) * block_size + block_i # current row index
                        si = div(slack_i, 2, RoundUp)
                        for i in si-δ+1:si-1
                            unsafe_push!(indices, slack_indices[2i])
                            unsafe_push!(values, -one(V))
                        end
                        let i=si-1+δ
                            for _ in (exp2 -1)*block_size+block_j+1:dim # the absolutes below are the same as to the right
                                unsafe_push!(indices, slack_indices[2i-1])
                                unsafe_push!(values, -one(V))
                                i += δ
                                δ += 1
                            end
                        end
                        fixstate = add_constr_fix!(state, fixstate, Indvals(indices, values), zero(V))
                    elseif length(lens) == 2
                        # Since the rhs is zero, these are just two nonnegative constraints on the two slack variables.
                        add_constr_nonnegative!(state, unchecked_iterator)
                    else
                        (rquad ? add_constr_rotated_quadratic! : add_constr_quadratic!)(state, to_soc!(indices, values, lens,
                            rquad))
                    end
                    empty!(indices)
                    empty!(values)
                    empty!(lens)
                elseif representation isa RepresentationPSD
                    if !matrix_indexing && complex && total_real && !ondiag
                        # We skipped the addition of an imaginary part because it is zero. But we have to tell this to the
                        # solver.
                        unsafe_push!(lens, zero(eltype(lens)))
                    end
                end
            end
            if representation isa RepresentationDSOS && l1
                # Simple case: only real values and we can just put the whole column in an ℓ₁ norm constraint
                (complex ? add_constr_l1_complex! : add_constr_l1!)(state, unchecked_iterator)
                empty!(indices)
                empty!(values)
                empty!(lens)
            end
        end
    end
    representation isa RepresentationDD && return # already did everything in the loop
    if representation isa RepresentationSDD
        add_constr_fix_finalize!(state, fixstate)
        return
    end
    if matrix_indexing
        return (complex ? add_constr_psd_complex! : add_constr_psd!)(
            state, dim, PSDMatrixCartesian{_get_offset(indextype)}(dim, Tri, finish!(rows), finish!(indices), finish!(values))
        )
    else
        let indices=finish!(indices), values=finish!(values)
            @inbounds if dim == 1
                @assert(length(lens) == 1)
                add_constr_nonnegative!(state, Indvals(indices, values))
            elseif dim == 2 && (rquad || quad)
                @assert(length(lens) == (complex ? 4 : 3))
                # Note: indices/values represent the vectorized PSD cone (lower triangle) with off-diagonals already
                # pre-multiplied by sqrt(2). To make this exactly equivalent to a rotated quadratic cone, we must bring the
                # last element to the second position.
                @views @unroll for x in (indices, values)
                    reverse!(x[lens[1]+1:end-lens[end]])
                    reverse!(x[end-lens[end]+1:end])
                    reverse!(x[lens[1]+1:end])
                end
                if complex
                    lens[2], lens[3], lens[4] = lens[4], lens[2], lens[3]
                else
                    lens[2], lens[3] = lens[3], lens[2]
                end
                (rquad ? add_constr_rotated_quadratic! : add_constr_quadratic!)(state, to_soc!(indices, values, lens, rquad))
            else
                (complex ? add_constr_psd_complex! : add_constr_psd!)(state, dim, IndvalsIterator(indices, values, lens))
            end
        end
    end
    return
end

# generic moment matrix constraint with complex-valued monomials involved in the grouping, but the solver does not support the
# complex PSD cone explicitly
function moment_add_matrix_helper!(state, T, V, grouping::AbstractVector{M} where M<:SimpleMonomial,
    constraint::AbstractMatrix{<:SimplePolynomial}, indextype::PSDIndextype{Tri},
    ::Tuple{Val{false},Val{false}}, representation::Union{<:RepresentationDD,RepresentationPSD}) where {Tri}
    matrix_indexing = indextype isa PSDIndextypeMatrixCartesian && representation isa RepresentationPSD
    frontdiag = false
    if matrix_indexing
        Tri ∈ (:L, :U) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            (Val(false), Val(false)), representation)))
        tri = :U # we always create the data in U format, as the PSDMatrixCartesian then has to compute the row/col indices
                 # based on the linear index - and the formula is slightly simpler for U.
    else
        Tri ∈ (:L, :U, :F) || throw(MethodError(moment_add_matrix_helper!, (state, T, V, grouping, constraint, indextype,
            (Val(false), Val(false))), representation))
        if representation isa RepresentationDD
            l1 = supports_lnorm(state)
            if l1
                frontdiag = true
                tri = :F
            else
                tri = :U
            end
        else
            tri = Tri
            scaling = sqrt(V(2))
        end
    end
    lg = length(grouping)
    block_size = LinearAlgebra.checksquare(constraint)
    dim = lg * block_size
    if dim == 1 || (dim == 2 && (supports_rotated_quadratic(state) || supports_quadratic(state)))
        # in these cases, we will rewrite the Hermitian PSD cone in terms of linear or quadratic constraints, so break off
        return moment_add_matrix_helper!(state, T, V, grouping, constraint, indextype, (Val(false), Val(true)), representation)
    end
    maxlen = maximum(length, constraint, init=0)
    colcount = tri === :F ? 4dim^2 : trisize(2dim)
    rows = FastVec{T}(buffer=2maxlen * colcount)
    indices = similar(rows)
    values = similar(rows, V)
    # Here we implement the default way: Z ⪰ 0 ⟺ [Re Z  -Im Z; Im Z  Re Z] ⪰ 0.
    # This is relatively simple, as "rows" encodes the position of the element in the matrix (in linear index form of the upper
    # triangle columnwise).
    dimT = T(dim)
    dim2 = 2dim
    items = tri === :F ? dimT^2 : trisize(dimT)
    i1 = zero(T)
    if tri === :U
        i2order = items
        i3 = items + dimT
    elseif tri === :L
        i2order = dimT
        i3 = (T(3) * dimT^2 + dimT) >> 1
    else
        i2order = dimT
        i2other = T(2) * items
        i3 = i2other + dimT
    end
    col = zero(T)
    @inbounds for (exp2, g₂) in enumerate(grouping)
        isreal_g₂ = isreal(g₂)
        for block_j in 1:block_size
            row = tri === :L ? col : zero(T)
            exp1_range = (tri === :F ? (1:lg) : (tri === :U ? (1:exp2) : (exp2:lg)))
            if tri === :U
                i2other = items + col + (row * (row + T(2) * dimT + one(T))) >> 1
            elseif tri === :L
                i2other = i2order
            end
            for (exp1, g₁) in zip(exp1_range, @view(grouping[exp1_range]))
                isreal_g₁ = isreal(g₁)
                for block_i in (tri === :F ? (1:block_size) : (tri === :U ? (1:(exp1 == exp2 ? block_j : block_size)) :
                                                                            ((exp1 == exp2 ? block_j : 1):block_size)))
                    total_real = block_i == block_j && (exp1 == exp2 || (isreal_g₁ && isreal_g₂))
                    ondiag = exp1 == exp2 && block_i == block_j
                    if frontdiag
                        # we must shuffle our diagonal elements to the front. Format is :F
                        if ondiag
                            _i1 = T(2) * dimT * col
                            _i3 = T(2) * items + T(2) * dimT * col
                        elseif row < col
                            _i1 = i1 + one(T)
                            _i3 = i3 + one(T)
                        else
                            _i1 = i1
                            _i3 = i3
                        end
                        _i2order = i2order # this is always below the diagonal
                        _i2other = i2other + one(T) # this is always above the diagonal
                    else
                        _i1 = i1
                        _i3 = i3
                        _i2order = i2order
                        _i2other = i2other
                    end
                    curlen = 0
                    for term_constr in constraint[block_i, block_j]
                        mon_constr = monomial(term_constr)
                        coeff_constr = coefficient(term_constr)
                        if representation isa RepresentationPSD && tri !== :F && !matrix_indexing && !ondiag
                            coeff_constr *= scaling
                        end
                        recoeff = real(coeff_constr)
                        imcoeff = imag(coeff_constr)
                        repart, impart, canonical = getreim(state, g₁, mon_constr, SimpleConjMonomial(g₂))
                        if total_real
                            if canonical
                                if !iszero(recoeff)
                                    unsafe_push!(rows, _i1, _i3)
                                    unsafe_push!(indices, repart, repart)
                                    if repart != impart
                                        recoeff *= V(2)
                                    end
                                    unsafe_push!(values, recoeff, recoeff)
                                    curlen += 2
                                end
                                if repart == impart
                                    @assert(iszero(imcoeff)) # else the polynomial on the diagonal would not be real-valued
                                    @assert(isreal(mon_constr))
                                elseif !iszero(imcoeff)
                                    unsafe_push!(rows, _i1, _i3)
                                    unsafe_push!(indices, impart, impart)
                                    imcoeff *= V(-2)
                                    unsafe_push!(values, imcoeff, imcoeff)
                                    curlen += 2
                                end
                            end
                        else
                            @assert(i2order != i2other)
                            if !iszero(recoeff)
                                @pushorupdate!(repart, _i1, recoeff, _i3, recoeff)
                                if repart != impart
                                    care = canonical ? recoeff : -recoeff
                                    # We don't really need the L/U distinction, as it does not matter where to place the
                                    # minus. However, to make testing simpler, we keep our promise.
                                    if tri === :L || tri === :F
                                        @pushorupdate!(impart, _i2order, care, _i2other, -care)
                                    else
                                        @pushorupdate!(impart, _i2other, care, _i2order, -care)
                                    end
                                end
                            end
                            if !iszero(imcoeff)
                                if tri === :L || tri === :F
                                    @pushorupdate!(repart, _i2order, imcoeff, _i2other, -imcoeff)
                                else
                                    @pushorupdate!(repart, _i2other, imcoeff, _i2order, -imcoeff)
                                end
                                if repart != impart
                                    care = canonical ? -imcoeff : imcoeff
                                    @pushorupdate!(impart, _i1, care, _i3, care)
                                end
                            end
                        end
                    end
                    row += one(T)
                    i1 += one(T)
                    if tri === :U
                        i2order += one(T)
                        i2other += dimT + row
                    elseif tri === :L
                        i2order += one(T)
                        i2other += T(2) * dimT - row
                    else
                        i2order += one(T)
                        i2other += one(T)
                    end
                    i3 += one(T)
                end
            end
            if tri === :U
                i2order += dimT
                i3 += dimT
            elseif tri === :L
                i1 += dimT
                i2order += dimT
            else
                i1 += dimT
                i2order += dimT
                i2other += dimT
                i3 += dimT
            end
            col += one(T)
        end
    end
    if matrix_indexing
        return add_constr_psd!(state, dim2, PSDMatrixCartesian{_get_offset(indextype)}(dim2, Tri, finish!(rows),
            finish!(indices), finish!(values)))
    else
        # We constructed everything in matrix form, as it was easier to assign arbitrary positions in this way; but now we need
        # to convert it to the iterative form. We cannot re-use rows for this, unfortunately. It may happen that some rows
        # (i.e., entries for the PSD matrix) are zero and don't arise in rows, although they will have to be present in lens.
        # So to avoid potentially overwriting data, we need a whole new vector.
        @inbounds let rows=finish!(rows), indices=finish!(indices), values=finish!(values)
            if representation isa RepresentationDD
                lens = FastVec{SimplePolynomials.smallest_unsigned(2maxlen)}(buffer=dim2)
                # Here we must work columnwise (by the actual columns). All diagonals were already put at the front of the
                # column.
                sort_along!(rows, indices, values)
                row = zero(T)
                j = 1
                remaining = length(rows)
                fullcol = 1
                if l1
                    fullcolstart = 1
                    while fullcol ≤ dim2
                        fullrow = 0
                        while !iszero(remaining)
                            currow = rows[j]
                            while currow > row
                                isempty(lens) && unsafe_push!(lens, zero(eltype(lens))) # in this cone, we only a zero entry if
                                                                                        # it is the first
                                row += one(T)
                                fullrow += 1
                                fullrow == dim2 && @goto colfull
                            end
                            curlen = one(eltype(lens))
                            j += 1
                            remaining -= 1
                            while !iszero(remaining) && rows[j] == currow
                                curlen += one(eltype(lens))
                                j += 1
                                remaining -= 1
                            end
                            unsafe_push!(lens, curlen)
                            row += one(T)
                            fullrow += 1
                            fullrow == dim2 && break
                        end
                        @label colfull
                        add_constr_l1!(state, IndvalsIterator(@view(indices[fullcolstart:j-1]),
                            @view(values[fullcolstart:j-1]), lens))
                        empty!(lens)
                        fullcolstart = j
                        fullcol += 1
                    end
                else
                    indices_buf = similar(indices, maxlen + dim2 -1) # hold the diagonal plus all slacks
                    values_buf = similar(values, length(indices_buf))
                    slack_indices = add_var_slack!(state, trisize(dim2 -1))
                    slack_i = 1
                    while fullcol ≤ dim2
                        fullrow = 1
                        while fullrow ≤ fullcol
                            currow = rows[j]
                            # Even if we don't have any element in the rows, we need to create slack variables fixed to zero -
                            # because later in the diagonal, we need to be able to access these slack variables with a simple
                            # index calculation.
                            while currow > row && fullcol > fullrow
                                add_constr_nonnegative!(state,
                                    IndvalsIterator(
                                        StackVec(slack_indices[slack_i], slack_indices[slack_i]),
                                        StackVec(one(V), -one(V)),
                                        1
                                    )
                                )
                                row += one(T)
                                fullrow += 1
                                slack_i += 1
                            end
                            itemstart = j
                            if currow == row
                                curlen = one(eltype(lens))
                                j += 1
                                remaining -= 1
                                while !iszero(remaining) && rows[j] == currow
                                    curlen += one(eltype(lens))
                                    j += 1
                                    remaining -= 1
                                end
                            else
                                curlen = zero(eltype(lens))
                                # No content on the diagonal - unfortunate user input: PSD constraint with a zero diagonal,
                                # which would be better off deleting the row and column and adding zero constraints on the
                                # deleted content. But this is likely to arise sometimes when we do the complex → real
                                # transformation.
                            end
                            if fullrow == fullcol
                                # We now need to add all the slack variables in the column to the linear constraint.
                                δ = fullcol # current row index
                                # Unfortunately, we cannot guarantee that the items before itemstart (which were already) used
                                # are sufficient to hold all slack variable indices (not even those that were above the
                                # diagonal, if some of the content there was zero). However, this is pretty likely once we are
                                # sufficiently far in the matrix. And once it has happened, it will always be the case. So we
                                # need to copy to a buffer only in an easy-to-check condition that will almost never fail
                                # during branch prediction.
                                if itemstart > dim2 -1
                                    copyto!(indices, itemstart - dim2 +1, slack_indices, slack_i - δ +1, δ -1)
                                    let i=slack_i-1+δ
                                        for (ib, col) in zip(Iterators.countfrom(itemstart - dim2 + δ), fullcol+1:dim2)
                                            indices[ib] = slack_indices[i]
                                            i += δ
                                            δ += 1
                                        end
                                    end
                                    fill!(@view(values[itemstart-dim2+1:itemstart-1]), -one(V))
                                    r = itemstart-dim2+1:j-1
                                    add_constr_nonnegative!(state, Indvals(@view(indices[r]), @view(values[r])))
                                else
                                    copyto!(indices_buf, 1, slack_indices, slack_i - δ +1, δ -1)
                                    let i=slack_i-1+δ
                                        for (ib, col) in zip(Iterators.countfrom(δ), fullcol+1:dim2)
                                            indices_buf[ib] = slack_indices[i]
                                            i += δ
                                            δ += 1
                                        end
                                    end
                                    copyto!(indices_buf, dim2, indices, itemstart, j - itemstart)
                                    fill!(@view(values_buf[1:dim2-1]), -one(V))
                                    copyto!(values_buf, dim2, values, itemstart, j - itemstart)
                                    r = 1:j-itemstart+dim2-1
                                    add_constr_nonnegative!(state, Indvals(@view(indices_buf[r]), @view(values_buf[r])))
                                end
                            else
                                # off-diagonal: Make the slack variable the absolute value of the matrix element. For this, we
                                # will need to add the slack index to the list of indices. Unless the user has some very stupid
                                # input with a vanishing (1, 1) matrix constraint, itemstart > 1, so the following condition
                                # can be perfectly predicted. But since we can't protect against stupidity, we have to take
                                # care of this very special case.
                                if itemstart > 1
                                    indices[itemstart-1] = slack_indices[slack_i]
                                    values[itemstart-1] = one(V)
                                    iv = Indvals(@view(indices[itemstart-1:j-1]), @view(values[itemstart-1:j-1]))
                                    add_constr_nonnegative!(state, iv)
                                    rmul!(@view(values[itemstart:j-1]), -one(V))
                                    add_constr_nonnegative!(state, iv)
                                else
                                    indices_buf[1] = slack_indices[slack_i]
                                    copyto!(indices_buf, 2, indices, itemstart, j - itemstart)
                                    values_buf[1] = one(V)
                                    copyto!(values_buf, 2, values, itemstart, j - itemstart)
                                    iv = Indvals(@view(indices_buf[1:j-itemstart]), @view(values_buf[1:j-itemstart]))
                                    add_constr_nonnegative!(state, iv)
                                    rmul!(@view(values_buf[2:j-itemstart]), -one(V))
                                    add_constr_nonnegative!(state, iv)
                                end
                                slack_i += 1
                            end
                            row += one(T)
                            fullrow += 1
                        end
                        fullcol += 1
                    end
                end
            elseif representation isa RepresentationPSD
                lens = Vector{SimplePolynomials.smallest_unsigned(2maxlen)}(undef, (Tri === :F ? 4dim^2 : trisize(2dim)))
                sort_along!(rows, indices, values)
                row = zero(T)
                i = 1
                j = 1
                remaining = length(rows)
                while !iszero(remaining)
                    currow = rows[j]
                    while currow > row
                        lens[i] = zero(eltype(lens))
                        i += 1
                        row += one(T)
                    end
                    curlen = one(eltype(lens))
                    j += 1
                    remaining -= 1
                    while !iszero(remaining) && rows[j] == currow
                        curlen += one(eltype(lens))
                        j += 1
                        remaining -= 1
                    end
                    lens[i] = curlen
                    i += 1
                    row += one(T)
                end
                fill!(@view(lens[i:end]), zero(eltype(lens)))

                add_constr_psd!(state, 2dim, IndvalsIterator(indices, values, lens))
            end
        end
    end
end

"""
    moment_add_matrix!(state, grouping::SimpleMonomialVector,
        constraint::Union{<:SimplePolynomial,<:AbstractMatrix{<:SimplePolynomial}},
        representation::RepresentationMethod=RepresentationPSD())

Parses a constraint in the moment hierarchy with a basis given in `grouping` (this might also be a partial basis due to
sparsity), premultiplied by `constraint` (which may be the unit polynomial for the moment matrix) and calls the appropriate
solver functions to set up the problem structure according to `representation`.

To make this function work for a solver, implement the following low-level primitives:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_l1!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_constr_l1_complex!`](@ref) (optional, then set [`supports_complex_lnorm`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_var_slack!`](@ref)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_equality!`](@ref), [`RepresentationMethod`](@ref).
"""
function moment_add_matrix!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::Union{P,<:AbstractMatrix{P}}, representation::RepresentationMethod=RepresentationPSD()) where {P<:SimplePolynomial}
    representation isa RepresentationDD{true} && !supports_quadratic(state) && !supports_complex_lnorm(state) &&
        error("The solver does not support quadratic or complex ℓ₁ cones, so a Hermitian representation via diagonally-dominant matrices is not possible")
    # TODO (unlikely): we could still do it if the solver supports only the rotated, but not the standard cone. But are there
    # any solvers in this category?
    representation isa RepresentationSDD && !supports_rotated_quadratic(state) && !supports_quadratic(state) &&
        error("The solver does not support (rotated) quadratic cones, so a representation via scaled diagonally-dominant matrices is not possible")
    return moment_add_matrix_helper!(
        state,
        Base.promote_op(mindex, typeof(state), monomial_type(P)),
        real(coefficient_type(P)),
        grouping,
        constraint isa AbstractMatrix ? constraint : ScalarMatrix(constraint),
        psd_indextype(state),
        (Val((length(grouping) == 1 || isreal(grouping)) &&
             (constraint isa SimplePolynomial || isreal(constraint))),
         Val(representation isa RepresentationDD{true} ||
             representation isa RepresentationSDD ||
             (representation isa RepresentationPSD && supports_complex_psd(state)))),
        representation
    )
end

"""
    moment_add_equality!(state, grouping::SimpleMonomialVector, constraint::SimplePolynomial)

Parses a polynomial equality constraint for moments and calls the appropriate solver functions to set up the problem structure.
`grouping` contains the basis that will be squared in the process to generate the prefactor.

To make this function work for a solver, implement the following low-level primitives:
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_fix_finalize!`](@ref) (optional)

Usually, this function does not have to be called explicitly; use [`moment_setup!`](@ref) instead.

See also [`moment_add_matrix!`](@ref).
"""
function moment_add_equality!(state, grouping::AbstractVector{M} where {M<:SimpleMonomial},
    constraint::P) where {Nr,Nc,I<:Integer,P<:SimplePolynomial{<:Any,Nr,Nc,<:SimpleMonomialVector{Nr,Nc,I}}}
    # We need to traverse all unique elements in groupings * groupings†. For purely complex-valued groupings, this is the full
    # list; as soon as we have a real variable present, it is smaller.
    # To avoid rehashings, get an overestimator of the total grouping size first.
    # TODO (maybe): In the first loop to populate unique_groupings, we determine whether the grouping is real-valued. So we
    # could instead populate two sets, saving isreal and a lot of conditionals in the second loop.
    unique_groupings = sizehint!(Set{FastKey{I}}(), iszero(Nc) ? trisize(length(grouping)) : length(grouping)^2)
    real_grouping = true
    totalsize = 0
    for (i, g₁) in enumerate(grouping)
        if !iszero(Nc)
            g₁real = !iszero(Nr) && isreal(g₁)
            # Consider the g₂ = ḡ₁ case separately in the complex case. Explanations below.
            let g₂=g₁
                prodidx = FastKey(monomial_index(g₁, SimpleConjMonomial(g₂)))
                indexug, sh = Base.ht_keyindex2_shorthash!(unique_groupings.dict, prodidx)
                if indexug ≤ 0
                    @inbounds Base._setindex!(unique_groupings.dict, nothing, prodidx, -indexug, sh)
                    totalsize += 1
                end
            end
        end
        # In the real case, we can skip the entries behind i as they would lead to duplicates.
        # In the complex case, we can also skip them, as they would lead to exact conjugates, which in the end give rise to the
        # same conditions (but note that i is already handled above).
        for g₂ in Iterators.take(grouping, iszero(Nc) ? i : i -1)
            # We don't use mindex, as this can have unintended side-effects on the solver state (such as creating a
            # representation for this monomial, although we don't even know whether we need it - if constraint does not contain
            # a constant term, this function must not automatically add all the squared groupings as monomials, even if they
            # will probably appear at some place).
            prodidx = FastKey(monomial_index(g₁, SimpleConjMonomial(g₂)))
            # We need to add the product to the set if it does not exists; we also need to count the number of conditions that
            # we get out of it.
            indexug, sh = Base.ht_keyindex2_shorthash!(unique_groupings.dict, prodidx)
            if indexug ≤ 0
                # It does not exist.
                @inbounds Base._setindex!(unique_groupings.dict, nothing, prodidx, -indexug, sh)
                # Assume we have a grouping g = (gᵣ + im*gᵢ) and a polynomial p = pᵣ + im*pᵢ, where the individual parts are
                # real-valued. Then, add_equality! means that g*p = 0 and ḡ*p = 0. Of course we can also conjugate everything.
                # We must split each constraint into its real and imaginary parts:
                # (I)   Re(g*p) = gᵣ*pᵣ - gᵢ*pᵢ
                # (II)  Im(g*p) = gᵣ*pᵢ + gᵢ*pᵣ
                # (III) Re(ḡ*p) = gᵣ*pᵣ + gᵢ*pᵢ
                # (IV)  Im(ḡ*p) = gᵣ*pᵢ - gᵢ*pᵣ
                # To analyze this (which would be easier if we added and subtracted the equalities, but in the
                # SimplePolynomials setup, the given form is most easy to handle), let's consider linear dependencies.
                # - If the constraint is real-valued, (III) is equal to (I) and (IV) is -(II), so we only take (I) and (II).
                # - If the grouping is real-valued, (III) is equal to (I) and (IV) is equal to (II); we only take (I) and (II).
                # - If both are real-valued, (III) is equal to (I) while (II) and (IV) are zero, so we only take (I).
                # - If both are complex-valued, all constraints are linearly independent.
                # Rearranging this, we always take (I); if at least one is complex-valued, we also take (II); if both are, we
                # take all. Note that we don't have to consider the conjugates of the groupings separately, as they only yield
                # a global sign in the zero-equality.
                # For this loop, this means that we will only check whether g₁*ḡ₂ belongs to a real-valued monomial, in which
                # case we add 1; or to a complex-valued monomial, in which case we add 2. After the loop, we multiply by 2 if
                # the constraint was also complex-valued.
                if iszero(Nc) || (!iszero(Nr) && g₁real && isreal(g₂)) # note that g₁ ≠ ḡ₂
                    totalsize += 1
                else
                    totalsize += 2
                    real_grouping = false
                end
            end
        end
    end

    real_constr = isreal(constraint)
    if !real_constr
        totalsize *= 2
    end

    constrstate = @inline add_constr_fix_prepare!(state, totalsize)
    V = real(coefficient_type(P))
    indices₁ = FastVec{Base.promote_op(mindex, typeof(state), monomial_type(P))}(buffer=2length(constraint))
    values₁ = similar(indices₁, V)
    # While we could conditionally define those variables only if the requirements are satisfied, the compiler might not be
    # able to infer that we only use them later on if the same conditions (potentially stricter) are met. So define them
    # always, but not using any memory.
    indices₂ = similar(indices₁, 0, buffer=real_grouping && real_constr ? 0 : 2length(constraint))
    values₂ = similar(indices₂, V)
    indices₃ = similar(indices₁, 0, buffer=real_grouping || real_constr ? 0 : 2length(constraint))
    values₃ = similar(indices₃, V)
    indices₄ = similar(indices₃)
    values₄ = similar(values₃)

    e = ExponentsAll{Nr+2Nc,I}()

    for grouping_idx in unique_groupings
        grouping = SimpleMonomial{Nr,Nc}(unsafe, e, convert(I, grouping_idx))
        real_grouping = isreal(grouping)
        skip₂ = real_constr && real_grouping
        for term_constr in constraint
            mon_constr = monomial(term_constr)
            coeff_constr = coefficient(term_constr)
            recoeff = real(coeff_constr)
            imcoeff = imag(coeff_constr)
            repart, impart, canonical = getreim(state, grouping, mon_constr)
            if !iszero(recoeff)
                pushorupdate!(indices₁, repart, values₁, recoeff)
                skip₂ || repart == impart || pushorupdate!(indices₂, impart, values₂, canonical ? recoeff : -recoeff)
            end
            if !iszero(imcoeff)
                repart == impart || pushorupdate!(indices₁, impart, values₁, canonical ? -imcoeff : imcoeff)
                skip₂ || pushorupdate!(indices₂, repart, values₂, imcoeff)
            end
            if !real_grouping && !real_constr
                repart₂, impart₂, canonical₂ = getreim(state, SimpleConjMonomial(grouping), mon_constr)
                if !iszero(recoeff)
                    pushorupdate!(indices₃, repart₂, values₃, recoeff)
                    repart₂ == impart₂ || pushorupdate!(indices₄, impart₂, values₄, canonical₂ ? recoeff : -recoeff)
                end
                if !iszero(imcoeff)
                    repart₂ == impart₂ || pushorupdate!(indices₃, impart₂, values₃, canonical₂ ? -imcoeff : imcoeff)
                    pushorupdate!(indices₄, repart₂, values₄, imcoeff)
                end
            end
        end
        constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₁, values₁), zero(V))
        empty!(indices₁); empty!(values₁)
        if !skip₂
            if !isempty(indices₂)
                constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₂, values₂), zero(V))
                empty!(indices₂); empty!(values₂)
            end
            if !real_grouping && !real_constr
                if !isempty(indices₃)
                    constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₃, values₃), zero(V))
                    empty!(indices₃); empty!(values₃)
                end
                if !isempty(indices₄)
                    constrstate = @inline add_constr_fix!(state, constrstate, Indvals(indices₄, values₄), zero(V))
                    empty!(indices₄); empty!(values₄)
                end
            end
        end
    end
    @inline add_constr_fix_finalize!(state, constrstate)
    return
end

"""
    moment_setup!(state, relaxation::AbstractRelaxation, groupings::RelaxationGroupings[; representation])

Sets up all the necessary moment matrices, variables, constraints, and objective of a polynomial optimization problem
`problem` according to the values given in `grouping` (where the first entry corresponds to the basis of the objective, the
second of the equality, the third of the inequality, and the fourth of the PSD constraints).

The following methods must be implemented by a solver to make this function work:
- [`mindex`](@ref)
- [`add_constr_nonnegative!`](@ref)
- [`add_constr_rotated_quadratic!`](@ref) (optional, then set [`supports_rotated_quadratic`](@ref) to `true`)
- [`add_constr_quadratic!`](@ref) (optional, then set [`supports_quadratic`](@ref) to `true`)
- [`add_constr_l1!`](@ref) (optional, then set [`supports_lnorm`](@ref) to `true`)
- [`add_constr_l1_complex!`](@ref) (optional, then set [`supports_complex_lnorm`](@ref) to `true`)
- [`add_constr_psd!`](@ref)
- [`add_constr_psd_complex!`](@ref) (optional, then set [`supports_complex_psd`](@ref) to `true`)
- [`psd_indextype`](@ref)
- [`add_constr_fix_prepare!`](@ref) (optional)
- [`add_constr_fix!`](@ref)
- [`add_constr_fix_finalize!`](@ref) (optional)
- [`fix_objective!`](@ref)
- [`add_var_slack!`](@ref)

!!! warning "Indices"
    The variable indices used in all solver functions directly correspond to the indices given back by [`mindex`](@ref).
    However, in a sparse problem there may be far fewer indices present; therefore, when the problem is finally given to the
    solver, care must be taken to eliminate all unused indices. The functionality provided by [`AbstractAPISolver`](@ref) and
    [`AbstractSparseMatrixSolver`](@ref) already takes care of this.

!!! info "Order"
    This function is guaranteed to set up the fixed constraints first, then followed by all the others. However, the order of
    nonnegative, quadratic, ``\\ell_1`` norm, and PSD constraints is undefined (depends on the problem).

!!! info "Representation"
    This function may also be used to describe simplified cones such as the (scaled) diagonally dominant one. The
    `representation` parameter can be used to define a representation that is employed for the individual groupings. This may
    either be an instance of a [`RepresentationMethod`](@ref) - which requires the method to be independent of the dimension of
    the grouping - or a callable. In the latter case, it will be passed as a first parameter an identifier[^3] of the current
    conic variable, and as a second parameter the side dimension of its matrix. The method must then return a
    [`RepresentationMethod`](@ref) instance.

    [^3]: This identifier will be a tuple, where the first element is a symbol - either `:objective`, `:nonneg`, or `:psd` - to
          indicate the general reason why the variable is there. The second element is an `Int` denoting the index of the
          constraint (and will be undefined for the objective, but still present to avoid extra compilataion). The last element
          is an `Int` denoting the index of the grouping within the constraint/objective.

See also [`sos_setup!`](@ref), [`moment_add_matrix!`](@ref), [`moment_add_equality!`](@ref),
[`RepresentationMethod`](@ref).
"""
function moment_setup!(state, relaxation::AbstractRelaxation{<:Problem{P}}, groupings::RelaxationGroupings;
    representation::Union{<:RepresentationMethod,<:Base.Callable}=RepresentationPSD()) where {P}
    problem = poly_problem(relaxation)
    T = Base.promote_op(mindex, typeof(state), monomial_type(P))
    V = real(coefficient_type(problem.objective))

    # fixed items
    # fix constant term to 1
    if isone(problem.prefactor)
        @inline add_constr_fix_finalize!(
            state,
            add_constr_fix!(
                state,
                add_constr_fix_prepare!(state, 1),
                Indvals(StackVec(mindex(state, constant_monomial(P))), StackVec(one(V))),
                one(V)
            )
        )
    else
        let buffer=length(problem.prefactor), indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
            for t in problem.prefactor
                # We know that if imaginary parts pop up somewhere, they will cancel out in the end, as the conjugates also
                # appear. So use the same strategy as always and only add canonical ones, but doubled.
                mon = monomial(t)
                coeff = coefficient(t)
                recoeff = real(coeff)
                imcoeff = imag(coeff)
                repart, impart, canonical = getreim(state, mon)
                if repart == impart
                    @assert(iszero(imcoeff)) # else the objective would not be real-valued
                    unsafe_push!(indices, repart)
                    unsafe_push!(values, recoeff)
                elseif canonical
                    if !iszero(recoeff)
                        unsafe_push!(indices, repart)
                        unsafe_push!(values, recoeff + recoeff)
                    end
                    if !iszero(imcoeff)
                        unsafe_push!(indices, impart)
                        unsafe_push!(values, -(imcoeff + imcoeff))
                    end
                end
            end
            @inline add_constr_fix_finalize!(
                state,
                add_constr_fix!(state, add_constr_fix_prepare!(state, 1), Indvals(indices, values), one(V))
            )
        end
    end
    for (groupingsᵢ, constrᵢ) in zip(groupings.zeros, problem.constr_zero)
        for grouping in groupingsᵢ
            moment_add_equality!(state, collect_grouping(grouping), constrᵢ)
        end
    end

    # SOS term for objective
    constantP = SimplePolynomial(constant_monomial(P), coefficient_type(problem.objective))
    for (i, grouping) in enumerate(groupings.obj)
        g = collect_grouping(grouping)
        moment_add_matrix!(state, g, constantP,
            representation isa RepresentationMethod ? representation : representation((:objective, 0, i), length(g)))
    end
    # localizing matrices
    for (i, (groupingsᵢ, constrᵢ)) in enumerate(zip(groupings.nonnegs, problem.constr_nonneg))
        for (j, grouping) in enumerate(groupingsᵢ)
            g = collect_grouping(grouping)
            moment_add_matrix!(state, g, constrᵢ,
                representation isa RepresentationMethod ? representation : representation((:nonneg, i, j), length(g)))
        end
    end
    for (i, (groupingsᵢ, constrᵢ)) in enumerate(zip(groupings.psds, problem.constr_psd))
        for (j, grouping) in enumerate(groupingsᵢ)
            g = collect_grouping(grouping)
            moment_add_matrix!(state, g, constrᵢ,
                representation isa RepresentationMethod ? representation :
                                                                  representation((:psd, i, j), length(g) * size(constrᵢ, 1)))
        end
    end

    # Riesz functional in the objective
    let buffer=length(problem.objective), indices=FastVec{T}(; buffer), values=FastVec{V}(; buffer)
        for t in problem.objective
            mon = monomial(t)
            coeff = coefficient(t)
            recoeff = real(coeff)
            imcoeff = imag(coeff)
            repart, impart, canonical = getreim(state, mon)
            # repart is the variable that is associated with the real part of the monomial mon;
            # impart is the variable associated with the imaginary part of the canonicalized monomial mon.
            if repart == impart
                @assert(iszero(imcoeff)) # else the objective would not be real-valued
                unsafe_push!(indices, repart)
                unsafe_push!(values, recoeff)
            elseif canonical # skip over noncanonicals - to have a real-valued objective, we know how they must look like
                if !iszero(recoeff)
                    unsafe_push!(indices, repart)
                    unsafe_push!(values, recoeff + recoeff)
                end
                if !iszero(imcoeff)
                    unsafe_push!(indices, impart)
                    unsafe_push!(values, -(imcoeff + imcoeff))
                end
            end
        end
        fix_objective!(state, Indvals(finish!(indices), finish!(values)))
    end

    return
end