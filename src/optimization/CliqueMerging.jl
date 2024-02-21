function merge_cliques!(cliques::AbstractVector{<:AbstractSet{T}}) where {T}
    # directly drop cliques of length 1 and 2. They are so efficient (linear vs. quadratic constraints) that we don't even
    # consider them in the merge process
    smallcliques = @view cliques[length.(cliques).≤2]
    cliques = @view cliques[length.(cliques).≥3]
    # first form the clique graph; this time, we work with the adjacency matrix
    n = length(cliques)
    n ≤ 1 && return [smallcliques; cliques]
    @inbounds adjmOwn = collect(@capture(i > j ? length($cliques[i])^3 + length(cliques[j])^3 -
                                                 length(cliques[i] ∪ cliques[j])^3 : 0
                                         for i in 1:n, j in 1:n))
    idxOwn = fill(true, n)
    GC.@preserve adjmOwn idxOwn begin
        adjm = unsafe_wrap(Array, pointer(adjmOwn), (n, n), own=false)
        idx = unsafe_wrap(Array, pointer(idxOwn), n, own=false)
        deleted = 0
        @inbounds while true
            # select two permissible cliques with the highest weight
            w, maxidx = findmax(adjm)
            i = maxidx[1]
            j = maxidx[2]
            # while clique graph contains positive weights
            w ≤ 0 && break
            # merge cliques
            union!(cliques[i], cliques[j])
            idx[j] = false
            # In every iteration, we have to search through n^2 elements, we have to reset n-1 elements and also recalculate
            # n-1 elements. The calculation a^3+b^3-c^3 itself consists of 6 multiplications, one addition and one subtraction
            # (and a couple of loads). Multiplication cost is about 3*addition cost, but with out-of-order execution this may
            # be reduced again, but assuming 20 cycles per calcuation (which is also conditional) is fair. Resetting to zero
            # can be done using ymm registers for four items in one step, but vmovups also has a latency of ~4 (varies greatly
            # between the architectures). But it is unrolled with four instructions following each other, which may partially
            # compensate. And we have the counter add, check and jump, so about 4.25(n-1) clock cycles for the reset would be
            # fair. In comparison, the @inbounds findmax requires about 8 clock cycles per element.
            # So the total cost is 8n^2 [search] + 4.25(n-1) [reset] + 20(n-1) [recalculate] for every iteration, until we are
            # done. If we instead restart the evaluation, we have to recalculate everything: 20n^2, but then the n is smaller.
            # So when is n * (8n^2 + 24.25(n +1)) ≥ deleted * (8n^2 + 24.25(n +1)) + 20(n - deleted)^2 +
            #                                       (n - deleted)*(8(n - deleted)^2 + 24.25(n - deleted +1))
            # As soon as n > deleted ≥ 2, we find this to be fulfilled. However, these theoretical considerations do not seem
            # to be particularly successful.
            deleted += 1
            if n > 100 && deleted == 50
                cliques = @view cliques[idx]
                n -= deleted
                # we already have enough space allocated at adjm; we will now simply overwrite it.
                adjm = unsafe_wrap(Array, pointer(adjmOwn), (n, n), own=false)
                adjm .= collect(@capture(i > j ? length($cliques[i])^3 + length(cliques[j])^3 -
                                                 length(cliques[i] ∪ cliques[j])^3 : 0
                                         for i in 1:n, j in 1:n))
                # same for idx
                idx = unsafe_wrap(Array, pointer(idxOwn), n, own=false)
                idx .= true
                deleted = 0
            else
                # update clique graph
                adjm[j, 1:j-1] .= 0
                adjm[j+1:n, j] .= 0
                # recompute weights for updated clique graph
                newLenCube = length(cliques[i])^3
                for k in 1:i-1
                    idx[k] && (adjm[i, k] = length(cliques[k])^3 + newLenCube - length(cliques[k] ∪ cliques[i])^3)
                end
                for k in i+1:n
                    idx[k] && (adjm[k, i] = newLenCube + length(cliques[k])^3 - length(cliques[k] ∪ cliques[i])^3)
                end
            end
        end
    end
    return [smallcliques; cliques[idx]]
end

_convert_clique(cl) = Set(monomial_index(x) for x in cl)
function _reconvert_clique(::Type{SMV}, clset) where {Nr,Nc,P<:Unsigned,SMV<:SimpleMonomialVector{Nr,Nc,P}}
    exponents_real, exponents_complex, exponents_conj =
        exponents_from_indices(P, Nr, Nc, clset, Val(SMV <: SimplePolynomials.SimpleDenseMonomialVector))
    return SMV(iszero(Nr) ? SimplePolynomials.absent : exponents_real, exponents_complex, exponents_conj)
end
function _reconvert_clique(::Type{SMV}, clset) where {Nr,P<:Unsigned,SMV<:SimpleMonomialVector{Nr,0,P}}
    exponents_real = exponents_from_indices(P, Nr, clset, Val(SMV <: SimplePolynomials.SimpleDenseMonomialVector))
    return SMV(exponents_real, SimplePolynomials.absent, SimplePolynomials.absent)
end

function merge_cliques(groupings::RelaxationGroupings{Nr,Nc,P,<:Any,MV}) where {Nr,Nc,P<:Unsigned,MV}
    dense = !(MV <: SimplePolynomials.SimpleSparseMonomialVectorOrView)
    M = dense ? Matrix{P} : SparseMatrixCSC{P,UInt}
    outtype_part = SimpleMonomialVector{Nr,Nc,P,M}
    outtype_full = outtype_part{iszero(Nr) ? SimplePolynomials.Absent : M,iszero(Nc) ? SimplePolynomials.Absent : M}
    newobj = let obj_merged=merge_cliques!(_convert_clique.(groupings.obj))
        newobj = similar(obj_merged, outtype_full)
        for (i, objᵢ) in enumerate(obj_merged)
            @inbounds newobj[i] = _reconvert_clique(outtype_part, objᵢ)
        end
        newobj
    end
    # it does not make sense to merge the zero constraints, as every unique product corresponds to a separate variable anyway.
    newnonnegs = similar(groupings.nonnegs, Vector{outtype_full})
    @inbounds for (k, nonneg) in groupings.nonnegs
        newnonnegs[k] = let nonneg_merged=merge_cliques!(_convert_clique.(nonneg))
            newnonneg = similar(nonneg, outtype_full)
            for (i, nonnegᵢ) in enumerate(nonneg_merged)
                newnonneg[i] = _reconvert_clique(outtype_part, nonnegᵢ)
            end
            newnonneg
        end
    end
    newpsds = similar(groupings.psds, Vector{outtype_full})
    @inbounds for (k, psd) in groupings.psds
        newpsds[k] = let psd_merged=merge_cliques!(_convert_clique.(psd))
            newpsd = similar(psd, outtype_full)
            for (i, psdᵢ) in enumerate(psd_merged)
                newpsd[i] = _reconvert_clique(outtype_part, psdᵢ)
            end
            newpsd
        end
    end
    return RelaxationGroupings(newobj, groupings.zeros, newnonnegs, newpsds, groupings.var_cliques)
end