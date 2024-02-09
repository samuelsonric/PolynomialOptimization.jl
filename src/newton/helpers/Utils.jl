toSigned(x::UInt8) = Core.bitcast(Int8, x)
toSigned(x::UInt16) = Core.bitcast(Int16, x)
toSigned(x::UInt32) = Core.bitcast(Int32, x)
toSigned(x::UInt64) = Core.bitcast(Int64, x)
toSigned(x::Signed) = x

function isless_degree(x::AbstractVector, y::AbstractVector)
    dx = sum(x)
    dy = sum(y)
    if dx == dy
        return isless(x, y)
    else
        return isless(dx, dy)
    end
end

function monomial_cut(mindeg, maxdeg, minmultideg, maxmultideg, batchsize)
    # to calculate the part-range-sizes, we have to duplicate some code from length(::MonomialIterator), but our particular
    # application here is too special to integrate it: We start calculating the length of the iterator starting from the left
    # and as soon as we hit the batchsize boundary, we know that all the parts to the right should be done by individual
    # tasks.
    cutat = 1
    occurrences = zeros(Int, maxdeg +1)
    @inbounds for deg₁ in minmultideg[1]:min(maxmultideg[1], maxdeg)
        occurrences[deg₁+1] = 1
    end
    nextround = similar(occurrences)
    restmax = sum(@view(maxmultideg[2:end]), init=0)
    for (minᵢ, maxᵢ) in Iterators.drop(zip(minmultideg, maxmultideg), 1)
        restmax -= min(maxᵢ, maxdeg)
        fill!(nextround, 0)
        for degᵢ in minᵢ:min(maxᵢ, maxdeg)
            for (degⱼ, occⱼ) in zip(Iterators.countfrom(0), occurrences)
                newdeg = degᵢ + degⱼ
                newdeg > maxdeg && break
                @inbounds nextround[newdeg+1] += occⱼ
            end
        end
        sum(@view(nextround[max(mindeg - restmax +1, 1):end])) > batchsize && break
        occurrences, nextround = nextround, occurrences
        cutat += 1
    end
    return cutat
end