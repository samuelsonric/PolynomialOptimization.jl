function restore_status!(fileprogress, workload::Vector{Int}, powers::Vector{T}) where {T<:Integer}
    lastprogress = UInt8[]
    seekstart(fileprogress)
    s = 2sizeof(Int) + sizeof(workload) + sizeof(powers)
    nb = readbytes!(fileprogress, lastprogress, s)
    GC.@preserve lastprogress begin
        if iszero(nb)
            return nothing
        elseif nb == s
            lpp = Ptr{Int}(pointer(lastprogress))
            unsafe_copyto!(pointer(powers), Ptr{T}(lpp + 2sizeof(Int)), length(powers))
            unsafe_copyto!(pointer(workload), lpp + 2sizeof(Int) + sizeof(powers), length(workload))
            return unsafe_load(lpp), unsafe_load(lpp, 2)
        else
            error("Unknown progress file format - please delete existing files.")
        end
    end
end