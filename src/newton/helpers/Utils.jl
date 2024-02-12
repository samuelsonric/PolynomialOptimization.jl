function isless_degree(x::AbstractVector, y::AbstractVector)
    dx = sum(x)
    dy = sum(y)
    if dx == dy
        return isless(x, y)
    else
        return isless(dx, dy)
    end
end