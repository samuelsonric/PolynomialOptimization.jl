struct InitialStateIterator{I,S,L}
    iter::I
    initial_state::S
    skip_length::L

    InitialStateIterator(iter, initial_state, skip_length::Union{<:Integer,Missing}=missing) =
        new{typeof(iter),typeof(initial_state),typeof(skip_length)}(iter, initial_state, skip_length)
end

Base.iterate(iter::InitialStateIterator, state=iter.initial_state) = iterate(iter.iter, state)
Base.IteratorSize(::Type{<:InitialStateIterator{<:Any,<:Any,Missing}}) = Base.SizeUnknown()
Base.IteratorSize(::Type{<:InitialStateIterator{I}}) where {I} = Base.drop_iteratorsize(Base.IteratorSize(I))
Base.IteratorEltype(::Type{<:InitialStateIterator{I}}) where {I} = Base.IteratorEltype(I)
Base.eltype(::Type{<:InitialStateIterator{I}}) where {I} = eltype(I)
Base.length(iter::InitialStateIterator{<:Any,<:Any,<:Integer}) = length(iter.iter) - iter.skip_length
Base.isdone(iter::InitialStateIterator, state=iter.initial_state) = Base.isdone(iter, state)