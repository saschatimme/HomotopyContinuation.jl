"""
    TaylorVector{N,T}

A data structure representing a vector of Taylor series with `N` terms.
Each element is an `NTuple{N,T}`.

    TaylorVector(T, N::Integer, n::Integer)

Create a vector of `n` `NTuple{N,T}`s.
"""
struct TaylorVector{N,T} <: AbstractVector{NTuple{N,T}}
    data::LinearAlgebra.Transpose{T,Matrix{T}}
    views::NTuple{N,SubArray{T,1,Matrix{T},Tuple{Int,UnitRange{Int}},true}}
end
TaylorVector{N}(TV::TaylorVector{M,T}) where {N,M,T} =
    TaylorVector{N,T}(TV.data, TV.views[1:N])
function TaylorVector{N}(data::Matrix) where {N}
    views = tuple((view(data, i, 1:size(data, 2)) for i = 1:N)...)
    TaylorVector(LinearAlgebra.transpose(data), views)
end
function TaylorVector{N}(T, n::Integer) where {N}
    TaylorVector{N}(zeros(T, N, n))
end

function Base.show(io::IO, ::MIME"text/plain", X::TaylorVector)
    summary(io, X)
    isempty(X) && return
    println(io, ":")
    Base.print_array(io, map(i -> X[i], 1:length(X)))
end

Base.length(TV::TaylorVector) = size(TV.data, 1)
Base.size(TV::TaylorVector) = (length(TV),)
Base.eltype(::Type{TaylorVector{N,T}}) where {N,T} = NTuple{N,T}
Base.IndexStyle(::TaylorVector) = Base.IndexLinear()

"""
    vectors(TV::TaylorVec{N})

Return the Taylor series as `N` seperate vectors.
"""
vectors(TV::TaylorVector) = TV.views

@generated function Base.getindex(TV::TaylorVector{N}, i::Integer) where {N}
    quote
        Base.@_propagate_inbounds_meta
        x = TV.data
        $(Expr(:tuple, (:(x[i, $k]) for k = 1:N)...))
    end
end
Base.getindex(TV::TaylorVector, i::Integer, j::Integer) = getindex(TV.data, i, j)

function Base.setindex!(TV::TaylorVector{N,T}, x, i::Integer) where {N,T}
    setindex!(TV, convert(NTuple{N,T}, x), i)
end
function Base.setindex!(TV::TaylorVector{1}, x::Number, i::Integer)
    TV.data[i] = x
    x
end
@generated function Base.setindex!(
    TV::TaylorVector{N,T},
    x::NTuple{N,T},
    i::Integer,
) where {N,T}
    quote
        Base.@_propagate_inbounds_meta
        d = TV.data
        $(Expr(:tuple, (:(d[i, $k]) for k = 1:N)...)) = x
        x
    end
end
Base.setindex!(TV::TaylorVector, x, i::Integer, j::Integer) = setindex!(TV.data, x, i, j)

function untuple(varname, tuplename, M)
    :($(Expr(:tuple, varname, (Symbol(varname, k) for k = 1:M-1)...)) = $tuplename)
end

function _impl_taylor_bivariate(K, M, N, op; inline = true)
    list = ModelKit.InstructionList()
    id = push!(list, (op, :x, :y))
    diff_map = ModelKit.DiffMap()
    for i = 1:M-1
        diff_map[:x, i] = Symbol(:x, i)
    end
    for j = 1:N-1
        diff_map[:y, j] = Symbol(:y, j)
    end
    dlist = ModelKit.univariate_diff!(list, K, diff_map)
    quote
        $(inline ? :(Base.@_inline_meta) : :())
        $(untuple(:x, :tx, M))
        $(untuple(:y, :ty, N))
        $(ModelKit.to_expr(dlist))
        $(Expr(:tuple, id, ((d = diff_map[id, k];
        d === nothing ? :(zero(x)) : d) for k = 1:K)...))
    end
end

@generated function taylor(
    ::Type{Val{op}},
    ::Type{Val{K}},
    tx::NTuple{M},
    ty::NTuple{N},
) where {op,K,M,N}
    _impl_taylor_bivariate(K, M, N, op)
end

make_ntuple(t) = t
make_ntuple(t::Number) = (t,)
make_ntuple(t::Tuple{A,B}) where {A,B} = convert(NTuple{2,promote_type(A, B)}, t)
make_ntuple(t::Tuple{A,B,C}) where {A,B,C} = convert(NTuple{3,promote_type(A, B, C)}, t)
make_ntuple(t::Tuple{A,B,C,D}) where {A,B,C,D} =
    convert(NTuple{4,promote_type(A, B, C, D)}, t)
make_ntuple(t::Tuple{A,B,C,D,E}) where {A,B,C,D,E} =
    convert(NTuple{5,promote_type(A, B, C, D, E)}, t)

taylor(op::Type{T1}, v::Type{T2}, tx, ty) where {T1,T2} =
    taylor(op::Type{T1}, v::Type{T2}, make_ntuple(tx), make_ntuple(ty))
function taylor(op::Type{T1}, v::Type{T2}, tx) where {T1,T2}
    taylor(op::Type{T1}, v::Type{T2}, make_ntuple(tx))
end

function _impl_taylor_pow(K, M)
    list = ModelKit.InstructionList()
    id = push!(list, (:^, :x, :r))
    diff_map = ModelKit.DiffMap()
    for i = 1:M-1
        diff_map[:x, i] = Symbol(:x, i)
    end
    dlist = ModelKit.univariate_diff!(list, K, diff_map)
    quote
        $(untuple(:x, :tx, M))
        $(ModelKit.to_expr(dlist))
        $(Expr(:tuple, id, ((d = diff_map[id, k];
        d === nothing ? :(zero(x)) : d) for k = 1:K)...))
    end
end

@generated function taylor(
    ::Type{Val{:^}},
    ::Type{Val{K}},
    tx::NTuple{M},
    r::Integer,
) where {K,M}
    _impl_taylor_pow(K, M)
end
taylor(op::Type{Val{:^}}, v::Type{Val{K}}, tx, r::Integer) where {K} =
    taylor(op, v, make_ntuple(tx), r)

function _impl_taylor_sqr(K, M)
    list = ModelKit.InstructionList()
    id = push!(list, (:^, :x, 2))
    diff_map = ModelKit.DiffMap()
    for i = 1:M-1
        diff_map[:x, i] = Symbol(:x, i)
    end
    dlist = ModelKit.univariate_diff!(list, K, diff_map)
    quote
        $(untuple(:x, :tx, M))
        $(ModelKit.to_expr(dlist))
        $(Expr(:tuple, id, ((d = diff_map[id, k];
        d === nothing ? :(zero(x)) : d) for k = 1:K)...))
    end
end


@generated function taylor(::Type{Val{:sqr}}, ::Type{Val{K}}, tx::NTuple{M}) where {K,M}
    _impl_taylor_sqr(K, M)
end
