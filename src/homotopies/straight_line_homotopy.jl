export StraightLineHomotopy

"""
    StraightLineHomotopy(G::System, F::System; gamma = 1.0)
    StraightLineHomotopy(G::AbstractSystem, F::AbstractSystem; gamma = 1.0)

Constructs the straight line homotopy ``H(x, t) = γ t G(x) + (1-t) F(x)`` where
``γ`` is `gamma`.
"""
struct StraightLineHomotopy{S<:AbstractSystem,T<:AbstractSystem} <: AbstractHomotopy
    start::S
    target::T
    γ::ComplexF64

    u::Vector{ComplexF64}
    ū::Vector{ComplexDF64}
    v̄::Vector{ComplexDF64}
    U::Matrix{ComplexF64}

    dv_start::LA.Transpose{ComplexF64,Matrix{ComplexF64}}
    dv_target::LA.Transpose{ComplexF64,Matrix{ComplexF64}}
end

function StraightLineHomotopy(
    start::System,
    target::System;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
)
    StraightLineHomotopy(
        fixed(start; compile = compile),
        fixed(target; compile = compile);
        kwargs...,
    )
end
function StraightLineHomotopy(
    start::AbstractSystem,
    target::AbstractSystem;
    γ = 1.0,
    gamma = γ,
)
    size(start) == size(target) ||
        throw(ArgumentError("Start and target do not have the same size, got $(size(start)) and $(size(target))"))

    m, n = size(start)
    u = zeros(ComplexF64, m)
    ū = zeros(ComplexDF64, m)
    v̄ = zeros(ComplexDF64, m)
    U = zeros(ComplexF64, m, n)

    dv_start = LA.transpose(zeros(ComplexF64, 5, m))
    dv_target = LA.transpose(zeros(ComplexF64, 5, m))

    StraightLineHomotopy(
        start,
        target,
        ComplexF64(gamma),
        u,
        ū,
        v̄,
        U,
        dv_start,
        dv_target,
    )
end

Base.size(H::StraightLineHomotopy) = size(H.start)

function Base.show(io::IO, mime::MIME"text/plain", H::StraightLineHomotopy)
    println(io, typeof(H), ":")
    println(io, "γ: ", H.γ)
    println(io, "\nG: ")
    show(io, mime, H.start)
    println(io, "\n\nF:")
    show(io, mime, H.target)
end

function ModelKit.evaluate!(u, H::StraightLineHomotopy, x::Vector{ComplexDF64}, t) where {T}
    evaluate!(H.v̄, H.start, x)
    evaluate!(H.ū, H.target, x)
    ts, tt = H.γ .* t, 1 - t
    for i = 1:size(H, 1)
        @inbounds u[i] = ts * H.v̄[i] + tt * H.ū[i]
    end
end

function ModelKit.evaluate!(u, H::StraightLineHomotopy, x, t)
    evaluate!(u, H.start, x)
    evaluate!(H.u, H.target, x)
    ts, tt = H.γ .* t, 1 - t
    for i = 1:size(H, 1)
        @inbounds u[i] = ts * u[i] + tt * H.u[i]
    end
    u
end

function ModelKit.evaluate_and_jacobian!(
    u,
    U,
    H::StraightLineHomotopy,
    x::AbstractVector{T},
    t,
) where {T}
    evaluate_and_jacobian!(u, U, H.start, x)
    evaluate_and_jacobian!(H.u, H.U, H.target, x)
    ts, tt = H.γ .* t, 1 - t
    for i = 1:size(H, 1)
        @inbounds u[i] = ts * u[i] + tt * H.u[i]
    end
    for j = 1:size(H, 2), i = 1:size(H, 1)
        @inbounds U[i, j] = ts * U[i, j] + tt * H.U[i, j]
    end
    nothing
end
function ModelKit.taylor!(u, ::Val{1}, H::StraightLineHomotopy, x, t)
    evaluate!(u, H.start, x)
    evaluate!(H.u, H.target, x)
    for i = 1:size(H, 1)
        @inbounds u[i] = H.γ * u[i] - H.u[i]
    end
    u
end
function ModelKit.taylor!(
    u,
    v::Val{K},
    H::StraightLineHomotopy,
    tx::TaylorVector,
    t,
) where {K}
    taylor!(H.dv_start, v, H.start, tx)
    taylor!(H.dv_target, v, H.target, tx)

    for i = 1:size(H, 1)
        start = H.γ * (H.dv_start[i, K] + t * H.dv_start[i, K+1])
        target = (1 - t) * H.dv_target[i, K+1] - H.dv_target[i, K]
        u[i] = start + target
    end
    u
end
