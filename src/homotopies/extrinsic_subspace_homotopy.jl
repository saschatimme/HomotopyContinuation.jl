export ExtrinsicSubspaceHomotopy

"""
    ExtrinsicSubspaceHomotopy
"""
mutable struct ExtrinsicSubspaceHomotopy{T<:AbstractSystem} <: AbstractHomotopy
    F::T
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    #cache
    A::Matrix{ComplexF64}
    b::Vector{ComplexF64}
    Ȧ::Matrix{ComplexF64}
    ḃ::Vector{ComplexF64}
    ū::Vector{ComplexDF64}
    t_cache::Base.RefValue{ComplexF64}
end

function ExtrinsicSubspaceHomotopy(
    F::System,
    start,
    target;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
)
    ExtrinsicSubspaceHomotopy(fixed(F; compile = compile), start, target)
end

function ExtrinsicSubspaceHomotopy(
    F::AbstractSystem,
    start::LinearSubspace,
    target::LinearSubspace;
    kwargs...,
)
    ExtrinsicSubspaceHomotopy(
        F,
        convert(LinearSubspace{ComplexF64}, start),
        convert(LinearSubspace{ComplexF64}, target);
        kwargs...,
    )
end

function ExtrinsicSubspaceHomotopy(
    F::AbstractSystem,
    start::LinearSubspace{ComplexF64},
    target::LinearSubspace{ComplexF64};
    kwargs...,
)
    A = copy(extrinsic(start).A)
    b = copy(extrinsic(start).b)

    Ȧ = extrinsic(start).A - extrinsic(target).A
    ḃ = extrinsic(start).b - extrinsic(target).b
    ū = zeros(ComplexDF64, size(A, 1))
    ExtrinsicSubspaceHomotopy(F, start, target, A, b, Ȧ, ḃ, ū, Ref(complex(NaN)))
end

Base.size(H::ExtrinsicSubspaceHomotopy) = (first(size(H.F)) + size(H.A, 1), last(size(H.F)))

start_parameters!(H::ExtrinsicSubspaceHomotopy, start) = parameters!(H, start, H.target)
target_parameters!(H::ExtrinsicSubspaceHomotopy, target) = parameters!(H, H.start, target)
parameters!(H::ExtrinsicSubspaceHomotopy, p, q) = set_subspaces!(H, p, q)
function set_subspaces!(
    H::ExtrinsicSubspaceHomotopy,
    start::LinearSubspace,
    target::LinearSubspace,
)
    H.start = start
    H.target = target
    H.Ȧ .= extrinsic(start).A .- extrinsic(target).A
    H.ḃ .= extrinsic(start).b .- extrinsic(target).b
    H.t_cache[] = NaN
    H
end

function tAb!(H::ExtrinsicSubspaceHomotopy, t)
    t == H.t_cache[] && return nothing

    A₁ = extrinsic(H.start).A
    A₀ = extrinsic(H.target).A
    b₁ = extrinsic(H.start).b
    b₀ = extrinsic(H.target).b
    if imag(t) == 0
        let t = real(t), t0 = (1.0 - real(t))
            @inbounds for i in eachindex(H.A)
                H.A[i] = t * A₁[i] + t0 * A₀[i]
            end
            @inbounds for i in eachindex(H.b)
                H.b[i] = t * b₁[i] + t0 * b₀[i]
            end
        end
    else
        t0 = (1.0 - t)
        @inbounds for i in eachindex(H.A)
            H.A[i] = t * A₁[i] + t0 * A₀[i]
        end
        @inbounds for i in eachindex(H.b)
            H.b[i] = t * b₁[i] + t0 * b₀[i]
        end
    end
    H.t_cache[] = t

    nothing
end


function ModelKit.evaluate!(u, H::ExtrinsicSubspaceHomotopy, x, t)
    tAb!(H, t)
    evaluate!(u, H.F, x)
    m = first(size(H.F))
    for i = 1:length(H.b)
        u[m+i] = -H.b[i]
    end
    for j = 1:size(H.A, 2)
        xj = x[j]
        for i = 1:size(H.A, 1)
            u[m+i] = muladd(H.A[i, j], xj, u[m+i])
        end
    end
    u
end
function ModelKit.evaluate!(u, H::ExtrinsicSubspaceHomotopy, x::Vector{ComplexDF64}, t)
    tAb!(H, t)
    evaluate!(u, H.F, x)
    m = first(size(H.F))
    for i = 1:length(H.b)
        H.ū[i] = -H.b[i]
    end
    for j = 1:size(H.A, 2)
        xj = x[j]
        for i = 1:size(H.A, 1)
            H.ū[i] = muladd(H.A[i, j], xj, H.ū[i])
        end
    end
    for i = 1:length(H.b)
        u[m+i] = H.ū[i]
    end

    u
end

function ModelKit.evaluate_and_jacobian!(u, U, H::ExtrinsicSubspaceHomotopy, x, t)
    tAb!(H, t)
    evaluate_and_jacobian!(u, U, H.F, x)
    m = first(size(H.F))
    for i = 1:length(H.b)
        u[m+i] = -H.b[i]
    end
    for j = 1:size(H.A, 2)
        xj = x[j]
        for i = 1:size(H.A, 1)
            u[m+i] = muladd(H.A[i, j], xj, u[m+i])
        end
    end
    for j = 1:size(H.A, 2), i = 1:size(H.A, 1)
        U[m+i, j] = H.A[i, j]
    end
    u
end

function ModelKit.taylor!(u, v::Val{1}, H::ExtrinsicSubspaceHomotopy, tx, t)
    m = first(size(H.F))
    for i = 1:m
        u[i] = zero(eltype(u))
    end
    for i = 1:length(H.b)
        u[m+i] = -H.ḃ[i]
    end
    for j = 1:size(H.Ȧ, 2)
        xⱼ, = tx[j]
        for i = 1:size(H.Ȧ, 1)
            u[m+i] = muladd(H.Ȧ[i, j], xⱼ, u[m+i])
        end
    end
    u
end

function ModelKit.taylor!(
    u,
    v::Val{K},
    H::ExtrinsicSubspaceHomotopy,
    tx::TaylorVector{K},
    t,
) where {K}
    m = first(size(H.F))
    taylor!(u, v, H.F, tx)
    for i = 1:length(H.b)
        u[m+i] = 0.0
    end
    for j = 1:size(H.Ȧ, 2)
        xⱼ = tx[j, K]
        for i = 1:size(H.Ȧ, 1)
            u[m+i] = muladd(H.Ȧ[i, j], xⱼ, u[m+i])
        end
    end
    u
end
