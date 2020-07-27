export CoefficientHomotopy

"""
    CoefficientHomotopy(
        F::Union{AbstractSystem,System};
        start_coefficients,
        target_coefficients,
    )

Construct the homotopy ``H(x, t) = ∑_{a ∈ Aᵢ} (c_a t + (1-t)d_a) x^a`` where ``c_a`` are
the start coefficients and ``d_a`` the target coefficients.
"""
struct CoefficientHomotopy{T<:AbstractSystem} <: AbstractHomotopy
    F::T
    start_coeffs::Vector{ComplexF64}
    target_coeffs::Vector{ComplexF64}
    #cache
    x::Vector{ComplexF64}
    t_cache::Base.RefValue{ComplexF64}
    t_taylor_cache::Base.RefValue{ComplexF64}
    coeffs::Vector{ComplexF64}
    dt_coeffs::Vector{ComplexF64}
    taylor_coeffs::TaylorVector{2,ComplexF64}
end

function CoefficientHomotopy(
    F;
    start_coefficients::AbstractVector,
    target_coefficients::AbstractVector,
)
    CoefficientHomotopy(F, start_coefficients, target_coefficients)
end
function CoefficientHomotopy(
    F::ModelKit.System,
    start_coeffs::AbstractVector,
    target_coeffs::AbstractVector;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
)
    @assert length(start_coeffs) == length(target_coeffs) == length(F.parameters)
    CoefficientHomotopy(fixed(F; compile = compile), start_coeffs, target_coeffs)
end
function CoefficientHomotopy(
    F::AbstractSystem,
    start_coeffs::AbstractVector,
    target_coeffs::AbstractVector,
)
    CoefficientHomotopy(
        F,
        Vector{ComplexF64}(start_coeffs),
        Vector{ComplexF64}(target_coeffs),
    )
end

function CoefficientHomotopy(
    F::AbstractSystem,
    start_coeffs::Vector{ComplexF64},
    target_coeffs::Vector{ComplexF64},
)
    m = length(target_coeffs)
    @assert length(start_coeffs) == m
    CoefficientHomotopy(
        F,
        start_coeffs,
        target_coeffs,
        zeros(ComplexF64, size(F, 2)),
        Ref(complex(NaN)),
        Ref(complex(NaN)),
        zeros(ComplexF64, m),
        start_coeffs - target_coeffs,
        TaylorVector{2}(ComplexF64, length(target_coeffs)),
    )
end

Base.size(H::CoefficientHomotopy) = size(H.F)

function start_parameters!(H::CoefficientHomotopy, p::AbstractVector)
    H.t_cache[] = NaN
    H.t_taylor_cache[] = NaN
    H.start_coeffs .= p
    H
end
function target_parameters!(H::CoefficientHomotopy, p::AbstractVector)
    H.t_cache[] = NaN
    H.t_taylor_cache[] = NaN
    H.target_coeffs .= p
    H
end

function coeffs!(H::CoefficientHomotopy, t)
    t == H.t_cache[] && return H.coeffs

    if isreal(t)
        s = real(t)
        s1 = 1.0 - s
        @inbounds for i = 1:length(H.start_coeffs)
            H.coeffs[i] = s * H.start_coeffs[i] + s1 * H.target_coeffs[i]
        end
    else
        t1 = 1.0 - t
        @inbounds for i = 1:length(H.start_coeffs)
            H.coeffs[i] = t * H.start_coeffs[i] + t1 * H.target_coeffs[i]
        end
    end
    H.t_cache[] = t

    H.coeffs
end

function ModelKit.evaluate!(u, H::CoefficientHomotopy, x, t)
    evaluate!(u, H.F, x, coeffs!(H, t))
end

function ModelKit.evaluate_and_jacobian!(u, U, H::CoefficientHomotopy, x, t)
    evaluate_and_jacobian!(u, U, H.F, x, coeffs!(H, t))
end

function ModelKit.taylor!(u, v::Val{1}, H::CoefficientHomotopy, x, t)
    evaluate!(u, H.F, x, H.dt_coeffs)
end


function taylor_coeffs!(H::CoefficientHomotopy, t::Union{ComplexF64,Float64})
    t == H.t_taylor_cache[] && return H.taylor_coeffs

    coeffs!(H, t)
    for i = 1:length(H.taylor_coeffs)
        H.taylor_coeffs[i, 1] = H.coeffs[i]
        H.taylor_coeffs[i, 2] = H.dt_coeffs[i]
    end
    H.t_taylor_cache[] = t

    H.taylor_coeffs
end


function ModelKit.taylor!(u, v::Val, H::CoefficientHomotopy, tx::TaylorVector, t)
    taylor!(u, v, H.F, tx, taylor_coeffs!(H, t))
end
