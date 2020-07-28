#=

Predictor

## Introduction

In an ideal world the predictor only uses an order (2,1) Padé approximant to generate
an initial guess for the Newton corrector.
The choice of Padé approximants as predictors is motived by the results in [1].
In particular, the ability to provide a trust region of the prediction method to
significantly decrease the likelihood of path jumping.
additionally, sice Padé approximants are rational functions, these perform very well
for diverging paths and close to singularities.

Now, for an Padé approximant of order (2,1) the first 3 derivatives are needed.
The most robust way to achieve this it to use automatic differentiation.
However, due to significant compile times, we also have to consider the case
that the Padé approximants are computed by numerical differentiation using the scheme
derived in [2]. The numerical differentiation scheme only uses O(h^2) approximations
since we are mostly constrained by the available precision.

However, the smaller the trust region is (i.e. the closer we are to a singularity) the
smaller we need to take the discretization in the numerical differentiation.
Thus, the error in the numerical differentiation is inversely corrolated to the trust
region.
Additionally, the third derivative needs to be sufficiently accurate. Otherwise
the predictor doesn't generate sufficiently accurate intial guesses resulting
in very small step sizes.
Therefore it can necessary to drop down to an Padé approximant of order (1,1).
Usually the arithmetic is sufficient to obtain for the second derivative a sufficiently
accurate approximation.

If also the second derivative is not available we use cubic hermite interpolation.
For diverging paths this works much much better than (explicit) Runge-Kutta methods.
The reason that we do not use expl. RK methods is that, then when numerical differentiation
has to give up, we are in a regime where the ODE of the path is *stiff*. Thus, expl.
RK just don't yield satisfying results.

## Endgame and coordinates

Additionally, the predictor computes in s-coordinates, that is for the path x(t)
the predictor internally computes with y(s) = x(s^m) where m is the *winding number*.
If m = 1 then this is the same as computing with x(t). However, during the endgame
`m` is possibly set.
All APIs expect input in t-coordinates and return results in t-coordinates.


[1]: A Robust Numerical Path Tracking Algorithm for Polynomial Homotopy Continuation,
 Telen, Van Barel, Verschelde https://arxiv.org/abs/1909.04984

[2]:Mackens, Wolfgang. "Numerical differentiation of implicitly defined space curves."
   Computing 41.3 (1989): 237-260.
=#

module PredictionMethod
@enum methods begin
    Pade21
    Hermite
end
end
using .PredictionMethod: PredictionMethod

"""
    Predict(H::AbstractHomotopy)

Construct a predictor. The predicor uses a Padé approximant of order (2,1) to produce
an initial guess for the Newton corrector.
For this, the predictor tries to compute the local derivatives up to order 3.
This is done using automatic differentiation for the derivatives up to order `3`.
Otherwise numerical differentiation is used.
"""
Base.@kwdef mutable struct Predictor
    method::PredictionMethod.methods = PredictionMethod.Pade21
    order::Int = 4
    use_hermite::Bool = true
    trust_region::Float64 = Inf
    local_error::Float64 = Inf
    cond_H_ẋ::Float64 = Inf

    tx⁰::TaylorVector{1,ComplexF64}
    tx¹::TaylorVector{2,ComplexF64}
    tx²::TaylorVector{3,ComplexF64}
    tx³::TaylorVector{4,ComplexF64}
    t::ComplexF64 = complex(NaN)
    tx_norm::Vector{Float64} = zeros(4)

    # finite diff
    xtemp::Vector{ComplexF64}
    u::Vector{ComplexF64}
    u₁::Vector{ComplexF64}
    u₂::Vector{ComplexF64}

    # for interpolation
    prev_tx¹::TaylorVector{2,ComplexF64}
    prev_t::ComplexF64 = complex(NaN)

    # endgame predictor
    winding_number::Int = 1
    s::ComplexF64 = complex(NaN)
    prev_s::ComplexF64 = complex(NaN)
    ty¹::TaylorVector{2,ComplexF64}
    prev_ty¹::TaylorVector{2,ComplexF64}
end


function Predictor(H::AbstractHomotopy)
    m, n = size(H)
    tx³ = TaylorVector{4}(ComplexF64, n)
    Predictor(
        tx⁰ = TaylorVector{1}(tx³),
        tx¹ = TaylorVector{2}(tx³),
        tx² = TaylorVector{3}(tx³),
        tx³ = tx³,
        xtemp = zeros(ComplexF64, n),
        u = zeros(ComplexF64, m),
        u₁ = zeros(ComplexF64, m),
        u₂ = zeros(ComplexF64, m),
        prev_tx¹ = TaylorVector{2}(ComplexF64, n),
        ty¹ = TaylorVector{2}(ComplexF64, n),
        prev_ty¹ = TaylorVector{2}(ComplexF64, n),
    )
end

function init!(predictor::Predictor)
    predictor.cond_H_ẋ = 1.0
    predictor.winding_number = 1

    predictor.t = predictor.prev_t = NaN
    predictor.s = predictor.prev_s = NaN
    predictor.trust_region = predictor.local_error = NaN
    predictor.use_hermite = true
    predictor
end

order(predictor::Predictor) = predictor.order
local_error(predictor::Predictor) = predictor.local_error
trust_region(predictor::Predictor) = predictor.trust_region

function winding_number!(P::Predictor, m)
    P.winding_number = m
    P
end

"""
    update!(
        P::Predictor,
        H::AbstractHomotopy,
        x,
        t,
        jacobian::Jacobian,
        norm::AbstractNorm,
        x̂ = nothing,
    )

Upadte the predictor with the new solution `(x,t)` of `H(x,t) = 0`.
This computes new local derivatives and chooses an appriopriate prediction method.
"""
function update!(
    predictor::Predictor,
    H::AbstractHomotopy,
    x,
    t,
    J::Jacobian,
    norm::AbstractNorm,
    x̂ = nothing, # the predicted value
)
    # The general strategy is as follows:
    #
    # If m = winding_number > 1 then we only use a hermite predictor.
    #
    # Otherwise we use a Padé predictor of order (2,1). For this we need the first three
    # derivativatives of the path x(t).
    # We obtain accurate derivatives by using automatic differentiation (AD) with the taylor!
    # api.

    @unpack u, tx⁰, tx¹, tx², tx³, ty¹, xtemp, tx_norm = predictor

    x⁰, x¹, x², x³ = vectors(tx³)
    y⁰, y¹ = vectors(ty¹)

    m = predictor.winding_number
    @inbounds for i = 1:length(xtemp)
        predictor.prev_tx¹[i, 1] = predictor.tx¹[i, 1]
        predictor.prev_tx¹[i, 2] = predictor.tx¹[i, 2]
    end

    predictor.prev_t, predictor.t = predictor.t, t

    if m > 1
        predictor.prev_s = predictor.s
        predictor.s = t_to_s_plane(t, m)
    end

    # we use the accuracy of the previous prediction for the local error estimate
    if isnothing(x̂)
        predictor.local_error = NaN
    else
        Δs = fast_abs(t - predictor.prev_t)
        predictor.local_error = norm(x̂, x) / Δs^predictor.order
    end

    x⁰ .= x
    tx_norm[1] = norm(x)
    if m > 1
        y⁰ .= x
    end

    # Compute ẋ always using AD
    taylor!(u, Val(1), H, x, t, true)
    # @show u[1:2]
    u .= .-u
    LA.ldiv!(xtemp, J, u)
    # Check error made in the linear algebra
    δ = fixed_precision_iterative_refinement!(xtemp, workspace(J), u, norm)
    predictor.cond_H_ẋ = cond_H_ẋ = δ / eps()
    # @show cond_H_ẋ
    tol_δ₁ = 1e-10

    δ > tol_δ₁ && iterative_refinement!(xtemp, J, u, InfNorm(); tol = tol_δ₁, max_iters = 5)
    tx_norm[2] = norm(xtemp)

    x¹ .= xtemp
    if m > 1
        if m == 2
            μ = 2 * predictor.s
        else
            μ = m * predictor.s^(m - 1)
        end
        y¹ .= μ .* xtemp

        predictor.method = PredictionMethod.Hermite
        predictor.order = 4
        predictor.trust_region = tx_norm[1] / tx_norm[2]
        if isnan(predictor.local_error)
            predictor.local_error = (tx_norm[2] / tx_norm[1])^3
        end
        return predictor
    end

    taylor!(u, Val(2), H, tx¹, t, true)
    u .= .-u
    LA.ldiv!(xtemp, J, u)
    tol_δ₂ = 1e-10
    δ > tol_δ₂ && iterative_refinement!(xtemp, J, u, norm; tol = tol_δ₂, max_iters = 4)
    tx_norm[3] = norm(xtemp)
    x² .= xtemp

    taylor!(u, Val(3), H, tx², t, true)
    u .= .-u
    LA.ldiv!(xtemp, J, u)
    tol_δ₃ = 1e-4
    δ > tol_δ₃ && iterative_refinement!(xtemp, J, u, norm; tol = tol_δ₃, max_iters = 3)
    tx_norm[4] = norm(xtemp)
    x³ .= xtemp

    τ = Inf
    for (i, (x, x¹, x², x³)) in enumerate(tx³)
        c, c¹, c², c³ = fast_abs.((x, x¹, x², x³))
        λ = max(1e-6, c¹)
        c¹ /= λ
        c² /= λ^2
        c³ /= λ^3
        tol = 1e-14 * max(c¹, c², c³)
        if !((c¹ ≤ tol && c² ≤ tol && c³ ≤ tol) || c² ≤ tol)
            τᵢ = (c² / c³) / λ
            if τᵢ < τ
                τ = τᵢ
            end
        end
    end
    if !isfinite(τ)
        τ = tx_norm[3] / tx_norm[4]
    end
    if !isfinite(τ)
        τ = tx_norm[1] / maximum(tx_norm)
    end

    predictor.method = PredictionMethod.Pade21
    predictor.order = 4
    predictor.trust_region = τ
    if isnan(predictor.local_error)
        predictor.local_error = ((inv(τ))^2)^2
    end

    predictor
end

predict!(x̂, pred::Predictor, H, x, t, Δt) = predict!(x̂, pred, pred.method, H, x, t, Δt)
function predict!(x̂, pred::Predictor, method::PredictionMethod.methods, H, x, t, Δt)
    m = pred.winding_number

    if method == PredictionMethod.Pade21
        λ = pred.trust_region
        λ² = λ^2
        λ³ = λ^3
        tol = 1e-12
        for (i, (x, x¹, x², x³)) in enumerate(pred.tx³)
            c, c¹, c², c³ = fast_abs.((x, x¹, x², x³))
            # check if only taylor series is used
            τ = tol * √(c^2 + (c¹ * λ)^2 + (c² * λ²)^2 + (c³ * λ³)^2)
            if c³ * λ³ ≤ τ || c² * λ² ≤ τ
                x̂[i] = x + Δt * (x¹ + Δt * x²)
            else
                δᵢ = 1 - Δt * x³ / x²
                x̂[i] = x + Δt * (x¹ + Δt * x² / δᵢ)
            end
        end
    elseif method == PredictionMethod.Hermite
        prev_s = t_to_s_plane(pred.prev_t, m)
        s = t_to_s_plane(t, m)
        s′ = t_to_s_plane(t + Δt, m)
        if m == 2
            prev_sm = 2 * prev_s
            sm = 2 * s
        else
            prev_sm = m * prev_s^(m - 1)
            sm = m * s^(m - 1)
        end
        for i = 1:length(pred.prev_tx¹)
            pred.prev_ty¹[i, 1] = pred.prev_tx¹[i, 1]
            pred.prev_ty¹[i, 2] = prev_sm * pred.prev_tx¹[i, 2]
        end
        for i = 1:length(pred.prev_tx¹)
            pred.ty¹[i, 1] = pred.tx¹[i, 1]
            pred.ty¹[i, 2] = sm * pred.tx¹[i, 2]
        end
        cubic_hermite!(x̂, pred.prev_ty¹, prev_s, pred.ty¹, s, s′)
    end

    x̂
end

"""
    t_to_s_plane(t::Complex, m::Int; branch::Int = 0)

Convert ``t = r * \\exp(√(-1)θ)`` to ``t = r^{1/m} * \\exp(√(-1)θ/m)`` where for
``r^{1/m}`` the positive real root is choosen.
The `branch` keyword allows to choose a different branch cut.
"""
function t_to_s_plane(t::Complex, m::Int; branch::Int = 0)
    r = fast_abs(t)
    if isreal(t) && real(t) > 0
        if branch > 0
            complex(nthroot(r, m), 2branch * π / m)
        else
            complex(nthroot(r, m))
        end
    else
        # angle returns a value in [-π,π], but we want [0,π]
        θ = mod2pi(angle(t)) + branch * 2π
        return nthroot(r, m) * cis(θ / m)
    end
end


function cubic_hermite!(x̂, tx¹₀, t₀, tx¹₁, t₁, t)
    if isreal(t₀) && isreal(t₁) && isreal(t)
        _cubic_hermite!(x̂, tx¹₀, real(t₀), tx¹₁, real(t₁), real(t))
    else
        _cubic_hermite!(x̂, tx¹₀, t₀, tx¹₁, t₁, t)
    end
end
@inline function _cubic_hermite!(x̂, tx¹₀, t₀, tx¹₁, t₁, t)
    s = (t - t₀) / (t₁ - t₀)
    h₀₀ = (1 + 2s) * (1 - s)^2
    h₁₀ = (t - t₀) * (1 - s)^2
    h₀₁ = s^2 * (3 - 2s)
    h₁₁ = (t - t₀) * s * (s - 1)
    @inbounds for i in eachindex(x̂)
        x̂[i] = h₀₀ * tx¹₀[i, 1] + h₁₀ * tx¹₀[i, 2] + h₀₁ * tx¹₁[i, 1] + h₁₁ * tx¹₁[i, 2]
    end
    x̂
end
