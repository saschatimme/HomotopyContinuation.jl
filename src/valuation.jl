"""
    Valuation

A data structure for approximating the valuation of a solution path ``x(t)``.
"""
mutable struct Valuation
    val_x::Vector{Float64}
    val_tẋ::Vector{Float64}
    Δval_x::Vector{Float64}
    Δval_tẋ::Vector{Float64}

    # Data for computing the val_x and val_tẋ and ν̈ by finite differences
    val_x_data::NTuple{2,Vector{Float64}}
    val_ẋ_data::NTuple{2,Vector{Float64}}
    logx_data::NTuple{2,Vector{Float64}}
    logẋ_data::NTuple{2,Vector{Float64}}
    logt_data::NTuple{2,Float64}
end

Valuation(x::AbstractVector) = Valuation(length(x))
Valuation(n::Integer) =
    Valuation((zeros(n) for i = 1:4)..., ((zeros(n), zeros(n)) for i = 1:4)..., (NaN, NaN))

function Base.show(io::IO, val::Valuation)
    print(io, "Valuation :")
    for field in [:val_x, :val_tẋ, :Δval_x, :Δval_tẋ]
        vs = [Printf.@sprintf("%#.4g", v) for v in getfield(val, field)]
        print(io, "\n • ", field, " → ", "[", join(vs, ", "), "]")
    end
end
Base.show(io::IO, ::MIME"application/prs.juno.inline", v::Valuation) = v

"""
    init!(val::Valuation)

Initialize the valuation estimator to start from scratch.
"""
function init!(val::Valuation)
    val.val_x .= 0.0
    val.val_tẋ .= 0.0
    val.Δval_x .= 0.0
    val.Δval_tẋ .= 0.0
    val.val_x_data[1] .= val.val_x_data[2] .= 0.0
    val.val_ẋ_data[1] .= val.val_ẋ_data[2] .= 0.0
    val.logx_data[1] .= val.logx_data[2] .= 0.0
    val.logẋ_data[1] .= val.logẋ_data[2] .= 0.0
    val.logt_data = (NaN, NaN)
    val
end


function ν(x, ẋ, t)
    u, v = reim(x)
    u¹, v¹ = reim(ẋ)
    xx = abs2(x)

    μ = u * u¹ + v * v¹
    l = μ / xx

    t * l
end

function ν_ν¹(x, ẋ, x², t)
    u, v = reim(x)
    u¹, v¹ = reim(ẋ)
    u², v² = reim(x²)

    xx = abs2(x)

    μ = u * u¹ + v * v¹
    l = μ / xx

    μ¹ = u * u² + u¹^2 + v * v² + v¹^2
    l¹ = μ¹ / xx - 2 * l^2

    t * l, t * l¹ + l
end


logabs(x) = log(fast_abs(x))

function update!(val::Valuation, pred::Predictor, t::Real)
    logt = log(t)
    ν₂, ν₁ = val.val_x_data
    logx₂, logx₁, = val.logx_data
    logẋ₂, logẋ₁, = val.logẋ_data
    logt₂, logt₁ = val.logt_data
    val_ẋ₂, val_ẋ₁ = val.val_ẋ_data

    # We use finite difference when the predictor is in hermite mode since then we only
    # have ẋ available
    diff = pred.winding_number > 1 && !isnan(logt₂)
    for (i, (xᵢ, ẋᵢ, x²ᵢ, x³ᵢ)) in enumerate(pred.tx³)
        logxᵢ = logabs(xᵢ)
        logẋᵢ = logabs(ẋᵢ)

        if diff
            νᵢ = ν(xᵢ, ẋᵢ, t)
            Δνᵢ = finite_diff(νᵢ, logt, ν₂[i], logt₂, ν₁[i], logt₁)
            val.val_x[i] = νᵢ
            val.Δval_x[i] = Δνᵢ

            val_ẋᵢ = finite_diff(logẋᵢ, logt, logẋ₂[i], logt₂, logẋ₁[i], logt₁)
            Δval_ẋᵢ = finite_diff(val_ẋᵢ, logt, val_ẋ₂[i], logt₂, val_ẋ₁[i], logt₁)
            val.val_tẋ[i] = val_ẋᵢ + 1
            val.Δval_tẋ[i] = Δval_ẋᵢ
        else
            νᵢ, ν¹ = ν_ν¹(xᵢ, ẋᵢ, 2x²ᵢ, t)
            val.val_x[i] = νᵢ
            val.Δval_x[i] = t * ν¹
            ν₂[i], ν₁[i] = νᵢ, ν₂[i]

            val_ẋᵢ, ν¹ = ν_ν¹(ẋᵢ, 2x²ᵢ, 6x³ᵢ, t)
            val.val_tẋ[i] = val_ẋᵢ + 1
            val.Δval_tẋ[i] = t * ν¹
        end
        logx₂[i], logx₁[i] = logxᵢ, logx₂[i]
        logẋ₂[i], logẋ₁[i] = logẋᵢ, logẋ₂[i]
    end

    val.logt_data = (logt, logt₂)

    val
end

"""
    finite_diff(ν₃, s₃, ν₂, s₂, ν₁, s₁)

Compute the derivative `ν̇` of a function ``ν(s)`` at `s₃` by using a finite
difference scheme. This uses the values `νᵢ = ν(sᵢ)` for ``i=1,…,3``.
Since we have a non-uniform grid, we need a more elaborate difference scheme.
The implementation follows the formulas derived in [^BS05].

[^BS05]: Bowen, M. K., and Ronald Smith. "Derivative formulae and errors for non-uniformly
  spaced points." Proceedings of the Royal Society A: Mathematical, Physical and Engineering
  Sciences 461.2059 (2005): 1975-1997.
"""
function finite_diff(ν, s, ν₂, s₂, ν₁, s₁)
    Δ₁, Δ₂, Δ₁₂ = s - s₁, s - s₂, s₁ - s₂
    (Δ₂ * ν₁) / (Δ₁₂ * Δ₁) - ((Δ₁₂ + Δ₂) * ν₂) / (Δ₁₂ * Δ₂) - (Δ₁₂ * ν) / (Δ₁ * Δ₂)
end

function at_infinity_tol!(
    at_infinity_tols,
    val::Valuation;
    finite_tol::Float64,
    zero_is_finite::Bool,
)
    @unpack val_x, val_tẋ, Δval_x, Δval_tẋ = val

    for (i, val_xᵢ) in enumerate(val_x)
        ε∞ = max(
            abs(1.0 - val_tẋ[i] / val_xᵢ),
            abs(Δval_x[i] / val_xᵢ),
            abs(Δval_tẋ[i] / val_tẋ[i]),
        )
        if isnan(ε∞)
            at_infinity_tols[i] = Inf
            continue
        end

        # ∞ check
        if val_xᵢ + ε∞ < -finite_tol
            at_infinity_tols[i] = ε∞
            # 0 check
        elseif !zero_is_finite && val_xᵢ - ε∞ > -finite_tol
            at_infinity_tols[i] = ε∞
        else
            at_infinity_tols[i] = Inf
        end
    end
    at_infinity_tols
end

function is_finite(
    val::Valuation;
    finite_tol::Float64,
    zero_is_finite::Bool,
    max_winding_number::Int,
)
    @unpack val_x, val_tẋ, Δval_x, Δval_tẋ = val

    δ = inv(max_winding_number)
    for (i, val_xᵢ) in enumerate(val_x)
        # Case a: val(x) = 0
        if abs(val_xᵢ) < finite_tol
            if !(abs(Δval_x[i]) < finite_tol) || val_tẋ[i] < 0.5δ
                return false
            end
            # Case b: val(x) = val(tẋ) > 0
        elseif zero_is_finite && val_xᵢ > (δ - finite_tol)
            ε∞ = max(
                abs(1.0 - val_tẋ[i] / val_xᵢ),
                abs(Δval_x[i] / val_xᵢ),
                abs(Δval_tẋ[i] / val_tẋ[i]),
            )
            if !(ε∞ < finite_tol)
                return false
            end
        else
            return false
        end
    end
    return true
end

function estimate_winding_number(val::Valuation; max_winding_number::Int)
    m = 1
    min_err = Inf
    for k = 1:max_winding_number
        err = check_winding_number(val, k)
        if err < min_err
            m = k
            min_err = err
        end
    end
    m, min_err
end

function check_winding_number(val::Valuation, m::Int)
    err = 0.0
    for vᵢ in val.val_tẋ
        mvᵢ = m * vᵢ
        errᵢ = abs(round(mvᵢ) - mvᵢ)
        err = ifelse(errᵢ > err, errᵢ, err)
    end
    err
end
