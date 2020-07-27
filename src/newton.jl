export newton, is_success, NewtonResult, NewtonCache

"""
    NewtonResult

Result returned by [`newton`](@ref).

## Fields

* `return_code::Symbol`: Can be `:success`, `:rejected` or `:max_iters`.
* `x::Vector{ComplexF64}`: The last value obtained.
* `accuracy::Float64`: Estimate of the absolute distance of `x` to a true zero.
* `iters::Int` Number of iterations performed.
* `contraction_ratio::Float64`: The value `|xᵢ - xᵢ₋₁| / |xᵢ₋₁ - xᵢ₋₂| `.
"""
struct NewtonResult
    return_code::Symbol
    x::Vector{ComplexF64}
    accuracy::Float64
    residual::Float64
    iters::Int
    contraction_ratio::Float64
end

Base.show(io::IO, ::MIME"application/prs.juno.inline", r::NewtonResult) = r
Base.show(io::IO, result::NewtonResult) = print_fieldnames(io, result)


"""
    is_success(R::NewtonResult)

Returns `true` if the [`newton`](@ref) was successfull.
"""
is_success(R::NewtonResult) = R.return_code == :success

"""
    solution(R::NewtonResult)

Return the solution stored in `R`.
"""
solution(R::NewtonResult) = R.x

"""
    NewtonCache(F::AbstractSystem)

Pre-allocates the necessary memory for [`newton`](@ref).
"""
struct NewtonCache{M}
    x::Vector{ComplexF64}
    Δx::Vector{ComplexF64}
    x_ext::Vector{ComplexDF64}
    J::M
    r::Vector{ComplexF64}
end
function NewtonCache(F::AbstractSystem)
    m, n = size(F)
    x = zeros(ComplexF64, n)
    Δx = copy(x)
    x_ext = zeros(ComplexDF64, n)
    if m ≥ n
        J = MatrixWorkspace(m, n)
    else
        J = zeros(ComplexF64, m, n)
    end
    r = zeros(ComplexF64, m)
    NewtonCache(x, Δx, x_ext, J, r)
end

"""
    newton(
        F::AbstractSystem,
        x₀::AbstractVector,
        p = nothing,
        norm::AbstractNorm = InfNorm(),
        cache::NewtonCache = NewtonCache(F);
        options...
    )

An implemenetation of a local Newton's method with various options to specify
convergence criteria.
Returns a [`NewtonResult`](@ref).
The computations are always performed in complex arithmetic with
double precision, i.e., using `Complex{Float64}`.
The optional `cache` argument pre-allocates the necessary memory.
This is useful if the method is called repeatedly.

### Options

* `atol::Float64 = 1e-8`: The method is declared converged if `norm(xᵢ₊₁ - xᵢ) < atol`.
* `rtol::Float64 = atol`: The method is declared converged if
  `norm(xᵢ₊₁ - xᵢ) < max(atol, rtol * norm(x₀)) `.
* `max_iters::Int = 20`: The maximal number of iterations.
* `extended_precision::Bool = false`: An optional use of extended precision for the evaluation
  of `F(x)`. This can increase the achievable accuracy.
* `contraction_factor::Float64 = 1.0`: The Newton updates have to satisfy
  ``|xᵢ₊₁ - xᵢ| < a^{2^{i-1}}|x₁ - x₀|`` for ``i ≥ 1`` where ``a`` is `contraction_factor`.
* `min_contraction_iters::Int = typemax(Int)`:  The minimal number of iterations the
  `contraction_factor` has to be satisfied. If after `min_contraction_iters` many
  iterations the contraction factor is not satisfied the step is accepted anyway.
* `max_abs_norm_first_update::Float64 = Inf`: The initial guess `x₀` is rejected if
  `norm(x₁ - x₀) >  max_abs_norm_first_update`
* `max_rel_norm_first_update::Float64 = max_abs_norm_first_update`: The initial guess `x₀`
  is rejected if `norm(x₁ - x₀) >  max_rel_norm_first_update * norm(x₀)`
"""
newton(f::System, args...; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...) =
    newton(fixed(f; compile = compile), args...; kwargs...)
function newton(
    F::AbstractSystem,
    x₀::AbstractVector,
    p = nothing,
    norm::AbstractNorm = InfNorm(),
    cache::NewtonCache = NewtonCache(F);
    extended_precision::Bool = false,
    atol::Float64 = 1e-8,
    rtol::Float64 = atol,
    max_iters::Int = 20,
    contraction_factor::Float64 = 1.0,
    min_contraction_iters::Int = typemax(Int),
    max_abs_norm_first_update::Float64 = Inf,
    max_rel_norm_first_update::Float64 = max_abs_norm_first_update,
)
    @unpack x, Δx, x_ext, J, r = cache
    x .= x₀
    a = contraction_factor
    norm_x = norm(x)
    norm_Δxᵢ = norm_Δxᵢ₋₁ = NaN
    res = NaN
    m, n = size(F)
    for i = 1:max_iters
        evaluate_and_jacobian!(r, matrix(J), F, x, p)
        if extended_precision
            x_ext .= x
            evaluate!(r, F, x_ext, p)
        end

        updated!(J)
        if m ≥ n
            LA.ldiv!(Δx, J, r)
        else
            LA.ldiv!(Δx, LA.qr!(J, Val(true)), r)
        end
        norm_Δxᵢ = norm(Δx)

        x .= x .- Δx

        if i > 1
            if norm_Δxᵢ < a * norm_Δxᵢ₋₁
                norm_Δxᵢ₋₁ = norm_Δxᵢ
                a *= a
            elseif i > min_contraction_iters || norm_Δxᵢ < max(atol, rtol * norm_x)
                return NewtonResult(
                    :success,
                    x,
                    norm_Δxᵢ,
                    norm(r),
                    i,
                    norm_Δxᵢ / norm_Δxᵢ₋₁,
                )
            else
                return NewtonResult(
                    :rejected,
                    x,
                    norm_Δxᵢ,
                    norm(r),
                    i,
                    norm_Δxᵢ / norm_Δxᵢ₋₁,
                )
            end
        elseif i == 1 && (
            norm_Δxᵢ > max_rel_norm_first_update * norm_x ||
            norm_Δxᵢ > max_abs_norm_first_update
        )
            return NewtonResult(:rejected, x, norm(r), norm_Δxᵢ, i, NaN)
        else
            norm_Δxᵢ₋₁ = norm_Δxᵢ
        end
    end
    return NewtonResult(:max_iters, x, norm_Δxᵢ, norm(r), max_iters, norm_Δxᵢ / norm_Δxᵢ₋₁)
end
