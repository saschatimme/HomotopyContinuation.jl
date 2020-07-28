export AffineChartSystem, on_affine_chart

"""
    AffineChartSystem(F::AbstractSystem, v::PVector{T,N})

Given a system ``F(x): (ℙ^{m_1} × ⋯ × ℙ^{m_N}) → ℂⁿ`` this creates a new affine
system ``F′`` which operates on the affine chart defined by the vector
``v ∈ ℙ^{m_1} × ⋯ × ℙ^{m_N}`` and the augmented conditions ``vᵀx = 1``.
"""
struct AffineChartSystem{S<:AbstractSystem,N} <: AbstractSystem
    system::S
    chart::PVector{ComplexF64,N}
end

ModelKit.variables(F::AffineChartSystem) = variables(F.system)
ModelKit.parameters(F::AffineChartSystem) = parameters(F.system)
ModelKit.variable_groups(F::AffineChartSystem) = variable_groups(F.system)

"""
    on_affine_chart(F::Union{System,AbstractSystem}, dimensions)

Construct an `AffineChartSystem` on a randomly generated chart `v`. Each entry is drawn
idepdently from a univariate normal distribution.
"""
on_affine_chart(
    F::System,
    dims = nothing;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
) = on_affine_chart(fixed(F; compile = compile), dims)
function on_affine_chart(
    F::AbstractSystem,
    dims = nothing,
    compile::Union{Bool,Symbol} = true,
)
    vargroups = variable_groups(F)
    if vargroups === nothing
        dims = [size(F, 2) - 1]
    else
        dims = length.(vargroups,) .- 1
    end
    chart = PVector(randn(ComplexF64, sum(dims) + length(dims)), tuple(dims...))
    AffineChartSystem(F, chart)
end

function Base.size(F::AffineChartSystem{<:AbstractSystem,N}) where {N}
    m, n = size(F.system)
    (m + N, n)
end

function set_solution!(x, F::AffineChartSystem, y)
    x .= y
    on_chart!(x, F.chart)
end
@inline function evaluate_chart!(u, v::PVector{<:Any,N}, x::AbstractVector) where {N}
    ranges = dimension_indices(v)
    for (k, range) in enumerate(ranges)
        out = zero(promote_type(eltype(u), eltype(x)))
        @inbounds for i in range
            out += v[i] * x[i]
        end
        u[k] = out - 1.0
    end
    nothing
end

@inline function jacobian_chart!(U, v::PVector{<:Any,N}, x::AbstractVector) where {N}
    ranges = dimension_indices(v)
    for j = 1:size(U, 2), i = 1:size(U, 1)
        U[i, j] = zero(eltype(U))
    end
    for (k, range) in enumerate(ranges)
        for j in range
            U[k, j] = v[j]
        end
    end
    nothing
end

function (F::AffineChartSystem{<:Any,N})(x::AbstractVector, p = nothing) where {N}
    v = PVector(x, dims(F.chart))
    u = F.system(x, p)
    append!(u, zeros(N))
    m = size(F.system, 1)
    evaluate_chart!(view(u, m+1:m+N), F.chart, v)
    u
end
function (F::AffineChartSystem{<:Any,N})(x, p = nothing) where {N}
    u = F.system(x, p)
    append!(u, zeros(N))
    m = size(F.system, 1)
    evaluate_chart!(view(u, m+1:m+N), F.chart, x)
    u
end

function ModelKit.evaluate!(u, F::AffineChartSystem{<:Any,N}, x, p = nothing) where {N}
    evaluate!(u, F.system, x, p)
    m = size(F.system, 1)
    evaluate_chart!(view(u, m+1:m+N), F.chart, x)
    u
end

function ModelKit.evaluate_and_jacobian!(
    u,
    U,
    F::AffineChartSystem{<:Any,N},
    x,
    p = nothing,
) where {N}
    evaluate_and_jacobian!(u, U, F.system, x, p)
    m = size(F.system, 1)
    evaluate_chart!(view(u, m+1:m+N), F.chart, x)
    jacobian_chart!(view(U, m+1:m+N, :), F.chart, x)
    nothing
end

function ModelKit.taylor!(u, v::Val, F::AffineChartSystem, tx, p = nothing)
    u .= zero(eltype(u))
    taylor!(u, v, F.system, tx, p)
    # affine chart part is always zero since it is an affine linear form
    u
end
