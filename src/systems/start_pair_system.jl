"""
    StartPairSystem(F)

A system which consider the system `F(x, p)` as a system `F([x p])`, i.e., variable and
parameter space are considered together.
"""
struct StartPairSystem{S<:AbstractSystem,C<:FiniteDiff.JacobianCache} <: AbstractSystem
    F::S
    # it's a little bit unnecessary to have these here, since views are theoretically
    # sufficient. However, we want to avoid any unnecessary recompilation
    # of the potential expensive evaluate! and evaluate_and_jacobian! functions.
    # So we give exactly the same types as in the tracking
    x::Vector{ComplexF64}
    p::Vector{ComplexF64}
    J::Matrix{ComplexF64}
    fd_cache::C
end

StartPairSystem(F::System; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    StartPairSystem(fixed(F; compile = compile))
function StartPairSystem(F::AbstractSystem)
    x = zeros(ComplexF64, size(F, 2))
    p = zeros(ComplexF64, nparameters(F))
    J = zeros(ComplexF64, size(F))
    fd_cache = FiniteDiff.JacobianCache(similar(p), zeros(ComplexF64, size(F, 1)))
    StartPairSystem(F, x, p, J, fd_cache)
end
Base.size(F::StartPairSystem) = (size(F.F, 1), size(F.F, 2) + nparameters(F.F))

function ModelKit.evaluate!(u, F::StartPairSystem, xp, ::Nothing = nothing)
    N, m = length(F.x), length(F.p)
    F.x .= view(xp, 1:N)
    F.p .= view(xp, N+1:N+m)
    evaluate!(u, F.F, F.x, F.p)
end


function ModelKit.evaluate_and_jacobian!(u, U, F::StartPairSystem, xp, ::Nothing = nothing)
    n, N, m = size(F.F, 1), length(F.x), length(F.p)
    F.x .= view(xp, 1:N)
    F.p .= view(xp, N+1:N+m)

    evaluate_and_jacobian!(u, F.J, F.F, F.x, F.p)
    view(U, :, 1:N) .= F.J

    fx = (v, q) -> evaluate!(v, F.F, F.x, q)
    FiniteDiff.finite_difference_jacobian!(view(U, :, N+1:N+m), fx, F.p, F.fd_cache)
end
