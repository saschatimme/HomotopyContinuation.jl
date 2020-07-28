export RandomizedSystem, square_up

"""
    RandomizedSystem(F::Union{System,AbstractSystem}, k::Integer) <: AbstractSystem

Given a ``n × N`` system `F` with ``n > N`` this constructs the system
``\\mathfrak{R}(F; k)(x) = [I A]⋅F(x)`` where ``I`` is a ``k × k`` identity matrix and
``A`` is random complex ``k × n`` matrix. See Chapter 13.5 in [^SW05] for more details.

    RandomizedSystem(F::Union{System,AbstractSystem}, A::Matrix{ComplexF64})

Explicitly provide the used randomization matrix `A`.

[^SW05]: Sommese, A. J., & Wampler, C. W. (2005). The Numerical Solution of Systems of Polynomials Arising in Engineering and Science. World Scientific.
"""
struct RandomizedSystem{S<:AbstractSystem} <: AbstractSystem
    system::S
    A::Matrix{ComplexF64}
    u::Vector{ComplexF64}
    ū::Vector{ComplexDF64}
    U::Matrix{ComplexF64}
    taylor_U::Matrix{ComplexF64}
end

function RandomizedSystem(
    F::Union{AbstractSystem,System},
    k::Integer;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
)
    n, N = size(F)
    #TODO: Should the rows of [I A] be unitary?
    A = randn(ComplexF64, k, n - k)
    RandomizedSystem(F, A; compile = compile)
end
RandomizedSystem(
    F::System,
    A::Matrix{ComplexF64};
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
) = RandomizedSystem(fixed(F; compile = compile), A)
function RandomizedSystem(
    F::AbstractSystem,
    A::Matrix{ComplexF64};
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
)
    n, N = size(F)
    n > N || throw(ArgumentError("Then given system is not overdetermined."))
    u = zeros(ComplexF64, n)
    ū = zeros(ComplexDF64, n)
    U = zeros(ComplexF64, n, N)
    taylor_U = zeros(ComplexF64, n, 5)
    RandomizedSystem(F, A, u, ū, U, taylor_U)
end

function Base.show(io::IO, mime::MIME"text/plain", F::RandomizedSystem)
    println(io, typeof(F), ":")
    println(io, "A:")
    show(io, mime, F.A)
    println(io, "\n\nF:")
    show(io, mime, F.system)
end

Base.size(F::RandomizedSystem) = (size(F.A, 1), last(size(F.system)))
ModelKit.variables(F::RandomizedSystem) = variables(F.system)
ModelKit.parameters(F::RandomizedSystem) = parameters(F.system)
ModelKit.variable_groups(F::RandomizedSystem) = variable_groups(F.system)

"""
    square_up(F::Union{System, AbstractSystem})

Creates the [`RandomizedSystem`](@ref) ``\\mathfrak{R}(F(x); N)`` where ``N`` is the number
of variables of `F`.
"""
square_up(
    F::Union{AbstractSystem,System};
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
) = RandomizedSystem(F, last(size(F)); compile = compile)

function randomize!(u, A, v::AbstractVector, ncols = 1)
    n, m = size(A, 1), length(v)

    @inbounds for i = 1:n
        u[i] = v[i]
    end
    @inbounds for j = 1:(m-n)
        vnj = v[n+j]
        for i = 1:n
            u[i] += A[i, j] * vnj
        end
    end
    u
end
function randomize!(U, A, V::AbstractMatrix, ncols = size(A, 1))
    n, m = size(A, 1), size(V, 1)

    for j = 1:ncols, i = 1:n
        U[i, j] = V[i, j]
    end
    for j = 1:ncols, k = 1:(m-n), i = 1:n
        U[i, j] += A[i, k] * V[n+k, j]
    end
    U
end

(F::RandomizedSystem)(x, p = nothing) = [LA.I F.A] * F.system(x, p)

function ModelKit.evaluate!(u, F::RandomizedSystem, x::Vector{ComplexDF64}, p = nothing)
    evaluate!(F.ū, F.system, x, p)
    randomize!(u, F.A, F.ū)
    u
end
function ModelKit.evaluate!(u, F::RandomizedSystem, x, p = nothing)
    evaluate!(F.u, F.system, x, p)
    randomize!(u, F.A, F.u)
    u
end

function ModelKit.evaluate_and_jacobian!(u, U, F::RandomizedSystem, x, p = nothing)
    evaluate_and_jacobian!(F.u, F.U, F.system, x, p)
    randomize!(u, F.A, F.u)
    randomize!(U, F.A, F.U)
end

function ModelKit.taylor!(u::AbstractVector, v::Val, F::RandomizedSystem, tx, p = nothing)
    taylor!(F.u, v, F.system, tx, p)
    randomize!(u, F.A, F.u)
end
function ModelKit.taylor!(
    U::AbstractMatrix,
    v::Val{N},
    F::RandomizedSystem,
    tx::TaylorVector,
    p = nothing,
) where {N}
    taylor!(F.taylor_U, v, F.system, tx, p)
    randomize!(U, F.A, F.taylor_U, N + 1)
end
