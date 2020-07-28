export SlicedSystem, slice

"""
    SlicedSystem(F::AbstractSystem, L::LinearSubspace)

Given a polynomial system `F` with the affine linear subspace `L`
defined by ``Ax = b`` this constructs the system `[F(x); A x - b] = 0`.
See also [`slice`](@ref).
"""
struct SlicedSystem{S<:AbstractSystem,T} <: AbstractSystem
    system::S
    affine::LinearSubspace{T}
end
SlicedSystem(
    F::System,
    A::LinearSubspace;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
) = SlicedSystem(fixed(F; compile = compile), A)

"""
    slice(F::System, L::LinearSubspace; compile = $(COMPILE_DEFAULT[]))
    slice(F::AbstractSystem, L::LinearSubspace)

Slice the zero set of the polynomial system `F` with the affine linear subspace `L`
defined by ``Ax = b``. This constructs the system `[F(x); A x - b] = 0`.
See also [`SlicedSystem`](@ref).
"""
slice(F::System, A::LinearSubspace; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    SlicedSystem(fixed(F; compile = compile), A)
slice(
    F::AbstractSystem,
    A::LinearSubspace;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
) = SlicedSystem(F, A)
Base.intersect(F::Union{System,AbstractSystem}, A::LinearSubspace) = SlicedSystem(F, A)
Base.intersect(A::LinearSubspace, F::Union{System,AbstractSystem}) = SlicedSystem(F, A)

ModelKit.variables(F::SlicedSystem) = variables(F.system)
ModelKit.parameters(F::SlicedSystem) = parameters(F.system)
ModelKit.variable_groups(F::SlicedSystem) = nothing

linear_subspace(A::SlicedSystem) = A.affine
system(A::SlicedSystem) = A.system

function Base.size(F::SlicedSystem)
    m, n = size(F.system)
    (m + codim(F.affine), n)
end

function (F::SlicedSystem)(x::AbstractVector, p = nothing)
    u = F.system(x, p)
    append!(u, extrinsic(F.affine)(x))
    u
end

function ModelKit.evaluate!(u, F::SlicedSystem, x, p = nothing)
    evaluate!(u, F.system, x, p)

    @unpack A, b = extrinsic(F.affine)
    m = size(F.system, 1)
    n, N = size(A)
    @inbounds for i = 1:n
        u[m+i] = -b[i]
    end
    @inbounds for j = 1:N
        xj = x[j]
        for i = 1:n
            u[m+i] += A[i, j] * xj
        end
    end

    u
end

function ModelKit.evaluate_and_jacobian!(u, U, F::SlicedSystem, x, p = nothing)
    evaluate_and_jacobian!(u, U, F.system, x, p)

    @unpack A, b = extrinsic(F.affine)
    m = size(F.system, 1)
    n, N = size(A)
    @inbounds for i = 1:n
        u[m+i] = -b[i]
    end
    @inbounds for j = 1:N
        xj = x[j]
        for i = 1:n
            u[m+i] += A[i, j] * xj
        end
    end

    @inbounds for j = 1:N, i = 1:n
        U[m+i, j] = A[i, j]
    end

    nothing
end

function ModelKit.taylor!(u, v::Val, F::SlicedSystem, tx, p = nothing)
    u .= zero(eltype(u))
    taylor!(u, v, F.system, tx, p)
    u
end
