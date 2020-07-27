export InterpretedSystem, InterpretedHomotopy

"""
    InterpretedSystem <: AbstractSystem

An [`AbstractSystem`](@ref) which automatically generates a program for the
fast evaluation of `F` and its Jacobian. The program is however, not compiled
but rather interpreted. See also [`CompiledSystem`](@ref).

    InterpretedSystem(F::System; optimizations = true)

Construct an `InterpretedSystem` from the given [`System`](@ref) `F`.
If `optimizations = true` then [`optimize`](@ref) is called on `F` before compiling.
"""
struct InterpretedSystem{T} <: AbstractSystem
    system::System
    eval_interpreter::Interpreter{T,1}
    eval_interpreter_cache::InterpreterCache{ComplexF64}
    eval_interpreter_cache_ext::InterpreterCache{ComplexDF64}
    jac_interpreter::Interpreter{T,2}
    jac_interpreter_cache::InterpreterCache{ComplexF64}
    taylor_caches::Tuple{
        InterpreterCache{NTuple{2,ComplexF64}},
        InterpreterCache{NTuple{3,ComplexF64}},
        InterpreterCache{NTuple{4,ComplexF64}},
    }
end

function InterpretedSystem(F::System; optimizations::Bool = true)
    F = optimizations ? optimize(F) : F
    eval_interpreter = evaluate_interpreter(F)
    jac_interpreter = jacobian_interpreter(F)
    T = promote_type(eltype(eval_interpreter), eltype(jac_interpreter))
    if !(T isa Rational)
        T = promote_type(T, Float64)
    end
    eval_interpreter = convert(Interpreter{T,1}, eval_interpreter)
    jac_interpreter = convert(Interpreter{T,2}, jac_interpreter)
    eval_interpreter_cache = InterpreterCache(ComplexF64, eval_interpreter)
    eval_interpreter_cache_ext = InterpreterCache(ComplexDF64, eval_interpreter)
    taylor_caches = (
        InterpreterCache(NTuple{2,ComplexF64}, eval_interpreter),
        InterpreterCache(NTuple{3,ComplexF64}, eval_interpreter),
        InterpreterCache(NTuple{4,ComplexF64}, eval_interpreter),
    )

    InterpretedSystem(
        F,
        eval_interpreter,
        eval_interpreter_cache,
        eval_interpreter_cache_ext,
        jac_interpreter,
        InterpreterCache(ComplexF64, jac_interpreter),
        taylor_caches,
    )
end

Base.size(F::InterpretedSystem) = size(F.system)
variables(F::InterpretedSystem) = variables(F.system)
parameters(F::InterpretedSystem) = parameters(F.system)
variable_groups(F::InterpretedSystem) = variable_groups(F.system)
Base.:(==)(F::InterpretedSystem, G::InterpretedSystem) = F.system == G.system

function Base.show(io::IO, F::InterpretedSystem)
    print(io, "Interpreted: ")
    show(io, F.system)
end

(F::InterpretedSystem)(x, p = nothing) = F.system(x, p)
function evaluate!(u, F::InterpretedSystem, x, p = nothing)
    execute!(u, F.eval_interpreter, x, p, F.eval_interpreter_cache)
end
function evaluate!(u, F::InterpretedSystem, x::AbstractVector{ComplexDF64}, p = nothing)
    execute!(u, F.eval_interpreter, x, p, F.eval_interpreter_cache_ext)
end
function evaluate_and_jacobian!(u, U, F::InterpretedSystem, x, p = nothing)
    execute!(u, U, F.jac_interpreter, x, p, F.jac_interpreter_cache)
    nothing
end
function jacobian!(U, F::InterpretedSystem, x, p = nothing, cache = F.jac_interpreter_cache)
    execute!(U, F.jac_interpreter, x, p, cache)
    nothing
end

for M = 1:3
    @eval function taylor!(
        u::AbstractVecOrMat,
        v::Val{$M},
        F::InterpretedSystem,
        x,
        p = nothing,
    )
        execute!(u, Val{$M}, F.eval_interpreter, x, p, F.taylor_caches[$M])
        u
    end
end

"""
    InterpretedHomotopy <: AbstractHomotopy

An [`AbstractHomotopy`](@ref) which automatically generates a program for the
fast evaluation of `H` and its Jacobian. The program is however, not compiled
but rather interpreted. See also [`CompiledHomotopy`](@ref).

    InterpretedHomotopy(H::Homotopy; optimizations = true)

Construct an `InterpretedHomotopy` from the given [`Homotopy`](@ref) `H`.
If `optimizations = true` then [`optimize`](@ref) is called on `H` before compiling.
"""
struct InterpretedHomotopy{T} <: AbstractHomotopy
    homotopy::Homotopy
    eval_interpreter::Interpreter{T,1}
    eval_interpreter_cache::InterpreterCache{ComplexF64}
    eval_interpreter_cache_ext::InterpreterCache{ComplexDF64}
    jac_interpreter::Interpreter{T,2}
    jac_interpreter_cache::InterpreterCache{ComplexF64}
    taylor_caches::Tuple{
        InterpreterCache{NTuple{2,ComplexF64}},
        InterpreterCache{NTuple{3,ComplexF64}},
        InterpreterCache{NTuple{4,ComplexF64}},
    }
end

function InterpretedHomotopy(H::Homotopy; optimizations::Bool = true)
    H = optimizations ? optimize(H) : H
    eval_interpreter = evaluate_interpreter(H)
    jac_interpreter = jacobian_interpreter(H)
    T = promote_type(eltype(eval_interpreter), eltype(jac_interpreter))
    if !(T isa Rational)
        T = promote_type(T, Float64)
    end
    eval_interpreter = convert(Interpreter{T,1}, eval_interpreter)
    jac_interpreter = convert(Interpreter{T,2}, jac_interpreter)
    eval_interpreter_cache = InterpreterCache(ComplexF64, eval_interpreter)
    eval_interpreter_cache_ext = InterpreterCache(ComplexDF64, eval_interpreter)
    taylor_caches = (
        InterpreterCache(NTuple{2,ComplexF64}, eval_interpreter),
        InterpreterCache(NTuple{3,ComplexF64}, eval_interpreter),
        InterpreterCache(NTuple{4,ComplexF64}, eval_interpreter),
    )

    InterpretedHomotopy(
        H,
        eval_interpreter,
        eval_interpreter_cache,
        eval_interpreter_cache_ext,
        jac_interpreter,
        InterpreterCache(ComplexF64, jac_interpreter),
        taylor_caches,
    )
end


Base.size(H::InterpretedHomotopy) = size(H.homotopy)
variables(H::InterpretedHomotopy) = variables(H.homotopy)
parameters(H::InterpretedHomotopy) = parameters(H.homotopy)

function Base.show(io::IO, H::InterpretedHomotopy)
    print(io, "Interpreted: ")
    show(io, H.homotopy)
end

(H::InterpretedHomotopy)(x, t, p = nothing) = H.homotopy(x, t, p)
function evaluate!(
    u,
    H::InterpretedHomotopy,
    x,
    t,
    p = nothing,
    cache = H.eval_interpreter_cache,
)
    execute!(u, H.eval_interpreter, x, t, p, cache)
end
function evaluate!(
    u,
    H::InterpretedHomotopy,
    x::AbstractVector{ComplexDF64},
    t,
    p = nothing,
)
    execute!(u, H.eval_interpreter, x, t, p, H.eval_interpreter_cache_ext)
end
function evaluate_and_jacobian!(
    u,
    U,
    H::InterpretedHomotopy,
    x,
    t,
    p = nothing,
    cache = H.jac_interpreter_cache,
)
    execute!(u, U, H.jac_interpreter, x, t, p, cache)
    nothing
end

function jacobian!(
    U,
    H::InterpretedHomotopy,
    x,
    t,
    p = nothing,
    cache = H.jac_interpreter_cache,
)
    execute!(U, H.jac_interpreter, x, t, p, cache)
    nothing
end

for M = 1:3
    @eval function taylor!(
        u::AbstractVecOrMat,
        v::Val{$M},
        H::InterpretedHomotopy,
        x,
        t,
        p = nothing,
    )
        execute!(u, Val{$M}, H.eval_interpreter, x, t, p, H.taylor_caches[$M])
        u
    end
end
