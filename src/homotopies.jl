# internal only
include("homotopies/toric_homotopy.jl")

# public, these should be linked on the top
include("homotopies/mixed_homotopy.jl")
include("homotopies/affine_chart_homotopy.jl")
include("homotopies/parameter_homotopy.jl")
include("homotopies/coefficient_homotopy.jl")
include("homotopies/extrinsic_subspace_homotopy.jl")
include("homotopies/intrinsic_subspace_homotopy.jl")
include("homotopies/straight_line_homotopy.jl")
include("homotopies/fixed_parameter_homotopy.jl")

"""
    fixed(H::Homotopy; compile::Union{Bool,Symbol} = $(COMPILE_DEFAULT[]))

Constructs either a [`CompiledHomotopy`](@ref) (if `compile = :all`), an
[`InterpretedHomotopy`](@ref) (if `compile = :none`) or a
[`MixedHomotopy`](@ref) (`compile = :mixed`).
"""
function fixed(H::Homotopy; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...)
    if compile == true || compile == :all
        CompiledHomotopy(F; kwargs...)
    elseif compile == false || compile == :none
        InterpretedHomotopy(F; kwargs...)
    elseif compile == :mixed
        MixedHomotopy(F; kwargs...)
    else
        error("Unknown argument $compile for keyword `compile`.")
    end
end
fixed(H::AbstractHomotopy; kwargs...) = H

function set_solution!(x::AbstractVector, H::AbstractHomotopy, y::AbstractVector, t)
    x .= y
end
get_solution(H::AbstractHomotopy, x::AbstractVector, t) = copy(x)

start_parameters!(H::AbstractHomotopy, p) = H
target_parameters!(H::AbstractHomotopy, p) = H
parameters!(H::AbstractHomotopy, p, q) = H

ModelKit.taylor!(u, v::Val, H::AbstractHomotopy, tx, t, incremental::Bool) =
    taylor!(u, v, H, tx, t)

struct WrappedHomotopy <: AbstractHomotopy
    size::Tuple{Int,Int}
    evaluate!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},Vector{ComplexF64},ComplexF64},
    }
    evaluate_ext!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},Vector{ComplexDF64},ComplexF64},
    }
    evaluate_and_jacobian!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},Matrix{ComplexF64},Vector{ComplexF64},ComplexF64},
    }
    taylor1!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},Vector{ComplexF64},ComplexF64,Bool},
    }
    taylor2!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},TaylorVector{2,ComplexF64},ComplexF64,Bool},
    }
    taylor3!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},TaylorVector{3,ComplexF64},ComplexF64,Bool},
    }

    set_solution!::FunctionWrapper{
        Nothing,
        Tuple{Vector{ComplexF64},Vector{ComplexF64},ComplexF64},
    }
    get_solution::FunctionWrapper{Vector{ComplexF64},Tuple{Vector{ComplexF64},ComplexF64}}

    start_parameters!::FunctionWrapper{Nothing,Tuple{Vector{ComplexF64}}}
    target_parameters!::FunctionWrapper{Nothing,Tuple{Vector{ComplexF64}}}
    parameters!::FunctionWrapper{Nothing,Tuple{Vector{ComplexF64},Vector{ComplexF64}}}
end

function WrappedHomotopy(H::AbstractHomotopy)
    WrappedHomotopy(
        size(H),
        (u, x, t) -> (evaluate!(u, H, x, t); nothing),
        (u, x, t) -> (evaluate!(u, H, x, t); nothing),
        (u, U, x, t) -> (evaluate_and_jacobian!(u, U, H, x, t); nothing),
        (u, x, t, inc) -> (taylor!(u, Val(1), H, x, t, inc); nothing),
        (u, x, t, inc) -> (taylor!(u, Val(2), H, x, t, inc); nothing),
        (u, x, t, inc) -> (taylor!(u, Val(3), H, x, t, inc); nothing),
        (x, y, t) -> (set_solution!(x, H, y, t); nothing),
        (x, t) -> get_solution(H, x, t),
        (p) -> (start_parameters!(H, p); nothing),
        (q) -> (target_parameters!(H, q); nothing),
        (p, q) -> (parameters!(H, p, q); nothing),
    )
end
Base.size(H::WrappedHomotopy) = H.size
ModelKit.evaluate!(u, H::WrappedHomotopy, x::Vector{ComplexF64}, t) =
    (H.evaluate!(u, x, t); u)
ModelKit.evaluate!(u, H::WrappedHomotopy, x::Vector{ComplexDF64}, t) =
    (H.evaluate_ext!(u, x, t); u)
ModelKit.evaluate_and_jacobian!(u, U, H::WrappedHomotopy, x::Vector{ComplexF64}, t) =
    (H.evaluate_and_jacobian!(u, U, x, t); u)
ModelKit.taylor!(u, v::Val{1}, H::WrappedHomotopy, tx, t, inc::Bool) = (H.taylor1!(u, tx, t, inc); u)
ModelKit.taylor!(u, v::Val{2}, H::WrappedHomotopy, tx, t, inc::Bool) = (H.taylor2!(u, tx, t, inc); u)
ModelKit.taylor!(u, v::Val{3}, H::WrappedHomotopy, tx, t, inc::Bool) = (H.taylor3!(u, tx, t, inc); u)

set_solution!(x::Vector{ComplexF64}, H::WrappedHomotopy, y::AbstractVector, t) =
    set_solution!(x, H, convert(Vector{ComplexF64}, y), convert(ComplexF64, t))
set_solution!(
    x::Vector{ComplexF64},
    H::WrappedHomotopy,
    y::Vector{ComplexF64},
    t::ComplexF64,
) = (H.set_solution!(x, y, t); x)

get_solution(H::WrappedHomotopy, x::AbstractVector, t::ComplexF64) =
    H.get_solution(convert(Vector{ComplexF64}, x), convert(ComplexF64, t))
get_solution(H::WrappedHomotopy, x::Vector{ComplexF64}, t::ComplexF64) =
    H.get_solution(x, t)

start_parameters!(H::WrappedHomotopy, p) = H.start_parameters!(p)
target_parameters!(H::WrappedHomotopy, p) = H.start_parameters!(p)
parameters!(H::WrappedHomotopy, p, q) = H.parameters!(p, q)
