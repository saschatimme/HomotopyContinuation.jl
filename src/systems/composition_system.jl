export CompositionSystem, compose

"""
    CompositionSystem(G::AbstractSystem, F::AbstractSystem)

Construct the system ``G(F(x;p);p)``. Note that the parameters are passed to ``G`` and
``F`` are identical.
"""
struct CompositionSystem{S1<:AbstractSystem,S2<:AbstractSystem} <: AbstractSystem
    # The system is g ∘ f
    g::S2
    f::S1

    f_u::Vector{ComplexF64}
    f_ū::Vector{ComplexDF64}
    f_U::Matrix{ComplexF64}
    g_U::Matrix{ComplexF64}
    tu⁴::TaylorVector{5,ComplexF64}
    tu³::TaylorVector{4,ComplexF64}
    tu²::TaylorVector{3,ComplexF64}
    tu¹::TaylorVector{2,ComplexF64}
end
function CompositionSystem(g::AbstractSystem, f::AbstractSystem)
    size(g, 2) == size(f, 1) ||
        throw(ArgumentError("Cannot create composition G ∘ F since the number of variabels of `G` and the number polynomials of `F` doesn't match."))
    f_u = zeros(ComplexF64, size(f, 1))
    f_ū = zeros(ComplexDF64, size(f, 1))
    f_U = zeros(ComplexF64, size(f))
    g_U = zeros(ComplexF64, size(g))

    tu⁴ = TaylorVector{5}(ComplexF64, size(f, 1))
    tu¹ = TaylorVector{2}(tu⁴)
    tu² = TaylorVector{3}(tu⁴)
    tu³ = TaylorVector{4}(tu⁴)

    CompositionSystem(g, f, f_u, f_ū, f_U, g_U, tu⁴, tu³, tu², tu¹)
end

Base.size(C::CompositionSystem) = (size(C.g, 1), size(C.f, 2))

ModelKit.variables(F::CompositionSystem) = variables(F.f)
ModelKit.parameters(F::CompositionSystem) = parameters(F.f)
ModelKit.variable_groups(F::CompositionSystem) = variable_groups(F.f)

function Base.show(io::IO, C::CompositionSystem)
    println(io, "Composition G ∘ F:")
    println(io, "F: ")
    show(io, C.f)
    println(io, "\nG: ")
    show(io, C.g)
end

(C::CompositionSystem)(x, p = nothing) = C.g(C.f(x, p), p)
function ModelKit.evaluate!(
    u,
    C::CompositionSystem,
    x::AbstractVector{ComplexF64},
    p = nothing,
)
    evaluate!(u, C.g, evaluate!(C.f_u, C.f, x, p), p)
end
function ModelKit.evaluate!(
    u,
    C::CompositionSystem,
    x::AbstractVector{ComplexDF64},
    p = nothing,
)
    evaluate!(u, C.g, evaluate!(C.f_ū, C.f, x, p), p)
end

function ModelKit.evaluate_and_jacobian!(u, U, C::CompositionSystem, x, p = nothing)
    evaluate_and_jacobian!(C.f_u, C.f_U, C.f, x, p)
    evaluate_and_jacobian!(u, C.g_U, C.g, C.f_u, p)
    LA.mul!(U, C.g_U, C.f_U)
    nothing
end

_get_tu(C::CompositionSystem, ::Val{1}) = C.tu¹
_get_tu(C::CompositionSystem, ::Val{2}) = C.tu²
_get_tu(C::CompositionSystem, ::Val{3}) = C.tu³
_get_tu(C::CompositionSystem, ::Val{4}) = C.tu⁴

function ModelKit.taylor!(u, v::Val, C::CompositionSystem, tx, p = nothing)
    tu = _get_tu(C, v)
    taylor!(tu.data, v, C.f, tx, p)
    taylor!(u, v, C.g, tu, p)
end

"""
    compose(G::Union{AbstractSystem,System}, F::Union{AbstractSystem,System})

Construct the composition ``G(F(x))``. You can also use the infix operator
`∘` (written by \\circ).


### Example
```julia
julia> @var a b c x y z

julia> g = System([a * b * c]);

julia> f = System([x+y, y + z, x + z]);

julia> compose(g, f)
Composition G ∘ F:
F:
ModelKitSystem{(0xbb16b481c0808501, 1)}:
Compiled: System of length 3
 3 variables: x, y, z

 x + y
 y + z
 x + z

G:
ModelKitSystem{(0xf0a2384a42428501, 1)}:
Compiled: System of length 1
 3 variables: a, b, c

 a*b*c


julia> (g ∘ f)([x,y,z])
1-element Array{Expression,1}:
 (x + z)*(y + z)*(x + y)
```
"""
compose(
    g::AbstractSystem,
    f::AbstractSystem;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
) = CompositionSystem(g, f)
compose(g::AbstractSystem, f::System; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    CompositionSystem(g, fixed(f; compile = compile))
function compose(g::System, f::System; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[])
    pg = parameters(g)
    pf = parameters(f)
    (isempty(pf) || isempty(pg) || pf == pg) ||
        throw(ArgumentError("Cannot construct a composition of two system with different sets of parameters."))
    CompositionSystem(fixed(g; compile = compile), fixed(f; compile = compile))
end
import Base: ∘
∘(g::Union{AbstractSystem,System}, f::Union{AbstractSystem,System}) = compose(g, f)
