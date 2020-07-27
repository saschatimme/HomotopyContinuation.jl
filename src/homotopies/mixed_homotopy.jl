struct MixedHomotopy{T,ID} <: AbstractHomotopy
    compiled::CompiledHomotopy{ID}
    interpreted::InterpretedHomotopy{T}
end
MixedHomotopy(H::Homotopy) = MixedHomotopy(CompiledHomotopy(H), InterpretedHomotopy(H))

Base.size(H::MixedHomotopy) = size(H.compiled)
ModelKit.variables(H::MixedHomotopy) = variables(H.interpreted)
ModelKit.parameters(H::MixedHomotopy) = parameters(H.interpreted)
ModelKit.variable_groups(H::MixedHomotopy) = variable_groups(H.interpreted)
ModelKit.Homotopy(H::MixedHomotopy) = Homotopy(H.interpreted)

function Base.show(io::IO, H::MixedHomotopy)
    print(io, "Hybrid: ")
    show(io, Homotopy(H))
end

(H::MixedHomotopy)(x, t, p = nothing) = H.interpreted(x, t, p)
ModelKit.evaluate!(u, H::MixedHomotopy, x::AbstractVector, t, p = nothing) =
    evaluate!(u, H.compiled, x, t, p)
ModelKit.evaluate_and_jacobian!(u, U, H::MixedHomotopy, x, t, p = nothing) =
    evaluate_and_jacobian!(u, U, H.compiled, x, t, p)
ModelKit.taylor!(u, v::Val, H::MixedHomotopy, tx, t, p = nothing) =
    taylor!(u, v, H.interpreted, tx, t, p)
