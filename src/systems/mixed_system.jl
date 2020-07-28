struct MixedSystem{T,ID} <: AbstractSystem
    compiled::CompiledSystem{ID}
    interpreted::InterpretedSystem{T}
end
MixedSystem(F::System; kwargs...) =
    MixedSystem(CompiledSystem(F; kwargs...), InterpretedSystem(F; kwargs...))

Base.size(F::MixedSystem) = size(F.compiled)
ModelKit.variables(F::MixedSystem) = variables(F.interpreted)
ModelKit.parameters(F::MixedSystem) = parameters(F.interpreted)
ModelKit.variable_groups(F::MixedSystem) = variable_groups(F.interpreted)
ModelKit.System(F::MixedSystem) = System(F.interpreted)
Base.:(==)(F::MixedSystem, G::MixedSystem) = F.interpreted == G.interpreted

function Base.show(io::IO, F::MixedSystem)
    print(io, "Mixed: ")
    show(io, System(F))
end

(F::MixedSystem)(x, p = nothing) = F.interpreted(x, p)
ModelKit.evaluate!(u, F::MixedSystem, x::AbstractVector, p = nothing) =
    evaluate!(u, F.compiled, x, p)
ModelKit.evaluate_and_jacobian!(u, U, F::MixedSystem, x, p = nothing) =
    evaluate_and_jacobian!(u, U, F.compiled, x, p)
ModelKit.taylor!(u, v::Val, F::MixedSystem, tx, p = nothing) =
    taylor!(u, v, F.interpreted, tx, p)
