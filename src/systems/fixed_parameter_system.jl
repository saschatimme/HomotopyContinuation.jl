export FixedParameterSystem, fix_parameters

"""
    FixedParameterSystem(F:AbstractSystem, parameters)

Construct a system from the given [`AbstractSystem`](@ref) `F` with the given `parameters`
fixed.
"""
struct FixedParameterSystem{S<:AbstractSystem,T} <: AbstractSystem
    system::S
    parameters::Vector{T}
end
FixedParameterSystem(F::AbstractSystem, p; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    FixedParameterSystem(F, p)
FixedParameterSystem(F::System, p; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    FixedParameterSystem(fixed(F; compile = compile), p)
Base.size(F::FixedParameterSystem) = size(F.system)

ModelKit.variables(F::FixedParameterSystem) = variables(F.system)
ModelKit.parameters(F::FixedParameterSystem) = Variable[]
ModelKit.variable_groups(F::FixedParameterSystem) = variable_groups(F.system)

(F::FixedParameterSystem)(x, p = nothing) = F.system(x, F.parameters)

ModelKit.evaluate!(u, F::FixedParameterSystem, x, p = nothing) =
    evaluate!(u, F.system, x, F.parameters)
ModelKit.evaluate_and_jacobian!(u, U, F::FixedParameterSystem, x, p = nothing) =
    evaluate_and_jacobian!(u, U, F.system, x, F.parameters)
ModelKit.taylor!(u, v::Val, F::FixedParameterSystem, tx, p = nothing) =
    taylor!(u, v, F.system, tx, F.parameters)

# Only for certification
ModelKit.jacobian!(U, F::FixedParameterSystem{<:InterpretedSystem}, x, p = nothing) =
    ModelKit.jacobian!(U, F.system, x, F.parameters)
ModelKit.jacobian!(U, F::FixedParameterSystem{<:InterpretedSystem}, x, p, cache) =
    ModelKit.jacobian!(U, F.system, x, F.parameters, cache)

"""
    fix_parameters(F::Union{System,AbstractSystem}, p; compile::Union{Bool,Symbol} = $(COMPILE_DEFAULT[]))

Fix the parameters of the given system `F`. Returns a [`FixedParameterSystem`](@ref).
"""
fix_parameters(F::Union{System,AbstractSystem}, p; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[]) =
    FixedParameterSystem(F, p; compile = compile)
