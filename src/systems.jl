export fixed

include("systems/mixed_system.jl")
include("systems/affine_chart_system.jl")
include("systems/composition_system.jl")
include("systems/fixed_parameter_system.jl")
include("systems/randomized_system.jl")
include("systems/sliced_system.jl")
include("systems/start_pair_system.jl")

"""
    fixed(F::System; compile::Union{Bool,Symbol} = $(COMPILE_DEFAULT[]))

Constructs either a [`CompiledSystem`](@ref) (if `compile = :all`), an
[`InterpretedSystem`](@ref) (if `compile = :none`) or a [`MixedSystem`](@ref) (`compile = :mixed`).
"""
function fixed(F::System; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...)
    if compile == true || compile == :all
        CompiledSystem(F; kwargs...)
    elseif compile == false || compile == :none
        InterpretedSystem(F; kwargs...)
    elseif compile == :mixed
        MixedSystem(F; kwargs...)
    else
        error("Unknown argument $compile for keyword `compile`.")
    end
end
fixed(F::AbstractSystem; kwargs...) = F

set_solution!(x, ::AbstractSystem, y) = (x .= y; x)
