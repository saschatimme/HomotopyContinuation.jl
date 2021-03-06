@reexport module ModelKit

export @var,
    @unique_var,
    AbstractHomotopy,
    AbstractSystem,
    CompiledHomotopy,
    CompiledSystem,
    Expression,
    Homotopy,
    InterpretedHomotopy,
    InterpretedSystem,
    System,
    TaylorVector,
    Variable,
    coefficients,
    coeffs_as_dense_poly,
    degree,
    degrees,
    differentiate,
    dense_poly,
    evaluate,
    evaluate!,
    evaluate_and_jacobian!,
    expand,
    exponents_coefficients,
    expressions,
    horner,
    is_homogeneous,
    jacobian,
    jacobian!,
    multi_degrees,
    parameters,
    nparameters,
    nvariables,
    monomials,
    optimize,
    parameters,
    subs,
    support_coefficients,
    rand_poly,
    taylor!,
    to_dict,
    to_number,
    variables,
    variable_groups,
    vectors

using ..DoubleDouble: ComplexDF64

using Arblib:
    Arblib,
    Arb,
    ArbRef,
    ArbVector,
    ArbRefVector,
    Acb,
    AcbRef,
    AcbVector,
    AcbRefVector,
    AcbRefMatrix
import LinearAlgebra
import MultivariatePolynomials:
    MultivariatePolynomials,
    coefficients,
    degree,
    differentiate,
    nvariables,
    monomials,
    subs,
    variables
using Parameters: @unpack
const MP = MultivariatePolynomials

include("model_kit/symengine.jl")
include("model_kit/symbolic.jl")

include("model_kit/taylor_vector.jl")

include("model_kit/instructions.jl")
include("model_kit/slp_interpreter.jl")
include("model_kit/slp_compiler.jl")
#
include("model_kit/abstract_system_homotopy.jl")
include("model_kit/compiled_system_homotopy.jl")
include("model_kit/interpreted_system_homotopy.jl")

end # module
