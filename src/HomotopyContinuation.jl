module HomotopyContinuation

export ModelKit

import DelimitedFiles
using DynamicPolynomials: @polyvar
import ElasticArrays: ElasticArray
import FiniteDiff
import LinearAlgebra
import LoopVectorization
using LRUCache: LRU
import MixedSubdivisions
import MultivariatePolynomials
const MP = MultivariatePolynomials
using Parameters: @unpack
import ProgressMeter
import Random
import Printf
import PrettyTables
using Reexport: @reexport
import StructArrays
import TreeViews

const LA = LinearAlgebra

include("DoubleDouble.jl")
using .DoubleDouble

include("interval_arithmetic.jl")
using .IntervalArithmetic

include("ModelKit.jl")
export @polyvar

using ProjectiveVectors: PVector, dims, dimension_indices

const COMPILE_DEFAULT = Ref(:mixed)

include("utils.jl")
include("norm.jl")
include("voronoi_tree.jl")
include("unique_points.jl")
include("linear_algebra.jl")
include("linear_subspaces.jl")
include("systems.jl")
include("homotopies.jl")
include("predictor.jl")
include("newton_corrector.jl")
include("newton.jl")
include("tracker.jl")
include("valuation.jl")
include("path_result.jl")
include("endgame_tracker.jl")
include("path_info.jl")
include("total_degree.jl")
include("binomial_system.jl")
include("polyhedral.jl")
include("overdetermined.jl")
include("result.jl")
include("solve.jl")
include("monodromy.jl")
include("witness_set.jl")
include("certification.jl")

function __init__()
    # Overwrite default IJulia behaviour. See
    # https://github.com/timholy/ProgressMeter.jl/issues/162
    if isdefined(ProgressMeter, :ijulia_behavior)
        ProgressMeter.ijulia_behavior(:clear)
    end
end

# include("precompile.jl")
# _precompile_()


end #module
