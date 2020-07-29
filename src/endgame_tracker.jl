export EndgameTracker, EndgameOptions, track

"""
    AbstractPathTracker

Supertype for path trackers.
"""
abstract type AbstractPathTracker end

Base.broadcastable(T::AbstractPathTracker) = Ref(T)

"""
    EndgameOptions(; options...)

Options controlling the behaviour of a [`EndgameTracker`](@ref).

## Options

* `at_infinity_check = true`: Whether divering paths should be truncated.
* `endgame_start = 0.1`: The point `t` in time where the endgame starts. Set it to `0.0`
  to disable the endgame.
* `only_nonsingular = false`: If `true` don't run the Cauchy endgame to handle singular
  solutions.
* `zero_is_at_infinity = false`: Whether paths going to a solution where at least one
  coordinates is zero should also be considered diverging.

### Parameters
These parameters control the behaviour during the endgame. See [^BT20] for details.

* `max_endgame_steps = 2000`: The maximal number of steps performed during the endgame.
* `max_winding_number = 6`: The maximal winding number which is attempted in the
 Cauchy endgame.
* `min_cond = 1e6`: The minimal condition number after which an endgame strategy is
  considered to be applied.
* `min_cond_growth = 1e4`: The minimal condition number growth after which an
   endgame strategy is considered to be applied.
* `min_coord_growth = 100`: The minimal relative growth of a coordinate necessary to
  to be considered going to infininity (resp. zero).
* `val_at_infinity_tol = 1e-3`: Tolerance on the valuation which has to be
  satisfied before a path is considered to diverge / go to infinity.
* `val_finite_tol = 1e-3`: Tolerance on the valuation which has to be satisfied
  before the Cauchy endgame is started.

[^BT20]: Breiding, P. and Timme, S. "Tropical Endgame", In preparation (2020)
"""
Base.@kwdef mutable struct EndgameOptions
    endgame_start::Float64 = 0.1
    max_endgame_steps::Int = 2000
    max_endgame_extended_steps::Int = 400
    # eg parameter
    min_cond::Float64 = 1e6
    min_cond_growth::Float64 = 1e4
    min_coord_growth::Float64 = 100.0
    zero_is_at_infinity::Bool = false
    at_infinity_check::Bool = true
    only_nonsingular::Bool = false
    # valuation etc
    val_finite_tol::Float64 = 0.05
    val_at_infinity_tol::Float64 = 0.01

    # singular solutions parameters
    max_winding_number::Int = 6
    singular_min_accuracy::Float64 = 1e-6
end

Base.show(io::IO, opts::EndgameOptions) = print_fieldnames(io, opts)
Base.show(io::IO, ::MIME"application/prs.juno.inline", opts::EndgameOptions) = opts

"""
    EndgameTrackerCode

The possible states an `EndgameTracker` can be in:

* `EndgameTrackerCode.tracking`
* `EndgameTrackerCode.success`
* `EndgameTrackerCode.at_infinity`
* `EndgameTrackerCode.at_zero`
* `EndgameTrackerCode.excess_solution`
* `EndgameTrackerCode.post_check_failed`
* `EndgameTrackerCode.polyhedral_failed`
* `EndgameTrackerCode.terminated_accuracy_limit`
* `EndgameTrackerCode.terminated_ill_conditioned`
* `EndgameTrackerCode.terminated_invalid_startvalue`
* `EndgameTrackerCode.terminated_max_winding_number`
* `EndgameTrackerCode.terminated_max_steps`
* `EndgameTrackerCode.terminated_step_size_too_small`
"""
module EndgameTrackerCode
using ..TrackerCode: TrackerCode

@enum codes begin
    tracking
    success
    at_infinity
    at_zero
    terminated_accuracy_limit
    terminated_invalid_startvalue
    terminated_ill_conditioned
    terminated_max_steps
    terminated_max_extended_steps
    terminated_max_winding_number
    terminated_step_size_too_small
    terminated_unknown
    post_check_failed
    excess_solution
    polyhedral_failed
end

function Base.convert(::Type{codes}, code::TrackerCode.codes)
    if code == TrackerCode.success
        return success
    elseif code == TrackerCode.terminated_max_steps
        return terminated_max_steps
    elseif code == TrackerCode.terminated_accuracy_limit
        return terminated_accuracy_limit
    elseif code == TrackerCode.terminated_ill_conditioned
        return terminated_ill_conditioned
    elseif code == TrackerCode.terminated_invalid_startvalue
        return terminated_invalid_startvalue
    elseif code == TrackerCode.terminated_step_size_too_small
        return terminated_step_size_too_small
    elseif code == TrackerCode.terminated_unknown
        return terminated_unknown
    else
        return tracking
    end
end

end


"""
    is_success(code::EndgameTrackerCode.codes)

Returns `true` if `status` indicates a success in tracking.
"""
is_success(code::EndgameTrackerCode.codes) = code == EndgameTrackerCode.success

"""
    is_success(code::EndgameTrackerCode.codes)

Returns `true` if `status` indicates that a path diverged towards infinity.
"""
is_at_infinity(code::EndgameTrackerCode.codes) = code == EndgameTrackerCode.at_infinity

"""
    is_tracking(code::EndgameTrackerCode.codes)

Returns `true` if `status` indicates the tracking is not going on.
"""
is_tracking(code::EndgameTrackerCode.codes) = code == EndgameTrackerCode.tracking

"""
    is_invalid_startvalue(code::EndgameTrackerCode.codes)

Returns `true` if the provided start value was not valid.
"""
is_invalid_startvalue(code::EndgameTrackerCode.codes) =
    code == EndgameTrackerCode.terminated_invalid_startvalue

###########
## State ##
###########

Base.@kwdef mutable struct EndgameTrackerState
    code::EndgameTrackerCode.codes = EndgameTrackerCode.tracking
    # modi
    singular_endgame::Bool = false

    val::Valuation
    winding_number::Union{Nothing,Int} = nothing

    solution::Vector{ComplexF64}
    accuracy::Float64 = NaN
    cond::Float64 = 1.0
    singular::Bool = false
    steps_eg::Int = 0
    ext_steps_eg_start::Int = 0

    jump_to_zero_failed::Tuple{Bool,Bool} = (false, false)
    last_point::Vector{ComplexF64} = copy(solution)
    last_t::Float64 = NaN

    # condition number
    row_scaling::Vector{Float64}
    col_scaling::Vector{Float64} = zeros(length(solution))
    cond_base::Vector{Float64} = copy(row_scaling)

    # at_infinity
    at_infinity_starts::Vector{Float64} = copy(col_scaling)
    at_infinity_tols::Vector{Float64} = copy(col_scaling)
    at_infinity_abs_coords::Vector{Float64} = copy(col_scaling)
    at_infinity_conds::Vector{Float64} = copy(col_scaling)

    # singular endgame
    samples::Vector{TaylorVector{2,ComplexF64}}
    sample_times::Vector{Float64} = zeros(3)
    sample_conds::Vector{Float64} = zeros(3)
    singular_start::Float64 = NaN
    singular_steps::Int = 0
    prediction::Vector{ComplexF64} = copy(solution)
    prev_prediction::Vector{ComplexF64} = copy(solution)
end

EndgameTrackerState(npolynomials::Integer, x::AbstractVector) = EndgameTrackerState(;
    val = Valuation(length(x)),
    solution = copy(x),
    row_scaling = zeros(npolynomials),
    samples = [TaylorVector{2}(ComplexF64, length(x)) for _ = 1:3],
)

"""
    EndgameTracker(tracker::Tracker; options = EndgameOptions())
    EndgameTracker(H::AbstractHomotopy; options = EndgameOptions())

A `EndgameTracker` combines a [`Tracker`](@ref) with an endgame. That is,
while a [`Tracker`](@ref) assumes that the solution path is non-singular and convergent, the endgame
allows to handle singular endpoints as well as diverging paths.
To compute singular solutions the *Cauchy endgame* used, for divering paths a strategy
based on the valuation of local Puiseux series expansion of the path is used.
See [^BT20] for a detailed description.
By convention, a `EndgameTracker` always tracks from ``t=1`` to ``t = 0``.
See [`EndgameOptions`](@ref) for the possible options.

[^BT20]: Breiding, P. and Timme, S. "Tropical Endgame", In preparation (2020)
"""
struct EndgameTracker{H<:AbstractHomotopy,M<:AbstractMatrix{ComplexF64}} <:
       AbstractPathTracker
    tracker::Tracker{H,M}
    state::EndgameTrackerState
    options::EndgameOptions
end

EndgameTracker(H::Homotopy; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...) =
    EndgameTracker(fixed(H; compile = compile); kwargs...)
function EndgameTracker(H::AbstractHomotopy; tracker_options = TrackerOptions(), kwargs...)
    EndgameTracker(Tracker(H; options = tracker_options); kwargs...)
end
function EndgameTracker(tracker::Tracker; options = EndgameOptions())
    if !(options isa EndgameOptions)
        options = EndgameOptions(; options...)
    end
    state = EndgameTrackerState(size(tracker.homotopy, 1), tracker.state.x)
    EndgameTracker(tracker, state, options)
end

Base.show(io::IO, x::EndgameTracker) = print(io, typeof(x), "()")
Base.show(io::IO, ::MIME"application/prs.juno.inline", x::EndgameTracker) = x
Base.broadcastable(T::EndgameTracker) = Ref(T)

function init!(
    endgame_tracker::EndgameTracker,
    x,
    t₁::Real;
    ω::Float64 = NaN,
    μ::Float64 = NaN,
    extended_precision::Bool = false,
)
    @unpack tracker, state, options = endgame_tracker

    tracker.options.min_rel_step_size = 0.0
    init!(tracker, x, t₁, 0.0; ω = ω, μ = μ, extended_precision = extended_precision)
    state.code = status(tracker)
    state.singular_endgame = false
    state.jump_to_zero_failed = (false, false)

    init!(state.val)
    state.winding_number = nothing

    state.solution .= NaN
    state.accuracy = NaN
    state.cond = NaN
    state.singular = false
    state.steps_eg = 0
    state.ext_steps_eg_start = typemax(Int)

    state.row_scaling .= 1
    state.col_scaling .= 1

    state.at_infinity_starts .= NaN

    state.singular_steps = 0

    endgame_tracker
end


function track!(
    endgame_tracker::EndgameTracker,
    x,
    t₁::Real;
    ω::Float64 = NaN,
    μ::Float64 = NaN,
    extended_precision::Bool = false,
    debug::Bool = false,
)
    init!(endgame_tracker, x, t₁; ω = ω, μ = μ, extended_precision = extended_precision)

    while is_tracking(endgame_tracker.state.code)
        step!(endgame_tracker, debug)
    end

    endgame_tracker.state.code
end


# The endgame tracker can be in three different states
# 1) Pre Endgame
# Just forward to a tracker step
#
# 2) Endgame
# Monitor valuation, and decide to possibly switch to singular eg, or cut off path.
# Each step! still corresponds to a single tracker step
#
# 3) Singular Endgame
#
# Here we expect a singular solution at the end. For this the tracker starts a geometric
# progression in which zero is approached.

function step!(endgame_tracker::EndgameTracker, debug::Bool = false)
    @unpack tracker, state, options = endgame_tracker
    # For performance reasons (and to catch degenerate situations) we only allow a maximal
    # number of steps in the endgame. Check if we reached this and if so terminate.
    if state.steps_eg ≥ options.max_endgame_steps
        return (state.code = EndgameTrackerCode.terminated_max_steps)
    elseif ext_steps(tracker.state) - state.ext_steps_eg_start >
           options.max_endgame_extended_steps
        # check if we had previously a singular solution attempt and return this
        if all(!isnan, state.solution) &&
           !isnothing(state.winding_number) &&
           state.accuracy < 1e-5
            state.cond =
                LA.cond(tracker, state.solution, 0.0, state.row_scaling, state.col_scaling)
            state.singular = true
            return (state.code = EndgameTrackerCode.success)
        end
        return (state.code = EndgameTrackerCode.terminated_max_extended_steps)
    end

    state.last_point .= tracker.state.x
    state.last_t = tracker.state.t

    # if we are in the singular_endgame actually, move there
    if state.singular_endgame
        return singular_endgame_step!(endgame_tracker, debug)
    end

    # keep track on whether we wanted to finish tracking / jump to zero
    is_jump_to_zero = iszero(tracker.state.t′)

    # perform a simple tracker step
    step_success = step!(tracker, debug)
    state.code = status(tracker)

    if !is_tracking(state.code)
        tracking_stopped!(endgame_tracker)
        return state.code
    end

    # Keep track on whether we attempted to jump to 0 and it failed.
    # This is indication that we have a solution on a reduced positive dimensional component
    # at the end of the path
    state.jump_to_zero_failed = (last(state.jump_to_zero_failed), is_jump_to_zero)

    # check if we can start the endgame
    t = real(tracker.state.t)
    t ≤ options.endgame_start || return state.code

    if state.steps_eg == 0
        state.ext_steps_eg_start = ext_steps(tracker.state)
    end
    state.steps_eg += 1

    # continue if we didn't make progress
    if !step_success
        return state.code
    end

    # Update the valuation to get more information about the path
    update!(state.val, tracker.predictor, t)
    debug && println(state.val)
    if check_finite!(state, options)
        switch_to_singular!(state, tracker; debug = debug)
        return state.code
    elseif check_at_infinity!(state, tracker, options; debug = debug)
        return state.code
    end

    return state.code
end


function check_finite!(state, options)
    is_finite(
        state.val;
        finite_tol = options.val_finite_tol,
        zero_is_finite = !options.zero_is_at_infinity,
        max_winding_number = options.max_winding_number,
    ) || return false

    m, merr =
        estimate_winding_number(state.val; max_winding_number = options.max_winding_number)
    if merr < options.val_finite_tol
        if m == 1 && !first(state.jump_to_zero_failed)
            return false
        else
            state.winding_number = m
            return true
        end
    else
        return false
    end
end

function check_at_infinity!(state, tracker, options; debug::Bool = false)
    options.at_infinity_check || return false
    # compute new tolerance
    at_infinity_tol!(
        state.at_infinity_tols,
        state.val;
        finite_tol = options.val_finite_tol,
        zero_is_finite = !options.zero_is_at_infinity,
    )
    κ = NaN
    t = real(tracker.state.t)
    debug && @show state.at_infinity_tols
    for (i, tolᵢ) in enumerate(state.at_infinity_tols)
        if tolᵢ < options.val_at_infinity_tol
            # Check if this is the first time the coordinate indicates going to infinity
            if isnan(state.at_infinity_starts[i])
                # check if we need to compute a row and column scaling
                # this is the case if we have no coordinate marked
                if all(isnan, state.at_infinity_starts)
                    state.col_scaling .= weights(tracker.state.norm)
                    row_scaling!(
                        state.row_scaling,
                        workspace(tracker.state.jacobian),
                        state.col_scaling,
                    )
                end
                κ = LA.cond(tracker.state.jacobian, state.row_scaling, state.col_scaling)
                state.at_infinity_conds[i] = κ
                state.at_infinity_abs_coords[i] = fast_abs(tracker.state.x[i])
                state.at_infinity_starts[i] = t

                # The coordinate is already marked as going to infinity, check
                # whether other criteria are also sufficient
            else
                if isnan(κ)
                    κ = LA.cond(
                        tracker.state.jacobian,
                        state.row_scaling,
                        state.col_scaling,
                    )
                end
                vᵢ = state.val.val_x[i]
                at_zero = state.val.val_x[i] > 0
                cond_growth = κ / state.at_infinity_conds[i]
                if at_zero
                    coord_growth =
                        state.at_infinity_abs_coords[i] / fast_abs(tracker.state.x[i])
                else
                    coord_growth =
                        fast_abs(tracker.state.x[i]) / state.at_infinity_abs_coords[i]
                end
                if debug
                    δt = t / state.at_infinity_starts[i]
                    @show i
                    @show coord_growth, 0.25^(4vᵢ)
                    @show κ, cond_growth, δt, log(cond_growth / δt)
                end

                if coord_growth > clamp(0.25^(4vᵢ), 20, options.min_coord_growth) &&
                   (cond_growth > options.min_cond_growth || κ > max(1e8, options.min_cond))
                    if at_zero
                        state.code = EndgameTrackerCode.at_zero
                    else
                        state.code = EndgameTrackerCode.at_infinity
                    end
                    return true
                end
            end
            # Coordinate was marked as going to infinity but it is no more
        elseif !isnan(state.at_infinity_starts[i])
            state.at_infinity_starts[i] = NaN
        end
    end
    false
end

function switch_to_singular!(state, tracker; debug::Bool)
    state.singular_endgame = true
    t = real(tracker.state.t)
    state.singular_start = t
    debug && printstyled("SWITCH TO SINGULAR"; color = :red, bold = true)

    if all(isone, state.row_scaling)
        state.col_scaling .= weights(tracker.state.norm)
        row_scaling!(
            state.row_scaling,
            workspace(tracker.state.jacobian),
            state.col_scaling,
        )
    end
    add_sample!(state, tracker, t)
    state.singular_steps = 0

    winding_number!(tracker.predictor, state.winding_number)
    state.at_infinity_conds[1] = state.sample_conds[1]

    tracker.state.keep_extended_prec = true
    return state
end
function switch_to_regular!(state, tracker)
    state.singular_endgame = false
    winding_number!(tracker.predictor, 1)
    init!(tracker, 0.0)
    return state
end


function singular_endgame_step!(endgame_tracker::EndgameTracker, debug::Bool = false)
    @unpack tracker, state, options = endgame_tracker

    λ = 0.25
    # move in geometric series forward
    t = real(tracker.state.t)
    init!(tracker, λ * t)
    max_steps = false
    while is_tracking(tracker.state.code)
        step!(tracker, debug)

        if (state.steps_eg += 1) ≥ options.max_endgame_steps
            state.code = EndgameTrackerCode.terminated_max_steps
            max_steps = true
            break
        elseif ext_steps(tracker.state) - state.ext_steps_eg_start >
               options.max_endgame_extended_steps
            state.code = EndgameTrackerCode.terminated_max_extended_steps
            max_steps = true
            break
        end
    end
    state.singular_steps += 1
    debug && @show λ * t
    debug && @show max_steps
    if max_steps
        @goto prediction
    end
    if !is_success(status(tracker))
        state.code = status(tracker)
        tracking_stopped!(endgame_tracker)
        return state.code
    end

    # Update the valuation to get more information about the path
    update!(state.val, tracker.predictor, λ * t)
    debug && println(state.val)
    # check that the valuation still is finite and that the winding number estimate
    # matches
    m̂, m̂_err =
        estimate_winding_number(state.val; max_winding_number = options.max_winding_number)
    if m̂ != state.winding_number || m̂_err > 0.1
        # if !check_finite!(state, options)
        debug && printstyled("SWITCH TO REGULAR"; color = :yellow, bold = true)
        switch_to_regular!(state, tracker)
        return state.code
    end

    add_sample!(state, tracker, λ * t)

    state.singular_steps ≥ 2 || return state.code

    # estimate solution at 0
    acc = predict_endpoint!(state, tracker.state.norm)
    if state.singular_steps == 2
        state.accuracy = acc
        state.solution .= state.prediction
        return state.code
    elseif acc < state.accuracy && state.accuracy > 1e-12
        state.accuracy = acc
        state.solution .= state.prediction
        return state.code
    end

    @label prediction
    m = state.winding_number
    κ = state.sample_conds[3]
    zero_cond = 1 / (m + 1)
    for i = 1:length(state.prediction)
        state.solution[i] = (state.val.val_x[i] < zero_cond) * state.prediction[i]
    end
    κ₀ = LA.cond(tracker, state.solution, 0.0, state.row_scaling, state.col_scaling)
    J₀_norm = inf_norm(workspace(tracker.state.jacobian), state.row_scaling)
    if debug
        @show κ κ₀ state.accuracy nanmax(κ₀, inv(J₀_norm))
    end
    if state.accuracy < options.singular_min_accuracy && (
        (
            (m > 1 && κ > options.min_cond && nanmax(κ₀, inv(J₀_norm)) > κ) ||
            (m == 1 && κ₀ > 1e12)
        ) ||
        # # some solution is better than none
        max_steps ||
        # handle univariate case
        (length(state.solution) == 1 && inv(J₀_norm) < options.min_cond)
    )
        state.cond = max(κ₀, inv(J₀_norm))
        state.singular = true
        return (state.code = EndgameTrackerCode.success)
    elseif !max_steps
        switch_to_regular!(state, tracker)
        return state.code
    end

    return state.code
end

function add_sample!(state, tracker, t)
    tx¹ = tracker.predictor.tx¹
    m = state.winding_number
    s = nthroot(t, m)
    # will transform the samples to s-plane
    μ = m * s^(m - 1)
    κ = LA.cond(tracker.state.jacobian, state.row_scaling, state.col_scaling)
    if state.singular_steps ≤ 2
        tyk = state.samples[state.singular_steps+1]
        for i = 1:length(tx¹)
            tyk[i, 1] = tx¹[i, 1]
            tyk[i, 2] = μ * tx¹[i, 2]
        end
        state.sample_times[state.singular_steps+1] = s
        state.sample_conds[state.singular_steps+1] = κ
    else
        # need to shift previous solutions to the left
        ty3 = state.samples[1]
        state.samples[1] = state.samples[2]
        state.sample_times[1] = state.sample_times[2]
        state.sample_conds[1] = state.sample_conds[2]
        state.samples[2] = state.samples[3]
        state.sample_times[2] = state.sample_times[3]
        state.sample_conds[2] = state.sample_conds[3]
        state.samples[3] = ty3
        for i = 1:length(tx¹)
            ty3[i, 1] = tx¹[i, 1]
            ty3[i, 2] = μ * tx¹[i, 2]
        end
        state.sample_times[3] = s
        state.sample_conds[3] = κ
    end
end

function predict_endpoint!(state, norm)
    state.singular_steps ≥ 2 || return Inf
    if state.singular_steps == 2
        cubic_hermite!(
            state.prediction,
            state.samples[1],
            state.sample_times[1],
            state.samples[2],
            state.sample_times[2],
            0.0,
        )
    end
    state.prev_prediction .= state.prediction
    cubic_hermite!(
        state.prediction,
        state.samples[2],
        state.sample_times[2],
        state.samples[3],
        state.sample_times[3],
        0.0,
    )
    p = state.sample_times[3] / state.sample_times[2]
    # Use error estimate from MSW92 (7)
    err = InfNorm()(state.prediction, state.prev_prediction) / abs((p^2)^2 - 1)
    norm_s = InfNorm()(state.prediction)
    if norm_s > 1e-8
        err /= norm_s
    end
    return err
end

function tracking_stopped!(endgame_tracker::EndgameTracker)
    @unpack tracker, state, options = endgame_tracker

    state.accuracy = tracker.state.accuracy
    if is_success(state.code) && state.accuracy > 1e-14
        refine_current_solution!(tracker; min_tol = 1e-14)
    end
    state.solution .= tracker.state.x
    state.winding_number = nothing
    # only update condition number for successfull paths
    if is_success(state.code)
        state.col_scaling .= weights(tracker.state.norm)
        row_scaling!(
            state.row_scaling,
            workspace(tracker.state.jacobian),
            state.col_scaling,
        )
        state.cond =
            LA.cond(tracker, state.solution, 0.0, state.row_scaling, state.col_scaling)
        state.singular = state.cond > 1e13 || state.accuracy > 1e-12
    end
end

#
# """
#     CauchyEndgameResult
#
# An enum indicating the result of the [`cauchy!`](@ref) computation.
#
# # Cases
# * `CAUCHY_SUCCESS`: The endgame was successfull.
# * `CAUCHY_TERMINATED_MAX_WINDING_NUMBER`: The endgame was terminated since the winding
#   number is larger than the provided threshold.
# * `CAUCHY_TERMINATED`: The endgame was terminated due to some other error in the path
#   tracking.
# """
# @enum CauchyEndgameResult begin
#     CAUCHY_SUCCESS
#     CAUCHY_TERMINATED_MAX_WINDING_NUMBER
#     CAUCHY_TERMINATED
# end
#
# """
#     cauchy!(state::EndgameTrackerState, tracker::Tracker, options::EndgameOptions)
#
# Try to predict the value of `x(0)` using the [`CauchyEndgame`](@ref).
# For this we track the polygon defined by ``te^{i2πk/n}`` until we end again at ``x``.
# Here ``n`` is the number of samples we take per loop, `samples_per_loop`.
# The computation gives up if we have a winding number larger than `max_winding_number`.
# It returns a tuple denoting the success ([`CauchyEndgameResult`](@ref)) the computed
# winding number `m::Int` and th expected accuracy of the solution.
#
# [Cauchy's integral formula]: https://en.wikipedia.org/wiki/Cauchy%27s_integral_formula
# """
# function cauchy!(state::EndgameTrackerState, tracker::Tracker, options::EndgameOptions)
#     @unpack last_point, prediction = state
#
#     # winding_number!(tracker.predictor, 1)
#
#     t = real(tracker.state.t)
#     # Mathemtically, we only need `n₀ = ceil(Int, log(eps()) / log(t))` many sample points
#     # to achieve the maximal accuracy since the error is ≈ t^n₀.
#     # However, we have to be careful that we are not getting too close to the singularity
#     # during tracking. E.g. with `n₀ = 3` we track fairly close to the origin for the
#     # first first and third path. So we require at least 8 sample points.
#     n₀ = max(ceil(Int, log(eps()) / log(t)), 8)
#     @unpack x, μ, ω = tracker.state
#
#     # # always use extended precision for cauchy endgame
#     prediction_acc = refine_current_solution!(tracker)
#     # fix tracker to not flip between extended precision and and mach. precision
#     tracker.state.keep_extended_prec = true
#     # disallow hermite predictor
#     tracker.predictor.use_hermite = false
#     tracker.predictor.branch = 0
#
#     state.last_point .= tracker.state.x
#     state.last_t = tracker.state.t
#     prediction .= 0.0
#     sample_point_acc = Inf
#     m = 1
#     Δθ = 2π / n₀
#     result = CAUCHY_TERMINATED_MAX_WINDING_NUMBER
#     while m ≤ options.max_winding_number
#         θⱼ = 0.0
#         tⱼ = complex(t, 0.0)
#         for j = 1:n₀
#             θⱼ += Δθ
#             if j == n₀
#                 # tracker.predictor.branch += 1
#                 tⱼ = complex(t, 0.0)
#             else
#                 tⱼ = t * cis(θⱼ)
#             end
#             # @show tⱼ
#             # @show t_to_s_plane(tⱼ, 3; branch = tracker.predictor.branch)
#             res = track!(tracker, tⱼ; debug = true)
#             sample_point_acc = tracker.state.accuracy
#             prediction_acc = max(prediction_acc, sample_point_acc)
#
#             if !is_success(res)
#                 result = CAUCHY_TERMINATED
#                 @goto _return
#             end
#             prediction .+= x
#         end
#         # Check that loop is closed
#         d = tracker.state.norm(last_point, x)
#         if d < 100 * max(prediction_acc, sample_point_acc)
#             n = n₀ * m
#             prediction .= prediction ./ n
#
#             result = CAUCHY_SUCCESS
#             break
#         end
#         m += 1
#     end
#
#
#     @label _return
#
#     init!(tracker, last_point, t, 0.0; ω = ω, μ = μ, keep_steps = true)
#
#     result, m, 100prediction_acc
# end

"""
    start_parameters!(tracker::EndgameTracker, p)

Set the start parameters of the homotopy of the tracker.
"""
start_parameters!(T::EndgameTracker, p) = (start_parameters!(T.tracker, p); T)

"""
    target_parameters!(tracker::EndgameTracker, p)

Set the target parameters of the homotopy of the tracker.
"""
target_parameters!(T::EndgameTracker, p) = (target_parameters!(T.tracker, p); T)

parameters!(T::EndgameTracker, p, q) = (parameters!(T.tracker, p, q); T)

function solution(endgame_tracker::EndgameTracker)
    get_solution(endgame_tracker.tracker.homotopy, endgame_tracker.state.solution, 0.0)
end


function PathResult(
    endgame_tracker::EndgameTracker,
    start_solution = nothing,
    path_number = nothing,
)
    @unpack tracker, state, options = endgame_tracker
    H = tracker.homotopy

    if is_success(state.code)
        t = 0.0
        solution = get_solution(H, state.solution, 0.0)
        evaluate!(tracker.corrector.r, H, state.solution, complex(0.0))
        residual = LA.norm(tracker.corrector.r, InfNorm())
    else
        t = real(tracker.state.t)
        solution = get_solution(H, tracker.state.x, t)
        evaluate!(tracker.corrector.r, H, tracker.state.x, complex(t))
        residual = LA.norm(tracker.corrector.r, InfNorm())
    end

    PathResult(
        return_code = Symbol(state.code),
        solution = solution,
        t = t,
        singular = state.singular,
        accuracy = state.accuracy,
        residual = residual,
        condition_jacobian = state.cond,
        winding_number = state.winding_number,
        last_path_point = (get_solution(H, state.last_point, state.last_t), state.last_t),
        valuation = t > options.endgame_start ? nothing : copy(state.val.val_x),
        start_solution = start_solution,
        path_number = path_number,
        ω = tracker.state.ω,
        μ = tracker.state.μ,
        extended_precision = tracker.state.extended_prec,
        accepted_steps = tracker.state.accepted_steps,
        rejected_steps = tracker.state.rejected_steps,
        steps_eg = state.steps_eg,
        extended_precision_used = tracker.state.used_extended_prec,
    )
end

"""
    track(endgame_tracker::EndgameTracker, x::AbstractVector, t::Real = 1.0;
          path_number = nothing, debug = false)

Track the given start solution `x` from `t` towards `0` using the given `endgame_tracker`.
Returns a [`PathResult`](@ref).

    track(endgame_tracker::EndgameTracker, r::PathResult, t::Real = 1.0;
          path_number = nothing, debug = false)

Track `solution(r)` from `t` towards `0` using the given `endgame_tracker`.
"""
function track(
    endgame_tracker::EndgameTracker,
    x,
    t₁::Real = 1.0;
    path_number::Union{Nothing,Int} = nothing,
    kwargs...,
)
    track!(endgame_tracker, x, t₁; kwargs...)
    PathResult(endgame_tracker, x, path_number)
end
function track(endgame_tracker::EndgameTracker, r::PathResult, t₁::Real; kwargs...)
    track(endgame_tracker, solution(r), t₁; ω = r.ω, μ = r.μ, kwargs...)
end
