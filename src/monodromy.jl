export monodromy_solve,
    find_start_pair,
    MonodromyOptions,
    MonodromyResult,
    is_success,
    is_heuristic_stop,
    parameters,
    permutations,
    trace
# verify_solution_completeness,
# solution_completeness_witnesses,

#####################
# Monodromy Options #
#####################


## Options
"""
    MonodromyOptions(; options...)

Options for [`monodromy_solve`](@ref).
"""
Base.@kwdef struct MonodromyOptions{D,F1,GA<:Union{Nothing,GroupActions},F2}
    check_startsolutions::Bool = true
    distance::D = EuclideanNorm()
    group_actions::GA = nothing
    loop_finished_callback::F1 = always_false
    parameter_sampler::F2 = independent_normal
    equivalence_classes::Bool = !isnothing(group_actions)
    # stopping heuristic
    trace_test_tol::Float64 = 1e-6
    trace_test::Bool = true
    target_solutions_count::Union{Nothing,Int} = nothing
    timeout::Union{Nothing,Float64} = nothing
    min_solutions::Union{Nothing,Int} = nothing
    max_loops_no_progress::Int = 5
    reuse_loops::Symbol = :all
    permutations::Bool = false
end

"""
    independent_normal(p::AbstractVector)

Sample a vector where each entry is drawn independently from the complex Normal distribution
by calling [`randn(ComplexF64)`](@ref).

    independent_normal(L::LinearSubspace)

Creates a random linear subspace by calling [`rand_subspace`](@ref).
"""
independent_normal(p::AbstractVector{T}) where {T} = randn(ComplexF64, length(p))
independent_normal(L::LinearSubspace) = rand_subspace(ambient_dim(L); dim = dim(L))

#############################
# Loops and Data Structures #
#############################

struct LoopTrackingJob
    id::Int
    loop_id::Int
end



#######################
# Loop data structure
#######################


struct MonodromyLoop{P<:Union{LinearSubspace{ComplexF64},Vector{ComplexF64}}}
    # p -> p₁ -> p₂ -> p
    p::P
    p₀₁::P # halfway
    p₁::P
    p₂::P
end

function MonodromyLoop(base::AbstractVector, options::MonodromyOptions)
    p = convert(Vector{ComplexF64}, base)
    p₁ = convert(Vector{ComplexF64}, options.parameter_sampler(p))
    p₂ = convert(Vector{ComplexF64}, options.parameter_sampler(p))

    MonodromyLoop(p, 0.5 .* (p₁ .- p), p₁, p₂)
end

function MonodromyLoop(base::LinearSubspace, options::MonodromyOptions)
    L = convert(LinearSubspace{ComplexF64}, base)
    # the second linear space is just a translation in order to perform a trace test
    # To still find new solutions quickly we translate the linear space
    # by a larger distance
    v = LA.rmul!(LA.normalize!(randn(ComplexF64, codim(L))), 5)
    L₀₁ = translate(L, v, Extrinsic)
    L₁ = translate(L₀₁, v, Extrinsic)
    L₂ = convert(LinearSubspace{ComplexF64}, options.parameter_sampler(L))

    MonodromyLoop(L, L₀₁, L₁, L₂)
end

##########################
## Monodromy Statistics ##
##########################

Base.@kwdef mutable struct MonodromyStatistics
    tracked_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    tracking_failures::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    generated_loops::Threads.Atomic{Int} = Threads.Atomic{Int}(0)
    solutions::Vector{Int} = Int[]
    permutations::Vector{Vector{Int}} = Vector{Int}[]
end

MonodromyStatistics(nsolutions::Int) =
    MonodromyStatistics(solutions_development = [nsolutions])


function Base.show(io::IO, S::MonodromyStatistics)
    println(io, "MonodromyStatistics")
    print_fieldname(io, S, :tracked_loops)
    print_fieldname(io, S, :tracking_failures)
    print_fieldname(io, S, :solutions)
end
Base.show(io::IO, ::MIME"application/prs.juno.inline", S::MonodromyStatistics) = S

# update routines
function loop_tracked!(stats::MonodromyStatistics)
    Threads.atomic_add!(stats.tracked_loops, 1)
    stats
end
function loop_failed!(stats::MonodromyStatistics)
    Threads.atomic_add!(stats.tracking_failures, 1)
    stats
end
function loop_finished!(stats, nsolutions)
    push!(stats.solutions, nsolutions)
end

function add_permutation!(stats::MonodromyStatistics, loop_id, start_id, end_id)
    perms = stats.permutations[loop_id]
    # ensure that perms has correct length
    while length(perms) < start_id
        push!(perms, 0)
    end
    perms[start_id] = end_id
    stats
end

function solutions_current_loop(stats::MonodromyStatistics, nsolutions)
    nsolutions - stats.solutions[end]
end
function solutions_last_loop(stats::MonodromyStatistics)
    length(stats.solutions) > 1 ? stats.solutions[end] - stats.solutions[end-1] : 0
end

function loops_no_change(stats::MonodromyStatistics, nsolutions)
    k = 0
    for i = length(stats.solutions):-1:1
        stats.solutions[i] == nsolutions || break
        k += 1
    end
    return max(k - 1, 0)
end

@noinline function make_showvalues(stats::MonodromyStatistics; queued::Int, solutions::Int)
    [
        ("tracked loops (queued)", "$(stats.tracked_loops[]) ($queued)"),
        (
            "solutions in current (last) loop",
            "$(solutions_current_loop(stats, solutions)) ($(solutions_last_loop(stats)))",
        ),
        (
            "generated loops (no change)",
            "$(stats.generated_loops[]) ($(loops_no_change(stats, solutions)))",
        ),
    ]
end



############
## Result ##
############

"""
    MonodromyResult

The monodromy result contains the result of the [`monodromy_solve`](@ref) computation.
"""
struct MonodromyResult{P,LP}
    returncode::Symbol
    results::Vector{PathResult}
    parameters::P
    loops::Vector{MonodromyLoop{LP}}
    statistics::MonodromyStatistics
    equivalence_classes::Bool
    seed::UInt32
    trace::Union{Nothing,Float64}
end

Base.show(io::IO, ::MIME"application/prs.juno.inline", x::MonodromyResult) = x
function Base.show(io::IO, result::MonodromyResult{T}) where {T}
    println(io, "MonodromyResult")
    println(io, "="^length("MonodromyResult"))
    println(io, "• return_code → :$(result.returncode)")
    if result.equivalence_classes
        println(io, "• $(nsolutions(result)) classes of solutions (modulo group action)")
    else
        println(io, "• $(nsolutions(result)) solutions")
    end
    println(io, "• $(result.statistics.tracked_loops[]) tracked loops")
    print(io, "• random_seed → ", sprint(show, result.seed))
    if !isnothing(result.trace)
        print(io, "\n• trace → ", sprint(show, result.trace))
    end
end

TreeViews.hastreeview(::MonodromyResult) = true
TreeViews.numberofnodes(::MonodromyResult) = 7
TreeViews.treelabel(io::IO, x::MonodromyResult, ::MIME"application/prs.juno.inline") =
    print(
        io,
        "<span class=\"syntax--support syntax--type syntax--julia\">MonodromyResult</span>",
    )
#
function TreeViews.nodelabel(
    io::IO,
    x::MonodromyResult,
    i::Int,
    ::MIME"application/prs.juno.inline",
)
    if i == 1
        if x.equivalence_classes
            print(io, "$(nsolutions(x)) solutions (modulo group action)")
        else
            print(io, "$(nsolutions(x)) solutions")
        end
    elseif i == 2
        print(io, "Return code")
    elseif i == 3
        print(io, "Tracked loops")
    elseif i == 4
        print(io, "Statistics")
    elseif i == 5
        print(io, "Parameters")
    elseif i == 6
        print(io, "Seed")
    elseif i == 7
        print(io, "Trace")
    end
end
function TreeViews.treenode(r::MonodromyResult, i::Integer)
    if i == 1
        return r.results
    elseif i == 2
        return r.returncode
    elseif i == 3
        return r.statistics.tracked_loops[]
    elseif i == 4
        return r.statistics
    elseif i == 5
        return r.parameters
    elseif i == 6
        return r.seed
    elseif i == 7
        return something(r.trace, missing)
    end
    missing
end

"""
    is_success(result::MonodromyResult)

Returns true if the monodromy computation achieved its target solution count.
"""
is_success(result::MonodromyResult) = result.returncode == :success

"""
    is_heuristic_stop(result::MonodromyResult)

Returns true if the monodromy computation stopped due to the heuristic.
"""
is_heuristic_stop(result::MonodromyResult) = result.returncode == :heuristic_stop

"""
    solutions(result::MonodromyResult)

Return all solutions.
"""
solutions(r::MonodromyResult) = solutions(r.results)

"""
    nsolutions(result::MonodromyResult)

Returns the number solutions of the `result`.
"""
nsolutions(r::MonodromyResult) = length(r.results)

"""
    results(result::MonodromyResult)

Returns the computed [`PathResult`](@ref)s.
"""
results(r::MonodromyResult) = r.results

"""
    nresults(result::MonodromyResult)

Returns the number of results computed.
"""
nresults(r::MonodromyResult) = length(r.results)

"""
    parameters(result::MonodromyResult)

Return the parameters corresponding to the given result `r`.
"""
ModelKit.parameters(r::MonodromyResult) = r.parameters

"""
    seed(result::MonodromyResult)

Return the random seed used for the computations.
"""
seed(r::MonodromyResult) = r.seed

"""
    trace(result::MonodromyResult)

Return the result of the trace test computed during the monodromy.
"""
trace(r::MonodromyResult) = r.trace


"""
    permutations(r::MonodromyResult; reduced=true)

Return the permutations of the solutions that are induced by tracking over the loops.
If `reduced=false`, then all permutations are returned.
If `reduced=true` then permutations without repetitions are returned.

If a solution was not tracked in the loop, then the corresponding entry is 0.

Example: monodromy loop for a varying line that intersects two circles.
```julia
using LinearAlgebra
@var x[1:2] a b c
c1 = (x - [2, 0]) ⋅ (x - [2, 0]) - 1
c2 = (x - [-2, 0]) ⋅ (x - [-2, 0]) - 1
F = [c1 * c2; a * x[1] + b * x[2] - c]
S = monodromy_solve(F, [[1, 0]], [1, 1, 1], parameters = [a, b, c], permutations = true)

permutations(S)
```

will return

```julia
2×2 Array{Int64,2}:
 1  2
 2  1
```

and `permutations(S, reduced = false)` returns

```julia
2×12 Array{Int64,2}:
 1  2  2  1  1  …  1  2  1  1  1
 2  1  1  2  2     2  1  2  2  2
```

"""
function permutations(r::MonodromyResult; reduced::Bool = true)
    π = r.statistics.permutations
    if reduced
        π = unique(π)
    end
    N = nresults(r)
    A = zeros(Int, N, length(π))
    for (j, πⱼ) in enumerate(π), i = 1:N
        A[i, j] = πⱼ[i]
    end

    A
end

#####################
## monodromy solve ##
#####################

mutable struct MonodromySolver{
    Tracker<:EndgameTracker,
    P,
    UP<:UniquePoints,
    MO<:MonodromyOptions,
}
    trackers::Vector{Tracker}
    loops::Vector{MonodromyLoop{P}}
    unique_points::UP
    unique_points_lock::ReentrantLock
    options::MO
    statistics::MonodromyStatistics
    trace::Vector{ComplexF64}
    trace_lock::ReentrantLock
end

function MonodromySolver(
    F::Union{System,AbstractSystem},
    parameters;
    intrinsic = nothing,
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    tracker_options = TrackerOptions(),
    options = MonodromyOptions(),
)
    if !isa(options, MonodromyOptions)
        options = MonodromyOptions(; options...)
    end

    if parameters isa LinearSubspace
        H = linear_subspace_homotopy(
            F,
            parameters,
            parameters;
            compile = compile,
            intrinsic = intrinsic,
        )
        P = LinearSubspace{ComplexF64}
    else
        H = parameter_homotopy(F; generic_parameters = parameters, compile = compile)
        P = Vector{ComplexF64}
    end
    egtracker = EndgameTracker(
        H;
        tracker_options = tracker_options,
        options = EndgameOptions(endgame_start = 0.0),
    )
    trackers = [egtracker]
    group_actions = options.equivalence_classes ? options.group_actions : nothing
    x₀ = zeros(ComplexF64, size(F, 2))
    #TODO: This is wrong for IntrinsicSubspaceHomotopy of projective varieties
    if !isnothing(group_actions) && (egtracker.tracker.homotopy isa AffineChartHomotopy)
        group_actions = let H = egtracker.tracker.homotopy, group_actions = group_actions
            (cb, s) -> apply_actions(v -> cb(set_solution!(v, H, v, 0)), group_actions, s)
        end
    end

    unique_points =
        UniquePoints(x₀, 1; metric = options.distance, group_actions = group_actions)
    trace = zeros(ComplexF64, length(x₀))
    MonodromySolver(
        trackers,
        MonodromyLoop{P}[],
        unique_points,
        ReentrantLock(),
        options,
        MonodromyStatistics(),
        trace,
        ReentrantLock(),
    )
end

function add_loop!(MS::MonodromySolver, p)
    push!(MS.loops, MonodromyLoop(p, MS.options))
    Threads.atomic_add!(MS.statistics.generated_loops, 1)
    if MS.options.permutations
        push!(MS.statistics.permutations, zeros(Int, length(MS.unique_points)))
    end
    MS
end
loop(MS::MonodromySolver, i) = MS.loops[i]
nloops(MS::MonodromySolver) = length(MS.loops)

"""
    monodromy_solve(F, [sols, p]; options..., tracker_options = TrackerOptions())

Solve a polynomial system `F(x;p)` with specified parameters and initial solutions `sols`
by monodromy techniques. This makes loops in the parameter space of `F` to find new solutions.
If `F` the parameters `p` only occur *linearly* in `F` it is eventually possible to compute
a *start pair* ``(x₀, p₀)`` automatically. In this case `sols` and `p` can be omitted and
the automatically generated parameters can be obtained with the [`parameters`](@ref) function
from the [`MonodromyResult`](@ref).

    monodromy_solve(F, [sols, L]; dim, codim, intrinsic = nothing, options...,
                                  tracker_options = TrackerOptions())

Solve the polynomial system `[F(x); L(x)] = 0` where `L` is a `[LinearSubspace`](@ref).
If `sols` and `L` are not provided it is necesary to provide `dim` or `codim` where `(co)dim`
is the expected (co)dimension of a component of `V(F)`.
See also [`linear_subspace_homotopy`](@ref) for the `intrinsic` option.

## Options

* `catch_interrupt = true`: If true catches interruptions (e.g. issued by pressing Ctrl-C)
  and returns the partial result.
* `check_startsolutions = true`: If `true`, we do a Newton step for each entry of `sols`for
  checking if it is a valid startsolutions. Solutions which are not valid are sorted out.
* `compile = $(COMPILE_DEFAULT[])`: If `true` then a `System` (resp. `Homotopy`) is compiled
  to a straight line program ([`CompiledSystem`](@ref) resp. [`CompiledHomotopy`](@ref))
  for evaluation. This induces a compilation overhead. If `false` then the generated program
  is only interpreted ([`InterpretedSystem`](@ref) resp. [`InterpretedHomotopy`](@ref)).
  This is slower than the compiled version, but does not introduce compilation overhead.
* `distance = EuclideanNorm()`: The distance function used for [`UniquePoints`](@ref).
* `loop_finished_callback = always_false`: A callback to end the computation. This function is
  called with all current [`PathResult`](@ref)s after a loop is exhausted.
  2 arguments. Return `true` if the compuation should be stopped.
* `equivalence_classes=true`: This only applies if there is at least one group action
  supplied. We then consider two solutions in the same equivalence class if we can transform
  one to the other by the supplied group actions. We only track one solution per equivalence
  class.
* `group_action = nothing`: A function taking one solution and returning other solutions if
  there is a constructive way to obtain them, e.g. by symmetry.
* `group_actions = nothing`: If there is more than one group action you can use this to chain
  the application of them. For example if you have two group actions `foo` and `bar` you can
  set `group_actions=[foo, bar]`. See [`GroupActions`](@ref) for details regarding the
  application rules.
* `max_loops_no_progress = 5`: The maximal number of iterations (i.e. loops generated)
  without any progress.
* `min_solutions`: The minimal number of solutions before a stopping heuristic is applied.
  By default no lower limit is enforced.
* `parameter_sampler = independent_normal`: A function taking the parameter `p` and
  returning a new random parameter `q`. By default each entry of the parameter vector
  is drawn independently from Normal distribution.
* `permutations = false`: Whether to keep track of the permutations induced by the loops.
* `resuse_loops = :all`: Strategy to reuse other loops for new found solutions. `:all`
  propagates a new solution through all other loops, `:random` picks a random loop, `:none`
  doesn't reuse a loop.
* `target_solutions_count`: The computation is stopped if this number of solutions is
  reached.
* `threading = true`: Enable multithreading of the path tracking.
* `timeout`: The maximal number of *seconds* the computation is allowed to run.
* `trace_test = true`: If `true` a trace test is performed to check whether all solutions
  are found. This is only applicable if monodromy is performed with a linear subspace.
  See also [`trace_test`](@ref).
* `trace_test_tol = 1e-10`: The tolerance for the trace test to be successfull.
  The trace is divided by the number of solutions before compared to the trace_test_tol.

"""
function monodromy_solve(F::Vector{<:ModelKit.Expression}, args...; parameters, kwargs...)
    monodromy_solve(System(F; parameters = parameters), args...; kwargs...)
end
function monodromy_solve(
    F::AbstractVector{<:MP.AbstractPolynomial},
    args...;
    parameters,
    variables = setdiff(MP.variables(F), parameters),
    variable_ordering = variables,
    kwargs...,
)
    sys = System(F, variables = variable_ordering, parameters = parameters)
    monodromy_solve(sys, args...; kwargs...)
end
function monodromy_solve(
    F::Union{System,AbstractSystem},
    args...;
    seed = rand(UInt32),
    tracker_options = TrackerOptions(),
    show_progress::Bool = true,
    threading::Bool = Threads.nthreads() > 1,
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    catch_interrupt::Bool = true,
    dim = nothing,
    codim = nothing,
    intrinsic = nothing,
    # monodromy options
    options = nothing,
    group_action = nothing,
    group_actions = isnothing(group_action) ? nothing : GroupActions(group_action),
    kwargs...,
)
    if isnothing(options)
        if !(group_actions isa GroupActions) && !isnothing(group_actions)
            group_actions = GroupActions(group_actions)
        end
        options = MonodromyOptions(; group_actions = group_actions, kwargs...)
    end

    if !isnothing(seed)
        Random.seed!(seed)
    end
    if length(args) == 0
        start_pair = find_start_pair(fixed(F; compile = compile))
        if isnothing(start_pair)
            error(
                "Cannot compute a start pair (x, p) using `find_start_pair(F)`." *
                " You need to explicitly pass a start pair.",
            )
        end
        x, p = start_pair
        S = [x]
        # if we have no parameters, then we intersect with a linear subspace.
        # For this we need to know the intended dimension or codimension.
        # We don't guess dim to catch the case that the user just forgot to pass
        # parameters
        if isnothing(p)
            if isnothing(dim) && isnothing(codim)
                error(
                    "Given system doesn't have any parameters. If you intended to intersect " *
                    "with a linear subspace it is necessary to provide a " *
                    "dimension (`dim`) or codimension (`codim`) of the component of interest.",
                )
            else
                projective = is_homogeneous(System(F))
                if !isnothing(codim)
                    codim += projective
                end
                p = rand_subspace(
                    x;
                    dim = codim,
                    codim = dim,
                    affine = !is_homogeneous(System(F)),
                )
            end
        end
    else
        S, p = args
    end
    MS = MonodromySolver(
        F,
        p;
        compile = compile,
        options = options,
        tracker_options = tracker_options,
        intrinsic = intrinsic,
    )
    monodromy_solve(
        MS,
        S,
        p,
        seed;
        show_progress = show_progress,
        threading = threading,
        catch_interrupt = catch_interrupt,
    )
end

"""
    find_start_pair(F; max_tries = 100, atol = 0.0, rtol = 1e-12)

Try to find a pair `(x,p)` for the system `F` such that `F(x,p) = 0` by randomly sampling
a pair `(x₀, p₀)` and performing Newton's method in variable *and* parameter space.
"""
find_start_pair(F::System; compile::Union{Bool,Symbol} = COMPILE_DEFAULT[], kwargs...) =
    find_start_pair(fixed(F; compile = compile); kwargs...)
function find_start_pair(
    F::AbstractSystem;
    max_tries::Int = 100,
    atol::Float64 = 0.0,
    rtol::Float64 = 1e-12,
)
    if nparameters(F) == 0
        SF = F
    else
        SF = StartPairSystem(F)
    end
    m = nparameters(F)
    n = nvariables(F)
    c = NewtonCache(SF)
    for i = 1:max_tries
        xp₀ = randn(ComplexF64, size(SF, 2))
        res = newton(SF, xp₀, nothing, InfNorm(), c, atol = atol, rtol = rtol)
        if is_success(res)
            x, p = res.x[1:n], res.x[n+1:end]
            res2 = newton(F, x, p, InfNorm())
            if is_success(res2)
                return res2.x, isempty(p) ? nothing : p
            end
        end
    end
    nothing
end

monodromy_solve(MS::MonodromySolver, x::AbstractVector{<:Number}, p, seed; kwargs...) =
    monodromy_solve(MS, [x], p, seed; kwargs...)

function monodromy_solve(
    MS::MonodromySolver,
    X::AbstractVector{<:AbstractVector{<:Number}},
    p,
    seed;
    show_progress::Bool = true,
    threading::Bool = Threads.nthreads() > 1,
    catch_interrupt::Bool = true,
)
    if !show_progress
        progress = nothing
    else
        if MS.options.equivalence_classes
            desc = "Solutions (modulo group action) found:"
        else
            desc = "Solutions found:"
        end
        progress = ProgressMeter.ProgressUnknown(; dt = 0.2, desc = desc, output = stdout)
        progress.tlast += 0.3
    end
    MS.statistics = MonodromyStatistics()
    empty!(MS.unique_points)
    if p isa LinearSubspace
        MS.trace .= 0.0
    else
        MS.trace .= Inf
    end
    results = check_start_solutions(MS, X, p)
    if isempty(results)
        @warn "None of the provided solutions is a valid start solution."
        retcode = :invalid_startvalue
    else
        retcode = :default
        try
            # convert to complex base
            if p isa LinearSubspace
                cp = convert(LinearSubspace{ComplexF64}, p)
            else
                cp = convert(Vector{ComplexF64}, p)
            end
            if threading
                retcode = threaded_monodromy_solve!(results, MS, cp, seed, progress)
            else
                retcode = serial_monodromy_solve!(results, MS, cp, seed, progress)
            end
        catch e
            if !catch_interrupt ||
               !(
                isa(e, InterruptException) ||
                (isa(e, TaskFailedException) && isa(e.task.exception, InterruptException))
            )
                rethrow(e)
            end
            retcode = :interrupted
        end
    end

    MonodromyResult(
        retcode,
        results,
        p,
        MS.loops,
        MS.statistics,
        MS.options.equivalence_classes,
        seed,
        p isa LinearSubspace ? LA.norm(MS.trace, Inf) / length(results) : nothing,
    )
end

function check_start_solutions(MS::MonodromySolver, X, p)
    tracker = MS.trackers[1]
    parameters!(tracker, p, p)
    results = PathResult[]
    for x in X
        res = track(tracker, x)
        if is_success(res)
            _, added = add!(MS, res, length(results) + 1)
            if added
                push!(results, res)
            end
        end
    end

    results
end

function add!(MS::MonodromySolver, res::PathResult, id)
    rtol = clamp(0.25 * inv(res.ω)^2, 1e-14, sqrt(res.accuracy))
    add!(MS.unique_points, solution(res), id; atol = 1e-14, rtol = rtol)
end

function serial_monodromy_solve!(
    results::Vector{PathResult},
    MS::MonodromySolver,
    p,
    seed,
    progress,
)
    queue = LoopTrackingJob[]
    tracker = MS.trackers[1]
    t₀ = time()
    retcode = :in_progress
    stats = MS.statistics
    while true
        loop_finished!(stats, length(results))

        if p isa LinearSubspace &&
           nloops(MS) > 0 &&
           MS.options.trace_test &&
           LA.norm(MS.trace, Inf) < length(results) * MS.options.trace_test_tol
            retcode = :success
            @goto _return
        end
        if isnothing(MS.options.target_solutions_count) &&
           loops_no_change(stats, length(results)) ≥ MS.options.max_loops_no_progress
            retcode = :heuristic_stop
            @goto _return
        end
        if length(results) == something(MS.options.target_solutions_count, -1)
            retcode = :success
            @goto _return
        end

        add_loop!(MS, p)
        MS.trace .= 0.0
        # schedule all jobs
        new_loop_id = nloops(MS)
        for i = 1:length(results)
            push!(queue, LoopTrackingJob(i, new_loop_id))
        end

        while !isempty(queue)
            job = popfirst!(queue)
            res = track(
                tracker,
                results[job.id],
                loop(MS, job.loop_id),
                MS.trace,
                MS.trace_lock;
                collect_trace = MS.options.trace_test && nloops(MS) == job.loop_id,
            )
            if !isnothing(res)
                loop_tracked!(stats)

                # 1) check whether solutions already exists
                id, got_added = add!(
                    MS.unique_points,
                    solution(res),
                    length(results) + 1;
                    atol = 1e-14,
                    rtol = uniqueness_rtol(res),
                )

                if MS.options.permutations
                    add_permutation!(stats, job.loop_id, job.id, id)
                end

                got_added || @goto _update
                # 2) doesn't exist, so add to results
                push!(results, res)

                # 3) schedule on same loop again
                push!(queue, LoopTrackingJob(id, job.loop_id))

                # 4) schedule on other loops
                if MS.options.reuse_loops == :all
                    for k = 1:nloops(MS)
                        k != job.loop_id || continue
                        push!(queue, LoopTrackingJob(id, k))
                    end
                elseif MS.options.reuse_loops == :random && nloops(MS) ≥ 2
                    k = rand(2:nloops(MS))
                    if k ≤ job.loop_id
                        k -= 1
                    end
                    push!(queue, LoopTrackingJob(id, k))
                end
            else
                loop_failed!(stats)
                if MS.options.permutations
                    add_permutation!(stats, job.loop_id, job.id, 0)
                end
            end
            # Update progress
            @label _update
            update_progress!(
                progress,
                stats;
                solutions = length(results),
                queued = length(queue),
            )

            if length(results) == something(MS.options.target_solutions_count, -1) &&
               # only terminate after a completed loop to ensure that we collect proper
               # permutation informations
               !MS.options.permutations
                retcode = :success
                @goto _return
            elseif !isnothing(MS.options.timeout) &&
                   time() - t₀ > (MS.options.timeout::Float64)
                retcode = :timeout
                @goto _return
            end
        end
    end

    @label _return

    update_progress!(
        progress,
        stats;
        finish = true,
        solutions = length(results),
        queued = length(queue),
    )

    return retcode
end

function threaded_monodromy_solve!(
    results::Vector{PathResult},
    MS::MonodromySolver,
    p,
    seed,
    progress,
)
    queue = Channel{LoopTrackingJob}(Inf)
    Threads.resize_nthreads!(MS.trackers)
    data_lock = ReentrantLock()
    t₀ = time()
    retcode = :in_progress
    stats = MS.statistics
    notify_lock = ReentrantLock()
    progress_lock = ReentrantLock()
    cond_queue_emptied = Threads.Condition(notify_lock)
    # make sure to avoid false sharing
    workers_idle = fill(true, 64 * Threads.nthreads())
    interrupted = Ref(false)
    queued = Ref(0)
    try
        workers = map(enumerate(MS.trackers)) do (tid, tracker)
            @tspawnat tid begin
                for job in queue
                    workers_idle[64*(tid-1)+1] = false
                    res = track(
                        tracker,
                        results[job.id],
                        loop(MS, job.loop_id),
                        MS.trace,
                        MS.trace_lock;
                        collect_trace = MS.options.trace_test && nloops(MS) == job.loop_id,
                    )
                    # @show tid, job
                    if !isnothing(res)
                        loop_tracked!(stats)

                        # 1) check whether solutions already exists
                        lock(data_lock)
                        id, got_added = add!(
                            MS.unique_points,
                            solution(res),
                            length(results) + 1;
                            atol = 1e-14,
                            rtol = uniqueness_rtol(res),
                        )

                        if MS.options.permutations
                            add_permutation!(stats, job.loop_id, job.id, id)
                        end

                        if !got_added
                            unlock(data_lock)
                            @goto _update
                        end
                        # 2) doesn't exist, so add to results
                        push!(results, res)
                        unlock(data_lock)

                        # 3) schedule on same loop again
                        push!(queue, LoopTrackingJob(id, job.loop_id))

                        # 4) schedule on other loops
                        if MS.options.reuse_loops == :all
                            for k = 1:nloops(MS)
                                k != job.loop_id || continue
                                push!(queue, LoopTrackingJob(id, k))
                            end
                        elseif MS.options.reuse_loops == :random && nloops(MS) ≥ 2
                            k = rand(2:nloops(MS))
                            if k ≤ job.loop_id
                                k -= 1
                            end
                            push!(queue, LoopTrackingJob(id, k))
                        end
                    else
                        loop_failed!(stats)
                        if MS.options.permutations
                            add_permutation!(stats, job.loop_id, job.id, 0)
                        end
                    end
                    # Update progress
                    @label _update
                    update_progress!(
                        progress,
                        stats;
                        solutions = length(results),
                        queued = Base.n_avail(queue),
                    )

                    # mark worker as idle
                    workers_idle[64*(tid-1)+1] = true

                    # if queue is empty, check whether all other are also waiting
                    if !isready(queue) && all(workers_idle)
                        lock(notify_lock)
                        notify(cond_queue_emptied)
                        unlock(notify_lock)
                    end

                    if length(results) ==
                       something(MS.options.target_solutions_count, -1) &&
                       # only terminate after a completed loop to ensure that we collect proper
                       # permutation informations
                       !MS.options.permutations
                        retcode = :success
                        lock(notify_lock)
                        notify(cond_queue_emptied)
                        unlock(notify_lock)
                    elseif !isnothing(MS.options.timeout) &&
                           time() - t₀ > (MS.options.timeout::Float64)
                        retcode = :timeout
                        lock(notify_lock)
                        notify(cond_queue_emptied)
                        unlock(notify_lock)
                    end
                end
            end
        end

        t = Threads.@spawn while !interrupted[]
            loop_finished!(stats, length(results))

            if loops_no_change(stats, length(results)) ≥ MS.options.max_loops_no_progress
                retcode = :heuristic_stop
                break
            end
            if length(results) == something(MS.options.target_solutions_count, -1)
                retcode = :success
                break
            end
            if p isa LinearSubspace &&
               nloops(MS) > 0 &&
               MS.options.trace_test &&
               LA.norm(MS.trace, Inf) < length(results) * MS.options.trace_test_tol
                retcode = :success
                break
            end

            add_loop!(MS, p)
            MS.trace .= 0.0
            # schedule all jobs
            new_loop_id = nloops(MS)
            for i = 1:length(results)
                push!(queue, LoopTrackingJob(i, new_loop_id))
            end

            lock(notify_lock)
            wait(cond_queue_emptied)
            unlock(notify_lock)

            retcode == :in_progress || break
        end

        wait(t)
    catch e
        close(queue)
        interrupted[] = true
        rethrow(e)
    finally
        queued[] = Base.n_avail(queue)
        close(queue)
    end

    update_progress!(
        progress,
        stats;
        finish = true,
        solutions = length(results),
        queued = queued[],
    )

    return retcode
end

function uniqueness_rtol(res::PathResult)
    # we should have a region of uniqueness of 0.5inv(ω) * norm(x)
    # the norm(x) factors comes from the fact that we computed with a
    # weighted norm.
    # To be more pessimistic we only consider 0.25inv(ω)^2
    # and require at most sqrt(res.accuracy) and at least 1e-14.
    clamp(0.25 * inv(res.ω)^2, 1e-14, sqrt(res.accuracy))
end
"""
    track(tracker, x, edge::LoopEdge, loop::MonodromyLoop, stats::MonodromyStatistics)

Track `x` along the edge `edge` in the loop `loop` using `tracker`. Record statistics
in `stats`.
"""
function track(
    egtracker::EndgameTracker,
    r::PathResult,
    loop::MonodromyLoop,
    trace,
    trace_lock;
    collect_trace::Bool = false,
)
    tracker = egtracker.tracker

    if loop.p isa LinearSubspace && collect_trace
        parameters!(tracker, loop.p, loop.p₀₁)
        retcode = track!(
            tracker,
            solution(r);
            ω = r.ω,
            μ = r.μ,
            extended_precision = r.extended_precision,
        )
        x₀₁ = solution(tracker)
        if !is_success(retcode)
            return nothing
        end

        parameters!(tracker, loop.p₀₁, loop.p₁)
        retcode = track!(
            tracker,
            x₀₁;
            ω = tracker.state.ω,
            μ = tracker.state.μ,
            extended_precision = tracker.state.extended_prec,
        )
        x₁ = solution(tracker)
        if !is_success(retcode)
            return nothing
        end

        lock(trace_lock)
        trace .+= (x₁ .- x₀₁) .- (x₀₁ .- solution(r))
        unlock(trace_lock)
    else
        parameters!(tracker, loop.p, loop.p₁)
        retcode = track!(
            tracker,
            solution(r);
            ω = r.ω,
            μ = r.μ,
            extended_precision = r.extended_precision,
        )
        if !is_success(retcode)
            return nothing
        end
        x₁ = solution(tracker)
    end

    parameters!(tracker, loop.p₁, loop.p₂)
    retcode = track!(
        tracker,
        x₁;
        ω = tracker.state.ω,
        μ = tracker.state.μ,
        extended_precision = tracker.state.extended_prec,
    )
    if !is_success(retcode)
        return nothing
    end

    x₂ = solution(tracker)
    parameters!(tracker, loop.p₂, loop.p)
    result = track(
        egtracker,
        x₂;
        ω = tracker.state.ω,
        μ = tracker.state.μ,
        extended_precision = tracker.state.extended_prec,
    )
    if is_success(result)
        result
    else
        nothing
    end
end

update_progress!(progress::Nothing, stats::MonodromyStatistics; kwargs...) = nothing
function update_progress!(
    progress,
    stats::MonodromyStatistics;
    queued::Int,
    solutions::Int,
    finish::Bool = false,
)
    if finish
        ProgressMeter.update!(progress, solutions)
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.finish!(progress; showvalues = showvalues)
    elseif time() > progress.tlast + progress.dt
        showvalues = make_showvalues(stats; queued = queued, solutions = solutions)
        ProgressMeter.update!(progress, solutions, showvalues = showvalues)
    end
    # yield()
    nothing
end

# #
# #
# # ##################
# # ## VERIFICATION ##
# # ##################
# # """
# #     verify_solution_completeness(F, res::MonodromyResult;
# #                                  parameters=..., trace_tol=1e-6, options...)
# #
# # Verify that the monodromy computation found all solutions by [`monodromy_solve`](@ref).
# # This uses a trace test as described in [^LRS18].
# # The trace is a numerical value which is 0 if all solutions are found, for this the
# # `trace_tol` keyword argument is used. The function returns `nothing` if some computation
# # couldn't be carried out. Otherwise returns a boolean. Note that this function requires the
# # computation of solutions to another polynomial system using monodromy. This routine can
# # return `false` although all solutions are found if this additional solution set is not
# # complete. The `options...` arguments can be everything which is accepted by `solve` and
# # `monodromy_solve`.
# #
# # ### Example
# #
# # ```
# # julia> @polyvar x y a b c;
# #
# # julia> f = x^2+y^2-1;
# #
# # julia> l = a*x+b*y+c;
# #
# # julia> res = monodromy_solve([f,l], [-0.6-0.8im, -1.2+0.4im], [1,2,3]; parameters=[a,b,c])
# # MonodromyResult
# # ==================================
# # • 2 solutions (0 real)
# # • return code → heuristic_stop
# # • 44 tracked paths
# # • seed → 367230
# #
# # julia> verify_solution_completeness([f,l], res; parameters=[a,b,c], trace_tol = 1e-8)
# # [ Info: Compute additional witnesses for completeness...
# # [ Info: Found 2 additional witnesses
# # [ Info: Compute trace...
# # [ Info: Norm of trace: 1.035918995391323e-15
# # true
# # ```
# #
# #     verify_solution_completeness(F, S, p; parameters=..., kwargs...)
# #
# # Verify the solution completeness using the computed solutions `S` to the parameter `p`.
# #
# #     verify_solution_completeness(TTS, S, W₁₀, p₀::Vector{<:Number}, l₀)
# #
# # Use the already computeded additional witnesses `W₁₀`. You want to obtain
# # `TTS`, `W₁₀` and `l₀` as the output from [`solution_completeness_witnesses`](@ref).
# #
# #
# # [^LRS18]:
# #     Leykin, Anton, Jose Israel Rodriguez, and Frank Sottile. "Trace test."
# #     Arnold Mathematical Journal 4.1 (2018): 113-125.
# # """
# # function verify_solution_completeness(F, R::MonodromyResult; kwargs...)
# #     verify_solution_completeness(F, solutions(R), R.parameters; kwargs...)
# # end
# #
# # function verify_solution_completeness(
# #     F,
# #     W₀₁::AbstractVector{<:AbstractVector},
# #     p₀::Vector{<:Number};
# #     show_progress = true,
# #     parameters = nothing,
# #     trace_tol = 1e-6,
# #     kwargs...,
# # )
# #     W₁₀, TTS, l₀ = solution_completeness_witnesses(
# #         F,
# #         W₀₁,
# #         p₀;
# #         show_progress = show_progress,
# #         parameters = parameters,
# #         kwargs...,
# #     )
# #     verify_solution_completeness(TTS, W₀₁, W₁₀, p₀, l₀; show_progress = show_progress)
# # end
# #
# # function verify_solution_completeness(
# #     TTS,
# #     W₀₁::AbstractVector{<:AbstractVector},
# #     W₁₀::AbstractVector{<:AbstractVector},
# #     p₀::Vector{<:Number},
# #     l₀;
# #     show_progress = true,
# #     trace_tol = 1e-6,
# #     kwargs...,
# # )
# #     # Combine W₀₁ and W₁₀.
# #     S = append!([[x; 0.0] for x in W₀₁], W₁₀)
# #     # To verify that we found all solutions we need move in the pencil
# #     if show_progress
# #         @info("Compute trace...")
# #     end
# #
# #     trace = track_and_compute_trace(TTS, S, l₀; kwargs...)
# #     if isnothing(trace)
# #         return nothing
# #     else
# #         show_progress && @info("Norm of trace: $(LinearAlgebra.norm(trace))")
# #         LinearAlgebra.norm(trace) < trace_tol
# #     end
# # end
# #
# # """
# #     solution_completeness_witnesses(F, S, p; parameters=..., kwargs...)
# #
# # Compute the additional necessary witnesses. Returns a triple `(W₁₀, TTS, l)`
# # containing the additional witnesses `W₁₀`, a trace test system `TTS` and
# # the parameters `l` for `TTS`.
# # """
# # function solution_completeness_witnesses(
# #     F,
# #     W₀₁,
# #     p₀::Vector{<:Number};
# #     parameters = nothing,
# #     show_progress = true,
# #     kwargs...,
# # )
# #     # generate another start pair
# #     q₀ = randn(ComplexF64, length(p₀))
# #     # Construct the trace test system
# #     TTS = TraceTestSystem(SPSystem(F; parameters = parameters), p₀, q₀ - p₀)
# #
# #     y₁ = solution(solve(F, W₀₁[1]; p₁ = p₀, p₀ = q₀, parameters = parameters, kwargs...)[1])
# #     # construct an affine hyperplane l(x) going through y₀
# #     l₁ = cis.(2π .* rand(length(y₁)))
# #     push!(l₁, -sum(l₁ .* y₁))
# #     # This is numerically sometimes not so nice. Let's move to a truly generic one.
# #     l₀ = randn(ComplexF64, length(l₁))
# #     y₀ = solution(solve(TTS, [y₁; 1]; p₁ = l₁, p₀ = l₀)[1])
# #
# #     if show_progress
# #         @info("Compute additional witnesses for completeness...")
# #     end
# #
# #     R₁₀ = monodromy_solve(TTS, y₀, l₀; max_loops_no_progress = 5, kwargs...)
# #     best_result = R₁₀
# #     best_params = l₀
# #     result_agreed = 0
# #     for i = 1:10
# #         k₀ = randn(ComplexF64, length(l₀))
# #         S_k₀ = solutions(solve(
# #             TTS,
# #             solutions(R₁₀);
# #             start_parameters = l₀,
# #             target_parameters = k₀,
# #         ))
# #         new_result = monodromy_solve(TTS, S_k₀, k₀; max_loops_no_progress = 5)
# #         if nsolutions(new_result) == nsolutions(best_result)
# #             result_agreed += 1
# #         elseif nsolutions(new_result) > nsolutions(best_result)
# #             best_result = new_result
# #             best_params = k₀
# #         end
# #         if result_agreed > 2
# #             break
# #         end
# #     end
# #
# #     W₁₀ = solutions(best_result)
# #
# #     if show_progress
# #         @info("Found $(length(W₁₀)) additional witnesses")
# #     end
# #     W₁₀, TTS, best_params
# # end
# #
# #
# # function track_and_compute_trace(TTS::TraceTestSystem, S, l₀; kwargs...)
# #     for i = 1:3
# #         TTP = TraceTestPencil(TTS, l₀)
# #         R₁ = solve(TTP, S, start_parameters = [0.0], target_parameters = [0.1], kwargs...)
# #         R₂ = solve(TTP, S, start_parameters = [0.0], target_parameters = [-.1], kwargs...)
# #         if nsolutions(R₁) ≠ length(S) || nsolutions(R₂) ≠ length(S)
# #             if i == 3
# #                 printstyled("Lost solutions $i times. Abort.\n", color = :red, bold = true)
# #                 return nothing
# #             end
# #             printstyled(
# #                 "Lost solutions, need to recompute trace...\n",
# #                 color = :yellow,
# #                 bold = true,
# #             )
# #         end
# #         s₁ = sum(solutions(R₁))
# #         s = sum(S)
# #         s₂ = sum(solutions(R₂))
# #         # Due to floating point errors, we compute the RELATIVE trace
# #         trace = ((s₁ .- s) .- (s .- s₂)) ./ max.(abs.(s₁ .- s), 1.0)
# #         return trace
# #     end
# #     nothing
# # end
