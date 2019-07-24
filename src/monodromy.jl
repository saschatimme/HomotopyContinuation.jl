export monodromy_solve, MonodromyResult, real_solutions, nreal, parameters


#####################
# Monodromy Options #
#####################
const monodromy_options_supported_keywords = [:distance, :identical_tol, :done_callback,
    :group_action,:group_actions, :group_action_on_all_nodes,
    :parameter_sampler, :equivalence_classes, :complex_conjugation, :check_startsolutions,
    :target_solutions_count, :timeout,
    :min_solutions, :max_loops_no_progress]

struct MonodromyOptions{F<:Function, F1<:Function, F2<:Tuple, F3<:Function}
    distance_function::F
    identical_tol::Float64
    done_callback::F1
    group_actions::GroupActions{F2}
    group_action_on_all_nodes::Bool
    parameter_sampler::F3
    equivalence_classes::Bool
    complex_conjugation::Bool
    check_startsolutions::Bool
    # stopping heuristic
    target_solutions_count::Int
    timeout::Float64
    min_solutions::Int
    max_loops_no_progress::Int
end

function MonodromyOptions(is_real_system;
    distance=euclidean_distance,
    identical_tol::Float64=1e-6,
    done_callback=always_false,
    group_action=nothing,
    group_actions=group_action === nothing ? nothing : GroupActions(group_action),
    group_action_on_all_nodes=false,
    parameter_sampler=independent_normal,
    equivalence_classes=true,
    complex_conjugation=is_real_system,
    check_startsolutions=true,
    # stopping heuristic
    target_solutions_count=nothing,
    timeout=float(typemax(Int)),
    min_solutions::Int=default_min_solutions(target_solutions_count),
    max_loops_no_progress::Int=10)

    if group_actions isa GroupActions
       actions = group_actions
    else
       if group_actions === nothing
           equivalence_classes = false
       end
       actions = GroupActions(group_actions)
    end


    MonodromyOptions(distance,identical_tol, done_callback, actions,
        group_action_on_all_nodes, parameter_sampler, equivalence_classes, complex_conjugation, check_startsolutions,
        target_solutions_count === nothing ? typemax(Int) : target_solutions_count,
        float(timeout),
        min_solutions,
        max_loops_no_progress)
end

default_min_solutions(::Nothing) = 1
function default_min_solutions(target_solutions_count::Int)
    div(target_solutions_count, 2)
end

always_false(x, sols) = false

has_group_actions(options::MonodromyOptions) = !(options.group_actions isa GroupActions{Tuple{}})


"""
    independent_normal(p::AbstractVector{T}) where {T}

Sample a vector where each entries is drawn independently from the univariate normal distribution.
"""
independent_normal(p::SVector{N, T}) where {N, T} = @SVector randn(T, N)
independent_normal(p::AbstractVector{T}) where {T} = randn(T, length(p))

##########################
## Monodromy Statistics ##
##########################

mutable struct MonodromyStatistics
    ntrackedpaths::Int
    ntrackingfailures::Int
    nreal::Int
    nparametergenerations::Int
    nsolutions_development::Vector{Int}
end

MonodromyStatistics(nsolutions::Int) = MonodromyStatistics(0, 0, 0, 1, [nsolutions])
function MonodromyStatistics(solutions)
    stats = MonodromyStatistics(length(solutions))
    for s in solutions
        if is_real_vector(s)
            stats.nreal +=1
        end
    end
    stats
end

Base.show(io::IO, S::MonodromyStatistics) = print_fieldnames(io, S)
Base.show(io::IO, ::MIME"application/prs.juno.inline", S::MonodromyStatistics) = S

# update routines
function trackedpath!(stats::MonodromyStatistics, retcode)
    if retcode == CoreTrackerStatus.success
        stats.ntrackedpaths += 1
    else
        stats.ntrackingfailures += 1
    end
end

function generated_parameters!(stats::MonodromyStatistics, nsolutions::Int)
    stats.nparametergenerations += 1
    push!(stats.nsolutions_development, nsolutions)
end

function finished!(stats, nsolutions)
    push!(stats.nsolutions_development, nsolutions)
end

function n_completed_loops_without_change(stats, nsolutions)
    k = 0
    for i in length(stats.nsolutions_development):-1:1
        if stats.nsolutions_development[i] != nsolutions
            return k
        end
        k += 1
    end
    return max(k - 1, 0)
end

function n_solutions_current_loop(statistics, nsolutions)
    nsolutions - statistics.nsolutions_development[end]
end

#############################
# Loops and Data Structures #
#############################

export Triangle, Petal

#######################
# Loop data structure
#######################

struct LoopEdge
    p₁::Vector{ComplexF64}
    p₀::Vector{ComplexF64}
    weights::Union{Nothing, NTuple{2, ComplexF64}}
end

function LoopEdge(p₁, p₀; weights=false)
    if weights
        γ = (randn(ComplexF64), randn(ComplexF64))
    else
        γ = nothing
    end
    LoopEdge(convert(Vector{ComplexF64}, p₁), convert(Vector{ComplexF64}, p₀), γ)
end

struct MonodromyLoop
    edges::Vector{LoopEdge}
end

function MonodromyLoop(base_p, nnodes::Int, options::MonodromyOptions; weights=true)
    edges = LoopEdge[]
    p₁ = base_p
    for i = 2:nnodes
        p₀ = options.parameter_sampler(p₁)
        push!(edges, LoopEdge(p₁, p₀; weights=weights))
        p₁ = p₀
    end
    push!(edges, LoopEdge(p₁, base_p; weights=weights))
    MonodromyLoop(edges)
end

##############
# Loop styles
##############
"""
    LoopStyle

Abstract type defining a style of a loop.
"""
abstract type LoopStyle end


"""
    Triangle(;useweights=true)

A triangle is a loop consisting of the main node and two addtional nodes.
If `weights` is true the edges are equipped with additional random weights.
Note that this is usually only necessary for real parameters.
"""
struct Triangle <: LoopStyle
    useweights::Bool
end
Triangle(;useweights=true) = Triangle(useweights)

function MonodromyLoop(strategy::Triangle, p, options::MonodromyOptions)
    MonodromyLoop(p, 3, options, weights=strategy.useweights)
end

"""
    Petal()

A petal is a loop consisting of the main node and one other node connected
by two edges with different random weights.
"""
struct Petal <: LoopStyle end
function MonodromyLoop(strategy::Petal, p, options::MonodromyOptions)
    MonodromyLoop(p, 2, options, weights=true)
end


"""
    regenerate!(loop::MonodromyLoop, options::MonodromyOptions, stats::MonodromyStatistics)

Regenerate all random parameters in the loop in order to introduce a new monodromy action.
"""
function regenerate!(loop::MonodromyLoop, options::MonodromyOptions, stats::MonodromyStatistics)
    main = mainnode(loop)

    # The first node is the main node and doesn't get touched
    store_points = options.group_action_on_all_nodes && has_group_actions(options)
    for i ∈ 2:length(loop.nodes)
        loop.nodes[i] = Node(options.parameter_sampler(main.p), loop.nodes[i], options;
                                        store_points=store_points, is_main_node=false)
    end
    loop.edges .= Edge.(loop.edges)
    generated_parameters!(stats, length(main.points)) # bookkeeping
end


"""
    track(tracker, x::AbstractVector, edge::LoopEdge, loop::MonodromyLoop, stats::MonodromyStatistics)

Track `x` along the edge `edge` in the loop `loop` using `tracker`. Record statistics
in `stats`.
"""
function track(tracker::CoreTracker, x::AbstractVector, loop::MonodromyLoop, stats::MonodromyStatistics)
    H = basehomotopy(tracker.homotopy)
    local retcode::CoreTrackerStatus.states
    y = x
    for e in loop.edges
        set_parameters!(H, (e.p₁, e.p₀), e.weights)
        retcode = track!(tracker, y)
        trackedpath!(stats, retcode)
        retcode == CoreTrackerStatus.success || break
        y = current_x(tracker)
    end
    retcode
end


#############
## Results ##
#############
"""
    MonodromyResult

The monodromy result contains the result of the `monodromy_solve` computation.
"""
struct MonodromyResult{N, T1, T2}
    returncode::Symbol
    solutions::Vector{SVector{N, T1}}
    parameters::Vector{T2}
    statistics::MonodromyStatistics
    equivalence_classes::Bool
end

Base.iterate(R::MonodromyResult) = iterate(R.solutions)
Base.iterate(R::MonodromyResult, state) = iterate(R.solutions, state)

Base.show(io::IO, ::MIME"application/prs.juno.inline", x::MonodromyResult) = x
function Base.show(io::IO, result::MonodromyResult{N, T}) where {N, T}
    println(io, "MonodromyResult")
    println(io, "==================================")
    if result.equivalence_classes
        println(io, "• $(nsolutions(result)) classes of solutions (modulo group action) ($(nreal(result)) real)")
    else
        println(io, "• $(nsolutions(result)) solutions ($(nreal(result)) real)")
    end
    println(io, "• return code → $(result.returncode)")
    println(io, "• $(result.statistics.ntrackedpaths) tracked paths")
end


TreeViews.hastreeview(::MonodromyResult) = true
TreeViews.numberofnodes(::MonodromyResult) = 5
TreeViews.treelabel(io::IO, x::MonodromyResult, ::MIME"application/prs.juno.inline") =
    print(io, "<span class=\"syntax--support syntax--type syntax--julia\">MonodromyResult</span>")

function TreeViews.nodelabel(io::IO, x::MonodromyResult, i::Int, ::MIME"application/prs.juno.inline")
    if i == 1
        if x.equivalence_classes
            print(io, "$(nsolutions(x)) classes of solutions (modulo group action)")
        else
            print(io, "$(nsolutions(x)) solutions")
        end
    elseif i == 2
        if x.equivalence_classes
            print(io, "$(nreal(x)) classes of real solutions")
        else
            print(io, "$(nreal(x)) real solutions")
        end
    elseif i == 3
        print(io, "Return code")
    elseif i == 4
        print(io, "Tracked paths")
    elseif i == 5
        print(io, "Parameters")
    end
end
function TreeViews.treenode(r::MonodromyResult, i::Integer)
    if i == 1
        return r.solutions
    elseif i == 2
        return real_solutions(r)
    elseif i == 3
        return r.returncode
    elseif i == 4
        return r.statistics.ntrackedpaths
    elseif i == 5
        return r.parameters
    end
    missing
end


"""
    mapresults(f, result::MonodromyResult; only_real=false, real_tol=1e-6)

Apply the function `f` to all entries of `MonodromyResult` for which the given conditions apply.

## Example
```julia
# This gives us all solutions considered real (but still as a complex vector).
real_solutions = mapresults(solution, R, only_real=true)
```
"""
function mapresults(f, R::MonodromyResult;
    only_real=false, real_tol=1e-6)
    [f(r) for r in R.solutions if
        (!only_real || is_real_vector(r, real_tol))]
end

"""
    solutions(result::MonodromyResult; only_real=false, real_tol=1e-6)

Return all solutions (as `SVector`s) for which the given conditions apply.

## Example
```julia
real_solutions = solutions(R, only_real=true)
```
"""
function solutions(R::MonodromyResult; kwargs...)
    mapresults(identity, R; kwargs...)
end

"""
    nsolutions(result::MonodromyResult)

Returns the number solutions of the `result`.
"""
nsolutions(res::MonodromyResult) = length(res.solutions)

"""
    real_solutions(res::MonodromyResult; tol=1e-6)

Returns the solutions of `res` whose imaginary part has norm less than 1e-6.
"""
function real_solutions(res::MonodromyResult; tol=1e-6)
    map(r -> real.(r), filter(r -> LinearAlgebra.norm(imag.(r)) < tol, res.solutions))
end

"""
    nreal(res::MonodromyResult; tol=1e-6)

Counts how many solutions of `res` have imaginary part norm less than 1e-6.
"""
function nreal(res::MonodromyResult; tol=1e-6)
    count(r -> LinearAlgebra.norm(imag.(r)) < tol, res.solutions)
end

"""
    parameters(r::MonodromyResult)

Return the parameters corresponding to the given result `r`.
"""
parameters(r::MonodromyResult) = r.parameters

#####################
## monodromy solve ##
#####################
"""
    MonodromyCache{FT<:FixedHomotopy, Tracker<:CoreTracker, NC<:AbstractNewtonCache}

Cache for monodromy loops.
"""
struct MonodromyCache{FT<:FixedHomotopy, Tracker<:CoreTracker, NC<:AbstractNewtonCache, AV<:AbstractVector}
    F::FT
    tracker::Tracker
    newton_cache::NC
    out::AV
end

struct MonodromySolver{T<:Number, UP<:UniquePoints, MO<:MonodromyOptions, MC<:MonodromyCache}
    parameters::Vector{T}
    solutions::UP
    loops::Vector{MonodromyLoop}
    options::MO
    statistics::MonodromyStatistics
    cache::MC
end


function MonodromySolver(F::Inputs, solution::Vector{<:Number}, p₀::AbstractVector{TP}; kwargs...) where {TP}
    MonodromySolver(F, [solution], p₀; kwargs...)
end
function MonodromySolver(F::Inputs, solutions::Vector{<:AbstractVector{<:Number}}, p₀::AbstractVector{TP}; kwargs...) where {TP}
    MonodromySolver(F, static_solutions(solutions), p₀; kwargs...)
end
function MonodromySolver(F::Inputs,
        startsolutions::Vector{<:SVector{NVars, <:Complex}},
        p::AbstractVector{TP};
        parameters=nothing,
        strategy=nothing,
        show_progress=true,
        showprogress=nothing, #deprecated
        kwargs...) where {TP, NVars}

    @deprecatekwarg showprogress show_progress

    if parameters !== nothing && length(p) ≠ length(parameters)
        throw(ArgumentError("Number of provided parameters doesn't match the length of initially provided parameter `p₀`."))
    end

    p₀ = Vector{promote_type(Float64, TP)}(p)

    optionskwargs, restkwargs = splitkwargs(kwargs, monodromy_options_supported_keywords)

    # construct tracker
    tracker = coretracker(F, startsolutions; parameters=parameters, generic_parameters=p₀, restkwargs...)
    if affine_tracking(tracker)
        HC = HomotopyWithCache(tracker.homotopy, tracker.state.x, 1.0)
        F₀ = FixedHomotopy(HC, 0.0)
    else
        # Force affine newton method
        patch_state = state(EmbeddingPatch(), tracker.state.x)
        HC = HomotopyWithCache(PatchedHomotopy(tracker.homotopy, patch_state), tracker.state.x, 1.0)
        F₀ = FixedHomotopy(HC, 0.0)
    end

    # Check whether homotopy is real
    is_real_system = numerically_check_real(tracker.homotopy, tracker.state.x)

    options = MonodromyOptions(is_real_system; optionskwargs...)
    # construct cache
    newt_cache = newton_cache(F₀, tracker.state.x)
    C =  MonodromyCache(F₀, tracker, newt_cache, copy(tracker.state.x))
    # construct UniquePoints
    if options.equivalence_classes
        uniquepoints = UniquePoints(eltype(startsolutions), options.distance_function;
                                    group_actions = options.group_actions,
                                    check_real = true)
    else
        uniquepoints = UniquePoints(eltype(startsolutions), options.distance_function; check_real = true)
    end
    # add only unique points that are true solutions
    if options.check_startsolutions
        for s in startsolutions
            y = verified_affine_vector(C, s, s, options)
            if y !== nothing
                add!(uniquepoints, y; tol=options.identical_tol)
            end
        end
    else
        add!(uniquepoints, startsolutions; tol=options.identical_tol)
    end
    statistics = MonodromyStatistics(uniquepoints)

    if strategy === nothing
        strategy = default_strategy(F, parameters, p; is_real_system=is_real_system)
    end

    # construct Loop
    loops = [MonodromyLoop(strategy, p₀, options)]

    MonodromySolver(p₀, uniquepoints, loops, options, statistics, C)
end

"""
    solutions(MS::MonodromySolver)

Get the solutions of the loop.
"""
solutions(MS::MonodromySolver) = MS.solutions.points

"""
    nsolutions(loop::MonodromyLoop)

Get the number solutions of the loop.
"""
nsolutions(MS::MonodromySolver) = length(solutions(MS))


"""
    monodromy_solve(F, sols, p; parameters=..., options..., pathtrackerkwargs...)

Solve a polynomial system `F(x;p)` with specified parameters and initial solutions `sols`
by monodromy techniques. This makes loops in the parameter space of `F` to find new solutions.

## Options
* `target_solutions_count=nothing`: The computations are stopped if this number of solutions is reached.
* `done_callback=always_false`: A callback to end the computation early. This function takes 2 arguments. The first one is the new solution `x` and the second one are all current solutions (including `x`). Return `true` if the compuation is done.
* `max_loops_no_progress::Int=10`: The maximal number of iterations (i.e. loops generated) without any progress.
* `group_action=nothing`: A function taking one solution and returning other solutions if there is a constructive way to obtain them, e.g. by symmetry.
* `strategy`: The strategy used to create loops. If `F` only depends linearly on `p` this will be [`Petal`](@ref). Otherwise this will be [`Triangle`](@ref) with weights if `F` is a real system.
* `show_progress=true`: Enable a progress meter.
* `distance_function=euclidean_distance`: The distance function used for [`UniquePoints`](@ref).
* `identical_tol::Float64=1e-6`: The tolerance with which it is decided whether two solutions are identical.
* `group_actions=nothing`: If there is more than one group action you can use this to chain the application of them. For example if you have two group actions `foo` and `bar` you can set `group_actions=[foo, bar]`. See [`GroupActions`](@ref) for details regarding the application rules.
* `group_action_on_all_nodes=false`: By default the group_action(s) are only applied on the solutions with the main parameter `p`. If this is enabled then it is applied for every parameter `q`.
* `parameter_sampler=independent_normal`: A function taking the parameter `p` and returning a new random parameter `q`. By default each entry of the parameter vector is drawn independently from the univariate normal distribution.
* `equivalence_classes=true`: This only applies if there is at least one group action supplied. We then consider two solutions in the same equivalence class if we can transform one to the other by the supplied group actions. We only track one solution per equivalence class.
* `check_startsolutions=true`: If `true`, we do a Newton step for each entry of `sols`for checking if it is a valid startsolutions. Solutions which are not valid are sorted out.
* `timeout=float(typemax(Int))`: The maximal number of *seconds* the computation is allowed to run.
* `min_solutions`: The minimal number of solutions before a stopping heuristic is applied. By default this is half of `target_solutions_count` if applicable otherwise 2.
"""
function monodromy_solve(args...; show_progress=true, kwargs...)
    monodromy_solve!(MonodromySolver(args...; kwargs...); show_progress=show_progress)
end

function default_strategy(F::MPPolyInputs, parameters, p₀::AbstractVector{TP}; is_real_system=false) where {TC,TP}
    # If F depends only linearly on the parameters a petal is sufficient
    vars = variables(F; parameters=parameters)
    if all(d -> d ≤ 1, maxdegrees(F; parameters=vars))
        Petal()
    # For a real system we should introduce some weights to avoid the discriminant
    elseif is_real_system
        Triangle(useweights=true)
    else
        Triangle(useweights=false)
    end
end
function default_strategy(F::AbstractSystem, parameters, p₀; is_real_system=false)
    # For a real system we should introduce some weights to avoid the discriminant
    if is_real_system
        Triangle(useweights=true)
    else
        Triangle(useweights=false)
    end
end

# convert vector of vectors to vector of svectors
static_solutions(V::Vector) = static_solutions(V, Val(length(V[1])))
function static_solutions(V::Vector, ::Val{N}) where {N}
    map(v -> complex.(float.(SVector{N}(v))), V)
end
function static_solutions(V::Vector{<:AbstractVector{<:Complex{<:AbstractFloat}}}, ::Val{N}) where {N}
    SVector{N}.(V)
end


function numerically_check_real(H::AbstractHomotopy, x)
    y = copy(x)
    Random.randn!(y)
    for i in eachindex(y)
        y[i] = real(y[i]) + 0.0im
    end
    t = rand()
    r = evaluate(H, y, t)
    isapprox(LinearAlgebra.norm(imag.(r), Inf), 0.0, atol=1e-14)
end

#################
## Actual work ##
#################

"""
    MonodromyJob

A `MonodromyJob` is consisting of a solution id and a loop id.
"""
struct MonodromyJob
    id::Int
    loop_id::Int
end

function monodromy_solve!(MS::MonodromySolver; show_progress=true)
    if nsolutions(MS) == 0
        @warn "None of the provided solutions is a valid start solution."
        return MonodromyResult(:invalid_startvalue,
                similar(MS.solutions.points, 0), MS.parameters, MS.statistics,
                MS.options.equivalence_classes)
    end
    # solve
    retcode = :not_assigned
    if show_progress
        if !MS.options.equivalence_classes
            desc = "Solutions found:"
        else
            desc = "Classes of solutions (modulo group action) found:"
        end
        progress = ProgressMeter.ProgressUnknown(desc; delay=0.5, clear_output_ijulia=true)
    else
        progress = nothing
    end

    n_blas_threads = single_thread_blas()
    try
        retcode = monodromy_solve!(MS, progress)
    catch e
        if (e isa InterruptException)
            retcode = :interrupt
        else
            rethrow(e)
        end
    end
    n_blas_threads > 1 && set_num_BLAS_threads(n_blas_threads)
    finished!(MS.statistics, nsolutions(MS))
    MonodromyResult(retcode, solutions(MS), MS.parameters, MS.statistics, MS.options.equivalence_classes)
end


function monodromy_solve!(MS::MonodromySolver, progress::Union{Nothing,ProgressMeter.ProgressUnknown})
    t₀ = time_ns()
    iterations_without_progress = 0 # stopping heuristic
    # intialize job queue
    queue = MonodromyJob.(1:nsolutions(MS), 1)

    n = nsolutions(MS)
    while n < MS.options.target_solutions_count
        retcode = empty_queue!(queue, MS, t₀, progress)

        if retcode == :done
            update_progress!(progress, nsolutions(MS), MS.statistics; finish=true)
            break
        elseif retcode == :timeout
            return :timeout
        elseif retcode == :invalid_startvalue
            return :invalid_startvalue
        end

        # Iterations heuristic
        n_new = nsolutions(MS)
        if n == n_new
            iterations_without_progress += 1
        else
            iterations_without_progress = 0
            n = n_new
        end
        if iterations_without_progress == MS.options.max_loops_no_progress &&
            n_new ≥ MS.options.min_solutions
            return :heuristic_stop
        end

        regenerate_loop_and_schedule_jobs!(queue, MS)
    end

    :success
end

function empty_queue!(queue, MS::MonodromySolver, t₀::UInt, progress)
    while !isempty(queue)
        job = pop!(queue)
        status = process!(queue, job, MS, progress)
        if status == :done
            return :done
        elseif status == :invalid_startvalue
            return :invalid_startvalue
        end
        update_progress!(progress, nsolutions(MS), MS.statistics)
        # check timeout
        if (time_ns() - t₀) > MS.options.timeout * 1e9 # convert s to ns
            return :timeout
        end
    end
    :incomplete
end

function verified_affine_vector(C::MonodromyCache, ŷ, x, options)
    # We distinguish solutions which have a distance larger than identical_tol
    # But due to the numerical error in the evaluation of the distance, we need to be a little bit
    # careful. Therefore, we require that the solutions should be one magnitude closer to
    # the true solutions as necessary
    tol = 0.1 * options.identical_tol
    result = newton!(C.out, C.F, ŷ, euclidean_norm, C.newton_cache,
                tol=tol, miniters=1, maxiters=3, simplified_last_step=false)

    if isconverged(result)
        return affine_chart(x, C.out)
    else
        return nothing
    end
end

affine_chart(x::SVector, y::PVector) = ProjectiveVectors.affine_chart!(x, y)
affine_chart(x::SVector{N, T}, y::AbstractVector) where {N, T} = SVector{N,T}(y)

function process!(queue, job::MonodromyJob, MS::MonodromySolver, progress)
    x = solutions(MS)[job.id]
    loop = MS.loops[job.loop_id]
    retcode = track(MS.cache.tracker, x, loop, MS.statistics)
    if retcode ≠ CoreTrackerStatus.success
        if retcode == CoreTrackerStatus.terminated_invalid_startvalue &&
           MS.statistics.ntrackedpaths == 0
            return :invalid_startvalue
        end
        return :incomplete
    end


    y = verified_affine_vector(MS.cache, current_x(MS.cache.tracker), x, MS.options)
    # is the solution at infinity?
    y !== nothing || return :incomplete

    add_and_schedule!(MS, queue, y, job) && return :done

    if MS.options.complex_conjugation
        add_and_schedule!(MS, queue, conj.(y), job) && return :done
    end

    if !MS.options.equivalence_classes
        apply_actions(MS.options.group_actions, y) do yᵢ
            add_and_schedule!(MS, queue, yᵢ, job)
        end
    end
    return :incomplete
end

"""
    add_and_schedule!(MS, queue, y, job)

Add `y` to the current `node` (if it not already exists) and schedule a new job to the `queue`.
Returns `true` if we are done. Otherwise `false`.
"""
function add_and_schedule!(MS::MonodromySolver, queue, y, job::MonodromyJob) where {N,T}
    k = add!(MS.solutions, y, Val(true); tol=MS.options.identical_tol)
    if k == NOT_FOUND || k == NOT_FOUND_AND_REAL
        # Check if we are done
        isdone(MS.solutions, y, MS.options) && return true
        push!(queue, MonodromyJob(nsolutions(MS), job.loop_id))
        # TODO: Schedule on other loops
    end
    MS.statistics.nreal += (k == NOT_FOUND_AND_REAL)
    false
end

function update_progress!(::Nothing, nsolutions, statistics::MonodromyStatistics; finish=false)
    nothing
end
function update_progress!(progress, nsolutions, statistics::MonodromyStatistics; finish=false)
    ProgressMeter.update!(progress, nsolutions, showvalues=(
        ("# paths tracked", statistics.ntrackedpaths),
        ("# loops generated", statistics.nparametergenerations),
        ("# completed loops without change", n_completed_loops_without_change(statistics, nsolutions)),
        ("# solutions in current loop", n_solutions_current_loop(statistics, nsolutions)),
        ("# real solutions", statistics.nreal),
    ))
    if finish
        ProgressMeter.finish!(progress)
    end
    nothing
end

function isdone(solutions::UniquePoints, x, options::MonodromyOptions)
    options.done_callback(x, solutions.points) ||
    length(solutions) ≥ options.target_solutions_count
end

function regenerate_loop_and_schedule_jobs!(queue, MS::MonodromySolver)
    es = MS.loops[1].edges
    loop = MonodromyLoop(MS.parameters, length(es), MS.options, weights=!isnothing(es[1].weights))
    MS.loops[1] = loop
    for id in 1:nsolutions(MS)
        push!(queue, MonodromyJob(id, 1))
    end
    generated_parameters!(MS.statistics, nsolutions(MS))
    nothing
end
