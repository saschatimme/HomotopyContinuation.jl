export polyhedral, PolyhedralTracker, PolyhedralStartSolutionsIterator, mixed_volume

using MixedSubdivisions: MixedCell, mixed_volume

"""
    PolyhedralStartSolutionsIterator

An iterator providing start solutions for the polyhedral homotopy.
"""
struct PolyhedralStartSolutionsIterator
    support::Vector{Matrix{Int32}}
    start_coefficients::Vector{Vector{ComplexF64}}
    lifting::Vector{Vector{Int32}}
    mixed_cells::Vector{MixedCell}
    BSS::BinomialSystemSolver
end

function PolyhedralStartSolutionsIterator(
    support::AbstractVector{<:AbstractMatrix{<:Integer}},
    coeffs::AbstractVector{<:AbstractVector{<:Number}},
    lifting = map(c -> zeros(Int32, length(c)), coeffs),
    mixed_cells = MixedCell[],
)
    BSS = BinomialSystemSolver(length(support))
    PolyhedralStartSolutionsIterator(
        convert(Vector{Matrix{Int32}}, support),
        coeffs,
        lifting,
        mixed_cells,
        BSS,
    )
end

Base.show(io::IO, C::PolyhedralStartSolutionsIterator) =
    print(io, "PolyhedralStartSolutionsIterator()")
Base.show(
    io::IO,
    ::MIME"application/prs.juno.inline",
    x::PolyhedralStartSolutionsIterator,
) = x
Base.IteratorSize(::Type{<:PolyhedralStartSolutionsIterator}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:PolyhedralStartSolutionsIterator}) = Base.HasEltype()
Base.eltype(iter::PolyhedralStartSolutionsIterator) = Tuple{MixedCell,Vector{ComplexF64}}

function compute_mixed_cells!(iter::PolyhedralStartSolutionsIterator)
    if isempty(iter.mixed_cells)
        res = MixedSubdivisions.fine_mixed_cells(iter.support)
        if isnothing(res) || isempty(res[1])
            throw(OverflowError("Cannot compute a start system."))
        end
        mixed_cells, lifting = res
        empty!(iter.mixed_cells)
        append!(iter.mixed_cells, mixed_cells)

        for (i, w) in enumerate(lifting)
            empty!(iter.lifting[i])
            append!(iter.lifting[i], w)
        end
    end
    iter
end

function Base.iterate(iter::PolyhedralStartSolutionsIterator)
    compute_mixed_cells!(iter)

    isempty(iter.mixed_cells) && return nothing

    cell = iter.mixed_cells[1]
    solve(iter.BSS, iter.support, iter.start_coefficients, cell)

    x = [iter.BSS.X[i, 1] for i = 1:length(iter.support)]
    el = (cell, x)
    next_state = size(iter.BSS.X, 2) == 1 ? (2, 1) : (1, 2)
    el, next_state
end

function Base.iterate(iter::PolyhedralStartSolutionsIterator, (i, j)::Tuple{Int,Int})
    i > length(iter.mixed_cells) && return nothing

    cell = iter.mixed_cells[i]
    j == 1 && solve(iter.BSS, iter.support, iter.start_coefficients, cell)
    x = [iter.BSS.X[i, j] for i = 1:length(iter.support)]
    el = (cell, x)
    next_state = size(iter.BSS.X, 2) == j ? (i + 1, 1) : (i, j + 1)
    return el, next_state
end

# Tracker
function polyhedral_system(support)
    m = 0
    p = Variable[]
    coeffs = map(support) do A
        c = variables(:c, m+1:m+size(A, 2))
        m += size(A, 2)
        append!(p, c)
        c
    end

    System(support, coeffs; variables = variables(:x, 1:length(support)), parameters = p)
end

"""
    PolyhedralTracker <: AbstractPathTracker

This tracker realises the two step approach of the polyhedral homotopy.
See also [`polyhedral`].
"""
struct PolyhedralTracker{H1<:ToricHomotopy,H2<:AbstractHomotopy,N,M} <: AbstractPathTracker
    toric_tracker::Tracker{H1,N,M}
    generic_tracker::EndgameTracker{H2,N,M}
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
end

"""
    polyhedral(F::Union{System, AbstractSystem};
        only_non_zero = false,
        endgame_options = EndgameOptions(),
        tracker_options = TrackerOptions())

Solve the system `F` in two steps: first solve a generic system derived from the support
of `F` using a polyhedral homotopy as proposed in [^HS95], then perform a
coefficient-parameter homotopy towards `F`.
This returns a path tracker ([`PolyhedralTracker`](@ref) or [`OverdeterminedTracker`](@ref)) and an iterator to compute the start solutions.

If `only_non_zero` is `true`, then only the solutions with non-zero coordinates are computed.
In this case the number of paths to track is equal to the
mixed volume of the Newton polytopes of `F`.

If `only_non_zero` is `false`, then all isolated solutions of `F` are computed.
In this case the number of paths to track is equal to the
mixed volume of the convex hulls of ``supp(F_i) ∪ \\{0\\}`` where ``supp(F_i)`` is the support
of ``F_i``. See also [^LW96].


    function polyhedral(
        support::AbstractVector{<:AbstractMatrix},
        coefficientss::AbstractVector{<:AbstractVector{<:Number}};
        kwargs...,
    )

It is also possible to provide directly the support and coefficients of the system `F` to be solved.

[^HS95]: Birkett Huber and Bernd Sturmfels. “A Polyhedral Method for Solving Sparse Polynomial Systems.” Mathematics of Computation, vol. 64, no. 212, 1995, pp. 1541–1555
[^LW96]: T.Y. Li and Xiaoshen Wang. "The BKK root count in C^n". Math. Comput. 65, 216 (October 1996), 1477–1484.

### Example

We consider a system `f` which has in total 6 isolated solutions,
but only 3 where all coordinates are non-zero.
```julia
@var x y
f = System([2y + 3 * y^2 - x * y^3, x + 4 * x^2 - 2 * x^3 * y])
tracker, starts = polyhedral(f; only_non_zero = false)
# length(starts) == 8
count(is_success, track.(tracker, starts)) # 6

tracker, starts = polyhedral(f; only_non_zero = true)
# length(starts) == 3
count(is_success, track.(tracker, starts)) # 3
```
"""
function polyhedral(F::AbstractSystem; kwargs...)
    _, n = size(F)
    @var x[1:n]
    polyhedral(System(F(x), x); kwargs...)
end
function polyhedral(
    f::System;
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    target_parameters = nothing,
    kwargs...,
)
    if target_parameters !== nothing
        return polyhedral(
            FixedParameterSystem(fixed(f; compile = compile), target_parameters);
            kwargs...,
        )
    end
    homogeneous = is_homogeneous(f)
    if homogeneous
        F = on_affine_chart(f; compile = compile)
        m, n = size(F)
        m ≥ n || throw(FiniteException(n - m))
        if m > n
            F = square_up(F; compile = compile)
        end
        @var x[1:n]
        support, target_coeffs = support_coefficients(System(F(x), x))
    else
        m, n = size(f)
        m ≥ n || throw(FiniteException(n - m))
        if m > n
            F = square_up(f; compile = compile)
            @var x[1:n]
            support, target_coeffs = support_coefficients(System(F(x), x))
        else
            support, target_coeffs = support_coefficients(f)
        end
    end
    tracker, starts = polyhedral(support, target_coeffs; compile = compile, kwargs...)
    if m > n
        tracker = OverdeterminedTracker(tracker, F)
    end
    tracker, starts
end

paths_to_track(f::AbstractSystem, val::Val{:polyhedral}) = paths_to_track(System(f), val)
function paths_to_track(
    f::System,
    ::Val{:polyhedral};
    only_torus::Bool = false,
    only_non_zero::Bool = only_torus,
)
    supp, coeffs = begin
        if is_homogeneous(f)
            F = on_affine_chart(f)
            m, n = size(F)
            m ≥ n || throw(FiniteException(n - m))
            if m > n
                F = square_up(F)
            end
            @var x[1:n]
            support_coefficients(System(F(x), x))
        else
            m, n = size(f)
            m ≥ n || throw(FiniteException(n - m))
            if m > n
                F = square_up(f)
                @var x[1:n]
                support_coefficients(System(F(x), x))
            else
                support_coefficients(f)
            end
        end
    end

    if only_non_zero
        MixedSubdivisions.mixed_volume(supp)
    else
        min_vecs = minimum.(supp, dims = 2)
        supp′ =
            map((A, mᵢ) -> iszero(mᵢ) ? A : [A zeros(Int32, size(A, 1), 1)], supp, min_vecs)
        MixedSubdivisions.mixed_volume(supp′)
    end
end
MixedSubdivisions.mixed_volume(f::Union{System,AbstractSystem}) =
    paths_to_track(f; start_system = :polyhedral, only_torus = true)

function polyhedral(
    support::AbstractVector{<:AbstractMatrix},
    target_coeffs::AbstractVector;
    kwargs...,
)
    start_coeffs = map(c -> rand_approx_unit(length(c)) .* LA.norm(c, Inf), target_coeffs)
    polyhedral(support, start_coeffs, target_coeffs; kwargs...)
end

"""
    rand_approx_unit(n::Integer)::Vector{ComplexF64}

This samples uniformly from the rectangle ``[0.9,1.1] × [0,2π]`` and transforms the sampled
values with the map ``(r, φ) ↦ r * e^{i φ}``.
"""
rand_approx_unit(n) = map(_ -> (0.9 + 0.2 * rand()) * cis2pi(rand()), 1:n)

cis2pi(x) = complex(cospi(2x), sinpi(2x))

function polyhedral(
    support::AbstractVector{<:AbstractMatrix},
    start_coeffs::AbstractVector,
    target_coeffs::AbstractVector;
    endgame_options = EndgameOptions(),
    tracker_options = TrackerOptions(),
    only_torus::Bool = false,
    only_non_zero::Bool = only_torus,
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
)
    unsupported_kwargs(kwargs)
    size.(support, 2) == length.(start_coeffs) == length.(target_coeffs) ||
        throw(ArgumentError("Number of terms do not coincide."))
    min_vecs = map(support) do A
        cols = view.(Ref(A), Ref(:), 1:size(A, 2))
        first(sort!(cols; lt = ModelKit.td_order, rev = true))
    end
    if !all(iszero, min_vecs)
        if only_non_zero
            support = map((A, v) -> A .- v, support, min_vecs)
        else
            # Add 0 to each support
            starts = Vector{ComplexF64}[]
            targets = Vector{ComplexF64}[]
            supp = eltype(support)[]
            for (i, A) in enumerate(support)
                if iszero(min_vecs[i])
                    push!(starts, start_coeffs[i])
                    push!(targets, target_coeffs[i])
                    push!(supp, A)
                else
                    push!(
                        starts,
                        [
                            start_coeffs[i]
                            randn(ComplexF64)
                        ],
                    )
                    push!(
                        targets,
                        [
                            target_coeffs[i]
                            0.0
                        ],
                    )
                    push!(supp, [A zeros(Int32, size(A, 1), 1)])
                end
            end
            support = supp
            start_coeffs = starts
            target_coeffs = targets
        end
    end

    F = fixed(polyhedral_system(support); compile = compile)
    H₁ = ToricHomotopy(F, start_coeffs)
    toric_tracker = Tracker(H₁; options = tracker_options)

    H₂ = begin
        p = reduce(append!, start_coeffs; init = ComplexF64[])
        q = reduce(append!, target_coeffs; init = ComplexF64[])
        CoefficientHomotopy(F, p, q)
    end
    generic_tracker =
        EndgameTracker(Tracker(H₂; options = tracker_options), options = endgame_options)

    S = PolyhedralStartSolutionsIterator(support, start_coeffs)
    tracker = PolyhedralTracker(toric_tracker, generic_tracker, S.support, S.lifting)

    tracker, S
end

function track(
    PT::PolyhedralTracker,
    start_solution::Tuple{MixedCell,<:AbstractVector{ComplexF64}};
    path_number::Union{Nothing,Int} = nothing,
    debug::Bool = false,
)
    cell, x∞ = start_solution
    H = PT.toric_tracker.homotopy
    # The tracker works in two stages
    # 1) Revert the totric degeneration by tracking 0 to 1
    #    Here we use the following strategy
    #    a) Reparameterize the path such that lowest power of t occuring is 1
    #    b)
    #      i) If maximal power is less than 10 track from 0 to 1
    #      ii) If maximal power is larger than 10 we can get some problems for values close to 1
    #          (since t^k is << 1 for t < 1 and k large)
    #          Therefore split in two stages:
    #           1) track from 0 to t₀ = clamp(0.1^(10 / max_weight), 0.9, 0.985)
    #           2) Second, reperamerize path s.t. max power of t is 10. This shifts the
    #              problem towards 0 again (where we have more accuracy available.)
    #              And then track to 1
    # 2) Perform coefficient homotopy with possible endgame
    min_weight, max_weight =
        update_weights!(H, PT.support, PT.lifting, cell, min_weight = 1.0)

    if debug
        println("Min-Max weight: ", min_weight, ", ", max_weight)
    end
    if max_weight < 10
        retcode = track!(
            PT.toric_tracker,
            x∞,
            0.0,
            1.0;
            ω = 20.0,
            μ = 1e-12,
            max_initial_step_size = 0.2,
            debug = debug,
        )
        @unpack μ, ω = PT.toric_tracker.state
    else
        t₀ = clamp(0.1^(10 / max_weight), 0.9, 1 - 1e-6)
        retcode = track!(
            PT.toric_tracker,
            x∞,
            0.0,
            t₀;
            ω = 20.0,
            μ = 1e-12,
            max_initial_step_size = 0.2,
            debug = debug,
        )

        @unpack μ, ω = PT.toric_tracker.state
        if is_success(retcode)
            min_weight, max_weight =
                update_weights!(H, PT.support, PT.lifting, cell, max_weight = 10.0)
            t_restart = t₀^(1 / min_weight)
            # set min_step_size to 0.0 to not accidentally loose a solution
            min_step_size = PT.toric_tracker.options.min_step_size
            PT.toric_tracker.options.min_step_size = 0.0

            retcode = track!(
                PT.toric_tracker,
                PT.toric_tracker.state.x,
                t_restart,
                1.0;
                ω = ω,
                μ = μ,
                τ = 0.1 * t_restart,
                keep_steps = true,
                debug = debug,
            )
            @unpack μ, ω = PT.toric_tracker.state
            PT.toric_tracker.options.min_step_size = min_step_size
        end
    end
    if !is_success(retcode)
        state = PT.toric_tracker.state
        return PathResult(
            return_code = :polyhedral_failed,
            solution = copy(state.x),
            start_solution = start_solution,
            t = real(state.t),
            accuracy = state.accuracy,
            singular = false,
            condition_jacobian = NaN,
            residual = NaN,
            winding_number = nothing,
            last_path_point = (copy(state.x), state.t),
            valuation = nothing,
            ω = state.ω,
            μ = state.μ,
            path_number = path_number,
            extended_precision = state.extended_prec,
            accepted_steps = state.accepted_steps,
            rejected_steps = state.rejected_steps,
            extended_precision_used = state.used_extended_prec,
        )
    end

    r = track(
        PT.generic_tracker,
        PT.toric_tracker.state.x;
        # Don't provide ω since this can be misleading a lead to a too large initial step
        # ω = ω,
        μ = μ,
        path_number = path_number,
        debug = debug,
    )
    r.start_solution = start_solution
    # report accurate steps
    r.accepted_steps += PT.toric_tracker.state.accepted_steps
    r.rejected_steps += PT.toric_tracker.state.rejected_steps

    r
end
