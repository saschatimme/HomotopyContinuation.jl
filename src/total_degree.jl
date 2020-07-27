export total_degree, bezout_number, total_degree_start_solutions

"""
    total_degree(
        F::System;
        parameters = nothing,
        gamma = cis(2π * rand()),
        tracker_options = TrackerOptions(),
        endgame_options = EndgameOptions(),
    )

Solve the system `F` using a total degree homotopy.
This returns a path tracker ([`EndgameTracker`](@ref) or [`OverdeterminedTracker`](@ref)) and an iterator to compute the start solutions.
If the system `F` has declared `variable_groups` then a multi-homogeneous
a start system following [^Wam93] will be constructed.

[^Wam93]: An efficient start system for multi-homogeneous polynomial continuation, Wampler, C.W. Numer. Math. (1993) 66: 517. https://doi.org/10.1007/BF01385710
"""
function total_degree(
    F::Union{System,AbstractSystem};
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
)
    if isa(F, AbstractSystem) || isnothing(variable_groups(F))
        return total_degree_variables(F; compile = compile, kwargs...)
    else
        return total_degree_variable_groups(F; compile = compile, kwargs...)
    end
end


function total_degree_variables(
    F::Union{System,AbstractSystem};
    target_parameters = nothing,
    γ = cis(2π * rand()),
    gamma = γ,
    tracker_options = TrackerOptions(),
    endgame_options = EndgameOptions(),
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
)
    unsupported_kwargs(kwargs)
    m, n = size(F)

    @unique_var x[1:n] t s[1:n]
    support, coeffs = support_coefficients(System(F(x, target_parameters), x))

    homogeneous = true
    D = zeros(Int, length(support))
    for (k, A) in enumerate(support)
        d = 0
        for j = 1:size(A, 2)
            dⱼ = 0
            for i = 1:size(A, 1)
                dⱼ += A[i, j]
            end
            homogeneous = homogeneous && (j == 1 || dⱼ == d)
            d = max(d, dⱼ)
        end
        D[k] = d
    end
    scaling = maximum.(abs ∘ float, coeffs)

    m ≥ (n - homogeneous) || throw(FiniteException(n - homogeneous - m))

    overdetermined = m > n - homogeneous
    # sort by descending degree for overdetermined systems
    if overdetermined && F isa System
        F = deepcopy(F)
        perm = sortperm(D; rev = true)
        permute!(F.expressions, perm)
        permute!(D, perm)
        permute!(scaling, perm)
    end

    if F isa System
        F = fixed(F; compile = compile)
    end
    if target_parameters !== nothing
        F = fix_parameters(F, target_parameters; compile = compile)
    end
    # if homogeneous put on on an affine chart
    if homogeneous && overdetermined
        F = on_affine_chart(F)
        push!(D, 1)
        push!(scaling, 1.0)
        homogeneous = false
        F = square_up(F)
        scaling = [LA.I F.A] * scaling
        D = D[1:n]
    elseif overdetermined
        F = square_up(F)
        scaling = [LA.I F.A] * scaling
        D = D[1:n]
    end
    if homogeneous
        G = fixed(
            System(s[1:n-1] .* (x[1:n-1] .^ D[1:n-1] .- x[end] .^ D[1:n-1]), x, s[1:n-1]);
            compile = compile,
        )
    else
        G = fixed(
            System(s .* (x .^ D .- 1), x, s);
            compile = compile,
            optimizations = false,
        )
    end
    G = fix_parameters(G, scaling)

    H = StraightLineHomotopy(G, F; γ = gamma)
    if homogeneous
        H = on_affine_chart(H)
    end
    T = EndgameTracker(H, tracker_options = tracker_options, options = endgame_options)
    if overdetermined
        T = OverdeterminedTracker(T, F)
    end
    starts = total_degree_start_solutions(D; homogeneous = homogeneous)

    T, starts
end


function total_degree_variable_groups(
    F::System;
    γ = cis(2π * rand()),
    gamma = γ,
    target_parameters = nothing,
    tracker_options = TrackerOptions(),
    endgame_options = EndgameOptions(),
    compile::Union{Bool,Symbol} = COMPILE_DEFAULT[],
    kwargs...,
)
    unsupported_kwargs(kwargs)
    m, n = size(F)
    homogeneous = is_homogeneous(F)
    D = multi_degrees(F)
    vargroups = variable_groups(F)
    projective_dims = length.(vargroups) .- homogeneous
    M = length(projective_dims)

    m ≥ (n - M * homogeneous) || throw(FiniteException(n - M * homogeneous - m))

    overdetermined = m > n - M * homogeneous

    F = fixed(F; compile = compile)
    if target_parameters !== nothing
        F = fix_parameters(F, target_parameters; compile = compile)
    end

    if homogeneous && overdetermined
        F = on_affine_chart(F, projective_dims)
        D = [D LA.I]
        homogeneous = false
        F = square_up(F)
        D = max.(D[:, 1:n], maximum(D[:, n+1:end], dims = 2))
        projective_dims .+= 1
    elseif overdetermined
        F = square_up(F)
        D = max.(D[:, 1:n], maximum(D[:, n+1:end], dims = 2))
    end

    g, C = multi_homogeneous_system(D, vargroups; homogeneous = homogeneous)
    P = parameters(g)
    p = randn(length(P))
    G = fix_parameters(fixed(g; compile = compile), p)
    starts = MultiBezoutSolutionsIterator(
        D,
        C(P => p),
        projective_dims;
        homogeneous = homogeneous,
    )
    H = StraightLineHomotopy(G, F; gamma = gamma)
    if homogeneous
        H = on_affine_chart(H, projective_dims)
    end
    T = EndgameTracker(H, tracker_options = tracker_options, options = endgame_options)
    if overdetermined
        T = OverdeterminedTracker(T, F)
    end
    T, starts
end

function multi_homogeneous_system(D, vargroups; homogeneous::Bool)
    M, n = size(D)
    projective_dims = length.(vargroups) .- homogeneous
    @unique_var c[1:n+M, 1:n]
    C = zeros(Expression, n + M, n)
    for i = 1:n
        j = 1
        for kⱼ in projective_dims
            for l = 1:kⱼ
                C[j, i] = begin
                    if (i == l && i ≤ kⱼ)
                        1
                    elseif i ≠ l && i ≤ kⱼ
                        0
                    else
                        c[j, i]
                    end
                end
                j += 1
            end
            C[j, i] = 1
            j += 1
        end
    end

    P = variables(C)

    if homogeneous
        Z = vargroups
    else
        Z = vcat.(vargroups, 1)
    end

    m, n = size(D)
    g = map(1:n) do i
        s = Ref(1)
        prod(1:m) do j
            dᵢⱼ = D[j, i]
            kⱼ = length(Z[j]) - 1
            if dᵢⱼ == 0
                s[] += kⱼ + 1
                return 1
            end
            bᵢⱼ = sum(1:kⱼ) do l
                cz = C[s[], i] * Z[j][l]
                s[] += 1
                cz
            end
            s[] += 1
            bᵢⱼ^dᵢⱼ - Z[j][end]^dᵢⱼ
        end
    end
    System(g; variable_groups = vargroups, parameters = P), C
end

struct TotalDegreeStartSolutionsIterator{Iter}
    degrees::Vector{Int}
    homogeneous::Bool
    iterator::Iter
end
function TotalDegreeStartSolutionsIterator(degrees, homogeneous::Bool)
    iterator = Iterators.product(map(d -> 0:(d-1), degrees)...)
    TotalDegreeStartSolutionsIterator(degrees, homogeneous, iterator)
end
function Base.show(io::IO, iter::TotalDegreeStartSolutionsIterator)
    print(io, "$(length(iter)) total degree start solutions for degrees $(iter.degrees)")
end

function Base.iterate(iter::TotalDegreeStartSolutionsIterator)
    indices, state = iterate(iter.iterator)
    _value(iter, indices), state
end
function Base.iterate(iter::TotalDegreeStartSolutionsIterator, state)
    it = iterate(iter.iterator, state)
    it === nothing && return nothing
    _value(iter, first(it)), last(it)
end

function _value(iter::TotalDegreeStartSolutionsIterator, indices)
    x = map((k, d) -> cis(2π * k / d), indices, iter.degrees)
    iter.homogeneous && push!(x, 1)
    x
end


Base.length(iter::TotalDegreeStartSolutionsIterator) = length(iter.iterator)
Base.eltype(iter::Type{<:TotalDegreeStartSolutionsIterator}) = Vector{Complex{Float64}}

"""
    total_degree_start_solutions(degrees)

Returns an iterator of the start solutions of the system
```math
\\begin{array}{rl}
    z_1^{d_1} - 1 &= 0 \\\\
    z_2^{d_2} - 1 &= 0 \\\\
    &\\vdots \\\\
    z_n^{d_n} - 1 &= 0 \\\\
\\end{array}
```
where ``d_i`` is `degrees[i]`.

## Example
```julia-repl
julia> iter = total_degree_start_solutions([2, 2])
4 total degree start solutions for degrees [2, 2]

julia> collect(iter)
4-element Array{Array{Complex{Float64},1},1}:
 [1.0 + 0.0im, 1.0 + 0.0im]
 [-1.0 + 1.2246467991473532e-16im, 1.0 + 0.0im]
 [1.0 + 0.0im, -1.0 + 1.2246467991473532e-16im]
 [-1.0 + 1.2246467991473532e-16im, -1.0 + 1.2246467991473532e-16im]
```
"""
total_degree_start_solutions(
    degrees::AbstractVector{<:Integer};
    homogeneous::Bool = false,
) = TotalDegreeStartSolutionsIterator(degrees, homogeneous)



###
### Multihomogeneous
###

struct MultiBezoutIndicesIterator
    D::Matrix{Int} # Degree matrix
    k::Vector{Int} # (projective) dimensions

    function MultiBezoutIndicesIterator(D::Matrix{Int}, k::Vector{Int})
        @assert size(D, 1) == length(k)
        new(D, k)
    end
end
Base.IteratorSize(::Type{MultiBezoutIndicesIterator}) = Base.SizeUnknown()
Base.eltype(::Type{MultiBezoutIndicesIterator}) = Tuple{Int,Vector{Int}}
function Base.iterate(
    iter::MultiBezoutIndicesIterator,
    state = [i for i = 1:length(iter.k) for j = 1:iter.k[i]],
)
    d = 0
    m, n = size(iter.D)
    p = state
    while d == 0
        state[1] > n && return nothing
        p, state = nextpermutation(1:m, state)
        let p = p, D = iter.D
            d = prod(i -> D[p[i], i], 1:n)::Int
        end
    end

    (d, p), state
end

# Adopted from Combinatorics.jl:
# https://github.com/JuliaMath/Combinatorics.jl/blob/8d9571402319799b29da2005a65b627e8771c1e4/src/permutations.jl#L47
function nextpermutation(m, state)
    n = length(state)
    perm = [m[state[i]] for i = 1:n]
    s = copy(state)
    i = n - 1
    while i >= 1 && s[i] >= s[i+1]
        i -= 1
    end
    if i > 0
        j = n
        while j > i && s[i] >= s[j]
            j -= 1
        end
        s[i], s[j] = s[j], s[i]
        reverse!(s, i + 1)
    else
        s[1] = n + 1
    end
    return (perm, s)
end

struct MultiBezoutSolutionsIterator
    indices::MultiBezoutIndicesIterator
    coeffs::Matrix{Float64}
    homogeneous::Bool
    # precomputed stuff and cache
    roots_of_unity::Matrix{ComplexF64} # Lower Triangular matrix storing the precomputed roots
    A::Vector{Matrix{Float64}}
    b::Vector{Vector{ComplexF64}}
    ranges::Vector{UnitRange{Int}}
end

function MultiBezoutSolutionsIterator(
    indices::MultiBezoutIndicesIterator,
    coeffs::Matrix;
    homogeneous::Bool,
)
    d_max = maximum(indices.D)
    roots_of_unity = zeros(ComplexF64, d_max, d_max)
    for i = 1:d_max, j = 0:i-1
        roots_of_unity[i, j+1] = cis(2π * j / i)
    end
    A = map(kⱼ -> zeros(kⱼ, kⱼ), indices.k)
    b = map(kⱼ -> zeros(ComplexF64, kⱼ), indices.k)
    r_stop = Ref(1)
    ranges = map(indices.k) do kⱼ
        range = r_stop[]:(kⱼ+r_stop[]-1)
        r_stop[] += kⱼ + 1
        range
    end
    MultiBezoutSolutionsIterator(indices, coeffs, homogeneous, roots_of_unity, A, b, ranges)
end
function MultiBezoutSolutionsIterator(
    D::Matrix,
    C::Matrix,
    proj_dims::Vector{Int};
    kwargs...,
)
    MultiBezoutSolutionsIterator(MultiBezoutIndicesIterator(D, proj_dims), C; kwargs...)
end

function Base.show(io::IO, iter::MultiBezoutSolutionsIterator)
    print(io, "Solutions iterator for a multi-homogeneous start system")
end
Base.show(io::IO, ::MIME"application/prs.juno.inline", x::MultiBezoutSolutionsIterator) = x
Base.IteratorSize(::Type{<:MultiBezoutSolutionsIterator}) = Base.SizeUnknown()
Base.eltype(::Type{MultiBezoutSolutionsIterator}) = Vector{ComplexF64}

function Base.iterate(iter::MultiBezoutSolutionsIterator)
    (_, perm), indices_state = iterate(iter.indices)
    dᵢ = ntuple(i -> iter.indices.D[perm[i], i], length(perm))
    d_iter = Iterators.product(map(dᵢ -> 1:dᵢ, dᵢ)...)
    q, d_state = iterate(d_iter)

    x = compute_solution(iter, perm, q, dᵢ)
    x, (perm, indices_state, dᵢ, d_iter, d_state)
end

function Base.iterate(iter::MultiBezoutSolutionsIterator, state)
    perm, indices_state, dᵢ, d_iter, d_state = state

    next_d_state = iterate(d_iter, d_state)
    if next_d_state !== nothing
        q, d_state = next_d_state
        x = compute_solution(iter, perm, q, dᵢ)
        return x, (perm, indices_state, dᵢ, d_iter, d_state)
    end

    # We exhausted our current batch. move indices forward
    indices_newstate = iterate(iter.indices, indices_state)
    # check whether we are completely done
    indices_newstate === nothing && return nothing

    (_, perm), indices_state = indices_newstate
    # Create new d_iter
    new_dᵢ::typeof(dᵢ) = let D = iter.indices.D, perm = perm
        ntuple(i -> D[perm[i], i], length(perm))
    end
    new_d_iter::typeof(d_iter) = Iterators.product(map(dᵢⱼ -> 1:dᵢⱼ, new_dᵢ)...)
    q, d_state = iterate(new_d_iter)
    x = compute_solution(iter, perm, q, new_dᵢ)
    return x, (perm, indices_state, new_dᵢ, new_d_iter, d_state)
end


function compute_solution(iter::MultiBezoutSolutionsIterator, perm, q, dᵢ)
    m, n = size(iter.indices.D)
    # for each variable group we have to solve a linear system
    t = 1
    if iter.homogeneous
        data = zeros(ComplexF64, n + m)
    else
        data = zeros(ComplexF64, n)
    end
    for j = 1:m
        kⱼ = iter.indices.k[j]
        Aⱼ = iter.A[j]
        bⱼ = iter.b[j]
        s = 1
        for i = 1:n
            perm[i] == j || continue
            range = iter.ranges[j]
            for (t, l) in enumerate(range)
                Aⱼ[s, t] = iter.coeffs[l, i]
            end
            bⱼ[s] = iter.roots_of_unity[dᵢ[i], q[i]]
            s += 1
        end

        LinearAlgebra.ldiv!(LinearAlgebra.lu!(Aⱼ), bⱼ)

        data[t:(t+kⱼ-1)] .= bⱼ
        if iter.homogeneous
            data[t+kⱼ] = 1
            t += kⱼ + 1
        else
            t += kⱼ
        end
    end

    data
end


function paths_to_track(f, ::Val{:total_degree})
    target_parameters = nparameters(f) > 0 ? zeros(nparameters(f)) : nothing
    _, starts = total_degree(f; target_parameters = target_parameters)
    if Base.IteratorSize(typeof(starts)) == Base.SizeUnknown()
        k = 0
        for s in starts
            k += 1
        end
        k
    else
        length(starts)
    end
end
@deprecate bezout_number(f::Union{System,AbstractSystem}) paths_to_track(
    f;
    start_system = :total_degree,
)
