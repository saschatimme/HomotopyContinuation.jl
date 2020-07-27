struct InstructionList
    instructions::Vector{Tuple{Symbol,Tuple{Symbol,Any,Any}}}
    var::Symbol
    n::Base.RefValue{Int}
end

function InstructionList(; var::Symbol = :_ι, n::Base.RefValue{Int} = Ref(0))
    InstructionList(Tuple{Symbol,Tuple{Symbol,Any,Any}}[], var, n)
end

function instruction_list(
    exprs::Vector{Expression};
    subexpressions::Dict{Expression,Expression} = Dict{Expression,Expression}(),
    perform_cse::Bool = true,
)
    v = InstructionList()
    PSE = Set{Symbol}()
    if perform_cse
        exprs, CSE = cse(exprs)
        merge!(CSE, subexpressions)
    else
        CSE = subexpressions
    end
    v, map(ex -> flat_expr!(v, ex, CSE, PSE), exprs)
end

Base.length(v::InstructionList) = v.n[]
function Base.show(io::IO, v::InstructionList)
    for (id, arg) in v.instructions
        println(io, :($id = $(Expr(:call, arg...))))
    end
end

function Base.push!(v::InstructionList, x)
    id = Symbol(v.var, v.n[] += 1)
    push!(v.instructions, (id, x))
    id
end
function Base.push!(v::InstructionList, x::Pair{Symbol,<:Tuple{Symbol,Any,Any}})
    push!(v.instructions, (first(x), last(x)))
    first(x)
end

function add!(list::InstructionList, @nospecialize(a), @nospecialize(b))
    if a === nothing
        b === nothing && return nothing
        return b
    else
        b === nothing && return a
        return push!(list, (:+, a, b))
    end
end
function sub!(list::InstructionList, @nospecialize(a), @nospecialize(b))
    if a === nothing
        b === nothing && return nothing
        return push!(list, (:-, b, nothing))
    else
        b === nothing && return a
        return push!(list, (:-, a, b))
    end
end

function mul!(list::InstructionList, @nospecialize(a), @nospecialize(b))
    (a === nothing || b === nothing) && return nothing
    if a == 1
        b == 1 && return a
        return b
    else
        b == 1 && return a
        return push!(list, (:*, a, b))
    end
end

function muladd!(
    list::InstructionList,
    @nospecialize(a),
    @nospecialize(b),
    @nospecialize(c),
)
    add!(list, c, mul!(list, a, b))
end

function div!(list::InstructionList, @nospecialize(a), @nospecialize(b))
    a === nothing && return nothing
    b == 1 && return a
    push!(list, (:/, a, b))
end

function pow!(list, @nospecialize(a), b::Int)
    @assert b ≥ 0
    b == 1 && return a
    b == 0 && return 1
    a == 1 && return a
    a === nothing && return nothing
    push!(list, (:^, a, b))
end

function flat_expr!(
    v,
    ex::Basic,
    # common sub expression
    CSE::Dict{Expression,Expression},
    # processed sub expression
    PSE::Set{Symbol} = Set{Symbol}(),
)
    t = class(ex)
    if t == :Symbol
        s = Symbol(to_string(ex))
        if haskey(CSE, ex) && s ∉ PSE
            push!(PSE, s)
            val = flat_expr!(v, CSE[ex], CSE, PSE)
            if val isa Symbol
                v.instructions[end] = (s, last(v.instructions[end]))
            else
                push!(v, s => val)
            end
        end
        return s
    elseif t == :Pow
        x, k = ModelKit.args(ex)
        return push!(v, (:^, flat_expr!(v, x, CSE, PSE), convert(Int, k)))
    elseif t == :Add || t == :Mul
        op = t == :Add ? :+ : :*
        vec = ModelKit.args(ex)
        op_arg = flat_expr!(v, vec[1], CSE, PSE)
        for i = 2:length(vec)
            op_arg = push!(v, (op, op_arg, flat_expr!(v, vec[i], CSE, PSE)))
        end
        return op_arg
    elseif t == :Div
        x, y = ModelKit.args(ex)
        return push!(v, (:/, flat_expr!(v, x, CSE, PSE), flat_expr!(v, y, CSE, PSE)))
    else
        return to_number(ex)
    end
end


struct DiffMap
    D::Dict{Tuple{Symbol,Int},Any}
end
DiffMap() = DiffMap(Dict{Tuple{Symbol,Int},Any}())

Base.getindex(D::DiffMap, s::Symbol, i::Int) = i == 0 ? s : get(D.D, (s, i), nothing)
Base.getindex(D::DiffMap, s, i) = i == 0 ? s : nothing
Base.setindex!(D::DiffMap, v::Any, s::Symbol, i::Int) = D.D[(s, i)] = v
Base.setindex!(D::DiffMap, v::Nothing, s::Symbol, i::Int) = nothing

function Base.diff(list::InstructionList, vars::Vector{Symbol}, f::Vector{Symbol})
    diff_map = DiffMap()
    for (i, v) in enumerate(vars)
        diff_map[v, i] = 1
    end
    dlist = diff!(list, length(vars), diff_map)

    J = Matrix{Union{Nothing,Symbol,Number}}(undef, length(f), length(vars))
    for (j, v) in enumerate(vars), (i, fi) in enumerate(f)
        J[i, j] = diff_map[fi, j]
    end
    dlist, J
end

"""
    diff!(list::InstructionList, N::Int, diff_map)

Returns an `InstructionList` list computing the derivatives of `N` variables.
The derivatives of the instructions are stored in `diff_map`.
"""
function diff!(list::InstructionList, N::Int, D::DiffMap)
    n = length(list)
    v = InstructionList(n = Ref(n))
    for (id, el) in list.instructions
        (op, arg1, arg2) = el
        if op == :^
            p2 = :NONE
            instr_added = false
            for ∂i = 1:N
                exp::Int = arg2
                arg1_∂i = D[arg1, ∂i]
                if arg1_∂i !== nothing
                    if p2 == :NONE
                        p1 = pow!(v, arg1, exp - 1)
                        p2 = mul!(v, exp, p1)
                    end
                    if !instr_added
                        if exp == 2
                            push!(v, id => (:^, arg1, 2))
                        else
                            push!(v, id => (:*, p1, arg1))
                        end
                        instr_added = true
                    end
                    D[id, ∂i] = mul!(v, p2, arg1_∂i)
                elseif p2 != :NONE && !instr_added
                    push!(v, id => el)
                    instr_added = true
                end

            end
            if !instr_added
                push!(v, id => el)
            end
        elseif op == :*
            for ∂i = 1:N
                ∂i == 1 && push!(v, id => el)
                a = mul!(v, D[arg1, ∂i], arg2)
                b = mul!(v, arg1, D[arg2, ∂i])
                D[id, ∂i] = add!(v, a, b)
            end
        elseif op == :/
            for ∂i = 1:N
                ∂i == 1 && push!(v, id => el)

                D[id, ∂i] = div!(
                    v,
                    sub!(v, mul!(v, D[arg1, ∂i], arg2), mul!(v, arg1, D[arg2, ∂i])),
                    pow!(v, arg2, 2),
                )
            end
        elseif op == :+
            for ∂i = 1:N
                ∂i == 1 && push!(v, id => el)
                D[id, ∂i] = add!(v, D[arg1, ∂i], D[arg2, ∂i])
            end
        elseif op == :-
            for ∂i = 1:N
                ∂i == 1 && push!(v, id => el)
                D[id, ∂i] = sub!(v, D[arg1, ∂i], D[arg2, ∂i])
            end
        end
    end
    v
end

## Higher order AD. This follows Chapter 13 of Griewank and Walther
"""
    univariate_diff!(list::InstructionList, K::Int, diff_map)

Returns an `InstructionList` list computing the first `K` derivatives.
The derivatives of the instructions are stored in `diff_map`.
"""
function univariate_diff!(list::InstructionList, K::Int, D::DiffMap)
    n = length(list)
    v = InstructionList(n = Ref(n))
    for (id, el) in list.instructions
        (op, arg1, arg2) = el
        if op == :^
            r = arg2
            arg1_1 = D[arg1, 1]
            if arg1_1 !== nothing && r == 2
                push!(v, id => el)
                for k = 1:K
                    w_k = nothing
                    # Compute ∑_{j=0}^k u_j U_{k - j} =
                    #         2(∑_{j=0}^div(k - 1,2) u_j U_{k - j}) +
                    #         iseven(k) * u_{k-1}^2
                    for j = 0:div(k - 1, 2)
                        w_k = muladd!(v, D[arg1, j], D[arg1, k-j], w_k)
                    end
                    w_k = mul!(v, 2, w_k)
                    if iseven(k)
                        w_k = add!(v, w_k, pow!(v, D[arg1, div(k, 2)], 2))
                    end
                    D[id, k] = w_k
                end
            elseif arg1_1 !== nothing
                w_0 = push!(v, id => (:^, arg1, r))
                u_0_inv = push!(v, (:inv_not_zero, arg1, nothing))
                # w_k is v_k in Griewank
                w = Symbol[]
                for k = 1:K
                    s = nothing
                    for j = 1:k
                        ũ_j = mul!(v, j, D[arg1, j])
                        w_kj = k == j ? w_0 : w[k-j]
                        s = muladd!(v, w_kj, ũ_j, s)
                    end
                    s = mul!(v, r, s)

                    t = nothing
                    for j = 1:k-1
                        u_kj = D[arg1, k-j]
                        w̃_j = mul!(v, j, w[j])
                        t = muladd!(v, u_kj, w̃_j, t)
                    end
                    w̃_k = mul!(v, u_0_inv, sub!(v, s, t))
                    w_k = div!(v, w̃_k, k)
                    push!(w, w_k)
                    D[id, k] = w_k
                end
            else
                push!(v, id => el)
            end
        elseif op == :*
            push!(v, id => el)
            for k = 1:K
                c_k = nothing
                for j = 0:k
                    c_k = muladd!(v, D[arg1, j], D[arg2, k-j], c_k)
                end
                # c_k === nothing && break
                D[id, k] = c_k
            end
        elseif op == :+
            push!(v, id => el)
            for k = 1:K
                D[id, k] = add!(v, D[arg1, k], D[arg2, k])
            end
        elseif op == :/
            push!(v, id => el)
            for k = 1:K
                s = nothing
                for j = 0:(k-1)
                    s = muladd!(v, D[id, j], D[arg2, k-j], s)
                end
                D[id, k] = div!(v, sub!(v, D[arg1, k], s), arg2)
            end
        end
    end
    v
end

@inline sqr(x) = x^2
@inline function sqr(z::Complex)
    x, y = reim(z)
    Complex((x + y) * (x - y), (x + x) * y)
end

quick_div(a, b::Complex) = Base.FastMath.div_fast(a, b)
quick_div(a, b) = a / b

"""
    inv_not_zero(x)

Invert x unless it is 0, then return 0.
"""
@inline inv_not_zero(x) = ifelse(iszero(x), x, @fastmath inv(x))

function unroll_pow(var, n)
    n == 0 && return :(one($var))
    n == 1 && return var
    n == 2 && return :(sqr($var))
    n == 3 && return :($var * sqr($var))
    n == 4 && return :(sqr(sqr($var)))
    n == 5 && return :($var * sqr(sqr($var)))

    # base to expansion shows up which power it is needed to compute
    d = digits(n, base = 2)
    x = :x
    exprs = map(2:length(d)) do i
        :(local $(Symbol(x, 1 << (i - 1))) = sqr($(Symbol(x, 1 << (i - 2)))))
    end
    prods = Symbol[]
    for (i, di) in enumerate(d)
        if !iszero(di)
            push!(prods, Symbol(x, 1 << (i - 1)))
        end
    end
    if length(prods) > 1
        push!(exprs, Expr(:call, :*, prods...))
    end
    Expr(:let, :($(Symbol(x, 1)) = $var), Expr(:block, exprs...))
end

function to_expr(
    list::InstructionList,
    var_map = Dict{Symbol,Union{Expr,Symbol}}(),
    assignements = Dict{Symbol,Vector{Expr}}(),
)
    block = Expr(:block)
    exprs = block.args
    for (id, (op, arg1, arg2)) in list.instructions
        if op == :^
            x::Union{Symbol,Expr} = arg1
            if arg2 isa Int
                k::Int = arg2
                if k < 0
                    push!(exprs, :($id = inv($(unroll_pow(get(var_map, x, x), -k)))))
                else
                    push!(exprs, :($id = $(unroll_pow(get(var_map, x, x), k))))
                end
            else
                push!(exprs, :($id = $(get(var_map, x, x))^$arg2))
            end
        elseif op == :/
            a = get(var_map, arg1, arg1)
            b = get(var_map, arg2, arg2)
            push!(exprs, :($id = quick_div($a, $b)))
        else
            a = get(var_map, arg1, arg1)
            if arg2 !== nothing
                b = get(var_map, arg2, arg2)
                push!(exprs, :($id = $(Expr(:call, op, a, b))))
            else
                push!(exprs, :($id = $(Expr(:call, op, a))))
            end
        end

        if haskey(assignements, id)
            append!(exprs, assignements[id])
        end
    end
    block
end
