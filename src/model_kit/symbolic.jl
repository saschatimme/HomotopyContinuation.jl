## Variable construction

Variable(name::Union{Symbol,AbstractString}, indices::Int...) =
    Variable("$(name)$(join(map_subscripts.(indices), "₋"))")

const SUBSCRIPTS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
const SUBSCRIPT_MAP = Dict([first(string(i)) => SUBSCRIPTS[i+1] for i = 0:9])
const SUBSCRIPT_TO_INT_MAP = Dict([SUBSCRIPTS[i+1] => i for i = 0:9])
map_subscripts(index) = join(SUBSCRIPT_MAP[c] for c in string(index))

function sort_key(v::Variable)
    name = string(ModelKit.name(v))
    sub_start = findnext(c -> c in ModelKit.SUBSCRIPTS, name, 1)
    if isnothing(sub_start)
        return name, Int[]
    end
    var_base = name[1:prevind(name, sub_start)]
    str_indices = split(name[sub_start:end], "₋")
    indices = map(str_indices) do str_index
        digits = map(s -> ModelKit.SUBSCRIPT_TO_INT_MAP[s], collect(str_index))
        n = length(digits)
        sum([digits[n-k+1] * 10^(k - 1) for k = 1:n])
    end
    var_base, reverse(indices)
end
Base.isless(a::Variable, b::Variable) = isless(sort_key(a), sort_key(b))
Base.sort!(vs::AbstractVector{Variable}; kwargs...) =
    permute!(vs, sortperm(sort_key.(vs); kwargs...))

Symbol(v::Variable) = name(v)


"""
    @var variable1 variable2 ...

Declare variables with the given names and automatically create the variable bindings.
The macro supports indexing notation to create `Array`s of variables.

## Examples

```julia-repl
julia> @var a b x[1:2] y[1:2,1:3]
(a, b, Variable[x₁, x₂], Variable[y₁₋₁ y₁₋₂ y₁₋₃; y₂₋₁ y₂₋₂ y₂₋₃])

julia> a
a

julia> b
b

julia> x
2-element Array{Variable,1}:
 x₁
 x₂

julia> y
2×3 Array{Variable,2}:
 y₁₋₁  y₁₋₂  y₁₋₃
 y₂₋₁  y₂₋₂  y₂₋₃
```
"""
macro var(args...)
    vars, exprs = buildvars(args; unique = false)
    :($(foldl((x, y) -> :($x; $y), exprs, init = :()));
    $(Expr(:tuple, esc.(vars)...)))
end

"""
    @unique_var variable1 variable2

This is similar to [`@var`](@ref) with the only difference that the macro automatically
changes the names of the variables to ensure uniqueness. However, the binding is still
to the declared name.
This is useful to ensure that there are no name collisions.

## Examples

```julia-repl
julia> @unique_var a b
(a#591, b#592)

julia> a
a#591

julia> b
b#592
```
"""
macro unique_var(args...)
    vars, exprs = buildvars(args; unique = true)
    :($(foldl((x, y) -> :($x; $y), exprs, init = :()));
    $(Expr(:tuple, esc.(vars)...)))
end

"""
    variables(prefix::Union{Symbol,String}, indices...)

Create an `Array` of variables with the given `prefix` and indices.
The expression  `@var x[1:3, 1:2]` is equivalent to  `x = variables(:x, 1:3, 1:2)`.
"""
function variables(prefix::Union{Symbol,String}, indices...)
    map(i -> Variable(prefix, i...), Iterators.product(indices...))
end

function buildvar(var; unique::Bool = false)
    if isa(var, Symbol)
        if unique
            # drop the first two ## from the gensym var
            varname = Symbol(String(gensym(var))[3:end])
        else
            varname = var
        end
        var, :($(esc(var)) = Variable($"$varname"))
    else
        isa(var, Expr) || error("Expected $var to be a variable name")
        Base.Meta.isexpr(var, :ref) ||
            error("Expected $var to be of the form varname[idxset]")
        (2 ≤ length(var.args)) || error("Expected $var to have at least one index set")
        varname = var.args[1]
        if unique
            prefix = String(gensym(varname))[3:end]
        else
            prefix = string(varname)
        end
        varname, :($(esc(varname)) = variables($prefix, $(esc.(var.args[2:end])...)))
    end
end

function buildvars(args; unique::Bool = false)
    vars = Symbol[]
    exprs = []
    for arg in args
        if arg isa Expr && arg.head == :tuple
            for _arg in arg.args
                var, expr = buildvar(_arg; unique = unique)
                push!(vars, var)
                push!(exprs, expr)
            end
        else
            var, expr = buildvar(arg; unique = unique)
            push!(vars, var)
            push!(exprs, expr)
        end
    end
    vars, exprs
end

"""
    unique_variable(var::Symbol, variables, parameters)

Create a variable with a unique name which doesn't clash with `variables` or
`parameters`.
If `var` is not possible the names `var##k` for `k=0,1,...` are tried until
one is possible,
"""
function unique_variable(var::Symbol, vars::Vector{Variable}, params::Vector{Variable})
    v = Variable(var)
    k = 0
    while (v in vars || v in params)
        v = Variable("$var##$k")
        k += 1
    end
    v
end


Base.adjoint(expr::Basic) = expr
Base.conj(expr::Basic) = expr
Base.transpose(expr::Basic) = expr
Base.broadcastable(v::Basic) = Ref(v)

"""
    variables(expr::Expression; parameters = Variable[])
    variables(exprs::AbstractVector{Expression}; parameters = Variable[])

Obtain all variables used in the given expression up to the ones declared in `parameters`.

## Example

```julia-repl
julia> @var x y a;
julia> variables(x^2 + y)
2-element Array{Variable,1}:
 x
 y

julia> variables([x^2 + a, y]; parameters = [a])
2-element Array{Variable,1}:
 x
 y
```
"""
variables(expr::Basic; kwargs...) = variables([expr]; kwargs...)
function variables(exprs::AbstractArray{<:Basic}; parameters = Variable[])
    S = Set{Variable}()
    for expr in exprs
        union!(S, _variables(expr))
    end
    setdiff!(sort!(collect(S)), parameters)
end

"""
    nvariables(expr::Expression; parameters = Variable[])
    nvariables(exprs::AbstractVector{Expression}; parameters = Variable[])

Obtain the number of variables used in the given expression not counting the the ones
declared in `parameters`.
"""
nvariables(exprs::Basic; kwargs...) = length(variables(exprs; kwargs...))
nvariables(exprs::AbstractArray{<:Basic}; kwargs...) = length(variables(exprs; kwargs...))

"""
    subs(expr::Expression, subsitutions::Pair...)
    subs(exprs::AbstractVector{<:Expression}, subsitutions::Pair...)

Apply the given substitutions to the given expressions.

## Examples
```julia-repl
@var x y

julia> subs(x^2, x => y)
y ^ 2

julia> subs(x * y, [x,y] => [x+2,y+2])
(x + 2) * (y + 2)

julia> subs([x + y, x^2], x => y + 2, y => x + 2)
2-element Array{Expression,1}:
 4 + x + y
 (2 + y)^2

# You can also use the callable syntax
julia> (x * y)([x,y] => [x+2,y+2])
 (x + 2) * (y + 2)
```
"""
subs(ex::Basic, args...) = subs(ex, ExpressionMap(args...))
subs(exs::AbstractArray{<:Basic}, args...) = subs.(exs, Ref(ExpressionMap(args...)))

"""
    evaluate(expr::Expression, subs...)
    evaluate(expr::AbstractArray{Expression}, subs...)

Evaluate the given expression.

## Example

```julia-repl
julia> @var x y;

julia> evaluate(x^2, x => 2)
4

julia> evaluate(x * y, [x,y] => [2, 3])
6

julia> evaluate([x^2, x * y], [x,y] => [2, 3])
2-element Array{Int64,1}:
 4
 6

# You can also use the callable syntax
julia> [x^2, x * y]([x,y] => [2, 3])
2-element Array{Int64,1}:
 4
 6
```
"""
function evaluate(expr::AbstractArray{<:Basic}, args...)
    out = map(to_number, subs(expr, args...))
    if isabstracttype(eltype(out))
        return to_smallest_eltype(out)
    else
        out
    end
end
evaluate(expr::Basic, args...) = to_number(subs(expr, args...))
(f::Union{Basic,AbstractArray{<:Basic}})(args...) = evaluate(f, args...)

"""
    evaluate!(u, expr::AbstractArray{Expression}, subs...)

Inplace form of [`evaluate`](@ref).
"""
function evaluate!(u::AbstractArray, expr::AbstractArray{<:Basic}, args...)
    map!(to_number, u, subs(expr, args...))
end

"""
    differentiate(expr::Expression, var::Variable, k = 1)
    differentiate(expr::AbstractVector{Expression}, var::Variable, k = 1)

Compute the `k`-th derivative of `expr` with respect to the given variable `var`.

    differentiate(expr::Expression, vars::AbstractVector{Variable})

Compute the partial derivatives of `expr` with respect to the given variable variables `vars`.
Retuns a `Vector` containing the partial derivatives.

    differentiate(exprs::AbstractVector{Expression}, vars::AbstractVector{Variable})

Compute the partial derivatives of `exprs` with respect to the given variable variables `vars`.
Returns a `Matrix` where the each row contains the partial derivatives for a given expression.
"""
function differentiate(expr::Basic, vars::AbstractVector{Variable})
    [differentiate(expr, v) for v in vars]
end
function differentiate(exprs::AbstractVector{<:Basic}, var::Variable, k::Int = 1)
    [differentiate(e, var, k) for e in exprs]
end
function differentiate(exprs::AbstractVector{<:Basic}, vars::AbstractVector{Variable})
    [differentiate(e, v) for e in exprs, v in vars]
end

"""
    monomials(variables::AbstractVector, d::Integer; affine = true)

Create all monomials of a given degree in the given `variables`.

```julia-repl
julia> @var x y
(x, y)

julia> monomials([x,y], 2; affine = false)
3-element Array{Operation,1}:
 x ^ 2
 x * y
 y ^ 2
```
"""
function monomials(
    vars::AbstractVector{<:Union{Variable,Expression}},
    d::Integer;
    affine::Bool = true,
    homogeneous::Bool = !affine,
)
    n = length(vars)
    exps = monomials_exponents(n, d; affine = !homogeneous)
    map(exps) do exp
        prod(i -> vars[i]^exp[i], 1:n)
    end
end
function monomials_exponents(n, d; affine::Bool)
    if affine
        pred = x -> sum(x) ≤ d
    else
        pred = x -> sum(x) == d
    end
    E = map(Iterators.filter(pred, Iterators.product(Iterators.repeated(0:d, n)...))) do e
        collect(e)
    end

    sort!(E, lt = td_order)
    E
end

function td_order(x, y)
    sx = sum(x)
    sy = sum(y)
    sx == sy ? x > y : sx > sy
end
function monomials(
    vars::AbstractVector{<:Union{Variable,Expression}},
    D::AbstractVector{<:Integer},
)
    D = sort(D; rev = true)
    M = monomials(vars, D[1]; affine = false)
    for i = 2:length(D)
        append!(M, monomials(vars, D[i]; affine = false))
    end
    M
end

"""
    dense_poly(vars::AbstractVector{Variable}, d::Integer;
               homogeneous = false,
               coeff_name::Symbol = gensym(:c))

Create a dense polynomial of degree `d` in the given variables `variables` where
each coefficient is a parameter. Returns a tuple with the first argument being the polynomial
and the second the parameters.

```julia-repl
julia> @var x y;

julia> f, c = dense_poly([x, y], 2, coeff_name = :q);

julia> f
 q₆ + x*q₄ + x^2*q₁ + y*q₅ + y^2*q₃ + x*y*q₂

julia> c
6-element Array{Variable,1}:
 q₁
 q₂
 q₃
 q₄
 q₅
 q₆
```
"""
function dense_poly(
    vars::AbstractVector{<:Union{Variable,Expression}},
    d::Integer;
    homogeneous::Bool = false,
    coeff_name::Symbol = gensym(:c),
)
    M = monomials(vars, d; affine = !homogeneous)
    c = Variable.(coeff_name, 1:length(M))
    sum(c .* M), c
end

"""
    coeffs_as_dense_poly(f, vars, d; homogeneous = false)

Given a polynomial `f` this returns a vector `c` of coefficients such that
`subs(dense_poly(vars, d; homogeneous = homogeneous), c) == f`.

## Example

```julia
@var x[1:3]
f, c = dense_poly(x, 3, coeff_name = :c)
g = x[1]^3+x[2]^3+x[3]^3-1
gc = coeffs_as_dense_poly(g, x, 3)
subs(f, c => gc) == g
```
```
true
```
"""
function coeffs_as_dense_poly(
    f::Expression,
    vars::AbstractVector{<:Union{Variable,Expression}},
    d::Integer;
    homogeneous::Bool = false,
)
    exps = monomials_exponents(length(vars), d; affine = !homogeneous)
    D = to_dict(expand(f), vars)
    to_smallest_eltype(map(exps) do exp
        if haskey(D, exp)
            to_number(D[exp])
        else
            false
        end
    end)
end


"""
    rand_poly(T = ComplexF64, vars::AbstractVector{Variable}, d::Integer; homogeneous::Bool = false)

Create a random dense polynomial of degree `d` in the given variables `variables`.
Each coefficient is sampled independently via `randn(T)`.

```julia-repl
julia> @var x y;

julia> rand_poly(Float64, [x, y], 2)
0.788764085756728 - 0.534507647623108*x - 0.778441366874946*y -
 0.128891763280247*x*y + 0.878962738754971*x^2 + 0.550480741774464*y^2
```
"""
function rand_poly(vars::AbstractVector, d::Integer; kwargs...)
    rand_poly(ComplexF64, vars, d; kwargs...)
end
function rand_poly(T, vars::AbstractVector, d::Integer; homogeneous::Bool = false)
    M = monomials(vars, d; affine = !homogeneous)
    sum(randn(T, length(M)) .* M)
end


"""
    expand(e::Expression)

Expand a given expression.

```julia-repl
julia> @var x y
(x, y)

julia> expand((x + y) ^ 2)
2*x*y + x^2 + y^2
```
"""
function expand(e::Basic)
    func_symbols = function_symbols(e)
    for x in func_symbols
        func_name = sym_func_name(x)
        if func_name == :det
            avec = collect(args(x))
            n = round(Int, sqrt(length(avec)))
            A = reshape(avec, n, n)
            e = subs(e, x => expand_det(A))
        else
            throw(ArgumentError(string("Unsupported function symbol: ", func_name)))
        end
    end
    symengine_expand(e)
end

"""
    to_dict(expr::Expression, vars::AbstractVector{Variable})

Return the coefficients of `expr` w.r.t. the given variables `vars`. Assumes that `expr`
is expanded.
"""
function to_dict(expr::Expression, vars::AbstractVector{Variable})
    mul_args, pow_args = ExprVec(), ExprVec()
    dict = Dict{Vector{Int},Expression}()

    if class(expr) == :Add
        for op in args(expr)
            to_dict_op!(dict, op, vars, mul_args, pow_args)
        end
    else
        to_dict_op!(dict, expr, vars, mul_args, pow_args)
    end

    dict
end

function to_dict_op!(dict, op, vars, mul_args, pow_args)
    cls = class(op)
    d = zeros(Int, length(vars))
    coeff = Expression(1)
    if cls == :Mul
        op_args = args!(mul_args, op)
        for arg in op_args
            arg_cls = class(arg)
            is_coeff = true
            if arg_cls == :Symbol
                for (i, v) in enumerate(vars)
                    if v == arg
                        d[i] = 1
                        is_coeff = false
                        break
                    end
                end
            elseif arg_cls == :Pow
                vec = args!(pow_args, arg)
                x = vec[1]
                for (i, v) in enumerate(vars)
                    if x == v
                        d[i] = convert(Int, vec[2])
                        is_coeff = false
                        break
                    end
                end
            end
            if is_coeff
                mul!(coeff, coeff, arg)
            end
        end
    elseif cls == :Symbol
        is_in_vars = false
        for (i, v) in enumerate(vars)
            if v == op
                d[i] = 1
                is_in_vars = true
                break
            end
        end
        if !is_in_vars
            coeff = copy(op)
        end
    elseif cls == :Pow
        vec = args!(pow_args, op)
        # check that base is one of the variables
        x = vec[1]
        is_var_pow = false
        for (i, v) in enumerate(vars)
            if x == v
                d[i] = convert(Int, vec[2])
                is_var_pow = true
                break
            end
        end
        if !is_var_pow
            coeff = copy(op)
        end
    else
        coeff = copy(op)
    end

    if haskey(dict, d)
        add!(dict[d], dict[d], coeff)
    else
        dict[d] = coeff
    end
    dict
end

"""
    exponents_coefficients(f::Expression, vars::AbstractVector{Variable}; expanded = false)

Return a matrix `M` containing the exponents for all occuring terms
(one term per column) and a vector `c` containing the corresponding coefficients.
Expands the given expression `f` unless `expanded = true`.
"""
function exponents_coefficients(
    f::Expression,
    vars::AbstractVector{Variable};
    expanded::Bool = false,
    unpack_coeffs::Bool = true,
)
    expanded || (f = expand(f))
    D = to_dict(f, vars)
    # make exponents to matrix
    m = length(D)
    E = collect(keys(D))
    perm = sortperm(E; lt = td_order)
    permute!(E, perm)
    n = length(vars)
    M = zeros(Int32, n, m)
    for (j, Eⱼ) in enumerate(E)
        for (i, dᵢ) in enumerate(Eⱼ)
            M[i, j] = dᵢ
        end
    end
    if unpack_coeffs
        coeffs = to_smallest_eltype(to_number.(values(D)))
    else
        coeffs = collect(values(D))
    end
    permute!(coeffs, perm)
    M, coeffs
end

"""
    coefficients(f::Expression, vars::AbstractVector{Variable}; expanded = false)

Return all coefficients of the given polynomial `f` for the given variables `vars`.
If `expanded = true` then this assumes that the expression `f` is already expanded,
e.g., with [`expand`](@ref).
"""
function coefficients(f::Expression, vars::AbstractVector{Variable}; expanded::Bool = false)
    expanded || (f = expand(f))
    D = to_dict(f, vars)
    m = length(D)
    E = collect(keys(D))
    perm = sortperm(E; lt = td_order)
    make_to_number = true
    for c in values(D)
        if !is_number(c)
            make_to_number = false
            break
        end
    end

    if make_to_number
        permute!(to_number.(values(D)), perm)
    else
        permute!(collect(values(D)), perm)
    end
end
coefficients(f::Expression, var::Variable; kwargs...) = coefficients(f, [var]; kwargs...)

"""
    degree(f::Expression, vars = variables(f); expanded = false)

Compute the degree of the expression `f`  in `vars`.
Unless `expanded` is `true` the expression is first expanded.
"""
function degree(
    f::Expression,
    vars::AbstractVector{Variable} = variables(f);
    expanded::Bool = false,
)
    if !expanded
        f = ModelKit.expand(f)
    end
    dicts = ModelKit.to_dict(f, vars)
    maximum(sum, keys(dicts))
end

"""
    degrees(f::AbstractVector{Expression}, vars = variables(f); expanded = false)

Compute the degrees of the expressions `f` in `vars`.
Unless `expanded` is `true` the expressions are first expanded.
"""
function degrees(
    f::AbstractVector{Expression},
    vars::AbstractVector{Variable} = variables(f);
    expanded::Bool = false,
)
    if !expanded
        f = expand.(f)
    end
    dicts = to_dict.(f, Ref(vars))
    maximum.(sum, keys.(dicts))
end


function expand_det(M::AbstractMatrix{<:Basic})
    m, n = size(M)
    m == n || error("Cannot take determinant of non-square matrix.")
    if m > 2
        return sum((-1)^(i - 1) * M[i, 1] * expand_det(M[1:end.!=i, 2:end]) for i = 1:m)
    elseif m == 2
        return M[1, 1] * M[2, 2] - M[2, 1] * M[1, 2]
    else # m == 1
        return M[1, 1]
    end
end

LinearAlgebra.det(M::AbstractMatrix{<:Union{Variable,Expression}}) =
    make_sym_func("det", vec(M))
LinearAlgebra.det(M::LinearAlgebra.Symmetric{<:Union{Variable,Expression}}) =
    LinearAlgebra.det(Matrix(M))

function is_homogeneous(f::Expression, vars::Vector{Variable}; expanded::Bool = false)
    if !expanded
        f = expand(f)
    end
    exponents = keys(to_dict(f, vars))
    d = sum(first(exponents))
    all(e -> sum(e) == d, exponents)
end
################
# Optimization #
################

"""
    horner(f::Expression, vars = variables(f))

Rewrite `f` using a multi-variate horner schema.

### Example
```julia-repl
julia> @var u v c[1:3]
(u, v, Variable[c₁, c₂, c₃])

julia> f = c[1] + c[2] * v + c[3] * u^2 * v^2 + c[3]u^3 * v
c₁ + v*c₂ + u^2*v^2*c₃ + u^3*v*c₃

julia> horner(f)
c₁ + v*(c₂ + u^3*c₃ + u^2*v*c₃)
```
"""
function horner(f::Expression, vars = variables(f))
    M, coeffs = exponents_coefficients(f, vars; expanded = true, unpack_coeffs = false)
    multivariate_horner(M, coeffs, vars)
end

function horner(coeffs::AbstractVector, var::Variable)
    d = length(coeffs) - 1
    h = copy(coeffs[d+1])
    @inbounds for k = d:-1:1
        mul!(h, h, var)
        add!(h, h, coeffs[k])
    end
    h
end

function multivariate_horner(
    M::AbstractMatrix,
    coeffs::AbstractVector,
    vars::AbstractVector{Variable},
)
    n, m = size(M)
    if m == 1
        fast_c = coeffs[1]
        for (i, var) in enumerate(vars)
            if M[i, 1] != 0
                if M[i, 1] == 1
                    fast_c *= var
                else
                    fast_c *= var^M[i, 1]
                end
            end
        end
        return fast_c::Expression
    elseif n == 1
        d = maximum(M)
        uni_coeff_indices = Union{Nothing,Int}[nothing for _ = 0:d]
        for j = 1:size(M, 2)
            uni_coeff_indices[M[1, j]+1] = j
        end

        var_coeffs = map(uni_coeff_indices) do ind
            if isnothing(ind)
                return zero(vars[1])
            else
                coeffs[ind]
            end
        end

        horner(var_coeffs, vars[1])
    else
        counts = zeros(Int, n)
        @inbounds for j = 1:size(M, 2), i = 1:size(M, 1)
            counts[i] += M[i, j] > 0
        end

        _, var_ind = findmax(counts)
        var = vars[var_ind]

        # compute degree in vars[var_ind]
        d = 0
        for j = 1:size(M, 2)
            d = max(d, M[var_ind, j])
        end

        # find coefficients
        coeff_indices = [Int[] for _ = 0:d]
        for j = 1:size(M, 2)
            push!(coeff_indices[M[var_ind, j]+1], j)
        end

        reduced_vars_ind = Int[]
        for i = 1:n
            if counts[i] > 0 && i != var_ind
                push!(reduced_vars_ind, i)
            end
        end
        reduced_vars = view(vars, reduced_vars_ind)

        var_coeffs = map(coeff_indices) do ind
            isempty(ind) && return zero(var)
            if length(ind) == 1
                j = first(ind)
                c = coeffs[j]
                for i in reduced_vars_ind
                    if M[i, j] != 0
                        if M[i, j] == 1
                            c *= vars[i]
                        else
                            c *= vars[i]^M[i, j]
                        end
                    end
                end
                return c::Expression
            end
            M_ind = view(M, reduced_vars_ind, ind)
            coeffs_ind = view(coeffs, ind)
            multivariate_horner(M_ind, coeffs_ind, reduced_vars)::Expression
        end
        horner(var_coeffs, var)
    end
end



#########################
## System and Homotopy ##
#########################


############
## System ##
############

function check_vars_params(f, vars, params)
    vars_params = params === nothing ? vars : [vars; params]
    Δ = setdiff(variables(f), vars_params)
    isempty(Δ) || throw(ArgumentError(
        "Not all variables or parameters of the system are given. Missing: " *
        join(Δ, ", "),
    ))
    if params !== nothing
        both = Set(vars) ∩ Set(params)
        if !isempty(both)
            error("$(join(both, ", ")) appear as parameters and variables")
        end
    end
    nothing
end

"""
    System(exprs::AbstractVector{Expression};
                variables = variables(exprssion),
                parameters = Variable[])

Create a system from the given [`Expression`](@ref)s `exprs`.
The `variables` determine also the variable ordering.
The `parameters` argument allows to declare certain [`Variable`](@ref)s as parameters.

    System(support::AbstractVector{<:AbstractMatrix{<:Integer}},
           coefficients::AbstractVector{<:AbstractVector};
           variables,
           parameters = Variable[])

Create a system from the given support and coefficients.

## Examples
```julia-repl
julia> @var x y;
julia> F = System([x^2, y^2]; variables = [y, x])
System of length 2
 2 variables: y, x

 x^2
 y^2

# Systems are callable.
# This evaluates F at y=2 and x=3
julia> F([2, 3])
2-element Array{Int64,1}:
 9
 4
```

It is also possible to declare parameters.
```julia-repl
julia> @var x y a b;
julia> F = System([x^2 + a, y^2 + b]; variables = [y, x], parameters = [a, b])
System of length 2
 2 variables: y, x
 2 parameters: a, b

 a + x^2
 b + y^2

julia> F([2, 3], [5, -2])
 2-element Array{Int64,1}:
  14
   2
```
"""
struct System
    expressions::Vector{Expression}
    variables::Vector{Variable}
    parameters::Vector{Variable}
    variable_groups::Union{Nothing,Vector{Vector{Variable}}}

    function System(
        exprs::Vector{Expression},
        vars::Vector{Variable},
        params::Vector{Variable},
        variable_groups::Union{Nothing,Vector{Vector{Variable}}} = nothing,
    )
        check_vars_params(exprs, vars, params)
        if !isnothing(variable_groups)
            vars == reduce(vcat, variable_groups) ||
                throw(ArgumentError("Variable groups and variables don't match."))
        end
        new(exprs, vars, params, variable_groups)
    end
end

System(exprs; kwargs...) = System(convert(Vector{Expression}, exprs); kwargs...)
function System(
    exprs::AbstractVector{Expression};
    parameters::Union{Nothing,Vector{Variable}} = nothing,
    variable_groups::Union{Nothing,Vector{Vector{Variable}}} = nothing,
    variables::Union{Nothing,Vector{Variable}} = nothing,
)
    vars, params = _default_vars_params(exprs, variables, parameters, variable_groups)
    System(convert(Vector{Expression}, exprs), vars, params, variable_groups)
end

function _default_vars_params(exprs, vars, parameters, variable_groups)
    all_vars = variables(exprs)
    if isnothing(variable_groups)
        parameters = something(parameters, Variable[])
        if isnothing(vars)
            setdiff(all_vars, parameters), parameters
        else
            vars, parameters
        end
    elseif isnothing(vars)
        vars = reduce(vcat, variable_groups)
        if isnothing(parameters)
            vars, setdiff(all_vars, vars)
        else
            vars, parameters
        end
    else
        if isnothing(parameters)
            vars, Variable[]
        else
            vars, parameters
        end
    end
end


function System(
    exprs::AbstractVector,
    variables::Vector{Variable};
    parameters::Vector{Variable} = Variable[],
)
    System(convert(Vector{Expression}, exprs), variables, parameters)
end
function System(
    exprs::AbstractVector,
    variables::Vector{Variable},
    parameters::Vector{Variable},
)
    System(convert(Vector{Expression}, exprs), variables, parameters)
end

function System(
    F::AbstractVector{<:MP.AbstractPolynomial};
    parameters = similar(MP.variables(F), 0),
    variables = setdiff(MP.variables(F), parameters),
    variable_groups = nothing,
)
    vars = map(variables) do v
        name, ind = MP.name_base_indices(v)
        Variable(name, ind...)
    end
    params = Variable[]
    for v in parameters
        name, ind = MP.name_base_indices(v)
        push!(params, Variable(name, ind...))
    end
    if variable_groups === nothing
        var_groups = nothing
    else
        var_groups = map(var_groups) do group
            map(group) do v
                name, ind = MP.name_base_indices(v)
                Variable(name, ind...)
            end
        end
    end
    variables_parameters = [variables; parameters]
    vars_params = [vars; params]
    G = map(F) do f
        sum(MP.terms(f)) do t
            c = MP.coefficient(t)
            c * prod(w^MP.degree(t, v) for (v, w) in zip(variables_parameters, vars_params))
        end
    end
    System(G, variables = vars, parameters = params, variable_groups = var_groups)
end

System(F::System) = F

function System(
    support::AbstractVector{<:AbstractMatrix{<:Integer}},
    coefficients::AbstractVector{<:AbstractVector};
    variables::AbstractVector{Variable},
    parameters::AbstractVector{Variable} = Variable[],
)

    System(
        map(support, coefficients) do A, c
            fi = Expression(0)
            for (k, a) in enumerate(eachcol(A))
                ModelKit.add!(fi, fi, c[k] * prod(variables .^ convert.(Int, a)))
            end
            fi
        end,
        variables,
        parameters,
    )
end

Base.hash(S::System, u::UInt64) =
    hash(S.expressions, hash(S.variables, hash(S.parameters, u)))

function Base.show(io::IO, F::System)
    if !get(io, :compact, false)
        println(io, "System of length $(length(F.expressions))")
        if isnothing(F.variable_groups)
            print(io, " $(length(F.variables)) variables: ", join(F.variables, ", "))
        else
            print(
                io,
                length(F.variables),
                " variables (",
                length(F.variable_groups),
                " groups): ",
                join("[" .* join.(F.variable_groups, Ref(", ")) .* "]", ", "),
            )
        end
        if !isempty(F.parameters)
            print(io, "\n $(length(F.parameters)) parameters: ", join(F.parameters, ", "))
        end
        print(io, "\n\n")
        for i = 1:length(F)
            print(io, " ", F.expressions[i])
            i < length(F) && print(io, "\n")
        end
    else
        print(io, "[")
        for i = 1:length(F)
            print(io, F.expressions[i])
            i < length(F) && print(io, ", ")
        end
        print(io, "]")
    end
end

evaluate(F::System, x::AbstractVector) = evaluate(F.expressions, F.variables => x)
function evaluate(F::System, x::AbstractVector, p::AbstractVector)
    evaluate(F.expressions, F.variables => x, F.parameters => p)
end
(F::System)(x::AbstractVector, p::Nothing = nothing) = evaluate(F, x)
(F::System)(x::AbstractVector, p::AbstractVector) = evaluate(F, x, p)

function Base.:(==)(F::System, G::System)
    F.expressions == G.expressions &&
        F.variables == G.variables &&
        F.parameters == G.parameters
end

Base.size(F::System) = (length(F.expressions), length(F.variables))
Base.size(F::System, i::Integer) = size(F)[i]
Base.length(F::System) = length(F.expressions)
Base.copy(F::System) = Base.deepcopy(F)

"""
    nvariables(F::System)

Returns the number of variables of the given system `F`.
"""
nvariables(F::System) = length(F.variables)

"""
    nparameters(F::System)

Returns the number of parameters of the given system `F`.
"""
nparameters(F::System) = length(F.parameters)

"""
    expressions(F::System)

Returns the expressions of the given system `F`.
"""
expressions(F::System) = F.expressions

"""
    variables(F::System)

Returns the variables of the given system `F`.
"""
variables(F::System) = F.variables

"""
    variable_groups(F::System)

Returns the variable groups of the given system `F`.
"""
variable_groups(F::System) = F.variable_groups

"""
    parameters(F::System)

Returns the parameters of the given system `F`.
"""
parameters(F::System) = F.parameters

"""
    degrees(F::System)

Return the degrees of the given system.
"""
degrees(F::System) = degrees(F.expressions, F.variables)

"""
    multi_degrees(F::System)

Return the degrees with respect to the given variable groups.
"""
function multi_degrees(F::System)
    if isnothing(F.variable_groups)
        reshape(degrees(F.expressions, F.variables), 1, nvariables(F))
    else
        reduce(vcat, map(vars -> degrees(F.expressions, vars)', F.variable_groups))
    end
end

"""
    support_coefficients(F::System)

Return the support of the system and the corresponding coefficients.
"""
function support_coefficients(F::System)
    supp_coeffs = exponents_coefficients.(F.expressions, Ref(F.variables))
    first.(supp_coeffs), last.(supp_coeffs)
end

"""
    is_homogeneous(F::System)

Checks whether a given system is homogeneous w.r.t to its variables
(resp. variable groups).
"""
function is_homogeneous(F::System)
    if isnothing(F.variable_groups)
        all(f -> is_homogeneous(f, F.variables), F.expressions)
    else
        all(F.expressions) do f
            g = expand(f)
            all(vars -> is_homogeneous(g, vars; expanded = true), F.variable_groups)
        end
    end
end

Base.iterate(F::System) = iterate(F.expressions)
Base.iterate(F::System, state) = iterate(F.expressions, state)


Base.push!(F::System, f::Expression) = (push!(F.expressions, f); F)
Base.append!(F::System, f::AbstractVector{Expression}) = (append!(F.expressions, f); F)

function Base.intersect(F::System, G::System)
    exprs = [
        F.expressions
        G.expressions
    ]
    vars = [
        F.variables
        setdiff(G.variables, F.variables)
    ]
    params = [
        F.parameters
        setdiff(G.parameters, F.parameters)
    ]
    System(exprs, vars, params, F.variable_groups)
end
function Base.intersect(F::System, G::AbstractVector{<:Expression})
    exprs = [F.expressions; G]
    vars = [
        F.variables
        setdiff(variables(G), F.variables)
    ]
    System(exprs, vars, F.parameters, F.variable_groups)
end
Base.intersect(F::AbstractVector{<:Expression}, G::System) = intersect(G, F)

"""
    optimize(F::System)

Optimize the evaluation cost of the given system `F`. This applies a multivariate horner
schema to the given expressions. See also [`horner`](@ref).

### Example

```julia
julia> f
System of length 4
 4 variables: z₁, z₂, z₃, z₄

 z₁ + z₂ + z₃ + z₄
 z₂*z₁ + z₂*z₃ + z₃*z₄ + z₄*z₁
 z₂*z₃*z₁ + z₂*z₃*z₄ + z₂*z₄*z₁ + z₃*z₄*z₁
 -1 + z₂*z₃*z₄*z₁

julia> optimize(f)
System of length 4
 4 variables: z₁, z₂, z₃, z₄

 z₁ + z₂ + z₃ + z₄
 (z₂ + z₄)*z₁ + (z₂ + z₄)*z₃
 z₁*(z₃*z₄ + (z₃ + z₄)*z₂) + z₂*z₃*z₄
 -1 + z₂*z₃*z₄*z₁
```
"""
function optimize(F::System)
    # horner can fail for rational functions
    try
        System(
            horner.(F.expressions, Ref(F.variables)),
            F.variables,
            F.parameters,
            F.variable_groups,
        )
    catch
        F
    end
end


## Conversion from MultivariatePolynomials
function system_with_coefficents_as_params(
    F::AbstractVector{<:MP.AbstractPolynomial};
    variables = MP.variables(F),
    variable_groups = nothing,
)
    vars = map(variables) do v
        name, ind = MP.name_base_indices(v)
        Variable(name, ind...)
    end
    if variable_groups === nothing
        var_groups = nothing
    else
        var_groups = map(var_groups) do group
            map(group) do v
                name, ind = MP.name_base_indices(v)
                Variable(name, ind...)
            end
        end
    end
    param_name = gensym(:c)
    k = 1
    params = Variable[]
    target_params = MP.coefficienttype(F)[]
    G = map(F) do f
        sum(MP.terms(f)) do t
            c = Variable(param_name, k)
            k += 1
            push!(params, c)
            push!(target_params, MP.coefficient(t))
            c * prod(w^MP.degree(t, v) for (v, w) in zip(variables, vars))
        end
    end
    sys = System(G, variables = vars, parameters = params, variable_groups = var_groups)
    sys, target_params
end

##############
## Homotopy ##
##############
"""
    Homotopy(exprs, vars, t, parameters = Variable[])

Create a homotopy `H(vars,t)` from the given [`Expression`](@ref)s `exprs` where `vars` are the given
variables and `t` is the dedicated variable parameterizing the family of systems.
The `parameters` argument allows to declare certain [`Variable`](@ref)s as parameters.

## Example
```julia-repl
julia> @var x y t;

julia> H = Homotopy([x + t, y + 2t], [y, x], t)
Homotopy in t of length 2
 2 variables: y, x

 t + x
 2*t + y

julia> H([2, 3], 0)
2-element Array{Int64,1}:
 3
 2


julia> H([2, 3], 1)
2-element Array{Int64,1}:
 4
 4
```

It is also possible to declare additional variables.
```julia-repl
julia> @var x y t a b;
julia> H = Homotopy([x^2 + t*a, y^2 + t*b], [x, y], t, [a, b])
Homotopy in t of length 2
 2 variables: x, y
 2 parameters: a, b

 a*t + x^2
 b*t + y^2
julia> H([2, 3], 1, [5, 2])
2-element Array{Int64,1}:
 9
 11
```
"""
struct Homotopy
    expressions::Vector{Expression}
    variables::Vector{Variable}
    t::Variable
    parameters::Vector{Variable}

    function Homotopy(
        exprs::Vector{Expression},
        vars::Vector{Variable},
        t::Variable,
        params::Vector{Variable},
    )
        check_vars_params(exprs, [vars; t], params)
        new(exprs, vars, t, params)
    end
end

function Homotopy(
    exprs::AbstractVector{Expression},
    variables::Vector{Variable},
    t::Variable;
    parameters::Vector{Variable} = Variable[],
)
    Homotopy(convert(Vector{Expression}, exprs), variables, t, parameters)
end
function Homotopy(exprs, variables::Vector{Variable}, t::Variable; kwargs...)
    Homotopy(convert(Vector{Expression}, exprs), variables, t; kwargs...)
end

function Base.show(io::IO, H::Homotopy)
    if !get(io, :compact, false)
        println(io, "Homotopy in ", H.t, " of length ", length(H.expressions))
        print(io, " $(length(H.variables)) variables: ", join(H.variables, ", "))
        if !isempty(H.parameters)
            print(io, "\n $(length(H.parameters)) parameters: ", join(H.parameters, ", "))
        end
        print(io, "\n\n")
        for i = 1:length(H)
            print(io, " ", H.expressions[i])
            i < length(H) && print(io, "\n")
        end
    else
        print(io, "[")
        for i = 1:length(H)
            print(io, H.expressions[i])
            i < length(H) && print(io, ", ")
        end
        print(io, "]")
    end
end

evaluate(H::Homotopy, x::AbstractVector, t) =
    evaluate(H.expressions, H.variables => x, H.t => t)
function evaluate(H::Homotopy, x::AbstractVector, t, p::AbstractVector)
    evaluate(H.expressions, H.variables => x, H.t => t, H.parameters => p)
end
(H::Homotopy)(x::AbstractVector, t) = evaluate(H, x, t)
(H::Homotopy)(x::AbstractVector, t, p::AbstractVector) = evaluate(H, x, t, p)

function Base.:(==)(H::Homotopy, G::Homotopy)
    H.expressions == G.expressions &&
        H.variables == G.variables &&
        H.parameters == G.parameters
end

Base.size(H::Homotopy) = (length(H.expressions), length(H.variables))
Base.size(H::Homotopy, i::Integer) = size(H)[i]
Base.length(H::Homotopy) = length(H.expressions)

"""
    nvariables(H::Homotopy)

Returns the number of variables of the given homotopy `H`.
"""
nvariables(H::Homotopy) = length(H.variables)

"""
    nparameters(H::Homotopy)

Returns the number of parameters of the given homotopy `H`.
"""
nparameters(H::Homotopy) = length(H.parameters)

"""
    expressions(H::Homotopy)

Returns the expressions of the given homotopy `H`.
"""
expressions(H::Homotopy) = H.expressions

"""
    variables(H::Homotopy)

Returns the variables of the given homotopy `H`.
"""
variables(H::Homotopy) = H.variables

"""
    parameters(H::Homotopy)

Returns the parameters of the given homotopy `H`.
"""
parameters(H::Homotopy) = H.parameters

"""
    to_smallest_eltype(A::AbstractArray)

Convert an array to the smallest eltype such that all elements still fit.

## Example
```julia
typeof(to_smallest_eltype(Any[2,3])) == Vector{Int}
```
"""
function to_smallest_eltype(A::AbstractArray)
    T = typeof(first(A))
    for a in A
        if !(typeof(a) <: T)
            T = promote_type(T, typeof(a))
        end
    end
    convert.(T, A)
end


function optimize(H::Homotopy)
    Homotopy(horner.(H.expressions, Ref(H.variables)), H.variables, H.t, H.parameters)
end
