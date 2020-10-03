using SymEngine_jll: libsymengine

function __init__()
    __init_constants()
    __init_type_ids()
end

###########
## TYPES ##
###########

"""
    Expression <: Number

A symbolic expression.

```julia-repl
julia> expr = (Variable(:x) + 1)^2
(1 + x)^2

julia> Expression(2)
2

julia> Expression(Variable(:x))
x
```
"""
mutable struct Expression <: Number
    ptr::Ptr{Cvoid}

    function Expression()
        z = new(C_NULL)
        ccall((:basic_new_stack, libsymengine), Nothing, (Ref{Expression},), z)
        finalizer(free!, z)
        return z
    end

    function Expression(v::Ptr{Cvoid})
        z = new(v)
        return z
    end
end

Expression(ex::Expression) = ex
Expression(T) = convert(Expression, T)

free!(b::Expression) =
    ccall((:basic_free_stack, libsymengine), Nothing, (Ref{Expression},), b)

"""
    ExpressionRef

Basically an `Expression` with the difference that it internally only contains
a reference to an `Expression`. If the underlying reference is freed there is
no guarantee that `ExpressionRef` points to valid memory.
"""
struct ExpressionRef
    ptr::Ptr{Cvoid}

    ExpressionRef(ptr::Ptr{Cvoid}) = new(ptr)
end
ExpressionRef() = ExpressionRef(Ptr{Cvoid}())
ExpressionRef(ex::Expression) = ExpressionRef(ex.ptr)
ExpressionRef(x) = convert(ExpressionRef, x)

"""
    Variable(name::Union{String,Symbol}, indices...) <: Number

A data structure representing a variable.

```julia-repl
julia> Variable(:a)
a

julia> Variable(:x, 1)
x₁

julia> Variable(:x, 10, 5)
x₁₀₋₅
```

### Equality and ordering

Variables are identified by their `name` and `indices`.
That is, two variables are equal if and only if they have the same `name` and `indices`.

```julia-repl
julia> Variable(:a) == Variable(:a)
true

julia> Variable(:a, 1) == Variable(:a, 2)
false
```

Similarly, variables are first ordered lexicographically by their name and then by their indices.
```julia-repl
julia> Variable(:a, 1) < Variable(:a, 2)
true

julia> Variable(:a, 1) < Variable(:b, 1)
true

julia> a = [Variable(:a, i, j) for i in 1:2, j in 1:2]
2×2 Array{Variable,2}:
 a₁₋₁  a₁₋₂
 a₂₋₁  a₂₋₂

julia> sort(vec(a))
4-element Array{Variable,1}:
 a₁₋₁
 a₂₋₁
 a₁₋₂
 a₂₋₂
```
"""
struct Variable <: Number
    ex::Expression

    function Variable(ex::Expression)
        @assert class(ex) == :Symbol
        new(ex)
    end
    Variable(s::Union{String,Symbol}) = new(Expression(s))
end

name(v::Variable) = Symbol(to_string(v))

const Basic = Union{Expression,ExpressionRef,Variable}

Base.convert(::Type{Expression}, v::Variable) = v.ex
Base.convert(::Type{Expression}, e::ExpressionRef) = copy(e)
Base.convert(::Type{ExpressionRef}, e::Expression) = ExpressionRef(e)
Base.convert(::Type{ExpressionRef}, v::Variable) = ExpressionRef(v.ex)

Base.promote_rule(::Type{<:Basic}, ::Type{<:Number}) = Expression
Base.promote_rule(::Type{Expression}, ::Type{Variable}) = Expression
Base.promote_rule(::Type{Expression}, ::Type{ExpressionRef}) = Expression
Base.promote_rule(::Type{Variable}, ::Type{ExpressionRef}) = Expression

function to_string(x::Basic)
    b = ExpressionRef(x)
    if b.ptr == C_NULL
        return ""
    end
    if class(b) == :FunctionSymbol && sym_func_name(b) == :det
        str_args = to_string.(args(b))
        n = round(Int, sqrt(length(str_args)))
        A = reshape(str_args, n, n)
        string("det([", join([join(row, " ") for row in eachrow(A)], "; "), "])")
    else
        a = ccall((:basic_str_julia, libsymengine), Cstring, (Ref{ExpressionRef},), b)
        str = unsafe_string(a)
        ccall((:basic_str_free, libsymengine), Nothing, (Cstring,), a)
        return str
    end
end

Base.show(io::IO, b::Basic) = print(io, to_string(b))
function Base.show(io::IO, mime::MIME"text/latex", b::Basic)
    print(io, "\$\$ ")
    print(io, to_string(b))
    print(io, " \$\$")
end

function Base.hash(ex::Basic, h::UInt)
    h2 = ccall((:basic_hash, libsymengine), UInt, (Ref{ExpressionRef},), ex)
    Base.hash_uint(3h - h2)
end

function _copy(x::Basic)
    y = Expression()
    ccall(
        (:basic_assign, libsymengine),
        Nothing,
        (Ref{Expression}, Ref{ExpressionRef}),
        y,
        x,
    )
    y
end
Base.copy(x::Basic) = _copy(x)
Base.copy(x::Variable) = Variable(_copy(x))

function Base.:(==)(b1::Basic, b2::Basic)
    ccall(
        (:basic_eq, libsymengine),
        Int,
        (Ref{ExpressionRef}, Ref{ExpressionRef}),
        b1,
        b2,
    ) == 1
end

Base.zero(::Basic) = zero(Expression)
Base.zero(::Type{<:Basic}) = Expression(0)
Base.one(::Basic) = one(Expression)
Base.one(::Type{<:Basic}) = Expression(1)

for op in [:im, :π, :ℯ, :γ, :catalan]
    @eval begin
        const $(Symbol("__", op)) = Expression(C_NULL)
    end
end
const SYMENGINE_CONSTANTS = Dict{Expression,Irrational}()

macro init_constant(op, libnm)
    tup = (Base.Symbol("basic_const_$libnm"), libsymengine)
    alloc_tup = (:basic_new_stack, libsymengine)
    op_name = Symbol("__", op)
    c = quote
        ccall($alloc_tup, Nothing, (Ref{Expression},), $op_name)
        ccall($tup, Nothing, (Ref{Expression},), $op_name)
        finalizer(free!, $op_name)
        $(
            op != :im ? :(SYMENGINE_CONSTANTS[$op_name] = Base.MathConstants.$op) :
                :(nothing)
        )
    end
    c
end

function __init_constants()
    @init_constant im I
    @init_constant π pi
    @init_constant ℯ E
    @init_constant γ EulerGamma
    @init_constant catalan Catalan
end

################
## arithmetic ##
################

## main ops
for (op, inplace, libnm) in [(:+, :add!, :add), (:-, :sub!, :sub), (:*, :mul!, :mul)]
    # $(Expr(:., :Base, QuoteNode(op)))
    @eval begin
        function $(inplace)(a::Expression, b1::Basic, b2::Basic)
            ccall(
                $((Symbol("basic_", libnm), libsymengine)),
                Nothing,
                (Ref{Expression}, Ref{ExpressionRef}, Ref{ExpressionRef}),
                a,
                b1,
                b2,
            )
            return a
        end
        function Base.$op(b1::Basic, b2::Basic)
            $inplace(Expression(), b1, b2)
        end
    end
end

function div!(a::Expression, b1::Basic, b2::Basic)
    # class(b2) ∈ NUMBER_TYPES || throw(ArgumentError("Cannot divide by " * string(b2)))
    ccall(
        (:basic_div, libsymengine),
        Nothing,
        (Ref{Expression}, Ref{ExpressionRef}, Ref{ExpressionRef}),
        a,
        b1,
        b2,
    )
    return a
end
Base.:(/)(b1::Basic, b2::Basic) = div!(Expression(), b1, b2)

Base.:(//)(x::Basic, y::Basic) = x / y
Base.:(//)(x::Basic, y::Integer) = x / Expression(y)

function Base.:(^)(x::Basic, k::Integer)
    a = Expression()
    ccall(
        (:basic_pow, libsymengine),
        Nothing,
        (Ref{Expression}, Ref{ExpressionRef}, Ref{Expression}),
        a,
        x,
        Expression(k),
    )
    return a
end
Base.:+(b::Basic) = b
Base.:-(b::Basic) = Expression(0) - b


# Functions
macro make_func(arg1, arg2)
    quote
        function $(esc(arg1))(b::Basic)
            a = Expression()
            ccall(
                ($(QuoteNode(arg2)), libsymengine),
                Base.Nothing,
                (Base.Ref{Expression}, Base.Ref{ExpressionRef}),
                a,
                b,
            )
            return a
        end
    end
end

@make_func symengine_expand basic_expand

##############################
## conversion to Expression ##
##############################

function Base.convert(::Type{Expression}, s::String)
    a = Expression()
    ccall((:symbol_set, libsymengine), Nothing, (Ref{Expression}, Ptr{Int8}), a, s)
    return a
end
Base.convert(::Type{Expression}, s::Symbol) = convert(Expression, string(s))

Base.convert(::Type{Expression}, x::Irrational{:γ}) = __γ
Base.convert(::Type{Expression}, x::Irrational{:π}) = __π
Base.convert(::Type{Expression}, x::Irrational{:ℯ}) = __ℯ
Base.convert(::Type{Expression}, x::Irrational{:catalan}) = __catalan


# basic number types
for (f, T) in
    [(:integer_set_si, Int), (:integer_set_ui, UInt), (:real_double_set_d, Float64)]
    @eval begin
        function Base.convert(::Type{Expression}, x::$T)
            a = Expression()
            ccall(($(QuoteNode(f)), libsymengine), Nothing, (Ref{Expression}, $T), a, x)
            return a
        end
    end
end

function Base.convert(::Type{Expression}, x::BigInt)
    a = Expression()
    ccall((:integer_set_mpz, libsymengine), Nothing, (Ref{Expression}, Ref{BigInt}), a, x)
    return a
end


Base.convert(::Type{Expression}, x::Union{Float16,Float32}) =
    convert(Expression, convert(Float64, x))
Base.convert(::Type{Expression}, x::AbstractFloat) =
    convert(Expression, convert(BigFloat, x))
function Base.convert(::Type{Expression}, x::BigFloat)
    if (x.prec <= 53)
        return convert(Expression, Float64(x))
    else
        a = Expression()
        ccall(
            (:real_mpfr_set, libsymengine),
            Nothing,
            (Ref{Expression}, Ref{BigFloat}),
            a,
            x,
        )
        return a
    end
end
Base.convert(::Type{Expression}, x::Int32) = convert(Expression, convert(Int, x))
Base.convert(::Type{Expression}, x::UInt32) = convert(Expression, convert(UInt, x))

Base.convert(::Type{Expression}, x::Integer) = Expression(BigInt(x))
Base.convert(::Type{Expression}, x::Rational) =
    Expression(numerator(x)) / Expression(denominator(x))
Base.convert(::Type{Expression}, x::Complex) =
    Expression(real(x)) + Expression(imag(x)) * __im

Base.complex(x::Expression, y::Expression) = x + y * __im
Base.complex(x::Expression, y::Real) = x + y * __im
Base.complex(x::Real, y::Expression) = x + y * __im


################################
## Iterating over expressions ##
################################

mutable struct ExpressionSet
    ptr::Ptr{Cvoid}

    function ExpressionSet()
        z = new(ccall((:setbasic_new, libsymengine), Ptr{Cvoid}, ()))
        finalizer(free!, z)
        z
    end
end

function free!(x::ExpressionSet)
    if x.ptr != C_NULL
        ccall((:setbasic_free, libsymengine), Nothing, (Ptr{Cvoid},), x.ptr)
        x.ptr = C_NULL
    end
    nothing
end

Base.length(s::ExpressionSet) =
    ccall((:setbasic_size, libsymengine), Int, (Ptr{Cvoid},), s.ptr)

function Base.getindex(s::ExpressionSet, n::Int)
    result = Expression()
    ccall(
        (:setbasic_get, libsymengine),
        Nothing,
        (Ptr{Cvoid}, Int, Ref{Expression}),
        s.ptr,
        n - 1,
        result,
    )
    result
end

_variables(ex::Variable) = [ex]
function _variables(ex::Expression)
    syms = ExpressionSet()
    ccall(
        (:basic_free_symbols, libsymengine),
        Nothing,
        (Ref{ExpressionRef}, Ptr{Cvoid}),
        ex,
        syms.ptr,
    )
    sort!([Variable(syms[i]) for i = 1:length(syms)])
end

function function_symbols(ex::Basic)
    syms = ExpressionSet()
    ccall(
        (:basic_function_symbols, libsymengine),
        Nothing,
        (Ptr{Cvoid}, Ref{ExpressionRef}),
        syms.ptr,
        ex,
    )
    [syms[i] for i = 1:length(syms)]
end

function differentiate(f::Basic, v::Variable)
    a = Expression()
    ret = ccall(
        (:basic_diff, libsymengine),
        Int,
        (Ref{Basic}, Ref{ExpressionRef}, Ref{ExpressionRef}),
        a,
        f,
        v,
    )
    return a
end
function differentiate(f::Basic, v::Variable, n::Int)
    n < 0 && throw(DomainError("n must be non-negative integer"))
    n == 0 && return f
    n == 1 && return differentiate(f, v)
    n > 1 && return differentiate(differentiate(f, v), v, n - 1)
end


# Get class of an Expression

const REAL_NUMBER_TYPES = [:Integer, :RealDouble, :Rational, :RealMPFR]
const COMPLEX_NUMBER_TYPES = [:Complex, :ComplexDouble, :ComplexMPC]
const NUMBER_TYPES = [
    REAL_NUMBER_TYPES
    COMPLEX_NUMBER_TYPES
]

function type_id(s::Basic)
    ccall((:basic_get_type, libsymengine), UInt, (Ref{ExpressionRef},), s)
end

function get_class_from_type_id(id::UInt)
    a = ccall((:basic_get_class_from_id, libsymengine), Ptr{UInt8}, (Int,), id)
    str = unsafe_string(a)
    ccall((:basic_str_free, libsymengine), Nothing, (Ptr{UInt8},), a)
    Symbol(str)
end

# prepopulate the dict
const TYPE_IDS = Dict{UInt,Symbol}()

function __init_type_ids()
    x = Expression("x")
    values = [
        x,
        Expression(1),
        Expression(0.5),
        Expression(2 // 3),
        Expression(big(0.5)),
        Expression(5 + 3im),
        Expression(0.5 + 0.2im),
        Expression(0.5 + big(0.2) * im),
        2x,
        x + 2,
        x^2,
    ]
    for v in values
        class(v)
    end
    nothing
end

function class(e::Basic)
    id = type_id(e)
    if haskey(TYPE_IDS, id)
        TYPE_IDS[id]
    else
        # add for futurue fast lookup
        cls = get_class_from_type_id(id)
        TYPE_IDS[id] = cls
        cls
    end
end


mutable struct ExprVec <: AbstractVector{ExpressionRef}
    ptr::Ptr{Cvoid}
    m::Union{Nothing,Ptr{ModelKit.ExpressionRef}}

    function ExprVec()
        ptr = ccall((:vecbasic_new, libsymengine), Ptr{Cvoid}, ())
        z = new(ptr, nothing)
        finalizer(free!, z)
        z
    end
end
function vec_set_ptr!(v::ExprVec)
    v.m = unsafe_load(Ptr{Ptr{ModelKit.ExpressionRef}}(v.ptr))
    v
end

function free!(x::ExprVec)
    if x.ptr != C_NULL
        ccall((:vecbasic_free, libsymengine), Nothing, (Ptr{Cvoid},), x.ptr)
        x.ptr = C_NULL
    end
end

Base.eltype(v::ExprVec) = ExpressionRef
Base.length(s::ExprVec) = ccall((:vecbasic_size, libsymengine), UInt, (Ptr{Cvoid},), s.ptr)
Base.size(s::ExprVec) = (length(s),)

function Base.getindex(v::ExprVec, n)
    @boundscheck checkbounds(v, n)
    if v.m === nothing
        vec_set_ptr!(v)
        unsafe_load(v.m, n)
    else
        unsafe_load(v.m, n)
    end
end

function Base.push!(v::ExprVec, x::Basic)
    ccall(
        (:vecbasic_push_back, libsymengine),
        Nothing,
        (Ptr{Cvoid}, Ref{ExpressionRef}),
        v.ptr,
        x,
    )
    v
end

args(ex::Basic) = args!(ExprVec(), ex)
function args!(vec::ExprVec, ex::Basic)
    ccall(
        (:basic_get_args, libsymengine),
        Nothing,
        (Ref{ExpressionRef}, Ptr{Cvoid}),
        ex,
        vec.ptr,
    )
    vec_set_ptr!(vec)
end

#
# mutable struct ExprMat <: DenseArray{Expression,2}
#     ptr::Ptr{Cvoid}
#
#     function ExprMat(m::Int, n::Int)
#         ptr =
#             ccall((:dense_matrix_new_rows_cols, libsymengine), Ptr{Cvoid}, (Int, Int), m, n)
#         z = new(ptr)
#         finalizer(free!, z)
#         z
#     end
# end
#
# function ExprMat(A::AbstractMatrix{<:Basic})
#     M = ExprMat(size(A)...)
#     for j in 1:size(A,2), i in 1:size(A,1)
#         M[i,j] = A[i,j]
#     end
#     M
# end
#
# function free!(x::ExprMat)
#     if x.ptr != C_NULL
#         ccall((:dense_matrix_free, libsymengine), Nothing, (Ptr{Cvoid},), x.ptr)
#         x.ptr = C_NULL
#     end
# end
#
# nrows(M::ExprMat) = ccall((:dense_matrix_rows, libsymengine), Int, (Ptr{Cvoid},), M.ptr)
# ncols(M::ExprMat) = ccall((:dense_matrix_cols, libsymengine), Int, (Ptr{Cvoid},), M.ptr)
# Base.size(M::ExprMat) = (nrows(M), ncols(M))
# # Base.promote_rule(::Type{CDenseMatrix}, ::Type{Matrix{T}} ) where {T <: Basic} = CDenseMatrix
#
# function Base.getindex(M::ExprMat, r::Int, c::Int)
#     @boundscheck checkbounds(M, r, c)
#     result = Expression()
#     ccall(
#         (:dense_matrix_get_basic, libsymengine),
#         Nothing,
#         (Ref{Expression}, Ptr{Cvoid}, UInt, UInt),
#         result,
#         M.ptr,
#         UInt(r - 1),
#         UInt(c - 1),
#     )
#     result
# end
# function Base.setindex!(M::ExprMat, x::Basic, r::Int, c::Int)
#     @boundscheck checkbounds(M, r, c)
#     ccall(
#         (:dense_matrix_set_basic, libsymengine),
#         Nothing,
#         (Ptr{Cvoid}, UInt, UInt, Ref{ExpressionRef}),
#         M.ptr,
#         UInt(r - 1),
#         UInt(c - 1),
#         x,
#     )
#     x
# end

function make_sym_func(name::String, x::Vector)
    a = Expression()
    args = ExprVec()
    for xᵢ in x
        push!(args, xᵢ)
    end
    ccall(
        (:function_symbol_set, libsymengine),
        Nothing,
        (Ref{Basic}, Ptr{Int8}, Ptr{Cvoid}),
        a,
        name,
        args.ptr,
    )
    return a
end

function sym_func_name(x::Basic)
    a = ccall(
        (:function_symbol_get_name, libsymengine),
        Ptr{UInt8},
        (Ref{ExpressionRef},),
        x,
    )
    string = unsafe_string(a)
    ccall((:basic_str_free, libsymengine), Nothing, (Cstring,), a)
    return Symbol(string)
end


################################
## conversion from Expression ##
################################

function Base.convert(::Type{Int}, n::Basic)
    @assert class(n) == :Integer
    ccall((:integer_get_si, libsymengine), Int, (Ref{ExpressionRef},), n)
end
Base.convert(::Type{Int32}, n::Basic) = convert(Int32, convert(Int, n))

function Base.convert(::Type{BigInt}, c::Basic)
    a = BigInt()
    ccall(
        (:integer_get_mpz, libsymengine),
        Nothing,
        (Ref{BigInt}, Ref{ExpressionRef}),
        a,
        c,
    )
    return a
end

function Base.convert(::Type{BigFloat}, c::Basic)
    a = BigFloat()
    ccall(
        (:real_mpfr_get, libsymengine),
        Nothing,
        (Ref{BigFloat}, Ref{ExpressionRef}),
        a,
        c,
    )
    return a
end

function Base.convert(::Type{Float64}, c::Basic)
    return ccall((:real_double_get_d, libsymengine), Cdouble, (Ref{ExpressionRef},), c)
end

function Base.real(x::Basic)
    if class(x) ∈ COMPLEX_NUMBER_TYPES
        a = Expression()
        ccall(
            (:complex_base_real_part, libsymengine),
            Nothing,
            (Ref{Expression}, Ref{ExpressionRef}),
            a,
            x,
        )
        return a
    else
        return x
    end
end
function Base.imag(x::Basic)
    if class(x) ∈ COMPLEX_NUMBER_TYPES
        a = Expression()
        ccall(
            (:complex_base_imaginary_part, libsymengine),
            Nothing,
            (Ref{Expression}, Ref{ExpressionRef}),
            a,
            x,
        )
        return a
    else
        return Expression(0)
    end
end

function _numer_denom(x::Basic)
    a, b = Expression(), Expression()
    ccall(
        (:basic_as_numer_denom, libsymengine),
        Nothing,
        (Ref{Expression}, Ref{Expression}, Ref{ExpressionRef}),
        a,
        b,
        x,
    )
    return a, b
end

is_number(ex::Expression) = class(ex) in NUMBER_TYPES

"""
    to_number(x::Expression)

Tries to unpack the `Expression` `x` to a native number type.

```julia-repl
julia> x = to_number(Expression(2))
2

julia> typeof(x)
Int64
"""
function to_number(x::Basic)
    cls = class(x)

    if cls == :Integer
        n = convert(BigInt, x)
        if typemin(Int32) ≤ n ≤ typemax(Int32)
            return convert(Int32, n)
        elseif typemin(Int64) ≤ n ≤ typemax(Int64)
            return convert(Int64, n)
        elseif typemin(Int128) ≤ n ≤ typemax(Int128)
            return convert(Int128, n)
        else
            return n
        end
    elseif cls == :RealDouble
        return convert(Float64, x)
    elseif cls == :Rational
        a, b = _numer_denom(x)
        return to_number(a) // to_number(b)
    elseif cls == :RealMPFR
        return convert(BigFloat, x)
    elseif cls == :Constant
        return SYMENGINE_CONSTANTS[x]
    elseif cls in COMPLEX_NUMBER_TYPES
        a, b = reim(x)
        return complex(to_number(a), to_number(b))
    else
        return x
    end
end

##########
## SUBS ##
##########

mutable struct ExpressionMap
    ptr::Ptr{Cvoid}

    function ExpressionMap()
        x = new(ccall((:mapbasicbasic_new, libsymengine), Ptr{Cvoid}, ()))
        finalizer(free!, x)
        x
    end
end

function free!(x::ExpressionMap)
    if x.ptr != C_NULL
        ccall((:mapbasicbasic_free, libsymengine), Nothing, (Ptr{Cvoid},), x.ptr)
        x.ptr = C_NULL
    end
end

ExpressionMap(args...) = ExpressionMap(ExpressionMap(), args...)
ExpressionMap(D::ExpressionMap) = D
function ExpressionMap(dict::Dict)
    c = ExpressionMap()
    for (key, value) in dict
        c[Expression(key)] = Expression(value)
    end
    return c
end
function ExpressionMap(
    D::ExpressionMap,
    (xs, ys)::Pair{<:AbstractArray{<:Basic},<:AbstractArray},
    args...,
)
    size(xs) == size(ys) ||
        throw(ArgumentError("Substitution arguments don't have the same size."))
    for (x, y) in zip(xs, ys)
        D[x] = y
    end
    ExpressionMap(D, args...)
end
function ExpressionMap(D::ExpressionMap, (x, y), args...)
    D[Expression(x)] = Expression(y)
    ExpressionMap(D, args...)
end


function Base.length(s::ExpressionMap)
    ccall((:mapbasicbasic_size, libsymengine), Int, (Ptr{Cvoid},), s.ptr)
end

Base.setindex!(s::ExpressionMap, v::Number, k::Basic) = setindex!(s, Expression(v), k)
function Base.setindex!(s::ExpressionMap, v::Basic, k::Basic)
    ccall(
        (:mapbasicbasic_insert, libsymengine),
        Nothing,
        (Ptr{Cvoid}, Ref{ExpressionRef}, Ref{ExpressionRef}),
        s.ptr,
        k,
        v,
    )
    v
end

function subs(ex::Basic, d::ExpressionMap)
    s = Expression()
    ccall(
        (:basic_subs, libsymengine),
        Nothing,
        (Ref{Expression}, Ref{ExpressionRef}, Ptr{Cvoid}),
        s,
        ex,
        d.ptr,
    )
    return s
end
#
# subs(ex::Basic, (k, v)::Pair{<:Basic,<:Number}) = subs(ex, k => Expression(v))
# function subs(ex::Basic, (k, v)::Pair{<:Basic,<:Basic})
#     s = Expression()
#     ccall(
#         (:basic_subs2, libsymengine),
#         Nothing,
#         (
#             Ref{Expression},
#             Ref{ExpressionRef},
#             Ref{ExpressionRef},
#             Ref{ExpressionRef},
#         ),
#         s,
#         ex,
#         k,
#         v,
#     )
#     return s
# end

function cse(exprs::Vector{Expression})
    vec = ExprVec()
    for ex in exprs
        push!(vec, ex)
    end

    replacement_syms = ExprVec()
    replacement_exprs = ExprVec()
    reduced_exprs = ExprVec()

    ccall(
        (:basic_cse, libsymengine),
        Nothing,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        replacement_syms.ptr,
        replacement_exprs.ptr,
        reduced_exprs.ptr,
        vec.ptr,
    )

    subs = Dict{Expression,Expression}()
    for (sᵢ, exᵢ) in zip(replacement_syms, replacement_exprs)
        subs[sᵢ] = exᵢ
    end
    map(Expression, reduced_exprs), subs
end
