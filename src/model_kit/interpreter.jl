@enum InterpreterOp::UInt32 begin
    OP_ADD
    OP_SUB
    OP_MUL
    OP_DIV
    OP_SQR
    OP_POW
    OP_INV_NOT_ZERO
end

function InterpreterOp(op::Symbol, arg1, arg2)
    if op == :+
        OP_ADD
    elseif op == :-
        OP_SUB
    elseif op == :*
        OP_MUL
    elseif op == :/
        OP_DIV
    elseif op == :^
        if arg2 == 2
            OP_SQR
        else
            OP_POW
        end
    elseif op == :inv_not_zero
        OP_INV_NOT_ZERO
    else
        throw(ArgumentError(String(op)))
    end
end

function is_univariate(op::InterpreterOp)
    op == OP_SQR || op == OP_INV_NOT_ZERO
end

@enum InterpreterData::UInt16 begin
    DATA_X
    DATA_P
    DATA_C
    DATA_I
    DATA_T
    NO_DATA
end

function make_op_block(
    assign,
    op,
    data1::InterpreterData,
    index1,
    data2::InterpreterData,
    index2,
)
    arg1 = gen_access(data1, index1)
    if data2 == NO_DATA
        quote
            if $op == OP_SQR
                $assign = sqr($arg1)
            elseif $op == OP_POW
                $assign = Base.power_by_squaring($arg1, $index2)
            elseif $op == OP_INV_NOT_ZERO
                $assign = inv_not_zero($arg1)
            end
        end
    else
        arg2 = gen_access(data2, index2)
        quote
            if $op == OP_MUL
                $assign = $arg1 * $arg2
            elseif $op == OP_ADD
                $assign = $arg1 + $arg2
            elseif $op == OP_SUB
                $assign = $arg1 - $arg2
            elseif $op == OP_DIV
                $assign = $arg1 / $arg2
            end
        end
    end
end

function make_taylor_op_block(
    assign,
    ord,
    op,
    data1::InterpreterData,
    index1,
    data2::InterpreterData,
    index2,
)
    arg1 = gen_access(data1, index1)
    if data2 == NO_DATA
        quote
            if $op == OP_SQR
                $assign = taylor(Val{:sqr}, $ord, $arg1)
            elseif $op == OP_POW
                $assign = taylor(Val{:^}, $ord, $arg1, $index2)
            end
        end
    else
        arg2 = gen_access(data2, index2)
        quote
            if $op == OP_MUL
                $assign = taylor(Val{:*}, $ord, $arg1, $arg2)
            elseif $op == OP_ADD
                $assign = taylor(Val{:+}, $ord, $arg1, $arg2)
            elseif $op == OP_SUB
                $assign = taylor(Val{:-}, $ord, $arg1, $arg2)
            elseif $op == OP_DIV
                $assign = taylor(Val{:/}, $ord, $arg1, $arg2)
            end
        end
    end
end


struct InterpreterArg
    data::InterpreterData
    ind::Int32
end
function InterpreterArg(instr_map, var_map, t, param_map, constants, arg)
    if haskey(instr_map, arg)
        return InterpreterArg(DATA_I, instr_map[arg])
    elseif haskey(var_map, arg)
        return InterpreterArg(DATA_X, var_map[arg])
    elseif haskey(param_map, arg)
        return InterpreterArg(DATA_P, param_map[arg])
    elseif t == arg
        return InterpreterArg(DATA_T, 0)
    else # constant
        c = findfirst(c -> c == arg, constants)
        if isnothing(c)
            push!(constants, arg)
            return InterpreterArg(DATA_C, length(constants))
        else
            return InterpreterArg(DATA_C, c)
        end
    end
end
Base.show(io::IO, arg::InterpreterArg) =
    print(io, "(", Symbol(arg.data), ", ", arg.ind, ")")

struct InterpreterInstruction
    op::InterpreterOp
    args::NTuple{2,InterpreterArg}
end
function Base.show(io::IO, instr::InterpreterInstruction)
    print(io, Symbol(instr.op), ": ", instr.args)
end

struct Interpreter{T,N}
    instructions::Vector{InterpreterInstruction}
    out::Array{InterpreterArg,N}
    constants::Vector{T}
    # only for evaluate_and_jacobian
    eval_out::Vector{InterpreterArg}
end

function Interpreter(
    list::InstructionList,
    out::AbstractArray,
    var_map::Dict{Symbol,Int},
    t::Union{Nothing,Symbol},
    param_map::Dict{Symbol,Int},
    eval_out::Vector,
)
    instructions = InterpreterInstruction[]
    output = Int[]
    instr_map = Dict{Symbol,Int}()
    constants = []
    for (id, (op, arg1, arg2)) in list.instructions
        OP = InterpreterOp(op, arg1, arg2)
        ARG1 = InterpreterArg(instr_map, var_map, t, param_map, constants, arg1)
        if isnothing(arg2) || is_univariate(OP)
            ARG2 = InterpreterArg(NO_DATA, Int32(0))
        elseif OP == OP_POW
            ARG2 = InterpreterArg(NO_DATA, Int32(arg2))
        else
            ARG2 = InterpreterArg(instr_map, var_map, t, param_map, constants, arg2)
        end
        push!(instructions, InterpreterInstruction(OP, (ARG1, ARG2)))
        instr_map[id] = length(instructions)
    end
    constants = isempty(constants) ? Float64[] : to_smallest_eltype(constants)
    output = map(id -> InterpreterArg(instr_map, var_map, t, param_map, constants, id), out)
    eval_output =
        map(id -> InterpreterArg(instr_map, var_map, t, param_map, constants, id), eval_out)

    Interpreter(instructions, output, constants, eval_output)
end


function Base.show(io::IO, I::Interpreter{T,N}) where {T,N}
    print(io, "Interpreter{$T,$N} with ", length(I.instructions), " instructions")
end

Base.eltype(::Interpreter{T}) where {T} = T
function Base.convert(::Type{Interpreter{T,N}}, I::Interpreter{S,N}) where {T,S,N}
    Interpreter(I.instructions, I.out, convert(Vector{T}, I.constants), I.eval_out)
end
show_instructions(I::Interpreter) = show_instructions(stdout, I)
function show_instructions(io::IO, I::Interpreter)
    for (i, instr) in enumerate(I.instructions)
        print(io, i, " ")
        show(io, instr)
        println(io)
    end
end

function evaluate_interpreter(F::Union{System,Homotopy})
    var_map = Dict((name(v), i) for (i, v) in enumerate(F.variables))
    param_map = Dict{Symbol,Int}()
    if !isnothing(F.parameters)
        for (i, v) in enumerate(F.parameters)
            param_map[name(v)] = i
        end
    end
    list, out = instruction_list(F.expressions)
    t = isa(F, Homotopy) ? name(F.t) : nothing
    Interpreter(list, out, var_map, t, param_map, out)
end

function jacobian_interpreter(F::Union{System,Homotopy})
    var_map = Dict((name(v), i) for (i, v) in enumerate(F.variables))
    param_map = Dict{Symbol,Int}()
    if !isnothing(F.parameters)
        for (i, v) in enumerate(F.parameters)
            param_map[name(v)] = i
        end
    end
    list, eval_out = instruction_list(F.expressions)
    dlist, J = diff(list, Symbol.(F.variables), eval_out)

    #replace nothings with 0
    t = isa(F, Homotopy) ? name(F.t) : nothing
    Interpreter(dlist, something.(J, 0), var_map, t, param_map, eval_out)
end

function taylor_interpreter(
    F::Union{System,Homotopy};
    order_out::Int,
    order_x::Int,
    order_p::Int,
)
    var_map = Dict((name(v), i) for (i, v) in enumerate(F.variables))
    param_map = Dict{Symbol,Int}()
    diff_map = DiffMap()
    t = isa(F, Homotopy) ? name(F.t) : nothing

    N = nvariables(F)
    for (i, v) in enumerate(F.variables)
        sv = name(v)
        var_map[sv] = i
        for k = 1:order_x
            dksv = Symbol(:__D__, sv, "__$(k)__")
            var_map[dksv] = k * N + i
            diff_map[sv, k] = dksv
        end
    end
    if !isnothing(F.parameters)
        m = length(F.parameters)
        for (i, v) in enumerate(F.parameters)
            sv = name(v)
            param_map[sv] = i
            for k = 1:order_p
                dksv = Symbol(:__D__, sv, "__$(k)__")
                param_map[dksv] = k * m + i
                diff_map[sv, k] = dksv
            end
        end
    end
    if !isnothing(t)
        diff_map[t, 1] = 1
    end

    list, eval_out = instruction_list(F.expressions)
    dlist = univariate_diff!(list, order_out, diff_map)
    outputs = map(id -> something(diff_map[id, order_out], 0), eval_out)
    Interpreter(dlist, outputs, var_map, t, param_map, eval_out)
end

struct InterpreterCache{T}
    instructions::Vector{T}
end
function InterpreterCache(::Type{T}, I::Interpreter) where {T}
    InterpreterCache(Vector{T}(undef, length(I.instructions)))
end
function InterpreterCache(
    I::Interpreter{T},
    x::AbstractArray,
    ::Nothing = nothing,
) where {T}
    InterpreterCache(promote_type(T, _eltype(x)), I)
end
function InterpreterCache(I::Interpreter{T}, x::AbstractArray, p::AbstractArray) where {T}
    InterpreterCache(promote_type(T, _eltype(x), _eltype(p)), I)
end
_eltype(x::TaylorVector{N,T}) where {N,T} = T
_eltype(x) = eltype(x)

Base.length(c::InterpreterCache) = length(c.instructions)
Base.similar(::Type{T}, I::InterpreterCache) where {T} =
    InterpreterCache(Vector{T}(undef, length(I)))

function data_block(make_op_block, name1, name2; parameters::Bool, t::Bool = false)
    if parameters && t
        inner_block = outer -> quote
            if $name2 == DATA_I
                $(make_op_block(outer, DATA_I))
            elseif $name2 == DATA_X
                $(make_op_block(outer, DATA_X))
            elseif $name2 == DATA_T
                $(make_op_block(outer, DATA_T))
            elseif $name2 == DATA_P
                $(make_op_block(outer, DATA_P))
            elseif $name2 == DATA_C
                $(make_op_block(outer, DATA_C))
            elseif $name2 == NO_DATA
                $(make_op_block(outer, NO_DATA))
            end
        end
        quote
            if $name1 == DATA_I
                $(inner_block(DATA_I))
            elseif $name1 == DATA_X
                $(inner_block(DATA_X))
            elseif $name1 == DATA_T
                $(inner_block(DATA_T))
            elseif $name1 == DATA_P
                $(inner_block(DATA_P))
            elseif $name1 == DATA_C
                $(inner_block(DATA_C))
            end
        end
    elseif parameters
        inner_block = outer -> quote
            if $name2 == DATA_I
                $(make_op_block(outer, DATA_I))
            elseif $name2 == DATA_X
                $(make_op_block(outer, DATA_X))
            elseif $name2 == DATA_P
                $(make_op_block(outer, DATA_P))
            elseif $name2 == DATA_C
                $(make_op_block(outer, DATA_C))
            elseif $name2 == NO_DATA
                $(make_op_block(outer, NO_DATA))
            end
        end
        quote
            if $name1 == DATA_I
                $(inner_block(DATA_I))
            elseif $name1 == DATA_X
                $(inner_block(DATA_X))
            elseif $name1 == DATA_P
                $(inner_block(DATA_P))
            elseif $name1 == DATA_C
                $(inner_block(DATA_C))
            end
        end
    elseif t
        inner_block = outer -> quote
            if $name2 == DATA_I
                $(make_op_block(outer, DATA_I))
            elseif $name2 == DATA_X
                $(make_op_block(outer, DATA_X))
            elseif $name2 == DATA_T
                $(make_op_block(outer, DATA_T))
            elseif $name2 == DATA_C
                $(make_op_block(outer, DATA_C))
            elseif $name2 == NO_DATA
                $(make_op_block(outer, NO_DATA))
            end
        end
        quote
            if $name1 == DATA_I
                $(inner_block(DATA_I))
            elseif $name1 == DATA_X
                $(inner_block(DATA_X))
            elseif $name1 == DATA_T
                $(inner_block(DATA_T))
            elseif $name1 == DATA_C
                $(inner_block(DATA_C))
            end
        end
    else
        inner_block = outer -> quote
            if $name2 == DATA_I
                $(make_op_block(outer, DATA_I))
            elseif $name2 == DATA_X
                $(make_op_block(outer, DATA_X))
            elseif $name2 == DATA_C
                $(make_op_block(outer, DATA_C))
            elseif $name2 == NO_DATA
                $(make_op_block(outer, NO_DATA))
            end
        end
        quote
            if $name1 == DATA_I
                $(inner_block(DATA_I))
            elseif $name1 == DATA_X
                $(inner_block(DATA_X))
            elseif $name1 == DATA_C
                $(inner_block(DATA_C))
            end
        end
    end
end

function gen_access(data, indname)
    if data == DATA_I
        :(instructions[$indname])
    elseif data == DATA_X
        :(x[$indname])
    elseif data == DATA_P
        :(p[$indname])
    elseif data == DATA_T
        :(t)
    elseif data == DATA_C
        :(constants[$indname])
    elseif data == NO_DATA
        nothing
    end
end


for (xarg, xval) in [(:(x::AbstractVector), :x), (:(tx::TaylorVector), :(tx.data))],
    (parg, pval) in [
        (:(p::Nothing), :(nothing)),
        (:(p::AbstractVector), :p),
        (:(tp::TaylorVector), :(tp.data)),
    ]

    @eval begin
        function execute!(
            u::AbstractArray,
            I::Interpreter,
            $xarg,
            $parg,
            cache::InterpreterCache = InterpreterCache(I, $xval, $pval),
        )
            _execute!(nothing, u, I, $xval, $pval, cache)
        end
        function execute!(
            u::AbstractArray,
            U::AbstractArray,
            I::Interpreter,
            $xarg,
            $parg,
            cache::InterpreterCache = InterpreterCache(I, $xval, $pval),
        )
            _execute!(u, U, I, $xval, $pval, cache)
        end

        # # homotopy versions
        function execute!(
            u::AbstractArray,
            I::Interpreter,
            $xarg,
            t,
            $parg,
            cache::InterpreterCache = InterpreterCache(I, $xval, $pval),
        )
            _execute!(nothing, u, I, $xval, t, $pval, cache)
        end
        function execute!(
            u::AbstractArray,
            U::AbstractArray,
            I::Interpreter,
            $xarg,
            t,
            $parg,
            cache::InterpreterCache = InterpreterCache(I, $xval, $pval),
        )
            _execute!(u, U, I, $xval, t, $pval, cache)
        end
    end
end

@eval begin
    function _execute!(
        u::Union{Nothing,AbstractVector},
        U::AbstractArray,
        I::Interpreter,
        x::AbstractArray,
        ::Nothing = nothing,
        cache::InterpreterCache = InterpreterCache(I, x),
    )
        instructions = cache.instructions
        constants = I.constants
        for (i, instr) in enumerate(I.instructions)
            op = instr.op
            arg1, arg2 = instr.args
            $(
                data_block(:(arg1.data), :(arg2.data); parameters = false) do data1, data2
                    make_op_block(
                        :(instructions[i]),
                        :op,
                        data1,
                        :(arg1.ind),
                        data2,
                        :(arg2.ind),
                    )
                end
            )
        end
        if I.out isa AbstractMatrix
            for j = 1:size(I.out, 2), i = 1:size(I.out, 1)
                @unpack data, ind = I.out[i, j]
                if data == DATA_I
                    U[i, j] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[i, j] = convert(eltype(U), x[ind])
                elseif data == DATA_C
                    U[i, j] = convert(eltype(U), constants[ind])
                end
            end
        else
            for k in eachindex(I.out)
                @unpack data, ind = I.out[k]
                if data == DATA_I
                    U[k] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[k] = convert(eltype(U), x[ind])
                elseif data == DATA_C
                    U[k] = convert(eltype(U), constants[ind])
                end
            end
        end

        if !isa(u, Nothing)
            for k in eachindex(I.eval_out)
                @unpack data, ind = I.eval_out[k]
                if data == DATA_I
                    u[k] = convert(eltype(u), instructions[ind])
                elseif data == DATA_X
                    u[k] = convert(eltype(u), x[ind])
                elseif data == DATA_C
                    u[k] = convert(eltype(u), constants[ind])
                end
            end
        end
        U
    end

    function _execute!(
        u::Union{Nothing,AbstractVector},
        U::AbstractArray,
        I::Interpreter,
        x::AbstractArray,
        p::AbstractArray,
        cache::InterpreterCache = InterpreterCache(I, x, p),
    )
        instructions = cache.instructions
        constants = I.constants
        for (i, instr) in enumerate(I.instructions)
            op = instr.op
            arg1, arg2 = instr.args
            $(
                data_block(:(arg1.data), :(arg2.data); parameters = true) do data1, data2
                    make_op_block(
                        :(instructions[i]),
                        :op,
                        data1,
                        :(arg1.ind),
                        data2,
                        :(arg2.ind),
                    )
                end
            )
        end

        if I.out isa AbstractMatrix
            for j = 1:size(I.out, 2), i = 1:size(I.out, 1)
                @unpack data, ind = I.out[i, j]
                if data == DATA_I
                    U[i, j] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[i, j] = convert(eltype(U), x[ind])
                elseif data == DATA_P
                    U[i, j] = convert(eltype(U), p[ind])
                elseif data == DATA_C
                    U[i, j] = convert(eltype(U), constants[ind])
                end
            end
        else
            for k in eachindex(I.out)
                @unpack data, ind = I.out[k]
                if data == DATA_I
                    U[k] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[k] = convert(eltype(U), x[ind])
                elseif data == DATA_P
                    U[k] = convert(eltype(U), p[ind])
                elseif data == DATA_C
                    U[k] = convert(eltype(U), constants[ind])
                end
            end
        end

        if !isa(u, Nothing)
            for k in eachindex(I.eval_out)
                @unpack data, ind = I.eval_out[k]
                if data == DATA_I
                    u[k] = convert(eltype(u), instructions[ind])
                elseif data == DATA_X
                    u[k] = convert(eltype(u), x[ind])
                elseif data == DATA_P
                    u[k] = convert(eltype(u), p[ind])
                elseif data == DATA_C
                    u[k] = convert(eltype(u), constants[ind])
                end
            end
        end
        U
    end

    # homotopy versions
    function _execute!(
        u::Union{Nothing,AbstractVector},
        U::AbstractArray,
        I::Interpreter,
        x::AbstractArray,
        t,
        ::Nothing = nothing,
        cache::InterpreterCache = InterpreterCache(I, x),
    )
        instructions = cache.instructions
        constants = I.constants
        for (i, instr) in enumerate(I.instructions)
            op = instr.op
            arg1, arg2 = instr.args
            $(
                data_block(
                    :(arg1.data),
                    :(arg2.data);
                    parameters = false,
                    t = true,
                ) do data1, data2
                    make_op_block(
                        :(instructions[i]),
                        :op,
                        data1,
                        :(arg1.ind),
                        data2,
                        :(arg2.ind),
                    )
                end
            )
        end
        if I.out isa AbstractMatrix
            for j = 1:size(I.out, 2), i = 1:size(I.out, 1)
                @unpack data, ind = I.out[i, j]
                if data == DATA_I
                    U[i, j] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[i, j] = convert(eltype(U), x[ind])
                elseif data == DATA_T
                    U[i, j] = convert(eltype(U), t)
                elseif data == DATA_C
                    U[i, j] = convert(eltype(U), constants[ind])
                end
            end
        else
            for k in eachindex(I.out)
                @unpack data, ind = I.out[k]
                if data == DATA_I
                    U[k] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[k] = convert(eltype(U), x[ind])
                elseif data == DATA_T
                    U[k] = convert(eltype(U), t)
                elseif data == DATA_C
                    U[k] = convert(eltype(U), constants[ind])
                end
            end
        end

        if !isa(u, Nothing)
            for k in eachindex(I.eval_out)
                @unpack data, ind = I.eval_out[k]
                if data == DATA_I
                    u[k] = convert(eltype(u), instructions[ind])
                elseif data == DATA_X
                    u[k] = convert(eltype(u), x[ind])
                elseif data == DATA_T
                    u[k] = convert(eltype(u), t)
                elseif data == DATA_C
                    u[k] = convert(eltype(u), constants[ind])
                end
            end
        end
        U
    end

    function _execute!(
        u::Union{Nothing,AbstractVector},
        U::AbstractArray,
        I::Interpreter,
        x::AbstractArray,
        t,
        p::AbstractArray,
        cache::InterpreterCache = InterpreterCache(I, x, p),
    )
        instructions = cache.instructions
        constants = I.constants
        for (i, instr) in enumerate(I.instructions)
            op = instr.op
            arg1, arg2 = instr.args
            $(
                data_block(
                    :(arg1.data),
                    :(arg2.data);
                    parameters = true,
                    t = true,
                ) do data1, data2
                    make_op_block(
                        :(instructions[i]),
                        :op,
                        data1,
                        :(arg1.ind),
                        data2,
                        :(arg2.ind),
                    )
                end
            )
        end

        if I.out isa AbstractMatrix
            for j = 1:size(I.out, 2), i = 1:size(I.out, 1)
                U[i, j] = begin
                    @unpack data, ind = I.out[i, j]
                    if data == DATA_I
                        convert(eltype(U), instructions[ind])
                    elseif data == DATA_X
                        convert(eltype(U), x[ind])
                    elseif data == DATA_T
                        convert(eltype(U), t)
                    elseif data == DATA_P
                        convert(eltype(U), p[ind])
                    elseif data == DATA_C
                        convert(eltype(U), constants[ind])
                    end
                end
            end
        else
            for k in eachindex(I.out)
                @unpack data, ind = I.out[k]
                if data == DATA_I
                    U[k] = convert(eltype(U), instructions[ind])
                elseif data == DATA_X
                    U[k] = convert(eltype(U), x[ind])
                elseif data == DATA_T
                    U[k] = convert(eltype(U), t)
                elseif data == DATA_P
                    U[k] = convert(eltype(U), p[ind])
                elseif data == DATA_C
                    U[k] = convert(eltype(U), constants[ind])
                end
            end
        end

        if !isa(u, Nothing)
            for k in eachindex(I.eval_out)
                @unpack data, ind = I.eval_out[k]
                if data == DATA_I
                    u[k] = convert(eltype(u), instructions[ind])
                elseif data == DATA_X
                    u[k] = convert(eltype(u), x[ind])
                elseif data == DATA_T
                    u[k] = convert(eltype(u), t)
                elseif data == DATA_P
                    u[k] = convert(eltype(u), p[ind])
                elseif data == DATA_C
                    u[k] = convert(eltype(u), constants[ind])
                end
            end
        end
        U
    end
end

# Taylor case

for has_t in [false, true], has_p in [false, true]
    @eval begin
        function execute!(
            u::AbstractVecOrMat,
            ord::Type{Val{K}},
            I::Interpreter,
            x::AbstractArray,
            $((has_t ? (:t₀,) : ())...),
            $((has_p ? (:(p::AbstractArray),) : (Expr(:kw, :(::Nothing), :nothing),))...),
            cache::InterpreterCache = InterpreterCache(NTuple{K + 1,eltype(x)}, I),
        ) where {K}
            instructions = cache.instructions
            constants = I.constants
            $((has_t ? (:(t = (t₀, one(t₀))),) : ())...)
            for (i, instr) in enumerate(I.instructions)
                op = instr.op
                arg1, arg2 = instr.args
                $(
                    data_block(
                        :(arg1.data),
                        :(arg2.data);
                        parameters = has_p,
                        t = has_t,
                    ) do data1, data2
                        make_taylor_op_block(
                            :(instructions[i]),
                            :ord,
                            :op,
                            data1,
                            :(arg1.ind),
                            data2,
                            :(arg2.ind),
                        )
                    end
                )
            end

            if u isa AbstractMatrix
                for k in eachindex(I.eval_out)
                    @unpack data, ind = I.eval_out[k]
                    if data == DATA_I
                        for (j, val_j) in enumerate(instructions[ind])
                            u[k, j] = val_j
                        end
                    elseif data == DATA_X
                        for j = 1:K+1
                            u[k, j] = length(x[ind]) ≥ j ? x[ind][j] : zero(eltype(u))
                        end
                    elseif data == DATA_C
                        u[k, 1] = constants[ind]
                        for j = 2:K+1
                            u[k, j] = zero(eltype(u))
                        end
                    else
                        $(
                            if has_p
                                quote
                                    if data == DATA_P
                                        for j = 1:K+1
                                            u[k, j] = length(p[ind]) ≥ j ? p[ind][j] :
                                                zero(eltype(u))
                                        end
                                    end
                                end
                            else
                                :(nothing)
                            end
                        )
                        $(
                            if has_t
                                quote
                                    if data == DATA_T
                                        u[k, 1] = t
                                        u[k, 2] = one(eltype(u))
                                        for j = 3:K+1
                                            u[k, j] = zero(eltype(u))
                                        end
                                    end
                                end
                            else
                                :(nothing)
                            end
                        )
                    end
                end
            else
                for k in eachindex(I.eval_out)
                    @unpack data, ind = I.eval_out[k]
                    if data == DATA_I
                        u[k] = last(instructions[ind])
                    elseif data == DATA_X
                        if length(x[ind]) ≥ K + 1
                            u[k] = x[ind][K+1]
                        else
                            u[k] = zero(eltype(u))
                        end
                    elseif data == DATA_C
                        u[k] = zero(eltype(u))
                    elseif data == DATA_T
                        if K == 1
                            u[k] = one(eltype(u))
                        else
                            u[k] = zero(eltype(u))
                        end
                    else
                        $(
                            if has_p
                                quote
                                    if data == DATA_P
                                        if length(p[ind]) ≥ K + 1
                                            u[k] = p[ind][K+1]
                                        else
                                            u[k] = zero(eltype(u))
                                        end
                                    end
                                end
                            else
                                :(nothing)
                            end
                        )
                    end
                end
            end
            u
        end
    end
end
