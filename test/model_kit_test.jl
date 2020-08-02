using HomotopyContinuation.ModelKit

@testset "ModelKit" begin
    @testset "SymEngine" begin
        @test Expression(MathConstants.catalan) isa Expression
        @test Expression(MathConstants.e) isa Expression
        @test Expression(MathConstants.pi) isa Expression
        @test Expression(MathConstants.γ) isa Expression
        @test Expression(Float16(2.1)) isa Expression
        @test Expression(Float32(2.1)) isa Expression
        @test Expression(big(2.1)) isa Expression
        @test Expression(Int32(2)) isa Expression
        @test Expression(Int16(2)) isa Expression
        @test Expression(Int128(2)) isa Expression
        @test Expression(2 // 3) isa Expression
        @test convert(BigFloat, Expression(big(2.1))) == big(2.1)
        @test ModelKit.is_number(Expression(2))
        @test ModelKit.to_number(Expression(big(2.1))) isa BigFloat
        @test ModelKit.to_number(Expression(2 // 3)) isa Rational{Int}
        @test ModelKit.to_number(Expression(2) // 3) isa Rational{Int}
        @test ModelKit.to_number(Expression(2) // Expression(3)) isa Rational{Int}
        @test ModelKit.to_number(Expression(big(2)^129)) isa BigInt
        @test complex(Expression(1), Expression(2)) == Expression(1) + im * 2
        @test complex(Expression(1), 2) == Expression(1) + im * 2
        @test complex(1, Expression(2)) == Expression(1) + im * 2
        @test +Expression(2) == 2
        @test Expression(HC.DoubleDouble.DoubleF64(2)) == big(2.0)
    end

    @testset "Variables" begin
        @var a b x[1:2] y[1:2, 1:3]

        @test a isa Variable
        @test b isa Variable
        @test x isa Vector{Variable}
        @test length(x) == 2
        @test y isa Matrix{Variable}
        @test size(y) == (2, 3)
        @test join(x, " ") == "x₁ x₂"

        @test reshape(sort(vec(y)), 2, 3) == y
        @var x[9:11]
        @test sort(x) == x

        @var c, d
        @test c isa Variable
        @test d isa Variable
        @test variables(c) == [c]
        @test c === c
        @test c !== copy(c)

        let
            c2 = @unique_var c
            @test c != c2
        end

        f = a + b
        @test variables(f) == [a, b]
        @test nvariables(f) == 2
        g = x[1]
        @test variables([f, g]) == [a, b, x[1]]
        @test nvariables([f, g]) == 3
    end

    @testset "Subs" begin
        @var x y w z u
        f = x^2 * (x + y * w)

        @test subs(f, x => z) == z^2 * (z + w * y)
        @test subs([f], x => z) == [z^2 * (z + w * y)]
        @test subs(f, [x, y] => [z^2, z + 2]) == z^4 * (w * (2 + z) + z^2)
        @test subs(f, [x, y] => [z^2, z + 2], w => u) == z^4 * (u * (2 + z) + z^2)
        @test subs(f, x => z^2, y => 3, w => u) == z^4 * (3 * u + z^2)
        @test subs(f, Dict(x => z^2, y => 3, w => u)) == z^4 * (3 * u + z^2)
    end

    @testset "Evaluation" begin
        @var x y w
        f = x^2 * (x + y * w)
        @test evaluate([f, f], [x, y, w] => [2, 3, -5]) == [-52, -52]
        @test [f, 2f]([x, y, w] => [2, 3, -5]) == [-52, -104]
    end

    @testset "Linear Algebra" begin
        @var x[1:2, 1:2]
        @test det(x[1:1, 1:1]) == x[1, 1]
        @test det(x) == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
        @test det(x') == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
        @test det(transpose(x)) == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
    end

    @testset "Differentiation" begin
        @var x y

        f = x^2 + y^2
        g = x^3 + y^3

        @test differentiate(f, x) == 2x
        @test differentiate(f, [x, y]) == [2x, 2y]

        @test differentiate([f, g], x) == [2x, 3 * x^2]
        @test differentiate([f, g], [x, y]) == [2x 2y; 3 * x^2 3 * y^2]
    end

    @testset "Expand" begin
        @var x y
        @test expand((x + y)^2) == 2 * x * y + x^2 + y^2
    end

    @testset "to_dict" begin
        @var x y a
        @test ModelKit.to_dict(x^2 + y * x * a + y * x, [x, y]) ==
              Dict([2, 0] => Expression(1), [1, 1] => a + 1)
        @test ModelKit.degree(x * y^5 + y^2, [x, y]) == 6
    end

    @testset "Modeling" begin
        @testset "Bottleneck" begin
            @var x y z
            f = [
                (0.3 * x^2 + 0.5z + 0.3x + 1.2 * y^2 - 1.1)^2 +
                (0.7 * (y - 0.5x)^2 + y + 1.2 * z^2 - 1)^2 - 0.3,
            ]

            I = let x = [x, y, z]
                n, m = length(x), length(f)
                @unique_var y[1:n] v[1:m] w[1:m]
                J = differentiate(f, x)
                f′ = f(x => y)
                J′ = J(x => y)
                Nx = (x - y) - J' * v
                Ny = (x - y) - J′' * w
                System([f; f′; Nx; Ny], [x; y; v; w])
            end
            @test I isa System
            @test size(I) == (8, 8)
        end

        @testset "Steiner" begin
            @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5]

            #tangential conics
            f = sum([a; 1] .* monomials([x; 1], 2; affine = false))
            ∇ = differentiate(f, x)
            #5 conics
            g = sum(c .* monomials([x; 1], 2; affine = false))
            ∇_2 = differentiate(g, x)
            #the general system
            #f_a_0 is tangent to g_b₀ at x₀
            function Incidence(f, a₀, g, b₀, x₀)
                fᵢ = f(x => x₀, a => a₀)
                ∇ᵢ = ∇(x => x₀, a => a₀)
                Cᵢ = g(x => x₀, c => b₀)
                ∇_Cᵢ = ∇_2(x => x₀, c => b₀)

                [fᵢ; Cᵢ; det([∇ᵢ ∇_Cᵢ])]
            end
            @var v[1:6, 1:5]
            I = vcat(map(i -> Incidence(f, a, g, v[:, i], y[:, i]), 1:5)...)
            F = System(I, [a; vec(y)], vec(v))
            @test size(F) == (15, 15)
        end

        @testset "Reach plane curve" begin
            @var x y
            f = (x^3 - x * y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5
            ∇ = differentiate(f, [x; y]) # the gradient
            H = differentiate(∇, [x; y]) # the Hessian

            g = ∇ ⋅ ∇
            v = [-∇[2]; ∇[1]]
            h = v' * H * v
            dg = differentiate(g, [x; y])
            dh = differentiate(h, [x; y])

            ∇σ = g .* dh - ((3 / 2) * h) .* dg

            F = System([v ⋅ ∇σ; f], [x, y])
            @test size(F) == (2, 2)
        end
    end

    @testset "Horner" begin
        # use fano scheme as test
        n = 4
        s = 1
        @var x[1:n]
        ν = monomials(x, 5; affine = false)
        N = length(ν)
        q₀ = randn(ComplexF64, N - 1)
        @var q[1:N-1]
        F = sum(q[i] * ν[i] for i = 1:N-1) + 1

        @var a[1:n-1] b[1:n-1] t
        L = [a; 1] .* t + [b; 0]

        G = subs(F, x => L)

        FcapL = collect(values(ModelKit.to_dict(expand(G), [t])))

        @test all(FcapL) do f
            g = horner(f)
            expand(g) == f
        end
        @test all(FcapL) do f
            g = horner(f, [a; b])
            expand(g) == f
        end
    end

    @testset "Rand / dense poly" begin
        @var x y
        f, c = dense_poly([x, y], 3)
        @test length(c) == 10
        g = rand_poly(Float64, [x, y], 3)
        @test subs(f, c => coefficients(g, [x, y])) == g
        _, coeffs = exponents_coefficients(g, [x, y]; expanded = true)
        @test subs(f, c => coeffs) == g

        @var x[1:3]
        f, c = dense_poly(x, 3, coeff_name = :c)
        g = x[1]^3 + x[2]^3 + x[3]^3 - 1
        gc = coeffs_as_dense_poly(g, x, 3)
        @test subs(f, c => gc) == g
    end

    @testset "System" begin
        @var x y a b
        f = [(x + y)^3 + x^2 + x + 5y + 3a, 2 * x^2 + b]
        F = System(f, [x, y], [b, a])
        @test nvariables(F) == 2
        @test nparameters(F) == 2
        @test variables(F) == [x, y]
        @test parameters(F) == [b, a]
        show_F = """
        System of length 2
         2 variables: x, y
         2 parameters: b, a

         3*a + x + 5*y + x^2 + (x + y)^3
         b + 2*x^2"""
        @test sprint(show, F) == show_F
        @test degrees(F) == [3, 2]

        T = CompiledSystem(F; optimizations = false)
        F2 = System(T)
        @test F == F2
        @test sprint(show, T) == "Compiled: $show_F"
        @test size(T) == size(F) == (2, 2)

        T2 = CompiledSystem(F; optimizations = true)
        F3 = System(T2)
        @test expand.(F.expressions) == expand.(F3.expressions)

        F4 = System(f, parameters = [b, a])
        @test variables(F4) == [x, y]

        @var x y a b
        f = Any[(x+y)^3+x^2+x+5y+3a, 2*x^2+b]
        F = System(f, [x, y], [b, a])
        @test F isa System
    end

    @testset "System variables groups + homogeneous" begin
        @var x y z
        f = System([
            (x^2 + y^2 + x * y - 3 * z^2) * (x + 3z),
            (x^2 + y^2 + x * y - 3 * z^2) * (y - x + 2z),
            2x + 5y - 3z,
        ])
        @test is_homogeneous(f)
        multi_degrees(f) == [1 1; 2 0]

        @var x y z v w
        g = System([x * y - 2v * w, x^2 - 4 * v^2], variable_groups = [[x, v], [y, w]])
        @test is_homogeneous(g)
        multi_degrees(g) == [1 1; 2 0]
    end

    @testset "Homotopy" begin
        @var x y z t

        h = [x^2 + y + z + 2t, 4 * x^2 * z^2 * y + 4z - 6x * y * z^2]
        H = Homotopy(h, [x, y, z], t)

        @test nvariables(H) == 3
        @test nparameters(H) == 0
        @test variables(H) == [x, y, z]
        @test parameters(H) == []
        show_H = """
        Homotopy in t of length 2
         3 variables: x, y, z

         2*t + y + z + x^2
         4*z - 6*x*y*z^2 + 4*x^2*y*z^2"""

        @test sprint(show, H) == show_H

        H_any = Homotopy(convert(Vector{Any}, h), [x, y, z], t)
        @test H_any isa Homotopy

        T = CompiledHomotopy(H)
        H2 = ModelKit.interpret(T)
        @test H == H2
        @test sprint(show, T) == "Compiled: $show_H"
        @test size(T) == size(H) == (2, 3)
    end

    @testset "CompiledHomotopy" begin
        @var x y a b c
        f = [(2 * x^2 + b^2 * y^3 + 2 * a * x * y)^3, (a + c)^4 * x + y^2]

        @var s sp[1:3] sq[1:3]
        g = subs(f, [a, b, c] => s .* sp .+ (1 .- s) .* sq)

        H = Homotopy(g, [x, y], s, [sp; sq])

        p = [5.2, -1.3, 9.3]
        q = [2.6, 3.3, 2.3]

        H = CompiledHomotopy(H)
        @test size(H) == (2, 2)

        v = [0.192, 2.21]
        t = 0.232

        u = zeros(ComplexF64, 2)
        U = zeros(ComplexF64, 2, 2)

        evaluate!(u, H, v, t, [p; q])
        @test u ≈ f([x, y] => v, [a, b, c] => t * p + (1 - t) * q)

        taylor!(u, Val(1), H, TaylorVector{1}(Matrix(v')), t, [p; q])
        @test u ≈ let
            @var s sp[1:3] sq[1:3]
            pt = s .* sp .+ (1 .- s) .* sq
            differentiate(subs(f, [a, b, c] => pt), s)(
                [x, y] => v,
                sp => p,
                sq => q,
                s => t,
            )
        end

        evaluate_and_jacobian!(u, U, H, v, t, [p; q])
        @test U ≈ differentiate(f, [x, y])([x, y] => v, [a, b, c] => t * p + (1 - t) * q)
    end

    @testset "Codegen helpers" begin
        @test ModelKit.sqr(3 + 2im) == (3 + 2im)^2
        @test ModelKit.sqr(3) == 3^2
    end

    @testset "Instructions" begin

        @testset "Simple dt" begin
            @var x a b t
            a = 0.5
            b = 0.1
            h = (x - (t - (a + im * b))^2) * (x + (t - (a + im * b))^2)
            list, _ = ModelKit.instruction_list([h])
            D = ModelKit.DiffMap()
            D[:t, 1] = 1
            slp = @eval ModelKit begin
                function dt_test1(x, t)
                    $(ModelKit.to_expr(ModelKit.univariate_diff!(list, 1, D)))
                end
            end
            expr1 = ModelKit.expand(ModelKit.dt_test1(x, t))
            expr2 = ModelKit.expand(ModelKit.differentiate(h, t))
            @test expr1 == expr2
        end
        @testset "Higher order pow diff" begin
            for d = 2:10
                @var x
                f = x^d
                list, _ = ModelKit.instruction_list([f])

                D = ModelKit.DiffMap()
                D[:x, 1] = :x1
                D[:x, 2] = :x2
                D[:x, 3] = :x3

                @eval ModelKit begin
                    function __diff_4_pow(x, x1, x2, x3)
                        $(ModelKit.to_expr(ModelKit.univariate_diff!(list, 4, D)))
                    end
                end

                @var x1 x2 x3 λ

                @test expand(
                    subs(
                        differentiate(
                            subs(f, x => x .+ λ .* x1 .+ λ^2 .* x2 + λ^3 .* x3),
                            λ,
                            4,
                        ),
                        λ => 0,
                    ) / 24,
                ) == expand(ModelKit.__diff_4_pow(x, x1, x2, x3))
            end
        end

        @testset "Higher order mul" begin
            @var x y
            f = x * y
            list, _ = ModelKit.instruction_list([f])
            D = ModelKit.DiffMap()
            D[:x, 1] = :x1
            D[:x, 2] = :x2
            D[:x, 3] = :x3
            D[:y, 1] = :y1
            D[:y, 2] = :y2
            D[:y, 3] = :y3

            @eval ModelKit begin
                function __diff_4_mul__(x, y, (x1, x2, x3), (y1, y2, y3))
                    $(ModelKit.to_expr(ModelKit.univariate_diff!(list, 4, D)))
                end
            end
            @var x1 x2 x3 y1 y2 y3 λ

            @test expand(
                subs(
                    differentiate(
                        subs(
                            f,
                            x => x .+ λ .* x1 .+ λ^2 .* x2 + λ^3 .* x3,
                            y => y .+ λ .* y1 .+ λ^2 .* y2 + λ^3 .* y3,
                        ),
                        λ,
                        4,
                    ),
                    λ => 0,
                ) / 24,
            ) == expand(ModelKit.__diff_4_mul__(x, y, (x1, x2, x3), (y1, y2, y3)))
        end

        @testset "Higher order plus" begin
            @var x y
            f = x + y
            list, _ = ModelKit.instruction_list([f])

            D = ModelKit.DiffMap()
            D[:x, 1] = :x1
            D[:x, 2] = :x2
            D[:x, 3] = :x3
            D[:y, 1] = :y1
            D[:y, 2] = :y2
            D[:y, 3] = :y3

            @eval ModelKit begin
                function __diff_3_plus__(x, y, (x1, x2, x3), (y1, y2, y3))
                    $(ModelKit.to_expr(ModelKit.univariate_diff!(list, 3, D)))
                end
            end

            @var x1 x2 x3 y1 y2 y3 λ

            @test expand(
                subs(
                    differentiate(
                        subs(
                            f,
                            x => x .+ λ .* x1 .+ λ^2 .* x2 + λ^3 .* x3,
                            y => y .+ λ .* y1 .+ λ^2 .* y2 + λ^3 .* y3,
                        ),
                        λ,
                        3,
                    ),
                    λ => 0,
                ) / 6,
            ) == expand(ModelKit.__diff_3_plus__(x, y, (x1, x2, x3), (y1, y2, y3)))
        end
    end


    @testset "Homotopy codegen (Katsura(3))" begin
        n = 3
        @var x[0:n] ẋ[0:n] ẍ[0:n] x3[0:n] t γ
        K = [
            (
                sum(x[abs(l)+1] * x[abs(m - l)+1] for l = -n:n if abs(m - l) <= n) - x[m+1] for m = 0:n-1
            )...,
            x[1] + 2 * sum(x[i+1] for i = 1:n) - 1,
        ]

        h = γ .* t .* [x[1:n] .^ 2 .- 1; x[n+1] - 1] + (1 - t) .* K
        H = ModelKit.Homotopy(h, x, t, [γ])
        TH = CompiledHomotopy(H)

        tx3 = TaylorVector{4}(zeros(Expression, 4, 4))
        y, y1, y2, y3 = vectors(tx3)
        y .= x
        y1 .= ẋ
        y2 .= ẍ
        y3 .= x3

        dt1 =
            ModelKit.taylor!(zeros(Expression, 4), Val(1), TH, TaylorVector{1}(tx3), t, [γ])
        dt2 =
            ModelKit.taylor!(zeros(Expression, 4), Val(2), TH, TaylorVector{2}(tx3), t, [γ])
        dt3 =
            ModelKit.taylor!(zeros(Expression, 4), Val(3), TH, TaylorVector{3}(tx3), t, [γ])
        dt4 =
            ModelKit.taylor!(zeros(Expression, 4), Val(4), TH, TaylorVector{4}(tx3), t, [γ])

        @var λ
        Hd1 = subs(H.expressions, t => t + λ)
        true_dt1 = subs(differentiate(Hd1, λ, 1), λ => 0)

        Hd2 = subs(H.expressions, x => x .+ λ .* ẋ, t => t + λ)
        true_dt2 = subs(differentiate(Hd2, λ, 2), λ => 0) / 2

        Hd3 = subs(H.expressions, x => x .+ λ .* ẋ .+ λ^2 .* ẍ, t => t + λ)
        true_dt3 = subs(differentiate(Hd3, λ, 3), λ => 0) / 6

        Hd4 = subs(H.expressions, x => x .+ λ .* ẋ .+ λ^2 .* ẍ .+ λ^3 .* x3, t => t + λ)
        true_dt4 = subs(differentiate(Hd4, λ, 4), λ => 0) / 24

        @test expand.(TH(x, t, [γ])) == expand.(h)
        @test expand.(ModelKit.jacobian(TH, x, t, [γ])) == expand.(differentiate(h, x))
        @test expand.(dt1) == expand.(true_dt1)
        @test expand.(dt2) == expand.(true_dt2)
        @test expand.(dt3) == expand.(true_dt3)
        @test expand.(dt4) == expand.(true_dt4)
    end

    @testset "taylor! - system" begin
        @var x[1:2] ẋ[1:2] ẍ[1:2] x3[1:2] p[1:2] ṗ[1:2]
        f = [(x[1] + x[2])^3 + x[1]^2 + x[1] + 5 * x[2] + 3 * p[1], 2 * x[1]^2 + p[2]]
        F = System(f, x, p)
        TF = CompiledSystem(F)

        tx = TaylorVector{4}([x'; ẋ'; ẍ'; x3'])
        tv = TaylorVector{5}(Expression, 2)
        v, v1, v2, v3, v4 = vectors(tv)
        @var λ

        ModelKit.taylor!(TaylorVector{3}(tv), TF, TaylorVector{2}(tx), p)
        Fd2 = subs(F.expressions, x => x .+ λ .* ẋ)
        true_v1 = subs(differentiate(Fd2, λ, 1), λ => 0)
        true_v2 = subs(differentiate(Fd2, λ, 2), λ => 0) / 2
        @test expand.(v) == expand.(f)
        @test expand.(v1) == expand.(true_v1)
        @test expand.(v2) == expand.(true_v2)

        ModelKit.taylor!(TaylorVector{4}(tv), TF, TaylorVector{3}(tx), p)
        Fd3 = subs(F.expressions, x => x .+ λ .* ẋ .+ λ^2 .* ẍ)
        true_v1 = subs(differentiate(Fd3, λ, 1), λ => 0)
        true_v2 = subs(differentiate(Fd3, λ, 2), λ => 0) / 2
        true_v3 = subs(differentiate(Fd3, λ, 3), λ => 0) / 6
        @test expand.(v) == expand.(f)
        @test expand.(v1) == expand.(true_v1)
        sub = variables(v2) => rand(1:10_000, 6)
        @test v2(sub) == true_v2(sub)
        @test v3(sub) == true_v3(sub)
    end
end
