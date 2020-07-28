@testset "Endgame" begin
    @testset "Cyclic 7" begin
        res = solve(cyclic(7); start_system = :total_degree, show_progress = false)
        @test nsolutions(res) == 924
    end

    @testset "Hyperbolic - 6,6" begin
        # 2 solutions with multiplicity 6, projective
        @var x z
        y = 1
        # This has two roots of multiplicity 6 at the hyperplane z=0
        # each root has winding number 3
        F = [
            0.75 * x^4 + 1.5 * x^2 * y^2 - 2.5 * x^2 * z^2 + 0.75 * y^4 - 2.5 * y^2 * z^2 + 0.75 * z^4
            10 * x^2 * z + 10 * y^2 * z - 6 * z^3
        ]
        res = solve(System(F); start_system = :total_degree)
        @test count(r -> r.winding_number == 3, path_results(res)) == 12
        @test nresults(res) == 2
        @test nsingular(res) == 2
        @test all(r -> multiplicity(r) == 6, results(res))
    end

    @testset "Singular 1" begin
        # 1 singular solution with multiplicity 3
        @var x y
        z = 1
        F = [
            x^2 + 2 * y^2 + 2 * im * y * z,
            (18 + 3 * im) * x * y + 7 * im * y^2 - (3 - 18 * im) * x * z - 14 * y * z -
            7 * im * z^2,
        ]
        result = solve(System(F); start_system = :total_degree)
        @test nsingular(result) == 1
        @test nresults(result; only_nonsingular = true) == 1
    end

    @testset "Wilkinson $d" for d in [12]
        @var x
        f = System([expand(prod(x - i for i = 1:d))])
        res = track.(total_degree(f, endgame_options = (only_nonsingular = true,))...)
        @test all(is_success, res)
        @test round.(Int, real.(sort(first.(solution.(res)); by = abs))) == 1:d
        @test maximum(abs.(imag.(first.(solution.(res))))) < 1e-4
    end

    @testset "(x-10)^$d" for d in [2, 6]
        @var x
        f = System([(x - 10)^d])
        res = track.(total_degree(f)...)
        @test count(r -> r.winding_number == d, res) == d
    end

    @testset "Beyond Polyhedral Homotopy Example" begin
        @var x y
        f = [2.3 * x^2 + 1.2 * y^2 + 3x - 2y + 3, 2.3 * x^2 + 1.2 * y^2 + 5x + 2y - 5]
        res = track.(total_degree(System(f))...)
        @test count(is_success, res) == 2
        @test count(is_at_infinity, res) == 2
    end

    @testset "Winding Number Family d=$d" for d = 2:2:6
        @var x y
        a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
        f1 = (a[1] * x^d + a[2] * y) * (a[3] * x + a[4] * y) + 1
        f2 = (a[1] * x^d + a[2] * y) * (a[5] * x + a[6] * y) + 1
        res = track.(total_degree(System([f1, f2]))...)
        @test count(is_success, res) == d + 1
    end

    @testset "Bacillus Subtilis" begin
        res = solve(bacillus())
        @test nsolutions(res) == 44
    end

    @testset "Mohab" begin
        # Communicated by Mohab Safey El Din
        @var x y z
        F = [
            -9091098778555951517 * x^3 * y^4 * z^2 +
            5958442613080401626 * y^2 * z^7 +
            17596733865548170996 * x^2 * z^6 - 17979170986378486474 * x * y * z^6 -
            2382961149475678300 * x^4 * y^3 - 15412758154771986214 * x * y^3 * z^3 +
            133,
            -10798198881812549632 * x^6 * y^3 * z - 11318272225454111450 * x * y^9 -
            14291416869306766841 * y^9 * z - 5851790090514210599 * y^2 * z^8 +
            15067068695242799727 * x^2 * y^3 * z^4 +
            7716112995720175148 * x^3 * y * z^3 +
            171,
            13005416239846485183 * x^7 * y^3 + 4144861898662531651 * x^5 * z^4 -
            8026818640767362673 * x^6 - 6882178109031199747 * x^2 * y^4 +
            7240929562177127812 * x^2 * y^3 * z +
            5384944853425480296 * x * y * z^4 +
            88,
        ]
        @time res =
            solve(System(F, [x, z, y]); start_system = :total_degree, show_progress = false)
        @test nnonsingular(res) == 693

        # test for γ value where path jumping happened with to loose default config
        @time res = solve(
            System(F, [x, z, y]),
            γ = -0.9132549847010242 + 0.4073884300256109im,
            start_system = :total_degree,
        )
        @test nnonsingular(res) == 693
    end

    @testset "Ill-conditioned solution - look's almost diverging" begin
        sys, q₀ = fano_quintic()
        solver, starts = solver_startsolutions(
            sys;
            target_parameters = q₀,
            start_system = :polyhedral,
            seed = 0x56db29a3,
        )
        S = collect(starts)
        res = track(solver, S[1785])
        @test is_success(res)
    end
end
