@testset "Extensive tests" begin
    @testset "Lines on a quintic surface in 3-space" begin
        sys, q₀, gamma = fano_quintic()
        @time res = solve(
            sys;
            gamma = gamma,
            target_parameters = q₀,
            start_system = :total_degree,
            show_progress = true,
        )
        @test nsolutions(res) == 2875
        @test ndistinct_certified(certify(sys, res, q₀)) == 2875

        @time poly_res = solve(sys; target_parameters = q₀, start_system = :polyhedral)
        @test nsolutions(poly_res) == 2875
    end

    @testset "3264" begin
        F = steiner()
        p = randn(ComplexF64, 30)
        res = solve(F; target_parameters = p)
        @test nresults(res; only_nonsingular = true) == 3264
        @test ndistinct_certified(certify(F, res, p)) == 3264

        real_conics = [
            10124547 // 662488724,
            8554609 // 755781377,
            5860508 // 2798943247,
            -251402893 // 1016797750,
            -25443962 // 277938473,
            1 // 1,
            520811 // 1788018449,
            2183697 // 542440933,
            9030222 // 652429049,
            -12680955 // 370629407,
            -24872323 // 105706890,
            1 // 1,
            6537193 // 241535591,
            -7424602 // 363844915,
            6264373 // 1630169777,
            13097677 // 39806827,
            -29825861 // 240478169,
            1 // 1,
            13173269 // 2284890206,
            4510030 // 483147459,
            2224435 // 588965799,
            33318719 // 219393000,
            92891037 // 755709662,
            1 // 1,
            8275097 // 452566634,
            -19174153 // 408565940,
            5184916 // 172253855,
            -23713234 // 87670601,
            28246737 // 81404569,
            1 // 1,
        ]

        real_res = solve(
            F,
            solutions(res);
            start_parameters = p,
            target_parameters = real_conics,
            tracker_options = (parameters = :conservative,),
        )

        @test nsolutions(real_res) == 3264
        real_cert = certify(F, real_res, real_conics)
        @test ndistinct_certified(real_cert) == 3264
        @test ndistinct_real_certified(real_cert) == 3264

        direct_real_res = solve(F; target_parameters = real_conics)
        @test nresults(direct_real_res; only_nonsingular = true) == 3264
    end
end
