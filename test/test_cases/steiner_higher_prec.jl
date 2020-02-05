# The following is a particular combination of generic parameters such that
# the goes close to a singularity. An increase in the precision of the residual
# is only necessary for 1 or 2 steps
@testset "Steiner High Precision" begin
    p = Complex{Float64}[
        -0.8760865081446109-0.8245161978967308im,
        -0.371989298448952+1.175807247192591im,
        -0.5889429565516365-0.6815249624614181im,
        0.05893625668644283-0.15242494424272904im,
        -0.04017905091825077-0.32922740646951476im,
        -1.1620912695313244-0.880462208384591im,
        -0.6128442732846304+0.2299959308120725im,
        0.04451305089395737-0.5613666038122553im,
        0.4275591895528981-0.4560517627610251im,
        -1.44959201510603+0.9148102094587954im,
        0.9642383181691287-1.028506672030121im,
        -0.04531547209868653+1.1658486088983617im,
        -0.7706473388214675+1.6213385645828058im,
        0.3905889030007126-0.837367437811382im,
        0.521976803491032+0.8715564019511688im,
        0.9913423747115939-0.9251461816473662im,
        -1.5976395911608239-0.6869570695244973im,
        0.3812500672931469-0.2640531551530478im,
        0.6050593029483629+1.0735839629840869im,
        0.24229108273519206+0.31651798241470175im,
        0.2343873046310603+0.9209493301878944im,
        -0.04243469809131875+0.2930962153305209im,
        -0.05095304883782437-0.07163754176755009im,
        0.055623959461651266+0.11458865587499502im,
        0.4749235468690426+1.3139464606178932im,
        -0.5340685054046019-0.05295858535911429im,
        0.6838868206371786-0.5368790063173585im,
        -0.6012500786841708-0.41545534317815064im,
        0.14315666179481693+0.7966975827842401im,
        0.06217289908817691-1.268704027512965im,
    ]

    q = [
        0.18802370367009363,
        -1.9260285280414342,
        0.6835970218915922,
        -0.5817725614568748,
        -0.5033365002810842,
        0.6894548688621689,
        0.03748633251693155,
        -0.12168863868778622,
        -1.6197914296308757,
        -2.2192790882018585,
        1.4133360740976828,
        1.6335108803709772,
        0.13316023698203722,
        0.6415334044239753,
        0.36899573610352765,
        -0.6026231774364099,
        1.086473021454091,
        -0.7385757989526022,
        1.4247229002869566,
        -0.2974094335738924,
        -0.2651970300878602,
        -1.0758341565830896,
        -0.2676109895616729,
        -1.0203047093858557,
        -0.2897673462238505,
        1.950855352655958,
        -1.3290748919968962,
        -0.6377519368264665,
        0.850257024568716,
        0.2844129100094346,
    ]

    s_p = Complex{Float64}[
        5.7580258277373275-4.533830475896814im,
        -4.471642985901334+11.790406705747543im,
        -3.1685796570857536-4.127973850160156im,
        -3.5677280267248253-1.4277434015840338im,
        1.0995526935530473-3.7157339191392107im,
        0.05304893282894325-8.310443430899932im,
        14.954165680284154-4.813664506429587im,
        -1.0940288937042102+1.525825504520944im,
        -1.008339359846099+0.3508156442116616im,
        0.07785841864101696-0.09487678203891686im,
        0.04803901724013503-0.35755534225201563im,
        -0.16148337320880576+0.09850690772683601im,
        -0.02859806047287574-0.3486320723561723im,
        -0.5931267530551303-0.49056820640225385im,
        -0.1700776249816018-1.1761079973154294im,
    ]

    s_q = Complex{Float64}[
        0.7971076145172316+0.5372402576809401im,
        0.2071466227304207+1.259009278293226im,
        -0.3398858196752808+0.3751960862136744im,
        -1.3675207743723174-0.09914031977945449im,
        -0.9246609053707422-1.25038329814984im,
        0.391715711840247+0.2299495066551252im,
        0.29477984269056495-0.26961698034518383im,
        1.0814832524360534-0.0564333571343061im,
        0.46580859060989904+0.5449535876793804im,
        -2.0668707771828707+4.099238639488696im,
        -1.2045014977210438-6.995994517223501im,
        0.6499685548019725+0.22568940175977828im,
        -1.094860871004778-2.07591025239193im,
        0.14092679017444423-0.6261846990179475im,
        0.8690167435198312-0.5157825273396847im,
    ]
    F = let
        @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5]

       #tangential conics
        f = a[1] * x[1]^2 + a[2] * x[1] * x[2] + a[3] * x[2]^2 + a[4] * x[1] +
            a[5] * x[2] + 1
        ∇ = differentiate(f, x)
       #5 conics
        g = c[1] * x[1]^2 + c[2] * x[1] * x[2] + c[3] * x[2]^2 + c[4] * x[1] +
            c[5] * x[2] + c[6]
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
        F = vcat(map(i -> Incidence(f, a, g, v[:, i], y[:, i]), 1:5)...)
        System(F, [a; vec(y)], vec(v))
    end

    tracker = Tracker(
        HC2.ParameterHomotopy(F, p, q),
        a = 0.05,
        high_precision = false,
    )
    failed_res = track(tracker, s_p, 1, 0)
    @test HC2.is_terminated(failed_res.returncode)
    @test !failed_res.high_precision_used
    tracker.options.high_precision = true
    # check that we can track back and forth
    r_q = track(tracker, s_p, 1, 0)
    @test HC2.is_success(r_q)
    @test r_q.high_precision_used
    @test HC2.solution(r_q) ≈ s_q
    r_p = track(tracker, r_q, 0, 1)
    @test HC2.is_success(r_p)
    @test r_p.high_precision_used
    @test HC2.solution(r_p) ≈ s_p
end
