using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "idmrg_mpomatrix" begin
  Random.seed!(1234)

  model = Model"ising"()
  model_kwargs = (J=-1.0, h=1.)

  function space_shifted(::Model"ising", q̃sz; conserve_qns=true)
    if conserve_qns
      return [QN("SzParity", 1 - q̃sz, 2) => 1, QN("SzParity", 0 - q̃sz, 2) => 1]
    else
      return [QN() => 2]
    end
  end

  function expect_prod(ψ::InfiniteCanonicalMPS, op1::String, op2::String, n::Int64)
    s = siteinds(only, ψ)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    ψp = dag(ψ)'
    S1 = op(op1, s[n]); S2 = op(op2, s[n+1])

    return tr(δ(l[n-1], prime(dag(l[n-1])))*ψ.AL[n]*S1*ψp.AL[n]*ψ.AL[n+1]*S2*ψp.AL[n+1]*ψ.C[n+1]*ψp.C[n+1])
  end

  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3

  initstate(n) = "↑"

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
  Hfinite = MPO(model, sfinite; model_kwargs...)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 10)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
  Szs_finite = expect(ψfinite, "Sz")

  function energy(ψ, h, n)
    ϕ = ψ[n] * ψ[n + 1] * ψ[n + 2]
    return (noprime(ϕ * h) * dag(ϕ))[]
  end

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
  orthogonalize!(ψfinite, nfinite)
  energy_finite = energy(ψfinite, hnfinite, nfinite)


  for nsite in 2:2:4
    dim_dmrg = nsite
    conserve_qns = true

    space_ = fill(space_shifted(model, 0; conserve_qns=conserve_qns), nsite)

    s = infsiteinds("S=1/2", nsite; space=space_)
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)

    dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg)
    advance_environments(dmrgStruc)
    advance_environments(dmrgStruc)
    inf_ener, _ = idmrg(dmrgStruc, nb_iterations = 100, maxdim = 40)
    Szs_infinite = [expect(dmrgStruc.ψ, "Sz", n) for n in 1:nsite]

    @test energy_finite ≈ inf_ener rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsite - 1)] ≈ Szs_infinite rtol = 1e-2
  end

  for dim_dmrg in 2:3
    nsite = 4
    conserve_qns = true

    space_ = fill(space_shifted(model, 0; conserve_qns=conserve_qns), nsite)

    s = infsiteinds("S=1/2", nsite; space=space_)
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)

    dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg)
    advance_environments(dmrgStruc)
    advance_environments(dmrgStruc)
    inf_ener, _ = idmrg(dmrgStruc, nb_iterations = 100, maxdim = 40)
    Szs_infinite = [expect(dmrgStruc.ψ, "Sz", n) for n in 1:nsite]

    @test energy_finite ≈ inf_ener rtol = 1e-4
    @test Szs_finite[nfinite:(nfinite + nsite - 1)] ≈ Szs_infinite rtol = 1e-2
  end
end



@testset "idmrg_mpomatrix_Heisenberg" begin
  Random.seed!(1234)

  model = Model"heisenberg"()
  function space_shifted_heis(::Model"heisenberg"; p = 1, q = 1, conserve_qns= true)
    if conserve_qns
      return [QN("Sz", q-p ) => 1, QN("Sz", -p) => 1];
    else
      return [QN() => 2]
    end
  end

  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3


  initstate(n) = isodd(n) ? "↑" : "↓"
  p = 1; q = 2;

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_sz=true)
  Hfinite = MPO(model, sfinite)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 40)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
  Szs_finite = expect(ψfinite, "Sz")

  function energy(ψ, h, n)
    ϕ = ψ[n] * ψ[n + 1]
    return (noprime(ϕ * h) * dag(ϕ))[]
  end

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(model, [sfinite[nfinite], sfinite[nfinite + 1]])
  orthogonalize!(ψfinite, nfinite)
  energy_finite = energy(ψfinite, hnfinite, nfinite)
  hnfinite = ITensor(model, [sfinite[nfinite+1], sfinite[nfinite + 2]])
  orthogonalize!(ψfinite, nfinite+1)
  energy_finite_2 = energy(ψfinite, hnfinite, nfinite+1)
  energy_finite = (energy_finite + energy_finite_2 )/2


  for nsite in 2:2:4
    dim_dmrg = nsite
    conserve_qns = true

    space_ = fill(space_shifted_heis(model; p = p, q=q, conserve_qns=conserve_qns), nsite)

    s = infsiteinds("S=1/2", nsite; space=space_)
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)

    dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg)
    advance_environments(dmrgStruc)
    advance_environments(dmrgStruc)
    inf_ener, _ = idmrg(dmrgStruc, nb_iterations = 100, maxdim = 40, output_level = 1)
    Szs_infinite = [expect(dmrgStruc.ψ, "Sz", n) for n in 1:nsite]

    @test energy_finite ≈ inf_ener rtol = 1e-4
    @test sum(Szs_finite[nfinite:(nfinite + nsite - 1)]) ≈ sum(Szs_infinite) atol = 1e-3
  end
end
