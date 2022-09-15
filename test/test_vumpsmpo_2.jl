using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "vumpsmpo" begin
  Random.seed!(1234)

  model = Model"heisenberg"()

  function space_shifted(::Model"heisenberg"; p=1, q=1, conserve_qns=true)
    if conserve_qns
      return [QN("Sz", q - p) => 1, QN("Sz", -p) => 1]
    else
      return [QN() => 2]
    end
  end

  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3

  p = 2
  q = 3
  initstate(n) = mod(n, 3) == 0 ? "↓" : "↑"

  # DMRG arguments
  Nfinite = 100
  sfinite = siteinds("S=1/2", Nfinite; conserve_sz=true)
  Hfinite = MPO(model, sfinite)
  ψfinite = randomMPS(sfinite, initstate)
  sweeps = Sweeps(20)
  setmaxdim!(sweeps, 10)
  setcutoff!(sweeps, 1E-10)
  energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
  Szs_finite = expect(ψfinite, "Sz")

  function energy(ψ, h, n)
    ϕ = ψ[n] * ψ[n + 1]
    return (noprime(ϕ * h) * dag(ϕ))[]
  end

  nfinite = Nfinite ÷ 2
  hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1])
  orthogonalize!(ψfinite, nfinite)
  energy_finite = energy(ψfinite, hnfinite, nfinite)
  #
  # for multisite_update_alg in ["sequential", "parallel"],
  #   conserve_qns in [true, false],
  #   nsite in [1, 2],
  #   time_step in [-Inf],
  #   expansion_space in [2]

  multisite_update_alg = "sequential"
  conserve_qns = true
  nsite = 6
  time_step = -Inf
  expansion_space = 2

  vumps_kwargs = (
    multisite_update_alg=multisite_update_alg,
    tol=tol,
    maxiter=maxiter,
    outputlevel=1,
    time_step=time_step,
  )
  subspace_expansion_kwargs = (
    cutoff=cutoff, maxdim=maxdim, expansion_space=expansion_space
  )

  space_ = fill(space_shifted(model; conserve_qns=conserve_qns, p=p, q=q), nsite)
  s = infsiteinds("S=1/2", nsite; space=space_)
  ψ = InfMPS(s, initstate)

  Hmpo = InfiniteMPOMatrix(model, s)
  # Alternate steps of running VUMPS and increasing the bond dimension
  ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
  for _ in 1:outer_iters
    ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
  end

  @test norm(contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...)) ≈
        0 atol = 1e-5

  H = InfiniteITensorSum(model, s)
  energy_infinite = expect(ψ, H)
  Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsite]

  @test energy_finite ≈ sum(energy_infinite) / nsite rtol = 1e-4
  @test Szs_finite[nfinite:(nfinite + nsite - 1)] ≈ Szs_infinite rtol = 1e-3
  #end
end
