using ITensors
using ITensorInfiniteMPS
using Test
using Random

@testset "idmrgmpo" begin
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

  nsite = 4
  dim_dmrg = 4
  conserve_qns = true
  initstate(n) = mod(n, 2) == 0 ? "↑" : "↓"
  #initstate(n) = mod(n, 2) == 0 ? "↑" : "↓"

  space_ = fill(space_shifted(model, 0; conserve_qns=conserve_qns), nsite)
  #space_ = [space_shifted(model, x; conserve_qns=conserve_qns) for x in 1:nsite]

  s = infsiteinds("S=1/2", nsite; space=space_)
  ψ = InfMPS(s, initstate)

  Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)

  dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg)
  advance_environments(dmrgStruc)
  inf_ener, _ = idmrg(dmrgStruc)

  dmrgStruc2 = iDMRGStructure(ψ, Hmpo, dim_dmrg)
  advance_environments(dmrgStruc2)
  inf_ener2, _ = idmrg(dmrgStruc2, mixer = true, α = 0.1)
  inf_ener2, _ = idmrg(dmrgStruc2, mixer = true, α = 0.01)
  inf_ener2, _ = idmrg(dmrgStruc2, mixer = true, α = 0.001)
  inf_ener2, _ = idmrg(dmrgStruc2, mixer = true, α = 0)





  model = Model"heisenberg"()
  function space_shifted_heis(::Model"heisenberg"; p = 1, q = 1, conserve_qns= true)
    if conserve_qns
      return [QN("Sz", q-p ) => 1, QN("Sz", -p) => 1];
    else
      return [QN() => 2]
    end
  end

  nsite = 4
  dim_dmrg = 4
  conserve_qns = true
  initstate(n) = mod(n, 2) == 0 ? "↑" : "↓"
  p = 1; q = 2;

  space_ = fill(space_shifted_heis(model; p=p, q=q, conserve_qns=conserve_qns), nsite)
  #space_ = [space_shifted(model, x; conserve_qns=conserve_qns) for x in 1:nsite]

  s = infsiteinds("S=1/2", nsite; space=space_)
  ψ = InfMPS(s, initstate)

  Hmpo = InfiniteMPOMatrix(model, s)

  dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg)
  advance_environments(dmrgStruc)
  ener_idmrg, _ = idmrg(dmrgStruc)


  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 20
  tol = 1e-8
  maxiter = 20
  outer_iters = 3
  vumps_kwargs = (
    multisite_update_alg="sequential",
    tol=tol,
    maxiter=maxiter,
    outputlevel=1,
    time_step=-Inf,
  )
  subspace_expansion_kwargs = (
    cutoff=cutoff, maxdim=maxdim, expansion_space=2
  )
  ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
  for _ in 1:outer_iters
    ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
  end
  H = InfiniteITensorSum(model, s)
  energy_infinite = expect(ψ, H)
  Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsite]




end

# XXX: orthogonalize is broken right now
## @testset "ITensorInfiniteMPS.jl" begin
##   @testset "Mixed canonical gauge" begin
##     N = 10
##     s = siteinds("S=1/2", N; conserve_szparity=true)
##     χ = 6
##     @test iseven(χ)
##     space = (("SzParity", 1, 2) => χ ÷ 2) ⊕ (("SzParity", 0, 2) => χ ÷ 2)
##     ψ = InfiniteMPS(ComplexF64, s; space=space)
##     randn!.(ψ)
##
##     ψ = orthogonalize(ψ, :)
##     @test prod(ψ.AL[1:N]) * ψ.C[N] ≈ ψ.C[0] * prod(ψ.AR[1:N])
##   end
## end
