using ITensors
using ITensorInfiniteMPS

##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-6
outer_iters = 4 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 4# Number of sites in the unit cell
J = 1.0
J₂ = 0.0
h = 1.0.;

function spin_space_shift(q̃nf; pop_p::Int64 = 1, pop_m::Int64 = 1)
    return [QN("Sz", 1 * pop_m) => 1, QN("Sz", -1 * pop_p) => 1];
end

heisenberg_space = fill(spin_space_shift(1, pop_p = (N+1)÷2, pop_m = N÷2), N);
s = infsiteinds("S=1/2", N; space=heisenberg_space);
initstate(n) = isodd(n) ? "↑" : "↓";
ψ = InfMPS(s, initstate);

model = Model("xx_extended");
H = InfiniteITensorSum(model, s, J=J, J₂ = J₂, h = h);

@show norm(contract(ψ.AL[1:N]..., ψ.C[N]) - contract(ψ.C[0], ψ.AR[1:N]...));

vumps_kwargs = (tol=vumps_tol, maxiter=max_vumps_iters)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)
ψ_0 = vumps(H, ψ; vumps_kwargs...)

for j in 1:outer_iters
    println("\nIncrease bond dimension")
    ψ_1 = subspace_expansion(ψ_0, H; subspace_expansion_kwargs...)
    println("Run VUMPS with new bond dimension")
    ψ_0 = vumps(H, ψ_1; vumps_kwargs...)
end


function ITensors.expect(ψ::InfiniteCanonicalMPS, o, n)
  return (noprime(ψ.AL[n] * ψ.C[n] * op(o, s[n])) * dag(ψ.AL[n] * ψ.C[n]))[]
end

function expect_two_site(ψ::InfiniteCanonicalMPS, h::ITensor, n1n2)
  n1, n2 = n1n2
  ϕ = ψ.AL[n1] * ψ.AL[n2] * ψ.C[n2]
  return (noprime(ϕ * h) * dag(ϕ))[]
end

Sz = [expect(ψ, "Sz", n) for n in 1:N]

bs = [(1, 2), (2, 3)]
energy_infinite = map(b -> expect_two_site(ψ, H[b], b), bs)



Nfinite = 100
sfinite = siteinds("S=1/2", Nfinite; conserve_sz=true)
Hfinite = MPO(model, sfinite; J=J, J₂ = 2, h = h)
ψfinite = randomMPS(sfinite, initstate; linkdims=10)
@show flux(ψfinite)
sweeps = Sweeps(15)
setmaxdim!(sweeps, maxdim)
setcutoff!(sweeps, cutoff)
energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps)
@show energy_finite_total / Nfinite

nfinite = Nfinite ÷ 2
orthogonalize!(ψfinite, nfinite)
Sz1_finite = expect(ψfinite[nfinite], "Sz")
orthogonalize!(ψfinite, nfinite + 1)
Sz2_finite = expect(ψfinite[nfinite + 1], "Sz")

@show Sz1_finite, Sz2_finite
