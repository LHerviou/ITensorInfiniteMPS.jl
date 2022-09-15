using ITensors
using ITensorInfiniteMPS



function fermions_space_shift(pos; p = 1, q = 1, conserve_momentum = true, momentum_shift = 1)
  if !conserve_momentum
    return [QN("Nf", - p ) => 1, QN("Nf", q - p) => 1];
  else
    return [QN(("Nf", - p ), ("NfMom", -p*pos + momentum_shift)) => 1, QN(("Nf", q - p), ("NfMom", (q - p)*pos + momentum_shift)) => 1];
  end
end


function fermion_momentum_translater(i::Index, n::Integer; N = N)
  ts = tags(i)
  translated_ts = translatecell(ts, n)
  new_i = replacetags(i, ts => translated_ts)
  for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch ), ("NfMom", mom + n*N*ch)),  new_i.space[j][2])
  end
  return new_i
end

function initstate(n)
  if mod(n, 3) == 1
    return 2
  else
    return 1
  end
end
##############################################################################
# VUMPS parameters
#

maxdim = 64 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 100 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-8
outer_iters = 4 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#
N = 6# Number of sites in the unit cell
p = 1
q = 3
conserve_momentum = true
momentum_shift = 1

fermionic_space = [fermions_space_shift(x; p=p, q=q, conserve_momentum = conserve_momentum, momentum_shift = momentum_shift) for x in 1:N]
s = infsiteinds("Fermion", N; space=fermionic_space, translater = fermion_momentum_translater);
ψ = InfMPS(s, initstate);

model = Model("fqhe_2b_pot");
Hmpo = InfiniteMPOMatrix(model, s, fermion_momentum_translater; Ly = 6., Vs = [1., 0], prec = 1e-6)


vumps_kwargs = (
multisite_update_alg="sequential",
tol=vumps_tol,
maxiter=max_vumps_iters,
outputlevel=1,
time_step=-Inf,
)
subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

# Alternate steps of running VUMPS and increasing the bond dimension
for _ in 1:outer_iters
  ψ = subspace_expansion(ψ, Hmpo; expansion_space = 6, subspace_expansion_kwargs...)
  ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
end
