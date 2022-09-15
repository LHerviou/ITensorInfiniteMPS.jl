using Revise
using Distributed
@everywhere using ITensors
@everywhere using ITensorInfiniteMPS

@everywhere function generate_Pfaffian(Ly::Float64)

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


#### Test 1/3
N = 8# Number of sites in the unit cell
p = 2
q = 4
conserve_momentum = true
momentum_shift = 2
dim_dmrg = 8


fermionic_space = [fermions_space_shift(x; p=p, q=q, conserve_momentum = conserve_momentum, momentum_shift = momentum_shift) for x in 1:N]
s = infsiteinds("Fermion", N; space=fermionic_space, translater = fermion_momentum_translater);
function initstate(n)
  if 1<=mod(n, 4) <= 2
    return 2
  else
    return 1
  end
end
ψ = InfMPS(s, initstate);

#ψ1 = InfMPS(s, n->[1, 2, 2, 1, 1, 1][n]);

ψ = InfMPS(s, initstate);

model = Model("generalized_fqhe");
model_kwargs = (Ly = 11., Vs_3b = 1.0)
Hmpo = InfiniteMPOMatrix(model, s, fermion_momentum_translater; model_kwargs...);

dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg);
#Pattern initialization?
temp = iDMRGStructure(ψ, Hmpo, 2);
for j in 1:length(temp.R)
  temp.R[j]= translatecell(ITensorInfiniteMPS.translater(temp), temp.R[j], 1)
end
j = N+1
while length(commoninds(temp.R[1], dmrgStruc.R[1]))== 0
  ITensorInfiniteMPS.apply_mpomatrix_right!(temp.R, temp.Hmpo[j], temp.ψ.AR[j])
  j-=1
  println(j)
end
advance_environments(dmrgStruc)
advance_environments(dmrgStruc)
advance_environments(dmrgStruc)

dmrgStruc.dmrg_sites = N
idmrg(dmrgStruc, nb_iterations = 10, maxdim = 40, output_level = 1, build_local_H = true)
dmrgStruc.dmrg_sites = 2
idmrg(dmrgStruc, nb_iterations = 30, maxdim = 40, output_level = 1, build_local_H = false)
ψ_15 = copy(dmrgStruc.ψ);
save(string("iDMRG_WF_Pfaffian_dim-40_Ly-", Ly, ".jld2"), "ψ", ψ_15);
dmrgStruc.dmrg_sites = N
idmrg(dmrgStruc, nb_iterations = 10, maxdim = 60, output_level = 1, build_local_H = true)
dmrgStruc.dmrg_sites = 2
idmrg(dmrgStruc, nb_iterations = 30, maxdim = 60, output_level = 1, build_local_H = false)
ψ_15 = copy(dmrgStruc.ψ);
save(string("iDMRG_WF_Pfaffian_dim-60_Ly-", Ly, ".jld2"), "ψ", ψ_15)
dmrgStruc.dmrg_sites = N
idmrg(dmrgStruc, nb_iterations = 10, maxdim = 80, output_level = 1, build_local_H = true)
dmrgStruc.dmrg_sites = 2
idmrg(dmrgStruc, nb_iterations = 30, maxdim = 80, output_level = 1, build_local_H = false)
ψ_15 = copy(dmrgStruc.ψ);
save(string("iDMRG_WF_Pfaffian_dim-80_Ly-", Ly, ".jld2"), "ψ", ψ_15)
end


Lys = vcat(11.1:0.1:16)

pmap(generate_Pfaffian, Lys)
