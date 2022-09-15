using ITensors
using ITensorInfiniteMPS

using PyPlot

#########################
rc("font", family="serif", serif="Times", size=16)
rc("axes", labelsize=16, titlesize=16)
rc("xtick", labelsize=16)
rc("ytick", labelsize=16)
rc("legend", fontsize=12, frameon=true)
rc("text",usetex="True")
colorsbis=["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"] ###color scheme for color blind people in principle
colorsbis=vcat(colorsbis, colorsbis); colorsbis=vcat(colorsbis, colorsbis)
rc("axes", prop_cycle=plt.cycler(color=colorsbis))

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

function compute_entanglement_spectrum(C::ITensor, ind::Index{Vector{Pair{QN, Int64}}}; prec = 1e-12, nb_qn = 2)
    entanglement_spectrum = Dict()
    n=1
    U, S, V = svd(C, ind)
    temp = inds(S)[1]
    for xind in temp.space
        local_ent = Float64[]
        for y in 1:xind[2]
          if S[n, n]^2 > prec
            append!(local_ent, S[n, n]^2)
          end
          n+=1
        end
        if length(local_ent)!=0
          entanglement_spectrum[( val.(xind[1])[1:min(length(xind[1]), nb_qn)] )] = local_ent
        end
    end
  return entanglement_spectrum
end


function compute_entanglement_spectrum(ψ::InfiniteCanonicalMPS; kwargs...)
    entanglement_spectrum = Dict()
    for j in 1:nsites(ψ)
      entanglement_spectrum[j] = compute_entanglement_spectrum(ψ.C[j], only(commoninds(ψ.C[j], ψ.AL[j])), kwargs...)
    end
  return entanglement_spectrum
end


function plot_orbital_entanglement_spectrum(coeff::Dict; n = -1, fig = true, c=colorsbis, marker = "_")
    if fig
        figure()
    end
    if n != -1
        for (k, v) in coeff
            if k[1]==n
                plot(k[2]*ones(length(v)), - log.(v), marker, c = c[1])
            end
        end
        p0, = plot([], [], marker, c = c[1])
        handles = ([n], [p0])
    else
        seen = Dict()
        for (k, v) in coeff
            seen[k[1]]=1
            plot(k[2]*ones(length(v)), - log.(v), marker, c=colorsbis[mod1(k[1], length(colorsbis))])
        end
        ns = sort(collect(keys(seen)))
        ps = [   plot([], [], marker, c = c=colorsbis[mod1(n, length(colorsbis))] )[1] for n in ns    ]
        handles = (ns, ps)
    end
    return handles
end

##############################################################################
# VUMPS parameters
#

maxdim = 2 # Maximum bond dimension
cutoff = 1e-6 # Singular value cutoff when increasing the bond dimension
max_vumps_iters = 4 # Maximum number of iterations of the VUMPS algorithm at each bond dimension
vumps_tol = 1e-8
outer_iters = 1 # Number of times to increase the bond dimension

##############################################################################
# CODE BELOW HERE DOES NOT NEED TO BE MODIFIED
#

#### Test 1/3
N = 6# Number of sites in the unit cell
p = 1
q = 3
conserve_momentum = true
momentum_shift = 1
dim_dmrg = 6


fermionic_space = [fermions_space_shift(x; p=p, q=q, conserve_momentum = conserve_momentum, momentum_shift = momentum_shift) for x in 1:N]
s = infsiteinds("Fermion", N; space=fermionic_space, translater = fermion_momentum_translater);
function initstate(n)
  if mod(n, 3) == 1
    return 2
  else
    return 1
  end
end
ψ = InfMPS(s, initstate);

#ψ1 = InfMPS(s, n->[1, 2, 2, 1, 1, 1][n]);

model = Model("fqhe_2b_pot");
Hmpo = InfiniteMPOMatrix(model, s, fermion_momentum_translater; Ly = 10., Vs = [1., 0], prec = 1e-6);

dmrgStruc = iDMRGStructure(ψ, Hmpo, dim_dmrg);
#Pattern initialization?
temp = iDMRGStructure(ψ, Hmpo, 1);
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
save("iDMRG_WF_Laughlin_13_dim-40_Ly-15.jld2", "ψ", ψ_15);
dmrgStruc.dmrg_sites = N
idmrg(dmrgStruc, nb_iterations = 10, maxdim = 60, output_level = 1, build_local_H = false)
dmrgStruc.dmrg_sites = 2
idmrg(dmrgStruc, nb_iterations = 30, maxdim = 60, output_level = 1, build_local_H = false)
ψ_15 = copy(dmrgStruc.ψ);
save("iDMRG_WF_Laughlin_13_dim-60_Ly-15.jld2", "ψ", ψ_15)
dmrgStruc.dmrg_sites = N
idmrg(dmrgStruc, nb_iterations = 6, maxdim = 80, output_level = 1, build_local_H = false)
dmrgStruc.dmrg_sites = 2
idmrg(dmrgStruc, nb_iterations = 30, maxdim = 80, output_level = 1, build_local_H = false)
ψ_15 = copy(dmrgStruc.ψ);
save("iDMRG_WF_Laughlin_13_dim-80_Ly-15.jld2", "ψ", ψ_15)

################################################################
################################################################
#### Test 1/2
N = 8# Number of sites in the unit cell
p = 2
q = 4
conserve_momentum = true
momentum_shift = 2
dim_dmrg = N


fermionic_space = [fermions_space_shift(x; p=p, q=q, conserve_momentum = conserve_momentum, momentum_shift = momentum_shift) for x in 1:N]
s = infsiteinds("Fermion", N; space=fermionic_space, translater = fermion_momentum_translater);
function initstate(n)
  if mod(n, 2) == 1
    return 2
  else
    return 1
  end
end

function initstate(n)
  if 1<=mod(n, 4) <= 2
    return 2
  else
    return 1
  end
end

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
idmrg(dmrgStruc, nb_iterations = 20, maxdim = 40, output_level = 1, build_local_H = false)
ψ_11 = copy(dmrgStruc.ψ)
save("iDMRG_WF_Pfaffian_dim-40_Ly-11.jld2", "ψ", ψ_11)
