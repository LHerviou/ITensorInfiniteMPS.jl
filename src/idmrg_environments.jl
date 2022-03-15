mutable struct iDMRGStructure
  ψ::InfiniteCanonicalMPS
  Hmpo::InfiniteMPOMatrix
  L::Vector{ITensor}
  R::Vector{ITensor}
  counter::Int64
  dmrg_sites::Int64
end
translater(IDM::iDMRGStructure) = translater(IDM.ψ)
nsites(IDM::iDMRGStructure) = nsites(IDM.ψ)
dmrg_sites(IDM::iDMRGStructure) = IDM.dmrg_sites
Base.copy(iDM::iDMRGStructure) = iDMRGStructure(copy(iDM.ψ), iDM.Hmpo, copy(iDM.L), copy(iDM.R), iDM.counter, iDM.dmrg_sites)


function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, dmrg_sites::Int64)
  N = dmrg_sites
  l = only(commoninds(ψ.AL[0], ψ.AL[1]))
  r = only(commoninds(ψ.AR[N+1], ψ.AR[N]))
  L = [ITensor(l, prime(dag(l))) for j in 1:size(Hmpo[1])[1]]
  for j = 2:length(L)-1
    for k in 1:j
      temp_l = filterinds(Hmpo[1][j, k], tags = "Link")
      for ind in temp_l
        if ind.dir == ITensors.In
          L[j] = ITensor(l, prime(dag(l)), dag(ind))
        end
      end
    end
  end
  R = [ITensor(r, prime(dag(r))) for j in 1:size(Hmpo[N])[2]]
  for j = 2:length(R)-1
    for k in j:length(R)
      temp_r = filterinds(Hmpo[N][k, j], tags = "Link")
      for ind in temp_r
        if ind.dir == ITensors.Out
          R[j] = ITensor(r, prime(dag(r)), dag(ind))
        end
      end
    end
  end
  L[end] = δ(l, prime(dag(l))); R[1] = δ(r, prime(dag(r)));
  return iDMRGStructure(copy(ψ), Hmpo, L, R, 1, dmrg_sites);
end

iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix) = iDMRGStructure(ψ, Hmpo, 2)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}, dmrg_sites::Int64) = iDMRGStructure(copy(ψ), Hmpo, L, R, 1, dmrg_sites)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}) = iDMRGStructure(copy(ψ), Hmpo, L, R, 1, 2)
iDMRGStructure(Hmpo::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS) = iDMRGStructure(ψ, Hmpo)



function apply_mpomatrix_left!(L::Vector{ITensor}, Hmpo::Matrix{ITensor})
  init = [false for j = 1:length(L)]
  for j = 1:length(L)
    for k = 1:j
      if isempty(L[j]) || isempty(Hmpo[j, k])
        continue
      end
      if !init[k] || isempty(L[k]) || k == j
        L[k] = L[j] * Hmpo[j, k]
        init[k] = true
      else
        L[k] +=  L[j] * Hmpo[j, k]
      end
    end
  end
  for j in 1:length(L)
   if !init[j]
      L[j] = ITensor(inds(L[j])...) * Hmpo[j, j]
    end
  end
end


function apply_mpomatrix_left!(L::Vector{ITensor}, Hmpo::Matrix{ITensor}, ψ::ITensor)
  ψp = dag(ψ)'
  init = [false for j = 1:length(L)]
  for j = 1:length(L)
    for k = 1:j
      if isempty(L[j]) || isempty(Hmpo[j, k])
        continue
      end
      if !init[k] || isempty(L[k]) || k == j
        L[k] = L[j] * Hmpo[j, k] * ψ * ψp
        init[k] = true
      else
        L[k] +=  L[j] * Hmpo[j, k]  * ψ * ψp
      end
    end
  end
  for j in 1:length(L)
   if !init[j]
      L[j] = ITensor(inds(L[j])...) * Hmpo[j, j]  * ψ * ψp
    end
  end
end


function build_left_mixer(L::Vector{ITensor}, Hmpo::Matrix{ITensor}, ψ::ITensor; maxdim = 20, cutoff = 1e-8)
  res = noprime(L[1] * Hmpo[1, 1] * ψ)
  for j in 2:length(L)
    temp = noprime(L[j] * Hmpo[j, j-1] * ψ)
    if order(temp) > 3
      l = combiner(only(uniqueinds(ψ, L[j], Hmpo[j, j-1])), only(uniqueinds(temp, ψ, L[j])), dir = ITensors.In)
      temp*=l
      res, (new_ind, )= ITensors.directsum(res, temp,
       uniqueinds(res, temp) ,
        uniqueinds(temp, res ); tags = [string(tags(only(uniqueinds(ψ, L[j], Hmpo[j, j-1]))))[2:end-1]])
        l = combiner(only(uniqueinds(res, ψ)))
        res*=l
      if maxdim < dim(new_ind)
        res, _, _ = svd(res, commoninds(ψ, res), left_tags = tags(only(uniqueinds(ψ, res)) ), maxdim = maxdim, cutoff = cutoff)
      end
    else
      res += temp
    end
  end
  return res
end

function apply_mpomatrix_right!(R::Vector{ITensor}, Hmpo::Matrix{ITensor})
  init = [false for j = 1:length(R)]
  for j = reverse(1:length(R))
    for k = reverse(j:length(R))
      if isempty(R[j]) || isempty(Hmpo[k, j])
        continue
      end
      if !init[k] || isempty(R[k]) || k == j
        R[k] = Hmpo[k, j] * R[j]
        init[k] = true
      else
        R[k] +=  Hmpo[k, j] * R[j]
      end
    end
  end
  for j in 1:length(R)
    if !init[j]
      R[j] =  Hmpo[j, j] * ITensor(inds(R[j])...)
    end
  end
end

function apply_mpomatrix_right!(R::Vector{ITensor}, Hmpo::Matrix{ITensor}, ψ::ITensor)
  ψp = dag(ψ)'
  init = [false for j = 1:length(R)]
  for j = reverse(1:length(R))
    for k = reverse(j:length(R))
      if isempty(R[j]) || isempty(Hmpo[k, j])
        continue
      end
      if !init[k] || isempty(R[k]) || k == j
        R[k] = Hmpo[k, j] * ψ * ψp * R[j]
        init[k] = true
      else
        R[k] +=  Hmpo[k, j] * ψ * ψp * R[j]
      end
    end
  end
  for j in 1:length(R)
    if !init[j]
      R[j] =  Hmpo[j, j] * ψ * ψp * ITensor(inds(R[j])...)
    end
  end
end

function (H::iDMRGStructure)(x)
  n = order(x) - 2
  L = H.L
  R = H.R
  start = mod1(H.counter, nsites(H))
  L = [L[j] * x for j in 1:length(L)]
  for j in 0:n-1
    apply_mpomatrix_left!(L, H.Hmpo[start+j])
  end
  result = L[1]*R[1]
  for j = 2:length(L)
    result+=L[j]*R[j]
  end
  return noprime(result)
end

function advance_environments(H::iDMRGStructure)
  N = nsites(H)
  nb_site = dmrg_sites(H)
  start = mod1(H.counter, N)
  for j in 0:N-1
    apply_mpomatrix_left!(H.L, H.Hmpo[start+j], H.ψ.AL[start + j])
  end
  for j in 0:N-1
    apply_mpomatrix_right!(H.R, H.Hmpo[start+nb_site-1-j], H.ψ.AR[start+nb_site-1-j])
  end
  for j in 1:length(H.R)
    H.R[j]= translatecell(translater(H), H.R[j], 1)
  end
  for j in 1:length(H.R)
    H.L[j]= translatecell(translater(H), H.L[j], -1)
  end
end


# function idmrg_step(iDM::iDMRGStructure; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10)
#   N = nsites(iDM)
#   nb_site = dmrg_sites(iDM)
#   if nb_site > N
#     error("iDMRG with a step size larger than the unit cell has not been implemented")
#   end
#   start = mod1(iDM.counter, N)
#   starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start+1]
#   for j = 3:nb_site
#     starting_state *= iDM.ψ.AR[start+j-1]
#   end
#   local_ener, new_x = eigsolve(iDM, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
#   U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start+1]))),
#   righttags = tags(only(commoninds(iDM.ψ.AR[start+1], iDM.ψ.AR[start]))))
#   err = 1 - norm(S2)
#   S2 = S2 / norm(S2)
#   iDM.ψ.AL[start] = U2
#   iDM.ψ.AR[start] = diag_ortho_polar(U2 * S2, iDM.ψ.C[start-1])
#   iDM.ψ.C[start] = S2
#   for j in 2:nb_site - 1
#     new_x = S2 * V2
#     linktags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j])))
#     U2, S2, V2 = svd(new_x, (only(commoninds(new_x, iDM.ψ.AL[start+j-1])), only(commoninds(new_x, iDM.ψ.AL[start+j-2])));
#       maxdim=maxdim, cutoff=cutoff, lefttags = linktags, righttags = linktags)
#     err += 1 - norm(S2)
#     S2 = S2 / norm(S2)
#     iDM.ψ.AL[start+j-1] = U2
#     iDM.ψ.AR[start+j-1] = diag_ortho_polar(U2 * S2, iDM.ψ.C[start+j-2])
#     iDM.ψ.C[start+j-1] = S2
#   end
#   iDM.ψ.AR[start+nb_site-1] = V2
#   iDM.ψ.AL[start+nb_site-1] = diag_ortho_polar(S2 * V2, iDM.ψ.C[start+nb_site-1])
#   apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start], iDM.ψ.AL[start])
#   iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
#   for j in reverse(nb_site-N+2:nb_site)
#     apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
#     iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
#   end
#   if start != N
#     for j in 1:length(iDM.R)
#       #replaceinds!(iDM.R[j], inds(iDM.R[j]), translatecell(translater(iDM), inds(iDM.R[j]), 1))
#       iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
#     end
#   else
#     for j in 1:length(iDM.R)
#       #replaceinds!(iDM.L[j], inds(iDM.L[j]), translatecell(translater(iDM), inds(iDM.L[j]), -1))
#       iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
#     end
#   end
#   iDM.counter+=1
#   return local_ener[1]/N, err
# end


function idmrg_step(iDM::iDMRGStructure; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site != N
    error("iDMRG with a step size different than the unit cell has not been implemented")
  end
  start = mod1(iDM.counter, N)
  starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start+1]
  for j = 3:nb_site
    starting_state *= iDM.ψ.AR[start+j-1]
  end
  local_ener, new_x = eigsolve(iDM, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
  U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start+1]))),
  righttags = tags(only(commoninds(iDM.ψ.AR[start+1], iDM.ψ.AR[start]))))
  err = 1 - norm(S2)
  S2 = S2 / norm(S2)
  iDM.ψ.AL[start] = U2
  iDM.ψ.AR[start] = diag_ortho_polar(U2 * S2, iDM.ψ.C[start-1])
  iDM.ψ.C[start] = S2
  for j in 2:nb_site - 1
    new_x = S2 * V2
    linktags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j])))
    U2, S2, V2 = svd(new_x, (only(commoninds(new_x, iDM.ψ.AL[start+j-1])), only(commoninds(new_x, iDM.ψ.AL[start+j-2])));
      maxdim=maxdim, cutoff=cutoff, lefttags = linktags, righttags = linktags)
    err += 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start+j-1] = U2
    iDM.ψ.AR[start+j-1] = diag_ortho_polar(U2 * S2, iDM.ψ.C[start+j-2])
    iDM.ψ.C[start+j-1] = S2
  end
  iDM.ψ.AR[start+nb_site-1] = V2
  iDM.ψ.AL[start+nb_site-1] = diag_ortho_polar(S2 * V2, iDM.ψ.C[start+nb_site-1])
  for j in 1:nb_site÷2
    apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
    iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
  end
  for j in reverse(nb_site÷2+1:nb_site)
    apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
    iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if start == 1
    for j in 1:length(iDM.R)
      iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
    end
  else
    for j in 1:length(iDM.R)
      iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
    end
  end
  iDM.counter+=nb_site÷2
  return local_ener[1]/N, err
end



function idmrg(iDM::iDMRGStructure; nb_iterations = 10, output_level = 0, mixer = false, α = 0.001, kwargs...)
  ener = 0; err = 0
  for j in 1:nb_iterations
    if !mixer
      ener, err = idmrg_step(iDM; kwargs...)
    else
      ener, err = idmrg_step_with_mixer(iDM; α = α, kwargs...)
    end
    if output_level == 1
      println("Energy after iteration $j is $ener")
    end
  end
  return ener, err
end


function test_validity_imps(ψ::InfiniteCanonicalMPS; prec = 1e-8)
  for j = 1:nsites(ψ)
    @assert norm(ψ.AL[j]*ψ.C[j] - ψ.C[j-1]*ψ.AR[j])<prec
  end
end

struct effectiveHam
  H::ITensor
end

function (H::effectiveHam)(x)
  return noprime(H.H*x)
end



function build_two_local_hamiltonian(Hl, Hr, L, R)
  tempL = copy(L)
  apply_mpomatrix_left!(tempL, Hl)
  apply_mpomatrix_left!(tempL, Hr)
  temp = tempL[1] *  R[1]
  for j in 2:length(tempL)
    temp += tempL[j] * R[j]
  end
  return effectiveHam(temp)
end


function idmrg_step_with_mixer(iDM::iDMRGStructure; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10, α = 0.1)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site != N
    error("iDMRG with a step size different than the unit cell has not been implemented")
  end
  start = mod1(iDM.counter, N)
  effective_Rs=[iDM.R for j in 1:N-1]
  local_ener=0
  err = 0
  for j in reverse(1:N-2)
    effective_Rs[j] = copy(effective_Rs[j+1])
    apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[start+j+1], iDM.ψ.AR[start+j+1])
  end
  effective_L = copy(iDM.L)
  for j = 1:N-1
    starting_state = iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j]
    temp_H = build_two_local_hamiltonian(iDM.Hmpo[start+j-1], iDM.Hmpo[start+j], effective_L, effective_Rs[j])
    local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
    #Cheap mixer (in code, but cannot solve our issue)
    U2, S2, V2 = svd(new_x[1]+α*temp_H(new_x[1]), commoninds(new_x[1], iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
    righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
    err += 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start+j-1] = U2
    iDM.ψ.AR[start+j-1] = diag_ortho_polar(U2 * S2, iDM.ψ.C[start+j-2])
    iDM.ψ.C[start+j-1] = S2
    iDM.ψ.AR[start+j] = V2
    iDM.ψ.AL[start+j] = diag_ortho_polar(S2 * V2, iDM.ψ.C[start+j])
    apply_mpomatrix_left!(effective_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
  end
  for j in 1:nb_site÷2
    apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
    iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
  end
  for j in reverse(nb_site÷2+1:nb_site)
    apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
    iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if start == 1
    for j in 1:length(iDM.R)
      iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
    end
  else
    for j in 1:length(iDM.R)
      iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
    end
  end
  iDM.counter+=nb_site÷2
  return local_ener[1]/N, err
end



function idmrg_step_with_mixer(iDM::iDMRGStructure; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10, α = 0.1)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site != N
    error("iDMRG with a step size different than the unit cell has not been implemented")
  end
  start = mod1(iDM.counter, N)
  effective_Rs=[iDM.R for j in 1:N-1]
  local_ener=0
  err = 0
  for j in reverse(1:N-2)
    effective_Rs[j] = copy(effective_Rs[j+1])
    apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[start+j+1], iDM.ψ.AR[start+j+1])
  end
  effective_L = copy(iDM.L)
  for j = 1:N-1
    starting_state = iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j]
    temp_H = build_two_local_hamiltonian(iDM.Hmpo[start+j-1], iDM.Hmpo[start+j], effective_L, effective_Rs[j])
    local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
    #Cheap mixer (in code, but cannot solve our issue)
    U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
    righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
    err += 1 - norm(S2)
    S2 = S2 / norm(S2)
    if α != 0
      temp_mix = build_left_mixer(effective_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
      temp_U, (new_right,) = ITensors.directsum(U2, α*temp_mix,
      commoninds(U2, S2),
      uniqueinds(temp_mix, iDM.ψ.AL[start+j-1]);
      tags = [string(tags(only(commoninds(U2, S2))))[2:end-1]]
       )
      Up, Sp, Vp =  svd(temp_U, commoninds(temp_U, iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
          righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
      Sp = Sp / norm(Sp)
      #Vp *= wδ(dag(only(uniqueinds(Vp, Sp))), only(commoninds(Up, Sp)))
      #l = combiner(only(uniqueinds(Vp, Sp)), tags = tags(only(uniqueinds(Vp, Sp))))
      #iDM.ψ.AL[start+j-1] =  Up * δ(inds(Sp)) * Vp * l #U2
      #iDM.ψ.C[start+j-1] =  (dag(l) * dag(Vp) * δ(dag(inds(Sp))))*Sp * (Vp * wδ(dag(only(uniqueinds(Vp, Sp))), dag(only(uniqueinds(S2, V2))))) * S2#S2
      iDM.ψ.AL[start+j-1] =  Up
      #iDM.ψ.AL[start+j-1] =  (Up * wδ(inds(Sp)...) * Vp) *  wδ(dag(only(uniqueinds(Vp, Sp))), only(commoninds(Up, Sp)))
      iDM.ψ.C[start+j-1] =  sqrt.(Sp) * (Vp * wδ(dag(only(uniqueinds(Vp, Sp))), dag(only(uniqueinds(S2, V2))))) * S2
      iDM.ψ.C[start+j-1] = iDM.ψ.C[start+j-1] / norm(iDM.ψ.C[start+j-1])
      #iDM.ψ.C[start+j-1] =  δ(only(commoninds(Sp, Up)), only(commoninds(Vp, Sp))) * wδ(dag(only(uniqueinds(Vp, Sp))), dag(only(uniqueinds(S2, V2)))) * S2
      #iDM.ψ.C[start+j-1] =   (wδ(only(uniqueinds(Vp, Sp)), only(commoninds(Sp, Up))) * ( dag(Vp) * wδ(only(commoninds(Vp, Sp)), only(commoninds(Up, Sp))) * Sp ) ) * Vp *wδ(dag(only(uniqueinds(Vp, Sp))), only(commoninds(U2, S2))) * S2#* (Sp * Vp * S2))

      iDM.ψ.AR[start+j-1] = ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
      #iDM.ψ.AR[start+j-1], iDM.ψ.C[start+j-2] = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
      iDM.ψ.AR[start+j] = V2
      iDM.ψ.AL[start+j] = ortho_polar(iDM.ψ.C[start+j-1] * V2, iDM.ψ.C[start+j])
      #iDM.ψ.AL[start+j], iDM.ψ.C[start+j] = diag_ortho_polar(iDM.ψ.C[start+j-1] * V2, iDM.ψ.C[start+j])
    else
      iDM.ψ.AL[start+j-1] =  U2
      iDM.ψ.C[start+j-1] =  S2
      iDM.ψ.AR[start+j] = V2
      #iDM.ψ.AR[start+j-1] = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
      iDM.ψ.AR[start+j-1], iDM.ψ.C[start+j-2] = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
      #println(norm(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1] - A * iDM.ψ.AR[start+j-1]))
      #println(inds(iDM.ψ.AR[start+j-1]))
      #println(inds(A))
      #println(inds(iDM.ψ.C[start+j-2]))
      iDM.ψ.AL[start+j], iDM.ψ.C[start+j] = diag_ortho_polar(iDM.ψ.C[start+j-1] * V2, iDM.ψ.C[start+j])
    end
    println(start+j-1)
    evaluate_unitarity(iDM.ψ)
    if j != N-1
      apply_mpomatrix_left!(effective_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
    end
  end
  for j in 1:nb_site÷2
    apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
    iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
  end
  for j in reverse(nb_site÷2+1:nb_site)
    apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
    iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if start == 1
    for j in 1:length(iDM.R)
      iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
    end
  else
    for j in 1:length(iDM.R)
      iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
    end
  end
  iDM.counter+=nb_site÷2
  return local_ener[1]/N, err
end




function evaluate_unitarity(ψ::InfiniteCanonicalMPS)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  println("Doing left")
  for j in 1:nsites(ψ)
    println(norm( δ(l[j-1], prime(dag(l[j-1]))) * ψ.AL[j] * δ(dag(s[j]), prime(s[j]))* dag(prime(ψ.AL[j])) - denseblocks(δ(l[j], prime(dag(l[j])))) ))
  end

  println("Doing right")
  for j in 1:nsites(ψ)
    println(norm( δ(dag(r[j]), prime(r[j])) * ψ.AR[j] * δ(dag(s[j]), prime(s[j]))* dag(prime(ψ.AR[j])) - denseblocks(δ(dag(r[j-1]), prime(r[j-1]))) ))
  end
end
