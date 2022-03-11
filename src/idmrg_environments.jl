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


struct effectiveHamiltonian_idmrg
  H::ITensor
end


function build_local_hamiltonian(iDM)

end

function idmrg_step(iDM::iDMRGStructure; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
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
  apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start], iDM.ψ.AL[start])
  iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
  for j in reverse(nb_site-N+2:nb_site)
    apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
    iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if start != N
    for j in 1:length(iDM.R)
      #replaceinds!(iDM.R[j], inds(iDM.R[j]), translatecell(translater(iDM), inds(iDM.R[j]), 1))
      iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
    end
  else
    for j in 1:length(iDM.R)
      #replaceinds!(iDM.L[j], inds(iDM.L[j]), translatecell(translater(iDM), inds(iDM.L[j]), -1))
      iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
    end
  end
  iDM.counter+=1
  return local_ener[1]/N, err
end


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



function idmrg(iDM::iDMRGStructure; nb_iterations = 10, kwargs...)
  ener = 0; err = 0
  for j in 1:nb_iterations
    ener, err = idmrg_step(iDM; kwargs...)
  end
  return ener, err
end


function test_validity_imps(ψ::InfiniteCanonicalMPS; prec = 1e-8)
  for j = 1:nsites(ψ)
    @assert norm(ψ.AL[j]*ψ.C[j] - ψ.C[j-1]*ψ.AR[j])<prec
  end
end
