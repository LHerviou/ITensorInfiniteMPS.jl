mutable struct iDMRGStructure{T}
  ψ::InfiniteCanonicalMPS
  Hmpo::T
  L::Vector{ITensor}
  R::Vector{ITensor}
  counter::Int64
  dmrg_sites::Int64
end
translater(IDM::iDMRGStructure) = translater(IDM.ψ)
nsites(IDM::iDMRGStructure) = nsites(IDM.ψ)
dmrg_sites(IDM::iDMRGStructure) = IDM.dmrg_sites
Base.copy(iDM::iDMRGStructure) = iDMRGStructure{typeof(iDM.Hmpo)}(copy(iDM.ψ), iDM.Hmpo, copy(iDM.L), copy(iDM.R), iDM.counter, iDM.dmrg_sites)


function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, dmrg_sites::Int64)
  N = nsites(ψ) #dmrg_sites
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
  return iDMRGStructure{InfiniteMPOMatrix}(copy(ψ), Hmpo, L, R, 1, dmrg_sites);
end

iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix) = iDMRGStructure{InfiniteMPOMatrix}(ψ, Hmpo, 2)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}, dmrg_sites::Int64) = iDMRGStructure{InfiniteMPOMatrix}(copy(ψ), Hmpo, L, R, 1, dmrg_sites)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}) = iDMRGStructure{InfiniteMPOMatrix}(copy(ψ), Hmpo, L, R, 1, 2)
iDMRGStructure(Hmpo::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS) = iDMRGStructure{InfiniteMPOMatrix}(ψ, Hmpo)

function (H::iDMRGStructure{InfiniteMPOMatrix})(x)
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

struct temporaryHamiltonian
  effectiveL::Vector{ITensor}
  effectiveR::Vector{ITensor}
  Hmpo::InfiniteMPOMatrix
  nref::Int64 #leftmostsite
end

function (H::temporaryHamiltonian)(x)
  n = order(x) - 2
  L = [H.effectiveL[j] * x for j in 1:length(H.effectiveL)]
  for j in 0:n-1
    apply_mpomatrix_left!(L, H.Hmpo[H.nref+j])
  end
  result = L[1]*H.effectiveR[1]
  for j = 2:length(L)
    result+=L[j]*H.effectiveR[j]
  end
  return noprime(result)
end

struct effectiveHam
  H::ITensor
end

function (H::effectiveHam)(x)
  return noprime(H.H*x)
end


struct effectiveHam_LR
  L::Vector{ITensor}
  R::Vector{ITensor}
end

function effectiveHam_LR(Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}, start::Int64, len::Int64)
  tempH = Hmpo[start]
  for j = 2:len
    @disable_warn_order tempH = tempH * Hmpo[start+j-1]
  end
  tempL = copy(L)
  @disable_warn_order  apply_mpomatrix_left!(tempL, tempH)
  return effectiveHam_LR(tempL, R)
end


function (H::effectiveHam_LR)(x)
  res = (H.L[1] * x) * H.R[1]
  for j = 2:length(H.L)
    res += (H.L[j] * x) * H.R[j]
  end
  return noprime(res)
end



function apply_mpomatrix_left!(L::Vector{ITensor}, Hmpo::Matrix{ITensor})
  init = [false for j = 1:length(L)]
  for j = 1:length(L)
    for k = 1:j
      if isempty(L[j]) || isempty(Hmpo[j, k])
        continue
      end
      if !init[k] || isempty(L[k])
        L[k] = L[j] * Hmpo[j, k]
        init[k] = true
      else
        L[k] +=  L[j] * Hmpo[j, k]
      end
    end
  end
  for j in 1:length(L)
   if !init[j]
      L[j] = ITensor(Float64, inds(L[j])..., inds(Hmpo[j, j])...)
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

function advance_environments(H::iDMRGStructure{InfiniteMPOMatrix})
  N = nsites(H)
  nb_site = N#dmrg_sites(H)
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

function idmrg_step_noupdate_sideC(iDM::iDMRGStructure{InfiniteMPOMatrix}; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10)
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


function idmrg_step(iDM::iDMRGStructure{InfiniteMPOMatrix}; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10, build_local_H = false)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site > N
    error("iDMRG with a step size different than the unit cell has not been implemented")
  end
  if nb_site == 1
    error("Single site dmrg has not been implemented")
  end
  if (N÷(nb_site÷2))*(nb_site÷2) != N
    error("We require that the (nb_site÷2) divides the unitcell length")
  end
  nbIterations =  (N - nb_site)÷(nb_site÷2) + 1#(N÷(nb_site÷2)) - 1
  original_start = mod1(iDM.counter, N)
  effective_Rs=[iDM.R for j in 1:nbIterations]
  local_ener=0
  err = 0
  site_looked = original_start + N-1
  for j in reverse(1:nbIterations-1)
      effective_Rs[j] = copy(effective_Rs[j+1])
      for k in 0:nb_site÷2-1
        apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked])
        site_looked -= 1
      end
  end
  adjust_left = 0
  adjust_right_most = 0
  start = original_start
  for count in 1:nbIterations
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start+1]
    for j = 3:nb_site
      starting_state *= iDM.ψ.AR[start+j-1]
    end
    if build_local_H
      temp_H = effectiveHam_LR(iDM.Hmpo, iDM.L, effective_Rs[count], start, nb_site)
      local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
    else
      temp_H = temporaryHamiltonian(iDM.L, effective_Rs[count], iDM.Hmpo, start)
      local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
    end
    U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start+1]))),
    righttags = tags(only(commoninds(iDM.ψ.AR[start+1], iDM.ψ.AR[start]))))
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start] = U2
    temp_R, temp_C = diag_ortho_polar_both(U2 * S2, iDM.ψ.C[start-1])
    if count == 1 #&& nbIterations > 1
      adjust_right_most = translatecell(translater(iDM), wδ(only(commoninds(iDM.ψ.AR[start], iDM.ψ.AR[start-1])), only(commoninds(temp_C, temp_R))), 1)
      for j in 1:length(iDM.R)
        effective_Rs[end][j] *= dag(adjust_right_most) #Also modify iDM.R
        effective_Rs[end][j] *= prime(adjust_right_most)
      end
    end
    iDM.ψ.AR[start - 1] *= wδ(only(commoninds(iDM.ψ.AR[start], iDM.ψ.AR[start-1])), only(commoninds(temp_C, temp_R)))
    iDM.ψ.AR[start] = temp_R
    iDM.ψ.C[start-1] = temp_C
    iDM.ψ.C[start] = S2
    for j in 2:nb_site - 1
      new_x = S2 * V2
      linktags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j])))
      U2, S2, V2 = svd(new_x, (only(commoninds(new_x, iDM.ψ.AL[start+j-1])), only(commoninds(new_x, iDM.ψ.AL[start+j-2])));
        maxdim=maxdim, cutoff=cutoff, lefttags = linktags, righttags = linktags)
      err += 1 - norm(S2)
      S2 = S2 / norm(S2)
      iDM.ψ.AL[start+j-1] = U2
      temp_R, temp_C = diag_ortho_polar_both(U2 * S2, iDM.ψ.C[start+j-2])
      iDM.ψ.AR[start+j-2] *= wδ(dag(only(uniqueinds(iDM.ψ.AR[start+j-2], iDM.ψ.AL[start+j-2], iDM.ψ.AR[start+j-3]))), only(commoninds(temp_C, temp_R)))
      iDM.ψ.AR[start+j-1] = temp_R
      iDM.ψ.C[start+j-2] = temp_C
      iDM.ψ.C[start+j-1] = S2
    end
    if count != nbIterations
      iDM.ψ.AR[start+nb_site-1] = V2
      temp_R, temp_C = diag_ortho_polar_both(S2 * iDM.ψ.AR[start+nb_site-1], iDM.ψ.C[start+nb_site-1])
      iDM.ψ.AL[start+nb_site-1] = temp_R
      iDM.ψ.C[start+nb_site-1] = temp_C
      adjust_left = wδ(only(commoninds(temp_C, temp_R)), dag(only(uniqueinds(iDM.ψ.AL[start+nb_site], iDM.ψ.AL[start+nb_site+1], iDM.ψ.AR[start+nb_site]))) )
      iDM.ψ.AL[start+nb_site] *= adjust_left
    else
      if nb_site == N
        adjust_right = wδ(dag(only(uniqueinds(V2, S2, iDM.ψ.AR[start+nb_site-1]))), dag(only(uniqueinds(iDM.ψ.AR[start+nb_site], iDM.ψ.AL[start+nb_site], iDM.ψ.C[start+nb_site]))))
        iDM.ψ.AR[start+nb_site-1] = V2 #*  adjust_right
        iDM.ψ.AR[start+nb_site-1] *=  adjust_right
      else
        iDM.ψ.AR[start+nb_site-1] = V2 #*  adjust_right
      end
      temp = wδ(only(uniqueinds(iDM.ψ.AR[start+nb_site], iDM.ψ.AL[start+nb_site], iDM.ψ.C[start+nb_site])), only(commoninds( iDM.ψ.C[start+nb_site], iDM.ψ.AL[start+nb_site])) )
      temp_R, temp_C = diag_ortho_polar_both(S2 * iDM.ψ.AR[start+nb_site-1], temp)
      iDM.ψ.AL[start+nb_site-1] = temp_R
      iDM.ψ.C[start+nb_site-1] = temp_C
      adjust_left = wδ(only(commoninds(temp_C, temp_R)), dag(only(uniqueinds(iDM.ψ.AL[start+nb_site], iDM.ψ.AL[start+nb_site+1], iDM.ψ.AR[start+nb_site]))) )
      iDM.ψ.AL[start+nb_site] *= adjust_left
    end
    for j in 1:nb_site÷2
      if j == 1 && nbIterations == 1
        apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], translatecell(translater(iDM), dag(adjust_left), -1)*iDM.ψ.AL[start+j-1])
      else
        apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
      end
      #iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
    end
    iDM.counter+=nb_site÷2
    start += nb_site÷2
  end
  for j in reverse((N-nb_site + nb_site÷2 + 1):N)
    apply_mpomatrix_right!(iDM.R, iDM.Hmpo[original_start+j-1], iDM.ψ.AR[original_start+j-1])
    iDM.R[end] -= local_ener[1]/length((N-nb_site + nb_site÷2 + 1):N) * denseblocks(δ(inds(iDM.R[end])...))
  end
  if start <= N
    for j in 1:length(iDM.R)
      iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
    end
  else
    for j in 1:length(iDM.R)
      iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
    end
  end
  return local_ener[1]/N, err
end



function idmrg(iDM::iDMRGStructure{InfiniteMPOMatrix}; nb_iterations = 10, output_level = 0, mixer = false, α = 0.001, kwargs...)
  ener = 0; err = 0
  for j in 1:nb_iterations
  #  if !mixer
      ener, err = idmrg_step(iDM; kwargs...)
  #  else
  #    ener, err = idmrg_step_with_mixer(iDM; α = α, kwargs...)
  #  end
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



function build_local_hamiltonian(H::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}, start::Int64, len::Int64)
  tempL = copy(L)
  for j in 0:len-1
    @disable_warn_order apply_mpomatrix_left!(tempL, H[start+j])
  end
  @disable_warn_order temp = tempL[1] *  R[1]
  for j in 2:length(tempL)
    @disable_warn_order temp += tempL[j] * R[j]
  end
  return effectiveHam(temp)
end



function build_local_hamiltonian_2(H::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}, start::Int64, len::Int64)
  temp_H = H[start]
  for j = 2:len
    temp_H = temp_H * H[start + j - 1]
  end
  tempL = copy(L)
  @disable_warn_order apply_mpomatrix_left!(tempL, temp_H)
  @disable_warn_order temp = tempL[1] *  R[1]
  for j in 2:length(tempL)
    @disable_warn_order temp += tempL[j] * R[j]
  end
  return effectiveHam(temp)
end



function build_local_hamiltonian_3(temp_H::Matrix{ITensor}, L::Vector{ITensor}, R::Vector{ITensor})
  tempL = copy(L)
  @disable_warn_order apply_mpomatrix_left!(tempL, temp_H)
  @disable_warn_order temp = tempL[1] *  R[1]
  for j in 2:length(tempL)
    @disable_warn_order temp += tempL[j] * R[j]
  end
  return effectiveHam(temp)
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

#
# function idmrg_step_with_mixer(iDM::iDMRGStructure{InfiniteMPOMatrix}; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10, α = 0.1)
#   N = nsites(iDM)
#   nb_site = dmrg_sites(iDM)
#   if nb_site != N
#     error("iDMRG with a step size different than the unit cell has not been implemented")
#   end
#   start = mod1(iDM.counter, N)
#   effective_Rs=[iDM.R for j in 1:N-1]
#   local_ener=0
#   err = 0
#   for j in reverse(1:N-2)
#     effective_Rs[j] = copy(effective_Rs[j+1])
#     apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[start+j+1], iDM.ψ.AR[start+j+1])
#   end
#   effective_L = copy(iDM.L)
#   for j = 1:N-1
#     starting_state = iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j]
#     temp_H = build_two_local_hamiltonian(iDM.Hmpo[start+j-1], iDM.Hmpo[start+j], effective_L, effective_Rs[j])
#     local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
#     #Cheap mixer (in code, but cannot solve our issue)
#     U2, S2, V2 = svd(new_x[1]+α*temp_H(new_x[1]), commoninds(new_x[1], iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
#     righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
#     err += 1 - norm(S2)
#     S2 = S2 / norm(S2)
#     iDM.ψ.AL[start+j-1] = U2
#     iDM.ψ.AR[start+j-1],  = diag_ortho_polar(U2 * S2, iDM.ψ.C[start+j-2])
#     iDM.ψ.C[start+j-1] = S2
#     iDM.ψ.AR[start+j] = V2
#     iDM.ψ.AL[start+j],  = diag_ortho_polar(S2 * V2, iDM.ψ.C[start+j])
#     apply_mpomatrix_left!(effective_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
#   end
#   for j in 1:nb_site÷2
#     apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
#     iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
#   end
#   for j in reverse(nb_site÷2+1:nb_site)
#     apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
#     iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
#   end
#   if start == 1
#     for j in 1:length(iDM.R)
#       iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
#     end
#   else
#     for j in 1:length(iDM.R)
#       iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
#     end
#   end
#   iDM.counter+=nb_site÷2
#   return local_ener[1]/N, err
# end
#
#
#
# function idmrg_step_with_mixer(iDM::iDMRGStructure{InfiniteMPOMatrix}; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10, α = 0.1)
#   N = nsites(iDM)
#   nb_site = dmrg_sites(iDM)
#   if nb_site != N
#     error("iDMRG with a step size different than the unit cell has not been implemented")
#   end
#   start = mod1(iDM.counter, N)
#   effective_Rs=[iDM.R for j in 1:N-1]
#   local_ener=0
#   err = 0
#   for j in reverse(1:N-2)
#     effective_Rs[j] = copy(effective_Rs[j+1])
#     apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[start+j+1], iDM.ψ.AR[start+j+1])
#   end
#   effective_L = copy(iDM.L)
#   for j = 1:N-1
#     starting_state = iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j]
#     temp_H = build_two_local_hamiltonian(iDM.Hmpo[start+j-1], iDM.Hmpo[start+j], effective_L, effective_Rs[j])
#     local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol)
#
#     U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
#     righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
#     err += 1 - norm(S2)
#     S2 = S2 / norm(S2)
#     if α != 0
#       temp_mix, bd= build_left_mixer(effective_L, iDM.Hmpo[start+j-1], U2; α = α, maxdim = maxdim)
#       #Up, Sp, Vp =  svd(temp_mix, commoninds(temp_mix, iDM.ψ.AL[start+j-1]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
#       #    righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
#       #Up2, Sp2, Vp2 = svd(Vp * wδ(dag(only(uniqueinds(Vp, Sp))), only(commoninds(S2, V2))) *V2, commoninds(Vp, Sp); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(Up, Sp))),
#       #    righttags = tags(only(commoninds(Sp, Vp))))d
#
#       UU, iDM.ψ.C[start+j-1], iDM.ψ.AR[start+j] = ITensors.qr(bd * S2 * V2, uniqueinds(bd, S2);
#         mindim = 2, maxdim=maxdim, lefttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))),
#           righttags = tags(only(commoninds(iDM.ψ.AL[start+j-1], iDM.ψ.AL[start+j]))))
#       iDM.ψ.AL[start+j-1] = temp_mix*UU
#       println(inds(UU))
#       #iDM.ψ.AL[start+j-1] =  temp_mix#Up
#       #iDM.ψ.C[start+j-1] =  bd*S2#Sp #* wδ(inds(Up2)...) * wδ(inds(Sp2)...)
#       #println(inds(iDM.ψ.AL[start+j-1]))
#       #println(inds(iDM.ψ.C[start+j-1]))
#       #println(inds(Vp2))
#       #println("Norm")
#       #println(norm(iDM.ψ.C[start+j-1]))
#     #  try
#       #temp = wδ(only(commoninds(iDM.ψ.C[start+j-2], iDM.ψ.AL[start+j-2])), dag(only(uniqueinds(iDM.ψ.C[start+j-2], iDM.ψ.AL[start+j-2]))))
#       iDM.ψ.AR[start+j-1], = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
#
#       #iDM.ψ.AR[start+j-2] *= temp * dag(te)
#       #println(inds(iDM.ψ.AR[start+j-2]))
#         #println(inds(iDM.ψ.C[start+j-2]))
#         #println(inds(iDM.ψ.C[start+j-1]))
#     #  catch e
#     #    iDM.ψ.AR[start+j-1], iDM.ψ.C[start+j-2] = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
#     #  end
#     #iDM.ψ.AR[start+j] = V2#Vp*wδ(dag(only(uniqueinds(Vp, Sp))), only(commoninds(S2, V2))) *V2
#       #println(dim.(inds(iDM.ψ.AR[start+j])))
#     #  try
#   #  temp = wδ(only(commoninds(iDM.ψ.C[start+j], iDM.ψ.AR[start+j+1])), dag(only(uniqueinds(iDM.ψ.C[start+j], iDM.ψ.AR[start+j+1]))))
#         iDM.ψ.AL[start+j], = diag_ortho_polar(iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j], iDM.ψ.C[start+j])
#   #      iDM.ψ.AL[start+j+1] *= temp * dag(te)
#   #      println(inds(iDM.ψ.AL[start+j+1]))
#
#     #  catch
#     #    iDM.ψ.AL[start+j], iDM.ψ.C[start+j] = diag_ortho_polar(iDM.ψ.C[start+j-1] * iDM.ψ.AR[start+j], iDM.ψ.C[start+j])
#     #  end
#     else
#       iDM.ψ.AL[start+j-1] =  U2
#       iDM.ψ.C[start+j-1] =  S2
#       iDM.ψ.AR[start+j] = V2
#       iDM.ψ.AR[start+j-1], _ = diag_ortho_polar(iDM.ψ.AL[start+j-1] * iDM.ψ.C[start+j-1], iDM.ψ.C[start+j-2])
#       iDM.ψ.AL[start+j], _ = diag_ortho_polar(iDM.ψ.C[start+j-1] * V2, iDM.ψ.C[start+j])
#     end
#     #println(start+ j -1)
#     #evaluate_unitarity_left(iDM.ψ, start+j-1)
#     #evaluate_unitarity_right(iDM.ψ, start+j-1)
#     #evaluate_unitarity_left(iDM.ψ, start+j)
#     #evaluate_unitarity_right(iDM.ψ, start+j)
#     if j != N-1
#       apply_mpomatrix_left!(effective_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
#     end
#   end
#   for j in 1:nb_site÷2
#     apply_mpomatrix_left!(iDM.L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
#     iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
#   end
#   for j in reverse(nb_site÷2+1:nb_site)
#     apply_mpomatrix_right!(iDM.R, iDM.Hmpo[start+j-1], iDM.ψ.AR[start+j-1])
#     iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
#   end
#   if start == 1
#     for j in 1:length(iDM.R)
#       iDM.R[j]= translatecell(translater(iDM), iDM.R[j], 1)
#     end
#   else
#     for j in 1:length(iDM.R)
#       iDM.L[j]= translatecell(translater(iDM), iDM.L[j], -1)
#     end
#   end
#   iDM.counter+=nb_site÷2
#   return local_ener[1]/N, err
# end
#



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


function evaluate_unitarity_left(ψ::InfiniteCanonicalMPS, j::Int64)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  println(norm( δ(l[j-1], prime(dag(l[j-1]))) * ψ.AL[j] * δ(dag(s[j]), prime(s[j]))* dag(prime(ψ.AL[j])) - denseblocks(δ(l[j], prime(dag(l[j])))) ))
  return  δ(l[j-1], prime(dag(l[j-1]))) * ψ.AL[j] * δ(dag(s[j]), prime(s[j]))* dag(prime(ψ.AL[j]))
end

function evaluate_unitarity_right(ψ::InfiniteCanonicalMPS, j::Int64)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  println(norm( δ(dag(r[j]), prime(r[j])) * ψ.AR[j] * δ(dag(s[j]), prime(s[j]))* dag(prime(ψ.AR[j])) - denseblocks(δ(dag(r[j-1]), prime(r[j-1]))) ))
end
