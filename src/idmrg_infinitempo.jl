
function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, dmrg_sites::Int64)
  N = nsites(ψ) #dmrg_sites
  l = only(commoninds(ψ.AL[0], ψ.AL[1]))
  r = only(commoninds(ψ.AR[N + 1], ψ.AR[N]))
  l_mpo = only(commoninds(Hmpo[0], Hmpo[1]))
  r_mpo = only(commoninds(Hmpo[N + 1], Hmpo[N]))
  tempL = ITensor(l_mpo)
  tempL[end] = 1.0
  L = δ(l, prime(dag(l))) * tempL
  tempR = ITensor(r_mpo)
  tempR[1] = 1.0
  R = δ(r, prime(dag(r))) * tempR

  return iDMRGStructure{InfiniteMPO,ITensor}(copy(ψ), Hmpo, L, R, 1, dmrg_sites)
end

function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO)
  return iDMRGStructure{InfiniteMPO,ITensor}(ψ, Hmpo, 2)
end
function iDMRGStructure(
  ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, L::ITensor, R::ITensor, dmrg_sites::Int64
)
  return iDMRGStructure{InfiniteMPO,ITensor}(copy(ψ), Hmpo, L, R, 1, dmrg_sites)
end
function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, L::ITensor, R::ITensor)
  return iDMRGStructure{InfiniteMPO,ITensor}(copy(ψ), Hmpo, L, R, 1, 2)
end
iDMRGStructure(Hmpo::InfiniteMPO, ψ::InfiniteCanonicalMPS) = iDMRGStructure(ψ, Hmpo)

function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor)
  return L * Hmpo
end

function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor, ψ::ITensor)
  ψp = dag(ψ)'
  return ((L * ψ) * Hmpo) * ψp
end

apply_mpomatrix_right(R::ITensor, Hmpo::ITensor) = apply_mpomatrix_left(R, Hmpo)
function apply_mpomatrix_right(R::ITensor, Hmpo::ITensor, ψ::ITensor)
  return apply_mpomatrix_left(R, Hmpo, ψ)
end

function (H::iDMRGStructure{InfiniteMPO,ITensor})(x)
  n = order(x) - 2
  L = H.L
  R = H.R
  start = mod1(H.counter, nsites(H))
  L = L * x
  for j in 0:(n - 1)
    L = apply_mpomatrix_left(L, H.Hmpo[start + j])
  end
  return noprime(L * R)
end

function (H::temporaryHamiltonian{InfiniteMPO,ITensor})(x)
  n = order(x) - 2
  L = H.effectiveL
  R = H.effectiveR
  start = H.nref
  L = L * x
  for j in 0:(n - 1)
    L = apply_mpomatrix_left(L, H.Hmpo[start + j])
  end
  return noprime(L * R)
end

function advance_environments(H::iDMRGStructure{InfiniteMPO})
  N = nsites(H)
  nb_site = N#dmrg_sites(H)
  start = mod1(H.counter, N)
  for j in 0:(N - 1)
    H.L = apply_mpomatrix_left(H.L, H.Hmpo[start + j], H.ψ.AL[start + j])
  end
  for j in 0:(N - 1)
    H.R = apply_mpomatrix_right(
      H.R, H.Hmpo[start + nb_site - 1 - j], H.ψ.AR[start + nb_site - 1 - j]
    )
  end
  H.R = translatecell(translator(H), H.R, 1)
  return H.L = translatecell(translator(H), H.L, -1)
end

function set_environments_defaultposition(H::iDMRGStructure{InfiniteMPO,ITensor})
  N = nsites(H)
  if mod1(H.counter, N) == 1
    println("Already at the correct position")
    return 0
  end
  nb_steps = mod(1 - H.counter, N)
  start = mod1(H.counter, N)
  for j in 0:(nb_steps - 1)
    H.L = apply_mpomatrix_left(H.L, H.Hmpo[start + j], H.ψ.AL[start + j])
  end
  for j in reverse((start + nb_steps - N):(start + N - 1))
    H.R = apply_mpomatrix_right(H.R, H.Hmpo[j], H.ψ.AR[j])
  end
  shift_cell = getcell(inds(H.L)[1]) - 0
  if shift_cell != 0
    H.L = translatecell(translator(H), H.L, -shift_cell)
  end
  shift_cell = getcell(inds(H.R)[1]) - 1
  if shift_cell != 0
    H.R = translatecell(translator(H), H.R, -shift_cell)
  end
  H.counter += nb_steps
  return 0
end

function idmrg_step(
  iDM::iDMRGStructure{InfiniteMPO,ITensor}; solver_tol=1e-8, maxdim=20, cutoff=1e-10
)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end
  if nb_site == 1
    #return idmrg_step_single_site(iDM; solver_tol, cutoff)
    error("Not fully implemented")
  end
  if (N ÷ (nb_site ÷ 2)) * (nb_site ÷ 2) != N
    error("We require that the (nb_site÷2) divides the unitcell length")
  end
  nbIterations = (N - nb_site) ÷ (nb_site ÷ 2) + 1#(N÷(nb_site÷2)) - 1
  original_start = mod1(iDM.counter, N)
  effective_Rs = [iDM.R for j in 1:nbIterations]
  local_ener = 0
  err = 0
  site_looked = original_start + N - 1
  for j in reverse(1:(nbIterations - 1))
    effective_Rs[j] = copy(effective_Rs[j + 1])
    for k in 0:(nb_site ÷ 2 - 1)
      effective_Rs[j] = apply_mpomatrix_right(
        effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked]
      )
      site_looked -= 1
    end
  end
  adjust_left = 0
  adjust_right_most = 0
  start = original_start
  current_L = copy(iDM.L)
  for count in 1:nbIterations
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
    for j in 3:nb_site
      starting_state *= iDM.ψ.AR[start + j - 1]
    end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
      current_L, effective_Rs[count], iDM.Hmpo, start
    )
    local_ener, new_x = eigsolve(
      temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
    )
    U2, S2, V2 = svd(
      new_x[1],
      commoninds(new_x[1], iDM.ψ.AL[start]);
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1]))),
      righttags=tags(only(commoninds(iDM.ψ.AR[start + 1], iDM.ψ.AR[start]))),
    )
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start] = U2
    iDM.ψ.C[start] = denseblocks(S2)
    iDM.ψ.AR[start] = ortho_polar(U2 * S2, iDM.ψ.C[start - 1])
    for j in 2:(nb_site - 1)
      new_x = S2 * V2
      linktags = tags(only(commoninds(iDM.ψ.AL[start + j - 1], iDM.ψ.AL[start + j])))
      U2, S2, V2 = svd(
        new_x,
        (
          only(commoninds(new_x, iDM.ψ.AL[start + j - 1])),
          only(commoninds(new_x, iDM.ψ.AL[start + j - 2])),
        );
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=linktags,
        righttags=linktags,
      )
      err += 1 - norm(S2)
      S2 = S2 / norm(S2)
      iDM.ψ.AL[start + j - 1] = U2
      iDM.ψ.C[start + j - 1] = denseblocks(S2)
      iDM.ψ.AR[start + j - 1] = ortho_polar(U2 * S2, iDM.ψ.C[start + j - 2])
    end
    iDM.ψ.AR[start+nb_site - 1] = V2
    iDM.ψ.AL[start+nb_site - 1] = ortho_polar(S2*V2, iDM.ψ.C[start + nb_site - 1])
    #Advance the left environment as long as we are not finished
    if count != nbIterations
      for j in 1:(nb_site ÷ 2)
        current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start + j - 1], iDM.ψ.AL[start + j - 1])
      end
      start += nb_site ÷ 2
    end
  end
  #By convention, we choose to advance half the unit cell
  for j in 1:(N ÷ 2)
    iDM.L = apply_mpomatrix_left(iDM.L, iDM.Hmpo[original_start + j - 1], iDM.ψ.AL[original_start + j - 1])
    tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start + j])))
    tempL[1] = 1.0
    iDM.L -=
      local_ener[1] / N * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start + j])...) * tempL
  end
  for j in reverse((N ÷ 2 + 1):N)#reverse((N-nb_site + nb_site÷2 + 1):N)
    iDM.R = apply_mpomatrix_right(
      iDM.R, iDM.Hmpo[original_start + j - 1], iDM.ψ.AR[original_start + j - 1]
    )
    tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start + j - 2])))
    tempR[end] = 1.0
    iDM.R -=
      local_ener[1] / N * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start + j - 2])...) * tempR
  end
  if original_start + N ÷ 2 >= N + 1
    iDM.L = translatecell(translator(iDM), iDM.L, -1)
  else
    iDM.R = translatecell(translator(iDM), iDM.R, 1)
  end

  iDM.counter += N ÷ 2
  return local_ener[1] / N, err
end




#### Mixing Pollmann version and mine, with alternating sweeps with and without expansion.
# Very important: We update environments _only_ with good MPS
function idmrg_step_with_noise_auxiliary(iDM::iDMRGStructure{InfiniteMPO,ITensor};solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 1e-6, kwargs...)
  N = nsites(iDM)
  original_start = mod1(iDM.counter, N)
  effective_Rs = [copy(iDM.R) for j in 1:N-1]
  local_ener = 0
  err = 0

  H_extension = get(kwargs, :MPO_extension, iDM.Hmpo)
  maxdim_subspace = get(kwargs, :maxdim_subspace, maxdim)

  #Building successive environments
  site_looked = original_start + N - 1
  for j in reverse(1:(N - 2))
    effective_Rs[j] = copy(effective_Rs[j + 1])
    effective_Rs[j] = apply_mpomatrix_right(effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked])
    site_looked -= 1
  end
  current_L = copy(iDM.L)
  for (count, start) in enumerate(original_start:original_start+N-2)
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
    if α != 0
      starting_state += max(α, 1e-6) * randomITensor(inds(starting_state)) #ensures that every sector has some non zero elements
    end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    current_L, effective_Rs[count], iDM.Hmpo, start
    )
    local_ener, new_x = eigsolve(
    temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
    )

    theta = new_x[1]
    left_indices = commoninds(theta, iDM.ψ.AL[start] )
    newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))

    if α != 0
      if H_extension == iDM.Hmpo
        env = current_L
      else
        left_link = only(commoninds(iDM.ψ.AL[start-1], iDM.ψ.AL[start]))
        left_link_mpo = only(commoninds(H_extension[start-1], H_extension[start]))
        env = randomITensor(left_link, dag(prime(left_link)), left_link_mpo)
      end
      U2, S2, V2 = subspace_expansion(theta, env, H_extension, (start, start+1); maxdim = maxdim_subspace, cutoff, newtags, α, svd_indices = left_indices)
    else
      U2, S2, V2 = svd(
      theta,
      left_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      #err = 1 - norm(S2)
      #S2 = S2 / norm(S2)
    end
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)

    iDM.ψ.AL[start] = U2
    iDM.ψ.C[start] = ITensors.denseblocks(S2)
    iDM.ψ.AR[start] = ortho_polar(U2 * S2, iDM.ψ.C[start - 1])
    iDM.ψ.AR[start+1] = V2
    iDM.ψ.AL[start+1] = ortho_polar(S2*V2, iDM.ψ.C[start + 1])
    #Advance the left environment as long as we are not finished
    current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start], iDM.ψ.AL[start])
  end
  #Now sweeping right
  effective_Ls = [iDM.L for j in 1:N-1]
  local_ener = 0
  err = 0
  for j in 2:N-1
    effective_Ls[j] = copy(effective_Ls[j - 1])
    effective_Ls[j] = apply_mpomatrix_right(
          effective_Ls[j], iDM.Hmpo[original_start + j - 2], iDM.ψ.AL[original_start + j - 2]
        )
  end
  current_R = copy(iDM.R)
  for (count, start) in enumerate(original_start+N-2:-1:original_start)
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
    if α != 0
      starting_state += max(α, 1e-6) * randomITensor(inds(starting_state)) #ensures that every sector has some non zero elements
    end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    effective_Ls[end+1-count], current_R, iDM.Hmpo, start
    )
    local_ener, new_x = eigsolve(
    temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
    )
    theta = new_x[1]
    right_indices = commoninds(theta, iDM.ψ.AR[start+1])
    newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))

    if α != 0
      if H_extension == iDM.Hmpo
        env = current_R
      else
        right_link = only(commoninds(iDM.ψ.AR[start+2], iDM.ψ.AR[start+1]))
        right_link_mpo = only(commoninds(H_extension[start+2], H_extension[start+1]))
        env = randomITensor(right_link, dag(prime(right_link)), right_link_mpo)
      end
      U2, S2, V2 = subspace_expansion(theta, env, H_extension, (start+1, start); maxdim = maxdim_subspace, cutoff, newtags, α, svd_indices = right_indices)
    else
      U2, S2, V2 = svd(
      theta,
      right_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
    end
    #This normalization can be so so after subspace expansion.
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start] = V2
    iDM.ψ.C[start] = ITensors.denseblocks(S2)
    iDM.ψ.AR[start] = ortho_polar(V2 * S2, iDM.ψ.C[start - 1])
    #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
    iDM.ψ.AR[start+1] = U2
    iDM.ψ.AL[start+1] = ortho_polar(S2*U2, iDM.ψ.C[start + 1])
    #Advance the left environment as long as we are not finished
    current_R = apply_mpomatrix_right(current_R, iDM.Hmpo[start+1], iDM.ψ.AR[start + 1])
  end
  return local_ener, err
end


function idmrg_step_with_noise(
  iDM::iDMRGStructure{InfiniteMPO,ITensor}; solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 1e-6, kwargs...
)
  N = nsites(iDM)
  nb_site = iDM.dmrg_sites
  if nb_site != 2
    error("For now, we assume 2 site DMRG for the noise")
  end
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end
  s = siteinds(only, iDM.ψ)

  #A first double sweep with noise
  if α != 0
    local_ener, err = idmrg_step_with_noise_auxiliary(
      iDM; solver_tol, maxdim, cutoff, α, kwargs...
      )
  else
    local_ener, err =   idmrg_step_with_noise_auxiliary(iDM; solver_tol, maxdim, cutoff, α=0, kwargs...)
  end

  #Updating environments
  original_start = mod1(iDM.counter, N)
  for j in 1:(N ÷ 2)
    iDM.L = apply_mpomatrix_left(iDM.L, iDM.Hmpo[original_start + j - 1], iDM.ψ.AL[original_start + j - 1])
    tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start + j])))
    tempL[1] = 1.0
    iDM.L -=
      local_ener[1] / N * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start + j])...) * tempL
  end
  for j in reverse((N ÷ 2 + 1):N)
    iDM.R = apply_mpomatrix_right(
      iDM.R, iDM.Hmpo[original_start + j - 1], iDM.ψ.AR[original_start + j - 1]
    )
    tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start + j - 2])))
    tempR[end] = 1.0
    iDM.R -=
      local_ener[1]/ N * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start + j - 2])...) * tempR
  end
  if original_start + N ÷ 2 >= N + 1
    iDM.L = translatecell(translator(iDM), iDM.L, -1)
  else
    iDM.R = translatecell(translator(iDM), iDM.R, 1)
  end

  iDM.counter += N ÷ 2
  return local_ener[1] / N , err
end
