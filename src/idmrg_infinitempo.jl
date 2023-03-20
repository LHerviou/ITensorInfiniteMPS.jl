
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


#
# function idmrg_step_with_noise(
#   iDM::iDMRGStructure{InfiniteMPO,ITensor}; solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 1e-6, maxdim_exp = 16
# )
#   N = nsites(iDM)
#   nb_site = dmrg_sites(iDM)
#   if nb_site != 2
#     error("For now, we assume 2 site DMRG for the noise")
#   end
#   if nb_site > N
#     error("iDMRG with a step size larger than the unit cell has not been implemented")
#   end
#   if nb_site == 1
#     #return idmrg_step_single_site(iDM; solver_tol, cutoff)
#     error("Not fully implemented")
#   end
#   if (N ÷ (nb_site ÷ 2)) * (nb_site ÷ 2) != N
#     error("We require that the (nb_site÷2) divides the unitcell length")
#   end
#   nbIterations = (N - nb_site) ÷ (nb_site ÷ 2) + 1#(N÷(nb_site÷2)) - 1
#   original_start = mod1(iDM.counter, N)
#   effective_Rs = [iDM.R for j in 1:nbIterations]
#   local_ener = 0
#   err = 0
#   site_looked = original_start + N - 1
#   for j in reverse(1:(nbIterations - 1))
#     effective_Rs[j] = copy(effective_Rs[j + 1])
#     for k in 0:(nb_site ÷ 2 - 1)
#       effective_Rs[j] = apply_mpomatrix_right(
#         effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked]
#       )
#       site_looked -= 1
#     end
#   end
#   adjust_left = 0
#   adjust_right_most = 0
#   start = original_start
#   current_L = copy(iDM.L)
#   for count in 1:nbIterations
#     #build the local tensor start .... start + nb_site - 1
#     starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
#     if norm(starting_state) == 0
#       println("Problem with norm of the initial guess")
#       starting_state = randomITensor(inds(starting_state))
#     end
#     #build_local_Hamiltonian
#     temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
#       current_L, effective_Rs[count], iDM.Hmpo, start
#     )
#     local_ener, new_x = eigsolve(
#       temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
#     )
#
#     #TODO assume 2 site idmrg. We follow Hubig logic, adapted for 2 sites
#     #TODO better with kernel?
#     #NL = nullspace(U2, commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start-1]); atol=1e-2, tags = "leftnullspace"))
#     #nL = only(filterinds(NL, tags =  "leftnullspace"))
#     #Suspicion that this will break at large everything
#     # ri1 = commoninds( iDM.ψ.AR[start + 1], iDM.ψ.AR[start + 2]);
#     # noise_term = noprime( (current_L * (α* new_x[1])) * (iDM.Hmpo[start]  * iDM.Hmpo[start+1]) )
#     # cri2 = combiner(ri1, only(commoninds(iDM.Hmpo[start+1], iDM.Hmpo[start+2])), dir = dir(ri1))
#     # ri2 = combinedind(cri2)
#     # noise_term = noise_term * cri2
#     # new_x, newr  = ITensors.directsum(
#     #         new_x[1] => ri1,
#     #         noise_term => ri2;
#     #         tags=("Right",),
#     #       )
#
#     U2, S2, V2 = svd(
#       new_x[1],
#       commoninds(new_x[1], iDM.ψ.AL[start]);
#       maxdim=maxdim,
#       cutoff=cutoff,
#       lefttags=tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1]))),
#       righttags=tags(only(commoninds(iDM.ψ.AR[start + 1], iDM.ψ.AR[start]))),
#     )
#     err = 1 - norm(S2)
#     S2 = S2 / norm(S2)
#
#     NL = nullspace(U2, commoninds(U2, S2); atol=1e-2, tags = "leftnullspace")
#     nL = only(filterinds(NL, tags =  "leftnullspace"))
#     ri1 = only(commoninds( S2, V2));
#     noise_term = noprime( (current_L * (U2 * (S2) ) ) * iDM.Hmpo[start] )
#     cri2 = combiner(ri1, only(commoninds(iDM.Hmpo[start], iDM.Hmpo[start+1])), dir = dir(ri1))
#     ri2 = combinedind(cri2)
#     noise_term = NL * noise_term * cri2
#     if norm(noise_term) == 0
#       println("No expansion at $start")
#       new_U = U2; new_V = V2; new_S = S2;
#     else
#       #noise_term =  noise_term * cri2
#       U2bis, S2bis, V2bis = svd(noise_term, nL, maxdim = maxdim_exp, cutoff = 1e-12 ) #I want this because my MPOs are stupid
#       #U2bis, S2bis, V2bis = svd(noise_term, commoninds(noise_term, U2), maxdim = maxdim_exp, cutoff = 1e-12 ) #I want this because my MPOs are stupid
#       S2bis = α * S2bis/norm(S2bis)
#     #println(norm(S2bis))
#
#       new_U, new_ur = ITensors.directsum(
#              U2 => commoninds(U2, S2),
#              dag(NL) * U2bis => commoninds(U2bis, S2bis);
#              tags=(tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start+1]))),),
#           )
#
#           new_S, (new_sl, new_sr) = ITensors.directsum(
#         S2 => (only(commoninds(S2, U2)), only(commoninds(S2, V2))),
#         S2bis => (only(commoninds(S2bis, U2bis)), only(commoninds(S2bis, V2bis)));
#                    tags=(tags(only(commoninds(S2, U2))),tags(only(commoninds(S2, V2))),),
#                 )
#         new_S = new_S / norm(new_S)
#
#
#       new_V, new_vl = ITensors.directsum(
#      V2 => commoninds(V2, S2),
#      ITensor(uniqueinds(V2, S2)..., commoninds(V2bis, S2bis)...) => commoninds(V2bis, S2bis);
#      tags=(tags(only(commoninds(V2, S2))),),
#      )
#      new_S = new_S * δ(dag(new_ur), dag(new_sl)) * δ(dag(new_vl), dag(new_sr))
#
#       new_U, new_S, nnew_V = svd(new_U*new_S, uniqueinds(new_U, new_S),
#       maxdim=maxdim,
#       cutoff=cutoff,
#       lefttags=tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1]))),
#       righttags=tags(only(commoninds(iDM.ψ.AR[start + 1], iDM.ψ.AR[start]))),
#     )
#       new_V = nnew_V * new_V
#     end
#     iDM.ψ.AL[start] = new_U
#     iDM.ψ.C[start] = ITensors.denseblocks(new_S)
#     iDM.ψ.AR[start] = ortho_polar(new_U * new_S, iDM.ψ.C[start - 1])
#     #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
#     iDM.ψ.AR[start+nb_site - 1] = new_V
#     iDM.ψ.AL[start+nb_site - 1] = ortho_polar(new_S*iDM.ψ.AR[start+nb_site - 1], iDM.ψ.C[start + 1])
#     #Advance the left environment as long as we are not finished
#     for j in 1:(nb_site ÷ 2)
#       current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start + j - 1], iDM.ψ.AL[start + j - 1])
#     end
#     start += nb_site ÷ 2
#   end
#   #Add one last optimization between L and 1 _without expension
#   current_R = iDM.R
#   for j in reverse(1:N-1)
#     site_looked = original_start + j
#     current_R = apply_mpomatrix_right(
#         current_R, iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked]
#       )
#   end
#   current_R = translatecell(translator(iDM), current_R, 1)
#   #build the local tensor start .... start + nb_site - 1
#   starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
#   # println(inds(starting_state))
#   # println()
#   # println(inds(current_L))
#   # println()
#   # println(inds(current_R))
#   #build_local_Hamiltonian
#   temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
#     current_L, current_R, iDM.Hmpo, start
#   )
#   local_ener, new_x = eigsolve(
#     temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
#   )
#
#   U2, S2, V2 = svd(
#     new_x[1],
#     commoninds(new_x[1], iDM.ψ.AL[start]);
#     maxdim=maxdim,
#     cutoff=cutoff,
#     lefttags=tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1]))),
#     righttags=tags(only(commoninds(iDM.ψ.AR[start + 1], iDM.ψ.AR[start]))),
#   )
#   err = 1 - norm(S2)
#   S2 = S2 / norm(S2)
#
#   iDM.ψ.AL[start] = U2
#   iDM.ψ.C[start] = ITensors.denseblocks(S2)
#   iDM.ψ.AR[start] = ortho_polar(U2 * S2, iDM.ψ.C[start - 1])
#   #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
#   iDM.ψ.AR[start+nb_site - 1] = V2
#   iDM.ψ.AL[start+nb_site - 1] = ortho_polar(S2*V2, iDM.ψ.C[start + 1])
#   #By convention, we choose to advance half the unit cell
#   iDM.L = apply_mpomatrix_left(current_L, iDM.Hmpo[original_start + N - 1], iDM.ψ.AL[original_start + N - 1])
#   tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start + N])))
#   tempL[1] = 1.0
#   iDM.L -=  local_ener[1]/2 * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start + N-1])...) * tempL
#   iDM.L = translatecell(translator(iDM), iDM.L, -1)
#
#   iDM.R = apply_mpomatrix_right(
#       current_R, iDM.Hmpo[original_start + N], iDM.ψ.AR[original_start + N]
#     )
#   tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start + N - 1])))
#   tempR[end] = 1.0
#   iDM.R -= local_ener[1] / 2 * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start + N-1])...) * tempR
#
#   iDM.counter += N
#   return local_ener[1] / N, err
# end



#### Testing Pollmann version
function idmrg_step_with_noise(
  iDM::iDMRGStructure{InfiniteMPO,ITensor}; solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 1e-6, maxdim_exp = 16, kwargs...
)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site != 2
    error("For now, we assume 2 site DMRG for the noise")
  end
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end
  s = siteinds(only, iDM.ψ)

  #Sweeping right
  nbIterations = N - 1#(N÷(nb_site÷2)) - 1
  original_start = mod1(iDM.counter, N)
  effective_Rs = [copy(iDM.R) for j in 1:nbIterations]
  local_ener = 0
  err = 0
  site_looked = original_start + N - 1
  for j in reverse(1:(nbIterations - 1))
    effective_Rs[j] = copy(effective_Rs[j + 1])
    effective_Rs[j] = apply_mpomatrix_right(effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked])
    site_looked -= 1
  end
  start = original_start
  current_L = copy(iDM.L)
  for count in 1:nbIterations
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
    if α != 0
      starting_state = randomITensor(inds(starting_state))
    end
    #starting_state = iDM.ψ.C[start-1] * iDM.ψ.AR[start] * iDM.ψ.AR[start + 1]
    if norm(starting_state) == 0
      println("Problem with norm of the initial guess")
      starting_state = randomITensor(inds(starting_state))
    end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
      current_L, effective_Rs[count], iDM.Hmpo, start
    )
    local_ener, new_x = eigsolve(
      temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
    )

    #TODO assume 2 site idmrg. We follow Hubig logic, adapted for 2 sites
    theta = new_x[1]
    left_indices = commoninds(theta, iDM.ψ.AL[start] )
    newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))

    if α != 0
      extension = noprime( ((current_L * theta) * iDM.Hmpo[start] ) * iDM.Hmpo[start+1] )
      extension = extension * (α / norm(extension))
      supp_index = only(commoninds(iDM.Hmpo[start+1], iDM.Hmpo[start+2]))
      dummy_index = Index(QN() => 1; dir=dir(supp_index))
      dum = ITensor(dummy_index); dum[1] = 1

      theta_extended, new_index = ITensors.directsum(
             theta  * dum => dummy_index,
             extension => supp_index;
             tags="Temporary",
          )
      closure = ITensor(new_index); closure[1] = 1;
      #println(norm(theta - theta_extended*dag(closure)))
      cc = combiner(new_index)
      U2, S2, V2 = svd(
        theta_extended*cc,
        left_indices;
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=newtags,
        righttags=newtags,
      )
      V2 = V2 * dag(cc) * dag(closure)
      temp_norm = norm(S2)#norm(U2*S2*V2) #TODO check
      err = 1 - temp_norm
      S2 = S2 / temp_norm
    else
      U2, S2, V2 = svd(
        theta,
        left_indices;
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=newtags,
        righttags=newtags,
        )
        err = 1 - norm(S2)
        S2 = S2 / norm(S2)
    end

    iDM.ψ.AL[start] = U2
    iDM.ψ.C[start] = ITensors.denseblocks(S2)
    iDM.ψ.AR[start] = ortho_polar(U2 * S2, iDM.ψ.C[start - 1])
    iDM.ψ.AR[start+1] = V2
    iDM.ψ.AL[start+1] = ortho_polar(S2*V2, iDM.ψ.C[start + 1])
    #Advance the left environment as long as we are not finished
    current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start], iDM.ψ.AL[start])
    start += 1
  end
  #Add one last optimization between L and 1
  current_R = copy(iDM.R)
  for j in reverse(1:N-1)
    site_looked = original_start + j
    current_R = apply_mpomatrix_right(
        current_R, iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked]
      )
  end
  current_R = translatecell(translator(iDM), current_R, 1)
  #starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
  starting_state = (iDM.ψ.AR[start] * iDM.ψ.C[start-1]) * iDM.ψ.AR[start + 1]
  if α != 0
    starting_state = randomITensor(inds(starting_state))
  end
  temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    current_L, current_R, iDM.Hmpo, start
  )
  local_ener, new_x = eigsolve(
    temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
  )
  theta = new_x[1]
  left_indices = commoninds(theta, iDM.ψ.AL[start] )
  newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))
  if false#α != 0
    extension = noprime( ((current_L * theta) * iDM.Hmpo[start] ) * iDM.Hmpo[start+1] )
    extension = extension * (α / norm(extension))
    supp_index = only(commoninds(iDM.Hmpo[start+1], iDM.Hmpo[start+2]))
    dummy_index = Index(QN() => 1; dir=dir(supp_index))
    dum = ITensor(dummy_index); dum[1] = 1

    theta_extended, new_index = ITensors.directsum(
           theta  * dum => dummy_index,
           extension => supp_index;
           tags="Temporary",
        )
    closure = ITensor(new_index); closure[1] = 1;
    U2, S2, V2 = svd(
      theta_extended,
      left_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
    )
    V2 = V2 * dag(closure)
    temp_norm = norm(U2*S2*V2)
    err = 1 - temp_norm
    S2 = S2 / temp_norm
  else
    U2, S2, V2 = svd(
    new_x[1],
    left_indices;
    maxdim=maxdim,
    cutoff=cutoff,
    lefttags=newtags,
    righttags=newtags,
  )
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
  end

  iDM.ψ.AL[start] = U2
  iDM.ψ.C[start] = ITensors.denseblocks(S2)
  iDM.ψ.AR[start] = ortho_polar(U2 * S2, iDM.ψ.C[start - 1])
  #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
  iDM.ψ.AR[start+1] = V2
  iDM.ψ.AL[start+1] = ortho_polar(S2*V2, iDM.ψ.C[start + 1])
  #By convention, we choose to advance half the unit cell
  iDM.L = apply_mpomatrix_left(current_L, iDM.Hmpo[original_start + N - 1], iDM.ψ.AL[original_start + N - 1])
  tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start + N])))
  tempL[1] = 1.0
  iDM.L -=  local_ener[1]/2 * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start + N-1])...) * tempL
  iDM.L = translatecell(translator(iDM), iDM.L, -1)
  iDM.R = apply_mpomatrix_right(
      current_R, iDM.Hmpo[original_start + N], iDM.ψ.AR[original_start + N]
    )
  tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start + N - 1])))
  tempR[end] = 1.0
  iDM.R -= local_ener[1]/2 * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start + N-1])...) * tempR

  #sweeping left
  nbIterations = N - 1
  effective_Ls = [iDM.L for j in 1:N-1]
  local_ener = 0
  err = 0
  for j in 2:N-1
    effective_Ls[j] = copy(effective_Ls[j - 1])
    effective_Ls[j] = apply_mpomatrix_right(
        effective_Ls[j], iDM.Hmpo[original_start + j - 2], iDM.ψ.AL[original_start + j - 2]
      )
  end
  start = start-1
  current_R = copy(iDM.R)
  for count in 1:nbIterations
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
    if α != 0
      starting_state = randomITensor(inds(starting_state))
    end
    if norm(starting_state) == 0
      println("Problem with norm of the initial guess")
      starting_state = randomITensor(inds(starting_state))
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
    #noise_term = noprime( current_L * theta * iDM.Hmpo[start] *iDM.Hmpo[start+1] )
    if α != 0
      extension = noprime( ((current_R * theta) * iDM.Hmpo[start+1] ) * iDM.Hmpo[start] )
      extension = extension * (α / norm(extension))
      supp_index = only(commoninds(iDM.Hmpo[start], iDM.Hmpo[start-1]))
      dummy_index = Index(QN() => 1; dir=dir(supp_index))
      dum = ITensor(dummy_index); dum[1] = 1

      theta_extended, new_index = ITensors.directsum(
             theta  * dum => dummy_index,
             extension => supp_index;
             tags="Temporary",
          )
      cc = combiner(new_index)
      closure = ITensor(new_index); closure[1] = 1;
      #println(norm(theta - theta_extended*dag(closure)))
      U2, S2, V2 = svd(
        theta_extended * cc,
        right_indices;
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=newtags,
        righttags=newtags,
      )
      V2 = V2 * dag(cc) * dag(closure)
      temp_norm = norm(S2)#norm(U2*S2*V2)
      err = 1 - temp_norm
      S2 = S2 / temp_norm
    else
      U2, S2, V2 = svd(
        theta,
        right_indices;
        maxdim=maxdim,
        cutoff=cutoff,
        lefttags=newtags,
        righttags=newtags,
        )
        err = 1 - norm(S2)
        S2 = S2 / norm(S2)
    end

    iDM.ψ.AL[start] = V2
    iDM.ψ.C[start] = ITensors.denseblocks(S2)
    iDM.ψ.AR[start] = ortho_polar(V2 * S2, iDM.ψ.C[start - 1])
    #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
    iDM.ψ.AR[start+nb_site - 1] = U2
    iDM.ψ.AL[start+nb_site - 1] = ortho_polar(S2*U2, iDM.ψ.C[start + 1])
    #Advance the left environment as long as we are not finished
    current_R = apply_mpomatrix_right(current_R, iDM.Hmpo[start+1], iDM.ψ.AR[start + 1])
    start -= 1
  end
  #Add one last optimization between L and 1
  current_L = iDM.L
  for j in 1:N-1
    site_looked = original_start + j - 1
    current_L = apply_mpomatrix_left(
        current_L, iDM.Hmpo[site_looked], iDM.ψ.AL[site_looked]
      )
  end
  current_L = translatecell(translator(iDM), current_L, -1)
  starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start + 1]
  if α != 0
    starting_state = randomITensor(inds(starting_state))
  end
  temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    current_L, current_R, iDM.Hmpo, start
  )
  local_ener, new_x = eigsolve(
    temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol
  )
  theta = new_x[1]
  right_indices = commoninds(theta, iDM.ψ.AR[start+1])
  newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))
  if false#α != 0
    extension = noprime( ((current_R * theta) * iDM.Hmpo[start+1] ) * iDM.Hmpo[start] )
    extension = extension * (α / norm(extension))
    supp_index = only(commoninds(iDM.Hmpo[start], iDM.Hmpo[start-1]))
    dummy_index = Index(QN() => 1; dir=dir(supp_index))
    dum = ITensor(dummy_index); dum[1] = 1

    theta_extended, new_index = ITensors.directsum(
           theta  * dum => dummy_index,
           extension => supp_index;
           tags="Temporary",
        )
    closure = ITensor(new_index); closure[1] = 1;
    U2, S2, V2 = svd(
      theta_extended,
      right_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
    )
    V2 = V2 * dag(closure)
    temp_norm = norm(U2*S2*V2)
    err = 1 - temp_norm
    S2 = S2 / temp_norm
  else
    U2, S2, V2 = svd(
      theta,
      right_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      err = 1 - norm(S2)
      S2 = S2 / norm(S2)
  end

  iDM.ψ.AL[start] = V2
  iDM.ψ.C[start] = ITensors.denseblocks(S2)
  iDM.ψ.AR[start] = ortho_polar(V2 * S2, iDM.ψ.C[start - 1])
  #println(norm(iDM.ψ.C[start - 1]* iDM.ψ.AR[start] - iDM.ψ.AL[start]*iDM.ψ.C[start]))
  iDM.ψ.AR[start + 1] = U2
  iDM.ψ.AL[start + 1] = ortho_polar(S2*U2, iDM.ψ.C[start + 1])
  #By convention, we choose to advance half the unit cell
  iDM.L = apply_mpomatrix_left(current_L, iDM.Hmpo[start], iDM.ψ.AL[start])
  tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[start+1])))
  tempL[1] = 1.0
  iDM.L -=  local_ener[1]/2 * δ(uniqueinds(iDM.L, iDM.Hmpo[start + 1])...) * tempL

  iDM.R = apply_mpomatrix_right(
      current_R, iDM.Hmpo[original_start], iDM.ψ.AR[original_start]
    )
  tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start])))
  tempR[end] = 1.0
  iDM.R -= local_ener[1] / 2 * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start - 1])...) * tempR
  iDM.R = translatecell(translator(iDM), iDM.R, 1)


  for j in 1:(N ÷ 2)
    iDM.L = apply_mpomatrix_left(iDM.L, iDM.Hmpo[original_start + j - 1], iDM.ψ.AL[original_start + j - 1])
    tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start + j])))
    tempL[1] = 1.0
    iDM.L -=
      local_ener[1] / 2 / N * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start + j])...) * tempL
  end
  for j in reverse((N ÷ 2 + 1):N)#reverse((N-nb_site + nb_site÷2 + 1):N)
    iDM.R = apply_mpomatrix_right(
      iDM.R, iDM.Hmpo[original_start + j - 1], iDM.ψ.AR[original_start + j - 1]
    )
    tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start + j - 2])))
    tempR[end] = 1.0
    iDM.R -=
      local_ener[1] / 2/ N * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start + j - 2])...) * tempR
  end
  if original_start + N ÷ 2 >= N + 1
    iDM.L = translatecell(translator(iDM), iDM.L, -1)
  else
    iDM.R = translatecell(translator(iDM), iDM.R, 1)
  end

  iDM.counter += N ÷ 2
  iDM.counter += N
  return local_ener[1] / N /2, err
end

#
# function temp_diag_inv(arr)
#   new_inds = dag.(inds(arr))
#   new_arr = ITensor(new_inds)
#   for x in 1:size(arr, 1)
#     new_arr[x, x] = 1/arr[x, x]
#   end
#   return new_arr
# end



#### Mixing Pollmann version and mine, with alternating sweeps with and without expansion.
#We update environments _only_ with good MPS
function idmrg_step_with_noise_auxiliary(iDM::iDMRGStructure{InfiniteMPO,ITensor};solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 1e-6, kwargs...)
  N = nsites(iDM)
  original_start = mod1(iDM.counter, N)
  effective_Rs = [copy(iDM.R) for j in 1:N-1]
  local_ener = 0
  err = 0
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

    #TODO assume 2 site idmrg. We follow Hubig logic, adapted for 2 sites
    theta = new_x[1]
    left_indices = commoninds(theta, iDM.ψ.AL[start] )
    newtags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start + 1])))

    if α != 0
      extension = noprime( ((current_L * theta) * iDM.Hmpo[start] ) * iDM.Hmpo[start+1] )
      extension = extension * (α / norm(extension))
      supp_index = only(commoninds(iDM.Hmpo[start+1], iDM.Hmpo[start+2]))
      dummy_index = Index(QN() => 1; dir=dir(supp_index))
      dum = ITensor(dummy_index); dum[1] = 1

      theta_extended, new_index = ITensors.directsum(
      theta  * dum => dummy_index,
      extension => supp_index;
      tags="Temporary",
      )
      closure = ITensor(new_index); closure[1] = 1;
      #println(norm(theta - theta_extended*dag(closure)))
      cc = combiner(new_index)
      U2, S2, V2 = svd(
      theta_extended*cc,
      left_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      V2 = V2 * dag(cc) * dag(closure)
      #temp_norm = norm(S2)#norm(U2*S2*V2) #TODO check
      #err = 1 - temp_norm
      #S2 = S2 / temp_norm
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
    #noise_term = noprime( current_L * theta * iDM.Hmpo[start] *iDM.Hmpo[start+1] )
    if α != 0
      extension = noprime( ((current_R * theta) * iDM.Hmpo[start+1] ) * iDM.Hmpo[start] )
      extension = extension * (α / norm(extension))
      supp_index = only(commoninds(iDM.Hmpo[start], iDM.Hmpo[start-1]))
      dummy_index = Index(QN() => 1; dir=dir(supp_index))
      dum = ITensor(dummy_index); dum[1] = 1

      theta_extended, new_index = ITensors.directsum(
      theta  * dum => dummy_index,
      extension => supp_index;
      tags="Temporary",
      )
      cc = combiner(new_index)
      closure = ITensor(new_index); closure[1] = 1;
      #println(norm(theta - theta_extended*dag(closure)))
      U2, S2, V2 = svd(
      theta_extended * cc,
      right_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      V2 = V2 * dag(cc) * dag(closure)
      #temp_norm = norm(S2)#norm(U2*S2*V2)
      #err = 1 - temp_norm
      #S2 = S2 / temp_norm
    else
      U2, S2, V2 = svd(
      theta,
      right_indices;
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
    if start == original_start
      theta = U2 * S2 * V2
      println(tr(dag(theta) * temp_H(theta)))
    end
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
    idmrg_step_with_noise_auxiliary(
      iDM; solver_tol, maxdim, cutoff, α, kwargs...
      )
  end

  local_ener, err =   idmrg_step_with_noise_auxiliary(iDM; solver_tol, maxdim, cutoff, α=0, kwargs...)

  original_start = mod1(iDM.counter, N)
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
