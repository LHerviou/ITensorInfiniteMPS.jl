
function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, dmrg_sites::Int64)
  N = nsites(ψ) #dmrg_sites
  l = only(commoninds(ψ.AL[0], ψ.AL[1]))
  r = only(commoninds(ψ.AR[N+1], ψ.AR[N]))
  l_mpo = only(commoninds(Hmpo[0], Hmpo[1]))
  r_mpo = only(commoninds(Hmpo[N+1], Hmpo[N]))
  tempL = ITensor(l_mpo); tempL[end] = 1.
  L = δ(l, prime(dag(l))) * tempL
  tempR = ITensor(r_mpo); tempR[1] = 1.
  R = δ(r, prime(dag(r))) * tempR

  return iDMRGStructure{InfiniteMPO, ITensor}(copy(ψ), Hmpo, L, R, 1, dmrg_sites);
end


iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO) = iDMRGStructure{InfiniteMPO, ITensor}(ψ, Hmpo, 2)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, L::ITensor, R::ITensor, dmrg_sites::Int64) = iDMRGStructure{InfiniteMPO, ITensor}(copy(ψ), Hmpo, L, R, 1, dmrg_sites)
iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPO, L::ITensor, R::ITensor) = iDMRGStructure{InfiniteMPO, ITensor}(copy(ψ), Hmpo, L, R, 1, 2)
iDMRGStructure(Hmpo::InfiniteMPO, ψ::InfiniteCanonicalMPS) = iDMRGStructure(ψ, Hmpo)


#test a preallocated temporaryHamiltonian
#RESULT: NO SPEEDING UP AS OF NOW

# struct temporaryHamiltonian_prealloc{Tmpo, Tenv}
#   effectiveL::Tenv
#   effectiveR::Tenv
#   Hmpo::Tmpo
#   nref::Int64 #leftmostsite
#   pre_allocated::Vector{ITensor}
# end
#
#
# function temporaryHamiltonian_prealloc{InfiniteMPO, ITensor}(L::ITensor, R::ITensor, Hmpo::InfiniteMPO, nref::Int64, x::ITensor)
#   n = order(x) - 2
#   pre_allocated = Vector{ITensor}()
#   start = nref
#   append!(pre_allocated, [randomITensor(uniqueinds(L, x)..., uniqueinds(x, L)...)])
#   for j in 0:n-1
#     append!(pre_allocated, [randomITensor(uniqueinds(pre_allocated[end], Hmpo[start+j])..., uniqueinds(Hmpo[start+j], pre_allocated[end])...)])
#   end
#   return temporaryHamiltonian_prealloc{InfiniteMPO, ITensor}(L, R, Hmpo, nref, pre_allocated)
# end






function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor)
  return L * Hmpo
end


function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor, ψ::ITensor)
  ψp = dag(ψ)'
  return ((L * ψ) * Hmpo ) * ψp
end

apply_mpomatrix_right(R::ITensor, Hmpo::ITensor) = apply_mpomatrix_left(R, Hmpo)
apply_mpomatrix_right(R::ITensor, Hmpo::ITensor, ψ::ITensor) = apply_mpomatrix_left(R, Hmpo, ψ)


function (H::iDMRGStructure{InfiniteMPO, ITensor})(x)
  n = order(x) - 2
  L = H.L
  R = H.R
  start = mod1(H.counter, nsites(H))
  L = L * x
  for j in 0:n-1
    L = apply_mpomatrix_left(L, H.Hmpo[start+j])
  end
  return noprime(L*R)
end


function (H::temporaryHamiltonian{InfiniteMPO, ITensor})(x)
  n = order(x) - 2
  L = H.effectiveL
  R = H.effectiveR
  start = H.nref
  L = L * x
  for j in 0:n-1
    L = apply_mpomatrix_left(L, H.Hmpo[start+j])
  end
  return noprime(L*R)
end

# function (H::temporaryHamiltonian_prealloc{InfiniteMPO, ITensor})(x)
#   n = order(x) - 2
#   L = H.effectiveL
#   R = H.effectiveR
#   start = H.nref
#   H.pre_allocated[1] .= L .* x
#   for j in 1:n
#     H.pre_allocated[j+1] .= H.pre_allocated[j] .* H.Hmpo[start+j-1]
#   end
#   return noprime(H.pre_allocated[end]*R)
# end


function advance_environments(H::iDMRGStructure{InfiniteMPO})
  N = nsites(H)
  nb_site = N#dmrg_sites(H)
  start = mod1(H.counter, N)
  for j in 0:N-1
    H.L = apply_mpomatrix_left(H.L, H.Hmpo[start+j], H.ψ.AL[start + j])
  end
  for j in 0:N-1
    H.R = apply_mpomatrix_right(H.R, H.Hmpo[start+nb_site-1-j], H.ψ.AR[start+nb_site-1-j])
  end
  H.R= translatecell(translator(H), H.R, 1)
  H.L= translatecell(translator(H), H.L, -1)
end


function set_environments_defaultposition(H::iDMRGStructure{InfiniteMPO, ITensor})
  N = nsites(H)
  if mod1(H.counter, N) == 1
    println("Already at the correct position")
    return 0
  end
  nb_steps = mod(1-H.counter, N)
  start = mod1(H.counter, N)
  for j in 0:nb_steps-1
    H.L = apply_mpomatrix_left(H.L, H.Hmpo[start+j], H.ψ.AL[start + j])
  end
  for j in reverse(start+nb_steps-N:start + N - 1)
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


function idmrg_step(iDM::iDMRGStructure{InfiniteMPO, ITensor}; solver_tol = 1e-8, maxdim = 20, cutoff = 1e-10)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end
  if nb_site == 1
    #return idmrg_step_single_site(iDM; solver_tol, cutoff)
    error("Not fully implemented")
  end
  if (N÷(nb_site÷2))*(nb_site÷2) != N
    error("We require that the (nb_site÷2) divides the unitcell length")
  end
  nbIterations =  (N - nb_site)÷(nb_site÷2) + 1#(N÷(nb_site÷2)) - 1
  original_start = mod1(iDM.counter, N)
  effective_Rs=[iDM.R for j in 1:nbIterations];
  local_ener=0
  err = 0
  site_looked = original_start + N-1
  for j in reverse(1:nbIterations-1)
      effective_Rs[j] = copy(effective_Rs[j+1])
      for k in 0:nb_site÷2-1
        effective_Rs[j] = apply_mpomatrix_right(effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked])
        site_looked -= 1
      end
  end
  adjust_left = 0
  adjust_right_most = 0
  start = original_start
  current_L = copy(iDM.L)
  for count in 1:nbIterations
    #build the local tensor start .... start + nb_site - 1
    starting_state = iDM.ψ.AL[start] * iDM.ψ.C[start] * iDM.ψ.AR[start+1]
    for j = 3:nb_site
      starting_state *= iDM.ψ.AR[start+j-1]
    end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO, ITensor}(current_L, effective_Rs[count], iDM.Hmpo, start);
    #temp_H = temporaryHamiltonian_prealloc{InfiniteMPO, ITensor}(current_L, effective_Rs[count], iDM.Hmpo, start, starting_state);
    local_ener, new_x = eigsolve(temp_H, starting_state, 1, :SR; ishermitian=true, tol=solver_tol);
    U2, S2, V2 = svd(new_x[1], commoninds(new_x[1], iDM.ψ.AL[start]); maxdim=maxdim, cutoff=cutoff, lefttags = tags(only(commoninds(iDM.ψ.AL[start], iDM.ψ.AL[start+1]))),
    righttags = tags(only(commoninds(iDM.ψ.AR[start+1], iDM.ψ.AR[start]))))
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
    iDM.ψ.AL[start] = U2
    temp_R, temp_C = diag_ortho_polar_both(U2 * S2, iDM.ψ.C[start-1])
    if count == 1 #&& nbIterations > 1
      adjust_right_most = translatecell(translator(iDM), wδ(only(commoninds(iDM.ψ.AR[start], iDM.ψ.AR[start-1])), only(commoninds(temp_C, temp_R))), 1)
      effective_Rs[end] *= dag(adjust_right_most) #Also modify iDM.R
      iDM.R *= dag(adjust_right_most)
      effective_Rs[end] *= prime(adjust_right_most)
      iDM.R *= prime(adjust_right_most)
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

    if count != nbIterations
      for j in 1:nb_site÷2
        if j == 1 && nbIterations == 1
          current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start+j-1], translatecell(translator(iDM), dag(adjust_left), -1)*iDM.ψ.AL[start+j-1])
        else
          current_L = apply_mpomatrix_left(current_L, iDM.Hmpo[start+j-1], iDM.ψ.AL[start+j-1])
        end
        #iDM.L[1] -= local_ener[1]/N * denseblocks(δ(inds(iDM.L[1])...))
      end
      start += nb_site÷2
    end
  end
  for j in 1:N÷2
    if j == 1
      iDM.L = apply_mpomatrix_left(iDM.L, iDM.Hmpo[original_start+j-1], translatecell(translator(iDM), dag(adjust_left), -1)*iDM.ψ.AL[original_start+j-1])
    else
      iDM.L = apply_mpomatrix_left(iDM.L, iDM.Hmpo[original_start+j-1], iDM.ψ.AL[original_start+j-1])
    end
    tempL = ITensor(only(commoninds(iDM.L, iDM.Hmpo[original_start+j]))); tempL[1] = 1.0
    iDM.L -= local_ener[1]/N * δ(uniqueinds(iDM.L, iDM.Hmpo[original_start+j])...) * tempL
  end
  for j in reverse(N÷2+1:N)#reverse((N-nb_site + nb_site÷2 + 1):N)
    if j == N && length(commoninds(iDM.R, iDM.ψ.AR[original_start+j-1]))==0
      error("Should not pop up")
    end
    iDM.R = apply_mpomatrix_right(iDM.R, iDM.Hmpo[original_start+j-1], iDM.ψ.AR[original_start+j-1])
    tempR = ITensor(only(commoninds(iDM.R, iDM.Hmpo[original_start+j-2]))); tempR[end] = 1.0
    iDM.R -= local_ener[1]/N * δ(uniqueinds(iDM.R, iDM.Hmpo[original_start+j-2])...) * tempR
    #iDM.R[end] -= local_ener[1]/N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if original_start + N÷2 >= N+1
    iDM.L = translatecell(translator(iDM), iDM.L, -1)
  else
    iDM.R = translatecell(translator(iDM), iDM.R, 1)
  end

  iDM.counter += N÷2
  return local_ener[1]/N, err
end
