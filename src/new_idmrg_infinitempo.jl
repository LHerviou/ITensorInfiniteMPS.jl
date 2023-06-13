function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor)
  return L * Hmpo
end

function apply_mpomatrix_left(L::ITensor, Hmpo::ITensor, ψ::ITensor)
  ψp = dag(ψ)'
  return contract(L, ψ, Hmpo, ψp; sequence = "automatic")
end

apply_mpomatrix_right(R::ITensor, Hmpo::ITensor) = apply_mpomatrix_left(R, Hmpo)
function apply_mpomatrix_right(R::ITensor, Hmpo::ITensor, ψ::ITensor)
  return apply_mpomatrix_left(R, Hmpo, ψ)
end

function (H::temporaryHamiltonian{InfiniteMPO,ITensor})(x)
  n = order(x) - 2
  L = H.effectiveL
  R = H.effectiveR
  start = H.nref
  if n == 2
    #ITensors.enable_contraction_sequence_optimization()
    #return noprime(contract(L, x, H.Hmpo[start], H.Hmpo[start + 1], R; sequence = :automatic) )
    temp = contract(L, x, H.Hmpo[start], H.Hmpo[start+1], R; sequence = "automatic") #L * x * H.Hmpo[start] * H.Hmpo[start + 1] * R
    #ITensors.disable_contraction_sequence_optimization()
    noprime!(temp)
    return temp
  end
  temp = contract(L, x, [H.Hmpo[start + j] for j in 0:n-1]..., R; sequence = "automatic")
  noprime!(temp)
  return temp
end


function advance_environments(Hmpo::InfiniteMPO, ψ::InfiniteCanonicalMPS, L::ITensor, R::ITensor; left_position = 1)
  N = nsites(Hmpo)
  #Starting position default to 1
  for j in 0:(N - 1)
    L = apply_mpomatrix_left(L, Hmpo[left_position + j], ψ.AL[left_position + j])
  end
  L = translatecell(translator(H), L, -1)
  for j in reverse(0:(N - 1))
    R = apply_mpomatrix_right(R, Hmpo[left_position + j], ψ.AR[left_position + j])
  end
  R = translatecell(translator(H), R, 1)
  return L, R
end

#TODO continue cleaning
function idmrg_step_with_noise_auxiliary_halfhalf(Hmpo::InfiniteMPO, ψ::InfiniteCanonicalMPS, L::ITensor, R::ITensor; solver_tol=1e-8, maxdim=20, cutoff=1e-10, α = 0., starting_position = 1, kwargs...)
  issymmetric = get(kwargs, :issymmetric, true)
  eager = get(kwargs, :eager, true)
  H_extension = get(kwargs, :MPO_extension, Hmpo)
  reduced_left_env = get(kwargs, :reduced_left_env, nothing)
  reduced_right_env = get(kwargs, :reduced_right_env, nothing)

  N = nsites(Hmpo)
  mid_chain = get(kwargs, :mid_chain, div(N, 2))

  effective_Rs = [copy(R) for j in 1:mid_chain]
  effective_Ls = ITensor[]
  local_ener = 0
  err = 0
  s = siteinds(only, ψ)

  #Starting the left sweep
  #Building successive environments
  site_looked = starting_position + N - 1
  while site_looked > starting_position + mid_chain
    effective_Rs[end] = apply_mpomatrix_right(effective_Rs[end], Hmpo[site_looked], ψ.AR[site_looked])
    site_looked -= 1
  end
  for j in reverse(1:mid_chain-1)
    effective_Rs[j] = copy(effective_Rs[j + 1])
    effective_Rs[j] = apply_mpomatrix_right(effective_Rs[j], Hmpo[site_looked], ψ.AR[site_looked])
    site_looked -= 1
  end

  current_L = copy(L)
  for (count, start) in enumerate(starting_position:starting_position+mid_chain-1)
    starting_state = ψ.C[start-1] * ψ.AR[start] * ψ.AR[start + 1]
    #if α != 0
    #  starting_state += max(α, 1e-6) * randomITensor(inds(starting_state)) #ensures that every sector has some non zero elements
    #end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    current_L, effective_Rs[count], Hmpo, start
    )
    local_ener, new_x, info = eigsolve(
    temp_H, starting_state, 1, :SR; issymmetric, ishermitian = true, tol=solver_tol, eager
    )
    #println(info)
    theta = new_x[1]
    left_indices = [only(commoninds(theta,  ψ.C[start-1] )) , s[start]]
    right_indices = uniqueinds(theta, left_indices)
    newtags = tags(only(commoninds(ψ.AL[start], ψ.AL[start + 1])))
    if α != 0
      if H_extension == Hmpo
        env = current_L
      elseif !isnothing(reduced_left_env)
        env = reduced_left_env
      else
        left_link = dag(left_indices[1])
        left_link_mpo = only(commoninds(H_extension[start-1], H_extension[start]))
        env = randomITensor(left_link, dag(prime(left_link)), left_link_mpo)
      end
      U2, S2, V2 = subspace_expansion(theta, env, H_extension, (start, start+1); maxdim, cutoff, newtags, α, svd_indices = left_indices)
      S2 = ITensors.denseblocks(S2)
    else
      U2, S2, V2 = svd(
      theta,
      left_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      S2 = ITensors.denseblocks(S2)
    end
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)
    #Updating the tensors.
    ψ.AL[start] = U2
    ψ.C[start] = S2
    if count != mid_chain #Only update those we are not coming back to
      ψ.AR[start] = ortho_polar(U2 * S2, ψ.C[start - 1])
    end
    ψ.AR[start+1] = V2
    if count == mid_chain
      ψ.AL[start+1] = ortho_polar(S2*V2, ψ.C[start + 1])
    end
    #Advance the left environment as long as we have not reached the central point
    if count != mid_chain
      #append!(effective_Ls, [current_L])
      current_L = apply_mpomatrix_left(current_L, Hmpo[start], ψ.AL[start])
      if !isnothing(reduced_left_env)
        reduced_left_env = apply_mpomatrix_left(reduced_left_env, H_extension[start], ψ.AL[start])
      end
    end
  end
  #Now we start sweeping from right to left up to the center bond
  effective_Ls = [copy(current_L)]
  for j in mid_chain:N-2
    current_L = apply_mpomatrix_left(current_L, Hmpo[starting_position+j-1], ψ.AL[starting_position+j-1])
    append!(effective_Ls, [copy(current_L)])
  end
  current_R = copy(R)
  for (count, start) in reverse(collect(enumerate(starting_position+mid_chain-1:starting_position+N-2)))
    #build the local tensor start .... start + nb_site - 1
    starting_state = ψ.AL[start] * ψ.AL[start + 1] * ψ.C[start+1]
    #if α != 0
    #  starting_state += max(α, 1e-6) * randomITensor(inds(starting_state)) #ensures that every sector has some non zero elements
    #end
    #build_local_Hamiltonian
    temp_H = temporaryHamiltonian{InfiniteMPO,ITensor}(
    effective_Ls[count], current_R, Hmpo, start
    )
    local_ener, new_x = eigsolve(
    temp_H, starting_state, 1, :SR;  issymmetric, ishermitian = true, tol=solver_tol, eager
    )
    theta = new_x[1]
    left_indices = commoninds(theta,  ψ.AL[start])
    right_indices = uniqueinds(theta, left_indices)
    newtags = tags(only(commoninds(ψ.AL[start], ψ.AL[start + 1])))
    if α != 0
      if H_extension == Hmpo
          env = current_R
      elseif !isnothing(reduced_right_env)
        env = reduced_right_env
      else
        right_link = dag(only(uniqueinds(right_indices, s[start+1])))
        right_link_mpo = only(commoninds(H_extension[start+2], H_extension[start+1]))
        env = randomITensor(right_link, dag(prime(right_link)), right_link_mpo)
      end
      V2, S2, U2 = subspace_expansion(theta, env, H_extension, (start+1, start); maxdim, cutoff, newtags, α, svd_indices = right_indices)
      S2 = ITensors.denseblocks(S2)
      if count == 1
        if H_extension == Hmpo
          env = effective_Ls[count]
        else
          left_link = only(commoninds(ψ.AL[start-1], ψ.AL[start]))
          #left_link = dag(left_indices[1])
          left_link_mpo = only(commoninds(H_extension[start-1], H_extension[start]))
          env = randomITensor(left_link, dag(prime(left_link)), left_link_mpo)
        end
        U2, _, _ = subspace_expansion(theta, env, H_extension, (start, start+1); maxdim, cutoff, newtags, α, svd_indices = left_indices)
        S2 = ITensors.dropzeros(dag(U2) * theta * dag(V2)) #Here S2 is not diagonal. Instead, we guarantee are sure that the environments are both well normalized
      end
    else
      U2, S2, V2 = svd(
      theta,
      left_indices;
      maxdim=maxdim,
      cutoff=cutoff,
      lefttags=newtags,
      righttags=newtags,
      )
      S2 = ITensors.denseblocks(S2)
    end
    err = 1 - norm(S2)
    S2 = S2 / norm(S2)

    ψ.AL[start] = U2
    ψ.C[start] = S2#ITensors.denseblocks(S2)
    if count == 1 #Only update when we are not coming back
      ψ.AR[start] = ortho_polar(U2 * S2, ψ.C[start - 1])
    end
    ψ.AR[start+1] = V2
    ψ.AL[start+1] = ortho_polar(S2*V2, ψ.C[start + 1])
    #Advance the rightt environment as long as we are not finished
    current_R = apply_mpomatrix_right(current_R, Hmpo[start+1], ψ.AR[start+1])
    if !isnothing(reduced_right_env)
        reduced_right_env = apply_mpomatrix_right(reduced_right_env, H_extension[start+1], ψ.AL[start+1])
      end
    #end
  end
  current_L = apply_mpomatrix_left(effective_Ls[1], Hmpo[starting_position + mid_chain-1], ψ.AL[starting_position + mid_chain-1])
  if !isnothing(reduced_left_env)
    reduced_left_env = apply_mpomatrix_left(reduced_left_env, H_extension[starting_position + mid_chain-1], ψ.AL[starting_position + mid_chain-1])
    return local_ener, err, current_L, current_R, reduced_left_env, reduced_right_env
  end
  return local_ener, err, current_L, current_R
end


function idmrg_step_with_noise(
  Hmpo::InfiniteMPO, ψ::InfiniteCanonicalMPS, L::ITensor, R::ITensor; dmrg_sites = 2, starting_position = 1, kwargs...
)
  N = nsites(ψ)
  nb_site = dmrg_sites
  mid_chain = get(kwargs, :mid_chain, N÷2)
  reduced_left_env = get(kwargs, :reduced_left_env, nothing)

  if nb_site != 2
    error("Only 2 site DMRG is implemented")
  end
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end

  #A first double sweep with noise
  if !isnothing(reduced_left_env)
    local_ener, err, L, R, reduced_left, reduced_right = idmrg_step_with_noise_auxiliary_halfhalf(Hmpo, ψ, L, R; kwargs...)
  else
    local_ener, err, L, R = idmrg_step_with_noise_auxiliary_halfhalf(Hmpo, ψ, L, R; kwargs...)
  end

  tempL = ITensor(only(commoninds(L, Hmpo[starting_position+mid_chain-1])))
  tempL[1] = 1.0
  L -= local_ener[1] * δ(uniqueinds(L, Hmpo[starting_position+mid_chain-1])...) * tempL

  if starting_position + mid_chain >= N + 1
    L = translatecell(translator(ψ), L, -1)
  else
    R = translatecell(translator(ψ), R, 1)
  end

  if !isnothing(reduced_left_env)
    local_ener[1] / N , err, L, R, reduced_left, reduced_right
  end
  return local_ener[1] / N , err, L, R
end
