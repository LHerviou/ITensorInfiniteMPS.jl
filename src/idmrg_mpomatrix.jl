mutable struct iDMRGStructure{Tmpo,Tenv}
  ψ::InfiniteCanonicalMPS
  Hmpo::Tmpo
  L::Tenv
  R::Tenv
  counter::Int64
  dmrg_sites::Int64
end
translator(IDM::iDMRGStructure) = translator(IDM.ψ)
nsites(IDM::iDMRGStructure) = nsites(IDM.ψ)
dmrg_sites(IDM::iDMRGStructure) = IDM.dmrg_sites
function Base.copy(iDM::iDMRGStructure)
  return iDMRGStructure{typeof(iDM.Hmpo),typeof(iDM.L)}(
    copy(iDM.ψ), iDM.Hmpo, copy(iDM.L), copy(iDM.R), iDM.counter, iDM.dmrg_sites
  )
end

struct temporaryHamiltonian{Tmpo,Tenv}
  effectiveL::Tenv
  effectiveR::Tenv
  Hmpo::Tmpo
  nref::Int64 #leftmostsite
end


function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, dmrg_sites::Int64)
  N = nsites(ψ) #dmrg_sites
  l = only(commoninds(ψ.AL[0], ψ.AL[1]))
  r = only(commoninds(ψ.AR[N + 1], ψ.AR[N]))
  L = [ITensor(l, prime(dag(l))) for j in 1:size(Hmpo[1])[1]]
  for j in 2:(length(L) - 1)
    for k in 1:j
      temp_l = filterinds(Hmpo[1][j, k]; tags="Link")
      for ind in temp_l
        if ind.dir == ITensors.In
          L[j] = ITensor(l, prime(dag(l)), dag(ind))
        end
      end
    end
  end
  R = [ITensor(r, prime(dag(r))) for j in 1:size(Hmpo[N])[2]]
  for j in 2:(length(R) - 1)
    for k in j:length(R)
      temp_r = filterinds(Hmpo[N][k, j]; tags="Link")
      for ind in temp_r
        if ind.dir == ITensors.Out
          R[j] = ITensor(r, prime(dag(r)), dag(ind))
        end
      end
    end
  end
  L[end] = δ(l, prime(dag(l)))
  R[1] = δ(r, prime(dag(r)))
  return iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}(
    copy(ψ), Hmpo, L, R, 1, dmrg_sites
  )
end

function iDMRGStructure(ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix)
  return iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}(ψ, Hmpo, 2)
end
function iDMRGStructure(
  ψ::InfiniteCanonicalMPS,
  Hmpo::InfiniteMPOMatrix,
  L::Vector{ITensor},
  R::Vector{ITensor},
  dmrg_sites::Int64,
)
  return iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}(
    copy(ψ), Hmpo, L, R, 1, dmrg_sites
  )
end
function iDMRGStructure(
  ψ::InfiniteCanonicalMPS, Hmpo::InfiniteMPOMatrix, L::Vector{ITensor}, R::Vector{ITensor}
)
  return iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}(copy(ψ), Hmpo, L, R, 1, 2)
end
iDMRGStructure(Hmpo::InfiniteMPOMatrix, ψ::InfiniteCanonicalMPS) = iDMRGStructure(ψ, Hmpo)

function apply_mpomatrix_left!(L::Vector{ITensor}, Hmpo::Matrix{ITensor})
  init = [false for j in 1:length(L)]
  for j in 1:length(L)
    for k in 1:j
      if isempty(L[j]) || isempty(Hmpo[j, k])
        continue
      end
      if !init[k] || isempty(L[k]) || k == j
        L[k] = L[j] * Hmpo[j, k]
        init[k] = true
      else
        L[k] .+= L[j] * Hmpo[j, k]
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
  init = [false for j in 1:length(L)]
  for j in 1:length(L)
    for k in 1:j
      if isempty(L[j]) || isempty(Hmpo[j, k])
        continue
      end
      if !init[k] || isempty(L[k]) || k == j
        L[k] = ((L[j] * ψ) * Hmpo[j, k]) * ψp
        init[k] = true
      else
        L[k] .+= ((L[j] * ψ) * Hmpo[j, k]) * ψp
      end
    end
  end
  for j in 1:length(L)
    if !init[j]
      L[j] = ITensor(inds(L[j])...) * ψ * Hmpo[j, j] * ψp
    end
  end
end

function apply_mpomatrix_right!(R::Vector{ITensor}, Hmpo::Matrix{ITensor})
  init = [false for j in 1:length(R)]
  for j in reverse(1:length(R))
    for k in reverse(j:length(R))
      if isempty(R[j]) || isempty(Hmpo[k, j])
        continue
      end
      if !init[k] || isempty(R[k]) || k == j
        R[k] = Hmpo[k, j] * R[j]
        init[k] = true
      else
        R[k] .+= Hmpo[k, j] * R[j]
      end
    end
  end
  for j in 1:length(R)
    if !init[j]
      R[j] = Hmpo[j, j] * ITensor(inds(R[j])...)
    end
  end
end

function apply_mpomatrix_right!(R::Vector{ITensor}, Hmpo::Matrix{ITensor}, ψ::ITensor)
  ψp = dag(ψ)'
  init = [false for j in 1:length(R)]
  for j in reverse(1:length(R))
    for k in reverse(j:length(R))
      if isempty(R[j]) || isempty(Hmpo[k, j])
        continue
      end
      if !init[k] || isempty(R[k]) || k == j
        R[k] = ((R[j] * ψ) * Hmpo[k, j]) * ψp
        init[k] = true
      else
        R[k] .+= ((R[j] * ψ) * Hmpo[k, j]) * ψp
      end
    end
  end
  for j in 1:length(R)
    if !init[j]
      R[j] = ITensor(inds(R[j])...) * ψ * Hmpo[j, j] * ψp
    end
  end
end

#TODO optimize
function (H::iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}})(x)
  n = order(x) - 2
  L = H.L
  R = H.R
  start = mod1(H.counter, nsites(H))
  L = [L[j] * x for j in 1:length(L)]
  for j in 0:(n - 1)
    apply_mpomatrix_left!(L, H.Hmpo[start + j])
  end
  result = L[1] * R[1]
  for j in 2:length(L)
    result .+= L[j] * R[j]
  end
  return noprime(result)
end

function (H::temporaryHamiltonian{InfiniteMPOMatrix,Vector{ITensor}})(x)
  n = order(x) - 2
  L = [H.effectiveL[j] * x for j in 1:length(H.effectiveL)]
  for j in 0:(n - 1)
    apply_mpomatrix_left!(L, H.Hmpo[H.nref + j])
  end
  result = L[1] * H.effectiveR[1]
  for j in 2:length(L)
    result .+= L[j] * H.effectiveR[j]
  end
  return noprime(result)
end

function advance_environments(H::iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}})
  N = nsites(H)
  nb_site = N#dmrg_sites(H)
  start = mod1(H.counter, N)
  for j in 0:(N - 1)
    apply_mpomatrix_left!(H.L, H.Hmpo[start + j], H.ψ.AL[start + j])
  end
  for j in 0:(N - 1)
    apply_mpomatrix_right!(
      H.R, H.Hmpo[start + nb_site - 1 - j], H.ψ.AR[start + nb_site - 1 - j]
    )
  end
  for j in 1:length(H.R)
    H.R[j] = translatecell(translator(H), H.R[j], 1)
  end
  for j in 1:length(H.R)
    H.L[j] = translatecell(translator(H), H.L[j], -1)
  end
end

function advance_environments(H::iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}, n::Int64)
  for j in 1:n
    advance_environments(H)
  end
end


function set_environments_defaultposition(
  H::iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}}
)
  N = nsites(H)
  if mod1(H.counter, N) == 1
    println("Already at the correct position")
    return 0
  end
  nb_steps = mod(1 - H.counter, N)
  start = mod1(H.counter, N)
  for j in 0:(nb_steps - 1)
    apply_mpomatrix_left!(H.L, H.Hmpo[start + j], H.ψ.AL[start + j])
  end
  for j in reverse((start + nb_steps - N):(start + N - 1))
    apply_mpomatrix_right!(H.R, H.Hmpo[j], H.ψ.AR[j])
  end
  shift_cell = getcell(inds(H.L[1])[1]) - 0
  if shift_cell != 0
    for j in 1:length(H.L)
      H.L[j] = translatecell(translator(H), H.L[j], -shift_cell)
    end
  end
  shift_cell = getcell(inds(H.R[1])[1]) - 1
  if shift_cell != 0
    for j in 1:length(H.R)
      H.R[j] = translatecell(translator(H), H.R[j], -shift_cell)
    end
  end
  H.counter += nb_steps
  return 0
end

function idmrg_step(
  iDM::iDMRGStructure{InfiniteMPOMatrix,Vector{ITensor}};
  solver_tol=1e-8,
  maxdim=20,
  cutoff=1e-10,
)
  N = nsites(iDM)
  nb_site = dmrg_sites(iDM)
  if nb_site > N
    error("iDMRG with a step size larger than the unit cell has not been implemented")
  end
  if nb_site == 1
    #return idmrg_step_single_site(iDM; solver_tol, cutoff)
    error("Single site idmrg not implemented")
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
      apply_mpomatrix_right!(effective_Rs[j], iDM.Hmpo[site_looked], iDM.ψ.AR[site_looked])
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
    temp_H = temporaryHamiltonian{InfiniteMPOMatrix,Vector{ITensor}}(
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
    if count == nbIterations
      starting_state = iDM.ψ.AL[start] * iDM.ψ.AL[start + 1] * iDM.ψ.C[start+1]
      for j in 3:nb_site
        starting_state *= iDM.ψ.AR[start + j - 1]
      end
      #println((dag(starting_state) * temp_H(starting_state))[1] / N)
    end
    #Advance the left environment as long as we are not finished
    if count != nbIterations
      for j in 1:(nb_site ÷ 2)
        apply_mpomatrix_left!(current_L, iDM.Hmpo[start + j - 1], iDM.ψ.AL[start + j - 1])
      end
      start += nb_site ÷ 2
    end

  end
  #By convention, we choose to advance half the unit cell
  for j in 1:(N ÷ 2)
    apply_mpomatrix_left!(iDM.L, iDM.Hmpo[original_start + j - 1], iDM.ψ.AL[original_start + j - 1])
    iDM.L[1] -= local_ener[1] / N * denseblocks(δ(inds(iDM.L[1])...))
  end
  for j in reverse((N ÷ 2 + 1):N)#reverse((N-nb_site + nb_site÷2 + 1):N)
    apply_mpomatrix_right!(
      iDM.R, iDM.Hmpo[original_start + j - 1], iDM.ψ.AR[original_start + j - 1]
    )
    iDM.R[end] -= local_ener[1] / N * denseblocks(δ(inds(iDM.R[end])...))
  end
  if original_start + N ÷ 2 >= N + 1
    for j in 1:length(iDM.L)
      iDM.L[j] = translatecell(translator(iDM), iDM.L[j], -1)
    end
  else
    for j in 1:length(iDM.R)
      iDM.R[j] = translatecell(translator(iDM), iDM.R[j], 1)
    end
  end
  iDM.counter += N ÷ 2
  return local_ener[1] / N, err
end

function idmrg(
  iDM::iDMRGStructure;
  nb_iterations=10,
  output_level=0,
  mixer=false,
  ener_tol=0,
  α=0.001,
  measure_entropies = false,
  kwargs...,
)
  eners = Float64[]
  errs = Float64[]
  entrs = Float64[]
  ener = 0
  err = 0
  for j in 1:nb_iterations
    if !mixer
      ener, err = idmrg_step(iDM; kwargs...)
    else
      mid_chain = get(kwargs, :mid_chain, nsites(iDM)÷2) #TODO think if this is the best way
      ener, err = idmrg_step_with_noise(iDM; α = α, mid_chain,  kwargs...)
      ener, err = idmrg_step_with_noise(iDM; α = α, kwargs..., mid_chain = nsites(iDM)-mid_chain)
    end
    append!(eners, ener)
    append!(errs, err)
    if measure_entropies
      append!(entrs, sum(entropies(iDM.ψ)) / nsites(iDM) )
    end
    if j > 5
      if maximum([abs(eners[end - j + 1] - eners[end - j]) for j in 1:5]) < ener_tol
        println("Early finish")
        return eners, errs, entrs
      end
    end

    if output_level == 1
      println("Energy after iteration $j is $ener")
      flush(stdout)
      flush(stderr)
    end
  end
  return eners, errs, entrs
end

function test_validity_imps(ψ::InfiniteCanonicalMPS; prec=1e-8)
  for j in 1:nsites(ψ)
    @assert norm(ψ.AL[j] * ψ.C[j] - ψ.C[j - 1] * ψ.AR[j]) < prec
  end
end

struct effectiveHam
  H::ITensor
end

function (H::effectiveHam)(x)
  return noprime(H.H * x)
end

function build_two_local_hamiltonian(Hl, Hr, L, R)
  tempL = copy(L)
  apply_mpomatrix_left!(tempL, Hl)
  apply_mpomatrix_left!(tempL, Hr)
  temp = tempL[1] * R[1]
  for j in 2:length(tempL)
    temp += tempL[j] * R[j]
  end
  return effectiveHam(temp)
end

function evaluate_unitarity(ψ::InfiniteCanonicalMPS)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  println("Doing left")
  for j in 1:nsites(ψ)
    println(
      norm(
        δ(l[j - 1], prime(dag(l[j - 1]))) *
        ψ.AL[j] *
        δ(dag(s[j]), prime(s[j])) *
        dag(prime(ψ.AL[j])) - denseblocks(δ(l[j], prime(dag(l[j])))),
      ),
    )
  end
  println("Doing right")
  for j in 1:nsites(ψ)
    println(
      norm(
        δ(dag(r[j]), prime(r[j])) *
        ψ.AR[j] *
        δ(dag(s[j]), prime(s[j])) *
        dag(prime(ψ.AR[j])) - denseblocks(δ(dag(r[j - 1]), prime(r[j - 1]))),
      ),
    )
  end
end

function evaluate_unitarity_left(ψ::InfiniteCanonicalMPS, j::Int64)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  println(
    norm(
      δ(l[j - 1], prime(dag(l[j - 1]))) *
      ψ.AL[j] *
      δ(dag(s[j]), prime(s[j])) *
      dag(prime(ψ.AL[j])) - denseblocks(δ(l[j], prime(dag(l[j])))),
    ),
  )
  return δ(l[j - 1], prime(dag(l[j - 1]))) *
         ψ.AL[j] *
         δ(dag(s[j]), prime(s[j])) *
         dag(prime(ψ.AL[j]))
end

function evaluate_unitarity_right(ψ::InfiniteCanonicalMPS, j::Int64)
  s = siteinds(only, ψ)
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  return println(
    norm(
      δ(dag(r[j]), prime(r[j])) *
      ψ.AR[j] *
      δ(dag(s[j]), prime(s[j])) *
      dag(prime(ψ.AR[j])) - denseblocks(δ(dag(r[j - 1]), prime(r[j - 1]))),
    ),
  )
end
