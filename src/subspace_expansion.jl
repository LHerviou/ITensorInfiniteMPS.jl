function replaceind_indval(IV::Tuple, iĩ::Pair)
  i, ĩ = iĩ
  return ntuple(n -> first(IV[n]) == i ? ĩ => last(IV[n]) : IV[n], length(IV))
end

function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteSum{MPO}, b::Tuple{Int,Int}; atol=1e-2
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  range_H = nrange(ψ, H[1])

  @assert range_H > 1 "Subspace expansion for InfiniteSum{MPO} is not defined for purely local Hamiltonians"

  if range_H > 2
    ψᴴ = dag(ψ)
    ψ′ = prime(ψᴴ)
  end

  if range_H == 2
    ψH2 = (ψ.AL[n1] * ψ.C[n1] * H[n1][1]) * (ψ.AR[n2] * H[n1][2])
    ψH2 = noprime(ψH2)
  else   # Should be a better version now
    ψH2 =
      ψ.AR[n2 + range_H - 2] * H[n1][end] * (ψ′.AR[n2 + range_H - 2] * δʳ(n2 + range_H - 2))
    common_sites = findsites(ψ, H[(n1, n2)])
    idx = length(common_sites) - 1
    for j in reverse(1:(range_H - 3))
      if n2 + j == common_sites[idx]
        ψH2 = ψH2 * ψ.AR[n2 + j] * H[n1][idx] * ψ′.AR[n2 + j]
        idx -= 1
      else
        ψH2 = ψH2 * ψ.AR[n2 + j] * (δˢ(n2 + j) * ψ′.AR[n2 + j])
      end
    end
    if common_sites[idx] == n2
      ψH2 = ψH2 * ψ.AR[n2] * H[n1][idx]
      idx -= 1
    else
      ψH2 = ψH2 * (ψ.AR[n2] * δˢ(n2))
    end
    if common_sites[idx] == n1
      ψH2 = ψH2 * (ψ.AL[n1] * ψ.C[n1]) * H[n1][idx]
      idx -= 1
    else
      ψH2 = ψH2 * ((ψ.AL[n1] * δˢ(n1)) * ψ.C[n1])
    end

    ψH2 = noprime(ψH2)
    for n in 1:(range_H - 2)
      temp_H2_right = δʳ(n2 + range_H - 2 - n)
      common_sites = findsites(ψ, H[n1 - n])
      idx = length(common_sites)
      for j in (n2 + range_H - 2 - n):-1:(n2 + 1)
        if j == common_sites[idx]
          temp_H2_right = temp_H2_right * ψ.AR[j] * H[n1 - n][idx] * ψ′.AR[j]
          idx -= 1
        else
          temp_H2_right = temp_H2_right * ψ.AR[j] * (δˢ(j) * ψ′.AR[j])
        end
      end
      if common_sites[idx] == n2
        temp_H2_right = temp_H2_right * ψ.AR[n2] * H[n1 - n][idx]
        idx -= 1
      else
        temp_H2_right = temp_H2_right * ψ.AR[n2] * δˢ(n2)
      end
      if common_sites[idx] == n1
        temp_H2_right = temp_H2_right * (ψ.AL[n1] * ψ.C[n1]) * H[n1 - n][idx]
        idx -= 1
      else
        temp_H2_right = temp_H2_right * ((ψ.AL[n1] * δˢ(n1)) * ψ.C[n1])
      end
      idx = n - idx + 1
      temp_H2_left = δˡ(n1 - n - 1)
      for j in reverse(1:n)
        if n1 - j == common_sites[idx]
          temp_H2_left = temp_H2_left * ψ.AL[n1 - j] * H[n1 - n][idx] * ψ′.AL[n1 - j]
          idx += 1
        else
          temp_H2_left = temp_H2_left * (ψ.AL[n1 - j] * δˢ(n1 - j)) * ψ′.AL[n1 - j]
        end
      end
      ψH2 = ψH2 + noprime(temp_H2_left * temp_H2_right)
    end
  end
  return ψH2
end

function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteSum{ITensor}, b::Tuple{Int,Int}; atol=1e-2
)
  n1, n2 = b
  lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  rⁿ¹ = commoninds(ψ.AR[n2], ψ.C[n1])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  range_H = nrange(H, 1)
  @assert range_H == 2 "Subspace expansion for InfiniteSum{ITensor} currently only works for 2-local Hamiltonians"

  if range_H == 2
    ψH2 = noprime(ψ.AL[n1] * H[n1] * ψ.C[n1] * ψ.AR[n2])
  end
  return ψH2
end

#Technically not exactly the nullspace subspace expansion for more than two states
function generate_twobody_nullspace(
  ψ::InfiniteCanonicalMPS, H::InfiniteMPOMatrix, b::Tuple{Int,Int}; atol=1e-2
)
  n_1, n_2 = b
  nbsite = n_2 - n_1
  pivot_site = n_1 + div(nbsite, 2)
  L, _ = left_environment(H, ψ)
  R, _ = right_environment(H, ψ)


  dₕ = length(L[n_1 - 1])
  temp_L = similar(L[n_1 - 1])
  for i in 1:dₕ
    non_empty_idx = dₕ
    while isempty(H[n_1][non_empty_idx, i]) && non_empty_idx >= i
      non_empty_idx -= 1
    end
    @assert non_empty_idx != i - 1 "Empty MPO"
    temp_L[i] =
      L[n_1 - 1][non_empty_idx] * (ψ.AL[n_1] * ψ.C[n_1]) * H[n_1][non_empty_idx, i]
    for j in reverse(i:(non_empty_idx - 1))
      if !isempty(H[n_1][j, i])
        temp_L[i] += L[n_1 - 1][j] * (ψ.AL[n_1] * ψ.C[n_1]) * H[n_1][j, i]
      end
    end
  end
  for x in (n_1 + 1):pivot_site
    new_temp_L = similar(temp_L)
    for i in 1:dₕ
      non_empty_idx = dₕ
      while isempty(H[x][non_empty_idx, i]) && non_empty_idx >= i
        non_empty_idx -= 1
      end
      @assert non_empty_idx != i - 1 "Empty MPO"
      new_temp_L[i] =
        temp_L[non_empty_idx] * ψ.AR[x]  * H[x][non_empty_idx, i]
      for j in reverse(i:(non_empty_idx - 1))
        if !isempty(H[x][j, i])
          new_temp_L[i] += temp_L[j] * ψ.AR[x] * H[x][j, i]
        end
      end
    end
    temp_L = new_temp_L
  end
  temp_R = similar(R[n_2 + 1])
  for i in 1:dₕ
    non_empty_idx = 1
    while isempty(H[n_2][i, non_empty_idx]) && non_empty_idx <= i
      non_empty_idx += 1
    end
    @assert non_empty_idx != i + 1 "Empty MPO"
    temp_R[i] = H[n_2][i, non_empty_idx] * (ψ.AR[n_2] * R[n_2 + 1][non_empty_idx])
    for j in (non_empty_idx + 1):i
      if !isempty(H[n_2][i, j])
        temp_R[i] += H[n_2][i, j] * (ψ.AR[n_2] * R[n_2+1][j])
      end
    end
  end
  for x in reverse(pivot_site+1:n_2-1)
    new_temp_R = similar(temp_R)
    for i in 1:dₕ
      non_empty_idx = 1
      while isempty(H[x][i, non_empty_idx]) && non_empty_idx <= i
        non_empty_idx += 1
      end
      @assert non_empty_idx != i + 1 "Empty MPO"
      new_temp_R[i] = H[x][i, non_empty_idx] * (ψ.AR[x] * temp_R[non_empty_idx])
      for j in (non_empty_idx + 1):i
        if !isempty(H[x][i, j])
          new_temp_R[i] += H[x][i, j] * (ψ.AR[x] * temp_R[j])
        end
      end
    end
    temp_R = new_temp_R
  end
  ψH2 = temp_L[1] * temp_R[1]
  for j in 2:dₕ
    ψH2 += temp_L[j] * temp_R[j]
  end
  return noprime(ψH2)
end

function subspace_expansion(
  ψ::InfiniteCanonicalMPS, H, b::Tuple{Int,Int}; maxdim, cutoff, atol=1e-2, kwargs...
)
  n1, n2 = b
  nbsite = n2 - n1
  if nbsite > nsites(ψ) + 1
    println("Subspace expansion not implemented if the range of the expansion is larger than nsites + 1")
    flush(stdout)
    flush(stderr)
    return (ψ.AL[n1:n2], ψ.C[n1:n2-1], ψ.AR[n1:n2...])
  end

  pivot_site = n1 + div(nbsite, 2)
  #lⁿ¹ = commoninds(ψ.AL[n1], ψ.C[n1])
  #rⁿ¹ = commoninds(ψ.AR[n1+1], ψ.C[n1])
  lⁿ¹ = commoninds(ψ.AL[pivot_site], ψ.C[pivot_site])
  rⁿ¹ = commoninds(ψ.AR[pivot_site+1], ψ.C[pivot_site])
  l = linkinds(only, ψ.AL)
  r = linkinds(only, ψ.AR)
  s = siteinds(only, ψ)
  δʳ(n) = δ(dag(r[n]), prime(r[n]))
  δˢ(n) = δ(dag(s[n]), prime(s[n]))
  δˡ(n) = δ(l[n], dag(prime(l[n])))

  dˡ = dim(lⁿ¹)
  dʳ = dim(rⁿ¹)
  @assert dˡ == dʳ
  if dˡ ≥ maxdim
    println(
      "Current bond dimension at bond $b is $dˡ while desired maximum dimension is $maxdim, skipping bond dimension increase at $b",
    )
    flush(stdout)
    flush(stderr)
    return (ψ.AL[n1:n2], ψ.C[n1:n2-1], ψ.AR[n1:n2])
  end
  maxdim -= dˡ

    temp_left = ψ.AL[n1]
    for j in n1+1:pivot_site
      temp_left *= ψ.AL[j]
    end
    NL = nullspace(temp_left, lⁿ¹; atol=atol, tags = "leftnullspace")
    temp_right = ψ.AR[n2]
    for j in n2-1:-1:pivot_site+1
      temp_right *= ψ.AR[j]
    end
    NR = nullspace(temp_right, rⁿ¹; atol=atol, tags = "rightnullspace")
    nL = only(filterinds(NL, tags =  "leftnullspace"))
    nR = only(filterinds(NR, tags =  "rightnullspace"))

  # NL = nullspace(ψ.AL[n1], lⁿ¹; atol=atol, tags = "leftnullspace")
  # temp_right = ψ.AR[n2]
  # for j in n2-1:-1:n1+1
  #   temp_right *= ψ.AR[j]
  # end
  # NR = nullspace(temp_right, rⁿ¹; atol=atol, tags = "rightnullspace")
  # nL = only(filterinds(NL, tags =  "leftnullspace"))
  # nR = only(filterinds(NR, tags =  "rightnullspace"))

  ψHN2 = generate_twobody_nullspace(ψ, H, b; atol=atol)  * NL * NR
  #Added due to crash during testing
  if norm(ψHN2.tensor) < 1e-12
    println(
      "The two-site subspace expansion produced a zero-norm expansion at $b. This is likely due to the long-range nature of the QN conserving Hamiltonian.",
    )
    flush(stdout)
    flush(stderr)
    return (ψ.AL[n1:n2], ψ.C[n1:n2-1], ψ.AR[n1:n2])
  end

  U, S, V = svd(ψHN2, nL; maxdim=maxdim, cutoff=cutoff, lefttags = "Right", righttags = "Left", kwargs...)
  if dim(S) == 0 #Crash before reaching this point
    return (ψ.AL[n1:n2], ψ.C[n1:n2-1], ψ.AR[n1:n2])
  end

  U, S, V = svd(dag(NL)*U*S*V*dag(NR), commoninds(ψ.AL[n1], NL); maxdim=maxdim, cutoff=cutoff, lefttags = "Right", righttags = "Left", kwargs...)


  #@show S[end, end]
  ALs = Vector{ITensor}(undef, n2-n1+1)
  ARs = Vector{ITensor}(undef, n2-n1+1)
  #U *= dag(NL); V *= dag(NR)
  #Us = Vector{ITensor}(undef, n2-n1); Us[1] = U
  ALs[1], newl = ITensors.directsum(
      ψ.AL[n1] => uniqueinds(ψ.AL[n1], NL),
      U => uniqueinds(U, ψ.AL[n1]);
      tags=("Right",),
    )


  for j in n1+1:n2-1
    new_U, new_S, V = svd(S*V, (only(uniqueinds(S, V)), s[j]); maxdim=maxdim, cutoff=cutoff, lefttags = "Right", righttags = "Left", kwargs...)
    #Needs ordering indices
    or = uniqueinds(ψ.AL[j], new_U)
    new = uniqueinds(new_U, ψ.AL[j])
      if dir(or[1]) != ITensors.Out
        or = reverse(or)
      end
      if dir(or[1]) != dir(new[1])
        new = reverse(new)
      end
      ALs[j-n1+1], newr  = ITensors.directsum(
              ψ.AL[j] => or,
              new_U => new;
              tags=("Left", "Right",),
            )
      smallAR = ortho_polar(new_U*new_S, denseblocks(S))
      or = uniqueinds(ψ.AR[j], smallAR)
      new = uniqueinds(smallAR, ψ.AR[j])
        if dir(or[1]) != ITensors.In
          or = reverse(or)
        end
        if dir(or[1]) != dir(new[1])
          new = reverse(new)
        end
        ARs[j-n1+1], newr  = ITensors.directsum(
                ψ.AR[j] => or,
                smallAR => new;
                tags=("Left", "Right",),
              )
       S = new_S
    end
  ARs[n2-n1+1], = ITensors.directsum(
    ψ.AR[n2] => uniqueinds(ψ.AR[n2], V),
    V => uniqueinds(V, ψ.AR[n2]);
    tags=("Left",),
  )

  Cs = Vector{ITensor}(undef, n2-n1)
  #Start from the left
  for x in n1:n2-1
    newl = only(filterinds(ALs[x-n1+1], tags = "Right"))
    newr = only(filterinds(ARs[x-n1+2], tags = "Left"))
    Cs[x-n1+1] = ITensor(dag(newl), dag(newr))
    ψCⁿ¹ = permute(ψ.C[x], commoninds(ψ.C[x], ψ.AL[x])..., uniqueinds(ψ.C[x], ψ.AL[x])...)
    for I in eachindex(ψCⁿ¹)
      v = ψCⁿ¹[I]
      if !iszero(v)
        Cs[x-n1+1][I] = ψCⁿ¹[I] #Is it still this one for n site?
      end
    end
  end
  #Using otho polar to fix the edges
  ARs[1] = ortho_polar(ALs[1]*Cs[1], ψ.C[n1-1])
  ALs[end] = ortho_polar(Cs[end]*ARs[end], ψ.C[n2])

  #Special case if the range of the perturbation is the same as the range of AR
  if nsites(ψ) == nbsite
    newr = commoninds(ALs[1], Cs[1])
    if nsites(ψ) > 1
      newl = commoninds(ALs[end], Cs[end])
      newAL = ITensor(translatecell(translator(ψ), newl..., -1), s[n1], newr...)
    else
      newAL = ITensor(translatecell(translator(ψ), dag(prime(newr))..., -1), s[n1], newr...)
    end
    tempAL = permute(ALs[1], commoninds(ALs[1], ψ.AL[n1-1])..., s[n1], newr...)
    for I in eachindex(tempAL)
      v = tempAL[I]
      if !iszero(v)
        newAL[I] = tempAL[I]
      end
    end
    ALs[1] = newAL
    newl = commoninds(ARs[end], Cs[end])
    if nsites(ψ) > 1
      newr = uniqueinds(ARs[1], ψ.AR[n1])
      newAR = ITensor(newl..., s[n2], translatecell(translator(ψ), newr..., 1))
    else
      newAR = ITensor(newl..., s[n2], translatecell(translator(ψ), prime(dag(newl))..., 1))
    end
    tempAR = permute(ARs[end], newl..., s[n2], commoninds(ARs[end], ψ.AR[n2+1])...)
    for I in eachindex(tempAR)
      v = tempAR[I]
      if !iszero(v)
        newAR[I] = tempAR[I]
      end
    end
    ARs[1] = translatecell(translator(ψ), newAR, -1)
  end

  #Only thing left is to join the indices AL-AL and AR-AR
  #Starting with ALs
  for j in 2:length(ALs)
    if nsites(ψ) == 1
      valid_l = only(filterinds(commoninds(ALs[1], Cs[1]), plev = 0))
      nl = combiner(valid_l, tags=tags(only(commoninds(ψ.AL[n1+j-2], ψ.C[n1+j-2]))))
      ALs[1] =noprime(translatecell(translator(ψ), prime(dag(nl)), -1) * ALs[1]* nl )
      Cs[1] *= dag(nl)
    else
    valid_l = only(commoninds(ALs[j-1], Cs[j-1]))
    if j < length(ALs)
      replace_l = only(filterinds(ALs[j], tags = "Left"))
      temp = wδ(dag(valid_l), dag(replace_l))
      ALs[j] *=  temp
    end
    nl = combiner(valid_l, tags=tags(only(commoninds(ψ.AL[n1+j-2], ψ.C[n1+j-2]))))
    ALs[j-1] *= nl
    Cs[j-1] *= dag(nl)
    ALs[j] *= dag(nl)
      if j == length(ALs) && nsites(ψ) == nbsite
        ALs[1] *= translatecell(translator(ψ), dag(nl), -1)
      end
    end
  end
  #Then doing ARs
  for j in reverse(1:length(ARs)-1)
    if nsites(ψ) == 1
      valid_r = only(filterinds(commoninds(ARs[1], Cs[1]), plev = 0))
      nr = combiner(valid_r, tags=tags(only(commoninds(ψ.AR[n1+j], ψ.C[n1+j-1]))))
      ARs[1] = noprime( nr * ARs[1] * translatecell(translator(ψ), prime(dag(nr)), 1) )
      Cs[1] *= dag(nr)
    else
    valid_r = only(commoninds(ARs[j+1], Cs[j]))
    if j > 1
      replace_r = only(filterinds(ARs[j], tags = "Right"))
      temp = wδ(dag(valid_r), dag(replace_r))
      ARs[j] *= temp
    end
    nr = combiner(valid_r, tags=tags(only(commoninds(ψ.AR[n1+j], ψ.C[n1+j-1]))))
    ARs[j+1] *= nr
    Cs[j] *= dag(nr)
    ARs[j] *= dag(nr)
      if j == length(ARs)-1 && nsites(ψ) == nbsite
        ARs[1] *= translatecell(translator(ψ), nr, -1)
      end
    end
  end
  #Fix the extremities

  return (ALs, Cs, ARs)
end



function subspace_expansion(
  theta::ITensor, env::ITensor, H::InfiniteMPO, b::Tuple{Int,Int}; maxdim, cutoff, newtags, α = 1e-2, svd_indices, kwargs...
)
  extension = noprime( ((env * theta) * H[b[1]] ) * H[b[2]] )
  extension = extension * (α / norm(extension)) #LH: I am not sure the renormalization by norm(extension) is really needed here
  supp_index = only(commoninds(H[b[2]], H[b[2] + b[2] - b[1]]))
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
  svd_indices;
  maxdim=maxdim,
  cutoff=cutoff/100,
  lefttags=newtags,
  righttags=newtags,
  use_relative_cutoff=false,
  use_absolute_cutoff = true,
  #mindim = maxdim
  )
  V2 = V2 * (dag(cc) * dag(closure))
  return U2, S2, V2
end


function subspace_expansion(ψ, H; kwargs...)
  range_subspace_expansion = get(kwargs, :range_subspace_expansion, 2)
  ψ = copy(ψ)
  N = nsites(ψ)
  AL = ψ.AL
  C = ψ.C
  AR = ψ.AR
  for n in 1:N
    n1, n2 = n, n + range_subspace_expansion-1
    ALs, Cs, ARs = subspace_expansion(ψ, H, (n1, n2); kwargs...)
    if N == 1
      #Here need to check that we went through the subspace expansion
      AL[n1] = ALs[1]
      C[n1] = Cs[1]
      AR[n2] = ARs[1]
    else
      for (idx, x) in enumerate(n1:min(n2, n1+N-1))
        AL[x] = ALs[idx]
        AR[x] = ARs[idx]
      end
      for (idx, x) in enumerate(n1:n2-1)
        C[x] = Cs[idx]
      end
    end
    ψ = InfiniteCanonicalMPS(AL, C, AR)
  end
  return ψ
end
