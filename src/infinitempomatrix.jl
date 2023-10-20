
mutable struct InfiniteMPOMatrix <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPOMatrix) = mpo.data.translator

# TODO better printing?
function Base.show(io::IO, M::InfiniteMPOMatrix)
  print(io, "$(typeof(M))")
  (length(M) > 0) && print(io, "\n")
  for i in eachindex(M)
    if !isassigned(M, i)
      println(io, "#undef")
    else
      A = M[i]
      println(io, "Matrix tensor of size $(size(A))")
      for k in 1:size(A)[1], l in 1:size(A)[2]
        if !isassigned(A, k + (size(A)[1] - 1) * l)
          println(io, "[($k, $l)] #undef")
        elseif isempty(A[k, l])
          println(io, "[($k, $l)] empty")
        else
          println(io, "[($k, $l)] $(inds(A[k, l]))")
        end
      end
    end
  end
end

function getindex(ψ::InfiniteMPOMatrix, n::Integer)
  return ψ.data[n]
end

function InfiniteMPOMatrix(arrMat::Vector{Matrix{ITensor}})
  return InfiniteMPOMatrix(arrMat, 0, size(arrMat)[1], false)
end

function InfiniteMPOMatrix(data::Vector{Matrix{ITensor}}, translator::Function)
  return InfiniteMPOMatrix(CelledVector(data, translator), 0, size(data)[1], false)
end

function InfiniteMPOMatrix(data::CelledVector{Matrix{ITensor}}, m::Int64, n::Int64)
  return InfiniteMPOMatrix(data, m, n, false)
end

function InfiniteMPOMatrix(data::CelledVector{Matrix{ITensor}})
  return InfiniteMPOMatrix(data, 0, size(data)[1], false)
end

function ITensors.siteinds(A::InfiniteMPOMatrix)
  return CelledVector([dag(only(filterinds(A[x][1, 1], plev = 0, tags = "Site"))) for x in 1:nsites(A)], translator(A))
end


function block_QR_for_left_canonical(H::Matrix{ITensor})
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  l1, l2 = size(H)
  (l1 != 3 || l2 != 3) &&
    error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  for y in 2:l2
    for x in 1:(y - 1)
      !isempty(H[x, y]) && norm(H[x, y]) >1e-14 &&
        error("Format of the InfiniteMPO Matrix incompatible with current implementation")
    end
  end

  #=Following https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147  (but with opposite conventions)
  we look for
  V      =   V'  0   x   R
  d  1       d'  1       t    1
  where the vectors (V', d') are orthogonal to each other, and to (1, 0)
  =#

  #We start by orthogonalizing with respect to 1
  # this corresponds to d' = d - tr(d)/tr(1) * 1, R = 1 and t = tr(d)/tr(1)
  s = commoninds(H[1, 1], H[end, end])
  new_H = copy(H)
  if isempty(new_H[3, 2])
    t1 = ITensor(uniqueinds(new_H[3, 2], s)...)
  else
    t1 = tr(new_H[3, 2]) / tr(new_H[3, 3])
  end
  new_H[3, 2] = new_H[3, 2] - t1 * H[3, 3]

  kept_inds = commoninds(new_H[3, 2], new_H[2, 2]);
  to_fuse_ind = only(uniqueinds(new_H[2, 2], kept_inds))
  dummy_index = Index(QN()=>1, dir = dir(to_fuse_ind), tags = tags(to_fuse_ind) )
  dummy_tensor = ITensor(dummy_index); dummy_tensor[1] = 1
  temp_M2, fused_ind= directsum(new_H[2, 2] => to_fuse_ind, new_H[3, 2]*dummy_tensor => dummy_index;
    tags = tags(dummy_index) )
  unfuse1, unfuse2 = ITensors.directsum_projectors(Float64, Float64, to_fuse_ind, dummy_index, fused_ind)
  unfuse2 = unfuse2 * dummy_tensor

  right_ind2 = only(uniqueinds(kept_inds, s))
  cL2 = combiner(fused_ind; tags=tags(fused_ind), dir = dir(fused_ind))
  cR2 = combiner(right_ind2)
  cLind2 = combinedind(cL2)
  cRind2 = combinedind(cR2)
  temp_M2 = cL2 * temp_M2 * cR2
  unfuse1 = unfuse1 * cL2
  unfuse2 = unfuse2 * cL2

  Q, R, new_right_ind = ITensorInfiniteMPS.myqr(
     temp_M2,
     uniqueinds(temp_M2, cRind2),
     tags=tags(right_ind2),
     positive=true,
     dir=dir(right_ind2),
     dilatation=sqrt(tr(H[3, 3])),
     full=false,
   );
   R = R * dag(cR2)
   new_H[2, 2] = dag(unfuse1) * Q
   new_H[3, 2] = dag(unfuse2) * Q
   #update the empty
   new_H[1, 2] = ITensor(commoninds(new_H[2, 2], new_H[3, 2]))

  #println(( norm(new_H[2, 2] * R - H[2, 2]), norm(new_H[3, 2] * R + new_H[3, 3] * t1 - H[3, 2])))
  T = fill(ITensor(), 3, 3)
  T[1, 1] = ITensor(1); T[3, 3] = ITensor(1)
  T[3, 1] = ITensor(); T[1, 3] = ITensor()
  T[2, 2] = R; T[3, 2] = t1
  left_ind = only(uniqueinds(R, t1))
  right_ind = only(commoninds(R, t1))
  T[2, 1] = ITensor(left_ind); T[2, 3] = ITensor(left_ind)
  T[1, 2] = ITensor(right_ind);
  return new_H, T #R, t1
end

# function apply_left_gauge_on_left(H::Matrix{ITensor}, R::ITensor, t::ITensor)
#   new_H = copy(H)
#   new_H[2, 1] = R * H[2, 1]
#   new_H[3, 1] = t * H[2, 1] + H[3, 1]
#   new_H[2, 2] = R * H[2, 2]
#   new_H[3, 2] = t * H[2, 2] + H[3, 2]
#   return new_H
# end

function apply_mps_to_mpo_on_site(H::Matrix{ITensor}, ψ::ITensor)
  res = Matrix{ITensor}(undef, size(H)...)
  ψd = prime(dag(ψ))
  for x in 1:length(H)
    res[x] = (H[x] * ψ) * ψd
  end
  return res
end

function block_QR_for_left_canonical(H::InfiniteMPOMatrix; left_env = nothing, method = :edge_update, ψ = nothing)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  Ts = Matrix{ITensor}[]
  for j in 1:nsites(H)
    new_H[j], T = block_QR_for_left_canonical(new_H[j])
    append!(Ts, [T])
    #We need to update the environment before the left update of site L
    if !isnothing(left_env) && j == nsites(H)
      if method == :edge_update
        isnothing(ψ) && error("State to project not defined")
        for k in 1:nsites(H)
          left_env .= left_env * apply_mps_to_mpo_on_site(new_H[k], ψ[k])
        end
        left_env = translatecell(translator(H), left_env, -1)
      end
    end
    new_H[j + 1] = T * new_H[j + 1]
  end
  return InfiniteMPOMatrix(new_H),   CelledVector(Ts, translator(new_H)), left_env
  #CelledVector(Rs, translator(new_H)),
  #CelledVector(ts, translator(new_H))
end

function check_convergence_left_canonical(newH, Ts; tol=1e-12, verbose = true)
  for x in 1:nsites(newH)
    li, ri = inds(Ts[x][2, 2])
    if dim(li)!=dim(ri)#li.space != ri.space
      if verbose
        println("Failed convergence due to dim change")
      end
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][3, 2])) > tol
      if verbose
        println("Failing convergence of H[3, 2] by $(norm(tr(newH[x][3, 2])))")
      end
      return false
    end
  end
  for x in 1:nsites(newH)
    primed_ind = inds(Ts[x][2, 2])[1]
    Td = prime(Ts[x][2, 2], primed_ind)
    normed = norm(Ts[x][2, 2]*dag(Td) - denseblocks(δ(primed_ind, prime(dag(primed_ind)))))
    if normed > tol
      if verbose
        println("Failing convergence of T by $normed")
      end
      return false
    end
  end
  return true
end

function left_canonical(H; tol=1e-12, max_iter=100, left_env = nothing, ψ = nothing, projection = -1,method = :edge_update, verbose = true, kwargs...)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end
  if method == :edge_update && isnothing(ψ)
    projection <= 0 && error("Not enough information to fix the boundaries")
    s = [dag(only(filterinds(inds(H[k][1, 1]), plev = 0, tags = "Site"))) for k in 1:nsites(H)]
    ψ = [ITensor(s[k]) for k in 1:nsites(H)]
    for k in 1:nsites(H)
      ψ[k][projection] = 1.0
    end
    ψ = CelledVector(ψ, translator(H))
  end
  if isnothing(left_env)
    left_ind = commoninds(ψ[0], ψ[1])
    if length(left_ind) == 0
      left_env = [ITensor(0), ITensor(commoninds(H[0][2, 2], H[1][2, 2])), ITensor(1)]
    else
      left_env = [ITensor(0, (left_ind, dag(prime(left_ind)))), ITensor(commoninds(H[0][2, 2], H[1][2, 2])..., left_ind, dag(prime(left_ind))), ITensor(1, left_ind, dag(prime(left_ind)))]
    end
  end
  #newH, Rs, ts, left_env = block_QR_for_left_canonical(H, left_env)
  newH, Ts, left_env = block_QR_for_left_canonical(H; left_env, ψ, method )
  if check_convergence_left_canonical(newH, Ts; tol, verbose)
    return newH, Ts, left_env
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Ts, left_env = block_QR_for_left_canonical(newH; left_env, ψ, method)
    cont = !check_convergence_left_canonical(newH, new_Ts; tol, verbose)
    for j in 1:nsites(newH)
      Ts[j] = new_Ts[j] * Ts[j]
    end
    j += 1
  end
  if cont
    println("Warning: reached max iterations before convergence")
  else
    println("Left canonicalized in $(j+1) iterations")
  end
  return newH, Ts, left_env
end


function block_QR_for_right_canonical(H::Matrix{ITensor})
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  l1, l2 = size(H)
  (l1 != 3 || l2 != 3) &&
    error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  for y in 2:l2
    for x in 1:(y - 1)
      !isempty(H[x, y]) && norm(H[x, y])!=0 &&
        error("Format of the InfiniteMPO Matrix incompatible with current implementation")
    end
  end

  #=Following https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147  (but with opposite conventions)
  we look for
  1      =   1  0   x  1
  b  V       t  R      b'  V'
  where the vectors (V', d') are orthogonal to each other, and to (1, 0)
  =#

  #We start by orthogonalizing with respect to 1
  # this corresponds to d' = d - tr(d)/tr(1) * 1, R = 1 and t = tr(d)/tr(1)
  s = commoninds(H[1, 1], H[end, end])
  new_H = copy(H)
  if isempty(new_H[2, 1])
    t1 = ITensor(uniqueinds(new_H[2, 1], s)...)
  else
    t1 = tr(new_H[2, 1]) / tr(H[1, 1])
  end
  new_H[2, 1] .= new_H[2, 1] .- t1 * H[1, 1]

  kept_inds = commoninds(new_H[2, 1], new_H[2, 2]);
  to_fuse_ind = only(uniqueinds(new_H[2, 2], kept_inds))
  dummy_index = Index(QN()=>1, dir = dir(to_fuse_ind), tags = tags(to_fuse_ind) )
  dummy_tensor = ITensor(dummy_index); dummy_tensor[1] = 1
  temp_M, fused_ind= directsum( new_H[2, 1]*dummy_tensor => dummy_index, new_H[2, 2] => to_fuse_ind;
    tags = tags(dummy_index) )
  unfuse2, unfuse1 = ITensors.directsum_projectors(Float64, Float64, dummy_index, to_fuse_ind, fused_ind)
  unfuse2 = unfuse2 * dummy_tensor

  left_ind = only(uniqueinds(kept_inds, s))
  cL = combiner(left_ind)
  cR = combiner(fused_ind; tags=tags(fused_ind))
  cLind = combinedind(cL)
  cRind = combinedind(cR)
  temp_M = cL * temp_M * cR
  unfuse1 = unfuse1 * cR
  unfuse2 = unfuse2 * cR

  Q, R, new_left_ind = myqr(
     temp_M,
     uniqueinds(temp_M, cLind),
     tags=tags(left_ind),
     positive=true,
     dir=dir(left_ind),
     dilatation=sqrt(tr(H[3, 3])),
     full=false,
   );
   R = R * dag(cL)
   new_H[2, 2] = dag(unfuse1) * Q
   new_H[2, 1] = dag(unfuse2) * Q
   #update the empty
   new_H[2, 3] = ITensor(commoninds(new_H[2, 2], new_H[2, 1]))

  T = fill(ITensor(), 3, 3)
  T[1, 1] = ITensor(1); T[3, 3] = ITensor(1)
  T[3, 1] = ITensor(); T[1, 3] = ITensor()
  T[2, 2] = R; T[2, 1] = t1;
  left_ind = only(commoninds(R, t1));
  right_ind = only(uniqueinds(R, t1));
  T[2, 3] = ITensor(left_ind);
  T[1, 2] = ITensor(right_ind); T[3, 2] = ITensor(right_ind);
  return new_H, T
end

function block_QR_for_right_canonical(H::InfiniteMPOMatrix; right_env = nothing, method = :edge_update, ψ = nothing)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  Ts = Matrix{ITensor}[]
  for j in reverse(1:nsites(H))
    new_H[j], T = block_QR_for_right_canonical(new_H[j])
    append!(Ts, [T])
    if !isnothing(right_env) && j == 1
      if method == :edge_update
        isnothing(ψ) && error("State to project not defined")
        for k in reverse(1:nsites(H))
          right_env = apply_mps_to_mpo_on_site(new_H[k], ψ[k]) * right_env
        end
        right_env = translatecell(translator(H), right_env, 1)
      end
    end
    new_H[j - 1] = new_H[j - 1] * T
  end
  return InfiniteMPOMatrix(new_H),
  CelledVector(reverse(Ts), translator(new_H)), right_env
end

function check_convergence_right_canonical(newH, Ts; tol=1e-12, verbose = true)
  for x in 1:nsites(newH)
    li, ri = inds(Ts[x][2, 2])
    if dim(li) != dim(ri)#li.space != ri.space
      if verbose
        println("Failed convergence due to dim change")
      end
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][2, 1])) > tol
      if verbose
        println("Failing convergence of H[2, 1] by $(norm(tr(newH[x][2, 1]))) ")
      end
      return false
    end
  end
  for x in 1:nsites(newH)
    primed_ind = inds(Ts[x][2, 2])[1]
    Td = prime(Ts[x][2, 2], primed_ind)
    normed = norm(Ts[x][2, 2]*dag(Td) - denseblocks(δ(primed_ind, prime(dag(primed_ind)))))
    if normed > tol
      if verbose
        println("Failing convergence of T by $(normed) ")
      end
      return false
    end
  end
  return true
end

function right_canonical(H; tol=1e-12, max_iter=100, right_env = nothing, ψ = nothing, projection = -1,method = :edge_update, verbose, kwargs...)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  if method == :edge_update && isnothing(ψ)
    projection <= 0 && error("Not enough information to fix the boundaries")
    s = [dag(only(filterinds(inds(H[k][1, 1]), plev = 0, tags = "Site"))) for k in 1:nsites(H)]
    ψ = [ITensor(s[k]) for k in 1:nsites(H)]
    for k in 1:nsites(H)
      ψ[k][projection] = 1.0
    end
    ψ = CelledVector(ψ, translator(H))
  end
  if typeof(ψ) == InfiniteCanonicalMPS
    ψ = ψ.AR
  end
  if isnothing(right_env)
    right_ind = commoninds(ψ[nsites(H)+1], ψ[nsites(H)])
    if length(right_ind) == 0
      right_env = [ITensor(1), ITensor(commoninds(H[nsites(H)+1][2, 2], H[nsites(H)][2, 2])), ITensor(0)]
    else
      right_env = [ITensor(1, right_ind, dag(prime(right_ind))), ITensor(commoninds(H[nsites(H)+1][2, 2], H[nsites(H)][2, 2])..., right_ind, dag(prime(right_ind))), ITensor(0, right_ind, dag(prime(right_ind)))]
    end
  end
  newH, Ts, right_env = block_QR_for_right_canonical(H; right_env, ψ, method)
  if check_convergence_right_canonical(newH, Ts; tol, verbose)
    println("Right canonicalized in 1 iterations")
    return newH, Ts, right_env
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Ts, right_env = block_QR_for_right_canonical(newH; right_env, ψ, method)
    cont = !check_convergence_right_canonical(newH, new_Ts; tol, verbose)
    for j in 1:nsites(newH)
      Ts[j] =  Ts[j] * new_Ts[j]
    end
    j += 1
  end
  if cont
    println("Warning: reached max iterations before convergence")
  else
    println("Right canonicalized in $j iterations")
  end
  return newH, Ts, right_env
end

function compress_impo(H::InfiniteMPOMatrix; left_env = nothing, right_env = nothing, verbose = true, kwargs...)
  split_blocks = get(kwargs, :splitblocks, true)
  tol = get(kwargs, :tol, 1e-12)

  smallH = make_block(H)
  if !isnothing(left_env)
    if length(left_env)>3
      links = [only(commoninds(left_env[j], H[1][j, j])) for j in 2:length(left_env)-1]
      temp, new_ind = directsum([left_env[j]=>links[j-1] for j in 2:length(left_env)-1]...,   tags = tags(links[1]) )
      replaceind!(temp, new_ind => only(commoninds(smallH[0][2, 2], smallH[1][2, 2])) )
      new_left_env = [left_env[1], temp, left_env[end]]
      left_env = new_left_env
    end
  end
  if !isnothing(right_env)
    if length(right_env)>3
      links = [only(commoninds(right_env[j], H[nsites(H)][j, j])) for j in 2:length(right_env)-1]
      temp, new_ind = directsum([right_env[j]=>links[j-1] for j in 2:length(right_env)-1]...,   tags = tags(links[1]) )
      replaceind!(temp, new_ind => only(commoninds(smallH[nsites(H)+1][2, 2], smallH[nsites(H)][2, 2])) )
      new_right_env = [right_env[1], temp, right_env[end]]
      right_env = new_right_env
    end
  end
  println("Starting left canonicalization")
  HL, Tl1, L = left_canonical(smallH; left_env, verbose, kwargs...)
  if !isnothing(right_env)
    right_env = Tl1[nsites(H)] * right_env
  end

  println("Starting right canonicalization")
  HR, Tr1, Rr = right_canonical(HL; right_env, verbose, kwargs...)
  Lr = L * Tr1[1]
  left_env = deepcopy(Lr)

  println("Starting second left canonicalization")
  HL, Ts, Ll = left_canonical(HR; left_env, verbose, kwargs... )
  Rl = Ts[nsites(HR)] * Rr
  #At this point, we hav<e HL[1]*Rs[1] = Rs[0] * HR[1] etc
  test_norm = maximum([norm(Ts[x][3, 2]) for x in 1:nsites(H)])
  if  test_norm> 1e-12
    error("Ts should be 0 at this point, instead it is $test_norm")
  end
  #TODO ? replace this by matrix formmulation?
  println("Starting truncation")
  Us = Vector{ITensor}(undef, nsites(H))
  Ss = Vector{ITensor}(undef, nsites(H))
  Vsd = Vector{ITensor}(undef, nsites(H))
  for x in 1:nsites(H)
    Us[x], Ss[x], Vsd[x] = svd(
      Ts[x][2, 2],
      commoninds(Ts[x][2, 2], HL[x][2, 2]);
      lefttags=tags(only(commoninds(Ts[x][2, 2], HL[x][2, 2]))),
      righttags=tags(only(commoninds(Ts[x][2, 2], HR[x + 1][2, 2]))),
      kwargs...
    )
    if split_blocks
      Us[x] = splitblocks(Us[x], commoninds(Us[x], Ss[x]))
      Vsd[x] = splitblocks(Vsd[x], commoninds(Vsd[x], Ss[x]))
    end
    #println(sum(diag(Ss[x]) .^ 2))
  end
  Us = CelledVector(Us, translator(H))
  Vsd = CelledVector(Vsd, translator(H))
  Ss = CelledVector(Ss, translator(H))
  newHL = copy(HL.data.data)
  newHR = copy(HR.data.data)
  for x in 1:nsites(H)
    ## optimizing the left canonical
    newHL[x][2, 1] = ITensors.dropzeros( dag(Us[x - 1]) * newHL[x][2, 1], tol = tol)
    newHL[x][3, 2] = ITensors.dropzeros( newHL[x][3, 2] * Us[x], tol = tol)
    newHL[x][2, 2] = ITensors.dropzeros( dag(Us[x - 1]) * newHL[x][2, 2] * Us[x], tol = tol)
    newHL[x][1, 2] = ITensor(commoninds(newHL[x][2, 2], newHL[x][3, 2]))
    newHL[x][2, 3] = ITensor(commoninds(newHL[x][2, 2], newHL[x][2, 1]))
    ## optimizing the right canonical
    newHR[x][2, 1] = ITensors.dropzeros( Vsd[x - 1] * newHR[x][2, 1], tol = tol)
    newHR[x][3, 2] = ITensors.dropzeros( newHR[x][3, 2] * dag(Vsd[x]), tol = tol)
    newHR[x][2, 2] = ITensors.dropzeros( Vsd[x - 1] * newHR[x][2, 2] * dag(Vsd[x]), tol = tol)
    newHR[x][1, 2] = ITensor(commoninds(newHR[x][2, 2], newHR[x][3, 2]) )
    newHR[x][2, 3] = ITensor(commoninds(newHR[x][2, 2], newHR[x][2, 1]) )
  end
  left_env_LC = deepcopy(Ll); left_env_LC[2] = left_env_LC[2] * Us[0];
  right_env_LC = deepcopy(Rl); right_env_LC[2] = dag(Us[nsites(H)]) * right_env_LC[2];
  left_env_RC = deepcopy(Lr); left_env_RC[2] = left_env_RC[2] * dag(Vsd[0]);
  right_env_RC = deepcopy(Rr); right_env_RC[2] = Vsd[nsites(H)] * right_env_RC[2];
  return (InfiniteMPOMatrix(newHL, translator(H)), left_env_LC, right_env_LC), (InfiniteMPOMatrix(newHR, translator(H)), left_env_RC, right_env_RC), Ss
end

#RETIRED
function make_block(H::InfiniteMPOMatrix)
  if size(H[1]) == (3, 3)
    println("Nothing to do")
    return H
  end
  #H = make_dummy_indices_explicit(H)
  lx, ly = size(H[1])
  # novel forn
  # 1
  # t  H
  # c  b  1
  temp = [
    matrixITensorToITensor(H[x], 2:(lx - 1), 2:(ly - 1); init_all=false) for
    x in 1:nsites(H)
  ]
  new_H = CelledVector([x[1] for x in temp], translator(H))
  lis = CelledVector([x[2] for x in temp], translator(H))
  ris = CelledVector([x[3] for x in temp], translator(H))
  #for t
  temp = [matrixITensorToITensor(H[x], 2:(lx - 1), 1; init_all=false) for x in 1:nsites(H)]
  new_t = CelledVector([x[1] for x in temp], translator(H))
  lis_t = CelledVector([x[2] for x in temp], translator(H))
  #for b
  temp = [matrixITensorToITensor(H[x], lx, 2:(ly - 1); init_all=false) for x in 1:nsites(H)]
  new_b = CelledVector([x[1] for x in temp], translator(H))
  lis_b = CelledVector([x[2] for x in temp], translator(H))

  #retags the right_links
  s = [only(dag(filterinds(commoninds(H[j][1, 1], H[j][end, end]), plev = 0))) for j in 1:nsites(H)]
  for j in 1:nsites(H)
    newTag = "Link,c=$(getcell(s[j])),n=$(getsite(s[j]))"
    temp = replacetags(ris[j], tags(ris[j]), newTag)
    new_H[j] = replaceinds(new_H[j], [ris[j]], [temp])
    ris[j] = replacetags(ris[j], tags(ris[j]), newTag)
  end
  # joining the indices
  for j in 1:nsites(H)
    temp = δ(dag(ris[j]), dag(lis[j + 1]))
    new_H[j + 1] *= temp
  end
  #Fixing the indices for b and t, recalling new_H[j] - t[j+1] and b[j]*new_H[j+1]
  for j in 1:nsites(H)
    temp = δ(dag(ris[j - 1]), dag(lis_t[j]))
    new_t[j] *= temp
    temp = δ(ris[j], dag(lis_b[j]))
    new_b[j] *= temp
  end

  new_mpo = [fill(op("Zero", s[x]), 3, 3) for x in 1:nsites(H)]
  for j in 1:nsites(H)
    new_mpo[j][1, 1] = H[j][1, 1]
    new_mpo[j][3, 3] = H[j][end, end]
    new_mpo[j][3, 1] = H[j][end, 1]
    new_mpo[j][2, 1] = new_t[j]
    new_mpo[j][2, 2] = new_H[j]
    new_mpo[j][3, 2] = new_b[j]
    new_mpo[j][2, 3] = ITensor(inds(new_t[j]) )
    new_mpo[j][1, 2] = ITensor(inds(new_b[j]) )
    #new_mpo[j][1, 3] is already good
  end
  return InfiniteMPOMatrix(new_mpo, translator(H))
end

function make_dummy_indices_explicit(H::InfiniteMPOMatrix)
  lx, ly = size(H[1])
  s = [commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  new_H = [fill(op("Zero", s[x]), lx, ly) for x in 1:nsites(H)]
  element = eltype(H[1][1, 1])

  for n in 1:nsites(H)
    s = commoninds(H[n][1, 1], H[n][end, end])
    #First, find all right indices, per column except first and last
    right_indices = []
    for y in 2:(ly - 1)
      for x in y:lx
        right_ind = filter(x -> dir(x) == ITensors.Out, uniqueinds(H[n][x, y], s))
        if length(right_ind) == 1
          append!(right_indices, right_ind)
          break
        end
      end
    end
    length(right_indices) != ly - 2 && error("Some right indices missing")
    #Second, find all left indices, per line except first and last
    left_indices = []
    for x in 2:(lx - 1)
      for y in 1:x
        left_ind = filter(x -> dir(x) == ITensors.In, uniqueinds(H[n][x, y], s))
        if length(left_ind) == 1
          append!(left_indices, left_ind)
          break
        end
      end
    end
    length(left_indices) != lx - 2 && error("Some left indices missing")
    #Now, filling the zeros
    for j in 1:lx
      for k in 1:ly
        if isempty(H[n][j, k])
          if (j == 1 || j == lx) && (k == 1 || k == ly)
            #just site indices
            new_H[n][j, k] = ITensor(element, s...)
          elseif (j == 1 || j == lx)
            #println(inds( ITensor(element, s..., right_indices[k-1])))
            new_H[n][j, k] = ITensor(element, s..., right_indices[k - 1])
          elseif (k == 1 || k == ly)
            new_H[n][j, k] = ITensor(element, s..., left_indices[j - 1])
          else
            new_H[n][j, k] = ITensor(
              element, s..., left_indices[j - 1], right_indices[k - 1]
            )
          end
        else
          new_H[n][j, k] = H[n][j, k]
        end
      end
    end
  end
  return InfiniteMPOMatrix(new_H, translator(H))
end


function convert_itensor_to_itensormatrix(tensor; kwargs...)
  if order(tensor) == 3
    return convert_itensor_3vector(tensor; kwargs...)
  elseif order(tensor) == 4
    return convert_itensor_33matrix(tensor; kwargs...)
  else
    error(
      "Conversion of ITensor into matrix of ITensor not planned for this type of tensors"
    )
  end
end

"Build the projectors on the three parts of the itensor used to split a MPO into an InfiniteMPOMatrix"
function build_three_projectors_from_index(is::Index; kwargs...)
  old_dim = dim(is)
  new_tags = get(kwargs, :tags, tags(is))
  split = get(kwargs, :split, true)
  #Build the local projectors.
  #We have to differentiate between the dense and the QN case
  #Note that as far as I know, the MPO even dense is always guaranteed to have identities at both corners
  #If it is not the case, my construction will not work
  top = onehot(dag(is) => 1)
  bottom = onehot(dag(is) => old_dim)
  if length(is.space) == 1
    new_ind = Index(is.space - 2; tags=new_tags)
    mat = zeros(new_ind.space, is.space)
    for x in 1:(new_ind.space)
      mat[x, x + 1] = 1
    end
    middle = ITensor(mat, new_ind, dag(is))
  else
    new_ind = Index(is.space[2:(end - 1)]; dir=dir(is), tags=new_tags)
    middle = ITensors.BlockSparseTensor(
      Float64,
      undef,
      Block{2}[Block(x, x + 1) for x in 1:length(new_ind.space)],
      (new_ind, dag(is)),
    )
    for x in 1:length(new_ind.space)
      dim_block = new_ind.space[x][2]
      ITensors.blockview(middle, Block(x, x + 1)) .= diagm(0 => ones(dim_block))
    end
    middle = itensor(middle)
    if !split
      cb = combiner(new_ind, dir = dir(new_ind), tags = tags(new_ind))
      middle =  cb * middle
    end
  end
  return top, middle, bottom
end

function convert_itensor_33matrix(tensor; leftdir=ITensors.In, kwargs...)
  @assert order(tensor) == 4
  left_ind = get(kwargs, :leftindex, nothing)
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  if isnothing(left_ind)
    temp = uniqueinds(tensor, sit)
    if dir(temp[1]) == leftdir
      left_ind = temp[1]
      right_ind = temp[2]
    else
      left_ind = temp[2]
      right_ind = temp[1]
    end
  else
    right_ind = only(uniqueinds(tensor, sit, left_ind))
  end
  left_dim = dim(left_ind)
  right_dim = dim(right_ind)
  #Build the local projectors.
  left_tags = get(kwargs, :left_tags, tags(left_ind))
  top_left, middle_left, bottom_left = build_three_projectors_from_index(
    left_ind; tags=left_tags, kwargs...
  )
  right_tags = get(kwargs, :righ_tags, tags(right_ind))
  top_right, middle_right, bottom_right = build_three_projectors_from_index(
    right_ind; tags=right_tags, kwargs...
  )

  matrix = fill(op("Zero", local_sit), 3, 3)
  for (idx_left, proj_left) in enumerate([top_left, middle_left, bottom_left])
    for (idx_right, proj_right) in enumerate([top_right, middle_right, bottom_right])
      matrix[idx_left, idx_right] = proj_left * tensor * proj_right
    end
  end
  return matrix
end

function convert_itensor_3vector(
  tensor; leftdir=ITensors.In, first=false, last=false, kwargs...
)
  @assert order(tensor) == 3
  #Identify the different indices
  sit = filterinds(inds(tensor); tags="Site")
  local_sit = dag(only(filterinds(sit; plev=0)))
  #A bit roundabout as filterinds does not accept dir
  old_ind = only(uniqueinds(tensor, sit))
  if dir(old_ind) == leftdir || last
    new_tags = get(kwargs, :left_tags, tags(old_ind))
    top, middle, bottom = build_three_projectors_from_index(old_ind; tags=new_tags, kwargs...)
    vector = fill(op("Zero", local_sit), 3, 1)
  else
    new_tags = get(kwargs, :right_tags, tags(old_ind))
    top, middle, bottom = build_three_projectors_from_index(old_ind; tags=new_tags, kwargs...)
    vector = fill(op("Zero", local_sit), 1, 3)
  end
  for (idx, proj) in enumerate([top, middle, bottom])
    vector[idx] = proj * tensor
  end
  return vector
end


function finding_indices(mat::Matrix{ITensor}; dir = ITensor.Out)
  if dir == ITensors.Out
    #Finding the left indices
    left_indices = Dict()
    for  j in 2:size(mat, 2)
      for k in 1:size(mat, 1)
        temp_ind = filter(x-> x.dir == ITensors.Out, filterinds(mat[k, j], tags = "Link"))
        if length(temp_ind) == 1
          left_indices[j] = only(temp_ind)
          continue
        end
      end
    end
    return left_indices
  else
    #Finding the right indices
    right_indices = Dict()
    for  j in 2:size(mat, 1)
      for k in 1:size(mat, 2)
        temp_ind = filter(x -> x.dir == ITensors.In, filterinds(mat[j, k], tags = "Link"))
        if length(temp_ind) == 1
          right_indices[j] = only(temp_ind)
          continue
        end
      end
    end
    return right_indices
  end
end


function Base.:*(A::Matrix{ITensor}, B::Matrix{ITensor})
  size(A, 2) != size(B, 1) && error("Matrix sizes are incompatible")
  C = Matrix{ITensor}(undef, size(A, 1), size(B, 2))
  for i in 1:size(A, 1)
    for j in 1:size(B, 2)
      #TODO something more resilient
      if isempty(A[i, 1]) || isempty(B[1, j])
        C[i, j] = ITensor(uniqueinds(A[i, 1], B[1, j])..., uniqueinds(B[1, j], A[i, 1])...)
      else
        C[i, j] = A[i, 1] * B[1, j]
      end
      for k in 2:size(A, 2)
        if isempty(A[i, k]) || isempty(B[k, j])
          continue
        end
        C[i, j] = C[i, j] + A[i, k] * B[k, j]
      end
    end
  end
  return C
end

function Base.:*(A::Matrix{ITensor}, B::Vector{ITensor})
  size(A, 2) != length(B) && error("Matrix sizes are incompatible")
  C = Vector{ITensor}(undef, size(A, 1))
  for i in 1:size(A, 1)
    if isempty(A[i, size(A, 2)]) || isempty(B[size(A, 2)])
      C[i] = ITensor(uniqueinds(A[i, size(A, 2)], B[size(A, 2)])..., uniqueinds(B[size(A, 2)], A[i, size(A, 2)])...)
      if length(inds(C[i]))==0
        C[i] = ITensor(0)
      end
    else
      C[i] = A[i, size(A, 2)] * B[size(A, 2)]
    end
    for k in reverse(1:size(A, 2)-1)
      if isempty(A[i, k]) || isempty(B[k])
        continue
      end
      #This is due to some stuff missing in NDTensors
      if length(inds(C[i]))==0
        temp = A[i, k] * B[k]
        if !isempty(temp)
          C[i] .+= temp[1]
        end
      else
        C[i] = C[i] + A[i, k] * B[k]
      end
    end
  end
  return C
end

function Base.:*(A::Vector{ITensor}, B::Matrix{ITensor})
  length(A) != size(B, 1) && error("Matrix sizes are incompatible")
  C = Vector{ITensor}(undef, size(B, 2))
  for j in 1:size(B, 2)
    if isempty(A[1]) || isempty(B[1, j])
      C[j] = ITensor(uniqueinds(A[1], B[1, j])..., uniqueinds(B[1, j], A[1])...)
      if length(dims(C[j]))==0
        C[j] = ITensor(0)
      end
    else
      C[j] = A[1] * B[1, j]
    end
    for k in 2:size(B, 1)
      if isempty(A[k]) || isempty(B[k, j])
        continue
      end
      if length(inds(C[j]))==0
        temp = A[k] * B[k, j]
        if !isempty(temp)
          C[j] .+= temp[1]
        end
      else
        C[j] = C[j] + A[k] * B[k, j]
      end
    end
  end
  return C
end

function scalar_product(A::Vector{ITensor}, B::Vector{ITensor})
  length(A) != length(B) && error("Vector sizes are incompatible")
  C = A[1] * B[1]
  for k in 2:length(B)
    if isempty(A[k]) || isempty(B[k])
      continue
    end
    C = C + A[k] * B[k]
  end
  return C
end

function ITensors.splitblocks!(H::InfiniteMPOMatrix)
  N = nsites(H)
  for j in 1:N
    ls = Dict()
    rs = Dict()
    for n in 1:length(H[j])
        H.data.data[j][n] = ITensors.splitblocks(H.data.data[j][n])
    end
  end
  return H
end

function Base.:+(A::InfiniteMPOMatrix, B::InfiniteMPOMatrix)
  #Asserts
  nsites(A) != nsites(B) && error("The two MPOs have different lengths, not implemented")
  N = nsites(A)
  s_A = siteinds(A); s_B = siteinds(B);
  for x in 1:N
    s_A[x] != s_B[x] && error("The site index are different, impossible to sum")
  end
  translator(A) != translator(B) && error("The two MPOs have different translation rules")
  #Building the sum
  sizes_A = size.(A); sizes_B = size.(B)
  new_MPOMatrices = Matrix{ITensor}[Matrix{ITensor}(undef, sizes_A[x][1] + sizes_B[x][1] - 2, sizes_A[x][2] + sizes_B[x][2] - 2) for x in 1:N] #The identities at the top corner are unchanged
  for j in 1:N
    #Filling up the A part
    for x in 1:sizes_A[j][1]-1, y in 1:sizes_A[j][2]-1
      new_MPOMatrices[j][x, y] = A[j][x, y]
    end
    for x in 1:sizes_A[j][1]-1
      new_MPOMatrices[j][x, end] = A[j][x, end]
    end
    for y in 1:sizes_A[j][2] -1
      new_MPOMatrices[j][end, y] = A[j][end, y]
    end
    #new_MPOMatrices[j][end, end] = A[j][end, end] Not needed as B automatically gives it
    #Filling up the B part
    x_shift, y_shift = sizes_A[j] .- 2
    for x in 2:sizes_B[j][1], y in 2:sizes_B[j][2]
      new_MPOMatrices[j][x_shift + x, y_shift + y] = B[j][x, y]
    end
    for x in 2:sizes_B[j][1]
      new_MPOMatrices[j][x_shift + x, 1] = B[j][x, 1]
    end
    for y in 2:sizes_B[j][2]
      new_MPOMatrices[j][1, y_shift + y] = B[j][1, y]
    end
  end
  #Filling up the rest
  for j in 1:N
    s_inds = inds(A[j][1, 1])
    new_MPOMatrices[j][1, end] = A[j][1, end] + B[j][1, end] #ITensor(s_inds)
    new_MPOMatrices[j][end, 1] = A[j][end, 1] + B[j][end, 1] #ITensor(s_inds)
    x_shift, y_shift = sizes_A[j] .- 2
    for x in 2:sizes_A[j][1]-1
      left_index = only(commoninds(A[j][x, x], A[j-1][x, x]))
      for y in 2:sizes_B[j][2]-1
        right_index = only(commoninds(B[j][y, y], B[j+1][y, y]))
        new_MPOMatrices[j][x, y_shift + y] = ITensor(left_index, s_inds..., right_index)
      end
    end
    for x in 2:sizes_B[j][1]-1
      left_index = only(commoninds(B[j][x, x], B[j-1][x, x]))
      for y in 2:sizes_A[j][2]-1
        right_index = only(commoninds(A[j][y, y], A[j+1][y, y]))
        new_MPOMatrices[j][x_shift + x,y] = ITensor(left_index, s_inds..., right_index)
      end
    end
  end
  #return new_MPOMatrices
  return InfiniteMPOMatrix(new_MPOMatrices, translator(A))
end
