
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
  t1 = tr(new_H[3, 2]) / tr(new_H[3, 3])
  new_H[3, 2] = new_H[3, 2] - t1 * H[3, 3]

  kept_inds = commoninds(new_H[3, 2], new_H[2, 2]);
  to_fuse_ind = only(uniqueinds(new_H[2, 2], kept_inds))
  dummy_index = Index(QN()=>1, dir = dir(to_fuse_ind), tags = tags(to_fuse_ind) )
  dummy_tensor = ITensor(dummy_index); dummy_tensor[1] = 1
  temp_M2, fused_ind= directsum(new_H[2, 2] => to_fuse_ind, new_H[3, 2]*dummy_tensor => dummy_index;
    tags = tags(dummy_index) )
  unfuse1, unfuse2 = ITensors.directsum_itensors(to_fuse_ind, dummy_index, fused_ind)
  unfuse2 = unfuse2 * dummy_tensor

  right_ind2 = only(uniqueinds(kept_inds, s))
  cL2 = combiner(fused_ind; tags=tags(fused_ind))
  cR2 = combiner(right_ind2)
  cLind2 = combinedind(cL2)
  cRind2 = combinedind(cR2)
  temp_M2 = cL2 * temp_M2 * cR2
  unfuse1 = unfuse1 * cL2
  unfuse2 = unfuse2 * cL2

  Q, R, new_right_ind = qr(
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

  #TODO decide whether I give R and t or a matrix?
  #println(( norm(new_H[2, 2] * R - H[2, 2]), norm(new_H[3, 2] * R + new_H[3, 3] * t1 - H[3, 2])))
  T = fill(ITensor(), 3, 3)
  T[1, 1] = ITensor(1); T[3, 3] = ITensor(1)
  T[3, 1] = ITensor(0); T[1, 3] = ITensor(0)
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

function block_QR_for_left_canonical(H::InfiniteMPOMatrix)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  # Rs = ITensor[]
  # ts = ITensor[]
  # for j in 1:nsites(H)
  #   new_H[j], R, t = block_QR_for_left_canonical(new_H[j])
  #   new_H[j + 1] = apply_left_gauge_on_left(new_H[j + 1], R, t)
  #   append!(Rs, [R])
  #   append!(ts, [t])
  #   #j == nsites(H)-1 && break
  # end
  Ts = Matrix{ITensor}[]
  for j in 1:nsites(H)
    new_H[j], T = block_QR_for_left_canonical(new_H[j])
    append!(Ts, [T])
    new_H[j + 1] = T * new_H[j + 1]
  end
  return InfiniteMPOMatrix(new_H),   CelledVector(Ts, translator(new_H))
  #CelledVector(Rs, translator(new_H)),
  #CelledVector(ts, translator(new_H))
end


function block_QR_for_left_canonical(H::InfiniteMPOMatrix, left_env::Vector{ITensor}; proj = 1)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  Rs = ITensor[]
  ts = ITensor[]
  for j in 1:nsites(H)
    new_H[j], R, t = block_QR_for_left_canonical(new_H[j])
    new_H[j + 1] = apply_left_gauge_on_left(new_H[j + 1], R, t)
    append!(Rs, [R])
    append!(ts, [t])
    #Now we advance the environment, assuming the block form
    s = filterinds(new_H[j][1, 1], tags = "Site", plev = 0)
    dummy_top = ITensor(dag(s)); dummy_top[proj] = 1
    dummy_down = dag(prime(dummy_top))
    if !isempty(new_H[j][2, 1])
      temp =  (new_H[j][2, 1] * dummy_top * dummy_down) * left_env[2]
      left_env[1] .+= isempty(temp) ? 0 : temp[1]
    end
    if !isempty(new_H[j][3, 1])
      temp = (new_H[j][3, 1]* dummy_top * dummy_down) * left_env[3]
      left_env[1] += isempty(temp) ? 0 : temp[1]
    end
    left_env[2] =  (new_H[j][2, 2]* dummy_top * dummy_down) * left_env[2]
    if !isempty(new_H[j][3, 2])
      left_env[2] += (new_H[j][3, 2]* dummy_top * dummy_down) * left_env[3]
    end
    #left_env[3] = left_env[3]
  end
  left_env = translatecell(translator(H), left_env, -1)
  return InfiniteMPOMatrix(new_H),
  CelledVector(Rs, translator(new_H)),
  CelledVector(ts, translator(new_H)),
  left_env
end

# function check_convergence_left_canonical(newH, Rs, ts; tol=1e-12)
#   return true
#   for x in 1:nsites(newH)
#     li, ri = inds(Rs[x])
#     if li.space != ri.space
#       return false
#     end
#   end
#   for x in 1:nsites(newH)
#     if norm(tr(newH[x][3, 2])) > tol
#       return false
#     end
#   end
#   for x in 1:nsites(newH)
#     if norm(Rs[x] - denseblocks(δ(inds(Rs[x])...))) > tol
#       return false
#     end
#   end
#   return true
# end

function check_convergence_left_canonical(newH, Ts; tol=1e-12)
  for x in 1:nsites(newH)
    li, ri = inds(Ts[x][2, 2])
    if dim(li)!=dim(ri)#li.space != ri.space
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][3, 2])) > tol
      return false
    end
  end
  for x in 1:nsites(newH)
    primed_ind = inds(Ts[x][2, 2])[1]
    Td = prime(Ts[x][2, 2], primed_ind)
    normed = norm(Ts[x][2, 2]*dag(Td) - denseblocks(δ(primed_ind, prime(dag(primed_ind)))))
    if normed > tol
      return false
    end
  end
  return true
end

# function left_canonical(H; tol=1e-12, max_iter=50)
#   l1, l2 = size(H[1])
#   if (l1 != 3 || l2 != 3)
#     H = make_block(H)
#   end
#   temp = ITensor(); temp[1] = 1; temp2 = ITensor(1);
#   left_env = [temp, ITensor(commoninds(H[0][2, 2], H[1][2, 2])), temp2]
#   newH, Rs, ts, left_env = block_QR_for_left_canonical(H, left_env)
#   if check_convergence_left_canonical(newH, Rs, ts; tol)
#     return newH, Rs, ts, left_env
#   end
#   j = 1
#   cont = true
#   while j <= max_iter && cont
#     newH, new_Rs, new_ts = block_QR_for_left_canonical(newH)
#     cont = !check_convergence_left_canonical(newH, new_Rs, new_ts; tol)
#     for j in 1:nsites(newH)
#       ts[j] = ts[j] + Rs[j] * new_ts[j]
#       Rs[j] = new_Rs[j] * Rs[j]
#     end
#     j += 1
#   end
#   if j == max_iter + 1 && cont
#     println("Warning: reached max iterations before convergence")
#   else
#     println("Left canonicalized in $(j+1) iterations")
#   end
#   return newH, Rs, ts, left_env
# end


function left_canonical(H; tol=1e-12, max_iter=50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end
  temp = ITensor(); temp[1] = 1; temp2 = ITensor(1);
  left_env = [temp, ITensor(commoninds(H[0][2, 2], H[1][2, 2])), temp2]
  #newH, Rs, ts, left_env = block_QR_for_left_canonical(H, left_env)
  newH, Ts = block_QR_for_left_canonical(H)
  if check_convergence_left_canonical(newH, Ts; tol)
    return newH, Ts#, left_env
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Ts = block_QR_for_left_canonical(newH)
    cont = !check_convergence_left_canonical(newH, new_Ts; tol)
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
  return newH, Ts
end


##########################################
# function block_QR_for_right_canonical(H::Matrix{ITensor})
#   #We verify that H is of the form
#   # 1
#   # b  V
#   # c  d  1
#   l1, l2 = size(H)
#   (l1 != 3 || l2 != 3) &&
#     error("Format of the InfiniteMPO Matrix incompatible with current implementation")
#   for y in 2:l2
#     for x in 1:(y - 1)
#       !isempty(H[x, y]) &&
#         error("Format of the InfiniteMPO Matrix incompatible with current implementation")
#     end
#   end
#
#   #=Following https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147  (but with opposite conventions)
#   we look for
#   1      =   1  0   x  1
#   b  V       t  R      b'  V'
#   where the vectors (V', d') are orthogonal to each other, and to (1, 0)
#   =#
#
#   #We start by orthogonalizing with respect to 1
#   # this corresponds to d' = d - tr(d)/tr(1) * 1, R = 1 and t = tr(d)/tr(1)
#   s = commoninds(H[1, 1], H[end, end])
#   new_H = copy(H)
#   t1 = tr(new_H[2, 1]) / tr(H[1, 1])
#   new_H[2, 1] .= new_H[2, 1] .- t1 * H[1, 1]
#
#   kept_inds = commoninds(new_H[2, 1], new_H[2, 2]);
#   to_fuse_ind = only(uniqueinds(new_H[2, 2], kept_inds))
#   dummy_index = Index(QN()=>1, dir = dir(to_fuse_ind), tags = tags(to_fuse_ind) )
#   dummy_tensor = ITensor(dummy_index); dummy_tensor[1] = 1
#   temp_M, fused_ind= directsum( new_H[2, 1]*dummy_tensor => dummy_index, new_H[2, 2] => to_fuse_ind;
#     tags = tags(dummy_index) )
#   unfuse2, unfuse1 = ITensors.directsum_itensors(dummy_index, to_fuse_ind, fused_ind)
#   unfuse2 = unfuse2 * dummy_tensor
#
#   left_ind = only(uniqueinds(kept_inds, s))
#   cL = combiner(left_ind)
#   cR = combiner(fused_ind; tags=tags(fused_ind))
#   cLind = combinedind(cL)
#   cRind = combinedind(cR)
#   temp_M = cL * temp_M * cR
#   unfuse1 = unfuse1 * cR
#   unfuse2 = unfuse2 * cR
#
#   Q, R, new_left_ind = qr(
#      temp_M,
#      uniqueinds(temp_M, cLind),
#      tags=tags(left_ind),
#      positive=true,
#      dir=dir(left_ind),
#      dilatation=sqrt(tr(H[3, 3])),
#      full=false,
#    );
#    R = R * dag(cL)
#    new_H[2, 2] = dag(unfuse1) * Q
#    new_H[2, 1] = dag(unfuse2) * Q
#
#   #TODO decide whether I give R and t or a matrix?
#   return new_H, R, t1
# end

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
  t1 = tr(new_H[2, 1]) / tr(H[1, 1])
  new_H[2, 1] .= new_H[2, 1] .- t1 * H[1, 1]

  kept_inds = commoninds(new_H[2, 1], new_H[2, 2]);
  to_fuse_ind = only(uniqueinds(new_H[2, 2], kept_inds))
  dummy_index = Index(QN()=>1, dir = dir(to_fuse_ind), tags = tags(to_fuse_ind) )
  dummy_tensor = ITensor(dummy_index); dummy_tensor[1] = 1
  temp_M, fused_ind= directsum( new_H[2, 1]*dummy_tensor => dummy_index, new_H[2, 2] => to_fuse_ind;
    tags = tags(dummy_index) )
  unfuse2, unfuse1 = ITensors.directsum_itensors(dummy_index, to_fuse_ind, fused_ind)
  unfuse2 = unfuse2 * dummy_tensor

  left_ind = only(uniqueinds(kept_inds, s))
  cL = combiner(left_ind)
  cR = combiner(fused_ind; tags=tags(fused_ind))
  cLind = combinedind(cL)
  cRind = combinedind(cR)
  temp_M = cL * temp_M * cR
  unfuse1 = unfuse1 * cR
  unfuse2 = unfuse2 * cR

  Q, R, new_left_ind = qr(
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
  T[3, 1] = ITensor(0); T[1, 3] = ITensor(0)
  T[2, 2] = R; T[2, 1] = t1;
  left_ind = only(commoninds(R, t1));
  right_ind = only(uniqueinds(R, t1));
  T[2, 3] = ITensor(left_ind);
  T[1, 2] = ITensor(right_ind); T[3, 2] = ITensor(right_ind);
  return new_H, T
end

# function apply_right_gauge_on_right(H::Matrix{ITensor}, R::ITensor, t::ITensor)
#   new_H = copy(H)
#   new_H[3, 2] = R * H[3, 2]
#   new_H[3, 1] = t * H[3, 2] + H[3, 1]
#   new_H[2, 2] = R * H[2, 2]
#   new_H[2, 1] = t * H[2, 2] + H[2, 1]
#   return new_H
# end

# function block_QR_for_right_canonical(H::InfiniteMPOMatrix)
#   #We verify that H is of the form
#   # 1
#   # b  V
#   # c  d  1
#   new_H = CelledVector(copy(H.data.data), translator(H))
#   Rs = ITensor[]
#   ts = ITensor[]
#   for j in reverse(1:nsites(H))
#     new_H[j], R, t = block_QR_for_right_canonical(new_H[j])
#     new_H[j - 1] = apply_right_gauge_on_right(new_H[j - 1], R, t)
#     append!(Rs, [R])
#     append!(ts, [t])
#   end
#   return InfiniteMPOMatrix(new_H),
#   CelledVector(reverse(Rs), translator(new_H)),
#   CelledVector(reverse(ts), translator(new_H))
# end

function block_QR_for_right_canonical(H::InfiniteMPOMatrix)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  Ts = Matrix{ITensor}[]
  for j in reverse(1:nsites(H))
    new_H[j], T = block_QR_for_right_canonical(new_H[j])
    new_H[j - 1] = new_H[j - 1] * T
    append!(Ts, [T])
  end
  return InfiniteMPOMatrix(new_H),
  CelledVector(reverse(Ts), translator(new_H))
end

# function check_convergence_right_canonical(newH, Rs, ts; tol=1e-12)
#   for x in 1:nsites(newH)
#     li, ri = inds(Rs[x])
#     if li.space != ri.space
#       return false
#     end
#   end
#   for x in 1:nsites(newH)
#     if norm(tr(newH[x][2, 1])) > tol
#       return false
#     end
#   end
#   for x in 1:nsites(newH)
#     if norm(Rs[x] - denseblocks(δ(inds(Rs[x])...))) > tol
#       return false
#     end
#   end
#   return true
# end


function check_convergence_right_canonical(newH, Ts; tol=1e-12)
  for x in 1:nsites(newH)
    li, ri = inds(Ts[x][2, 2])
    if dim(li) != dim(ri)#li.space != ri.space
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][2, 1])) > tol
      return false
    end
  end
  for x in 1:nsites(newH)
    primed_ind = inds(Ts[x][2, 2])[1]
    Td = prime(Ts[x][2, 2], primed_ind)
    normed = norm(Ts[x][2, 2]*dag(Td) - denseblocks(δ(primed_ind, prime(dag(primed_ind)))))
    if normed > tol
      return false
    end
  end
  return true
end


# function right_canonical(H; tol=1e-12, max_iter=50)
#   l1, l2 = size(H[1])
#   if (l1 != 3 || l2 != 3)
#     H = make_block(H)
#   end
#
#   newH, Rs, ts = block_QR_for_right_canonical(H)
#   if check_convergence_right_canonical(newH, Rs, ts; tol)
#     println("Right canonicalized in 1 iterations")
#     return newH, Rs, ts
#   end
#   j = 1
#   cont = true
#   while j <= max_iter && cont
#     newH, new_Rs, new_ts = block_QR_for_right_canonical(newH)
#     cont = !check_convergence_right_canonical(newH, new_Rs, new_ts; tol)
#     for j in 1:nsites(newH)
#       ts[j] = ts[j] + Rs[j] * new_ts[j]
#       Rs[j] = new_Rs[j] * Rs[j]
#     end
#     j += 1
#   end
#   if j == max_iter + 1 && cont
#     println("Warning: reached max iterations before convergence")
#   else
#     println("Right canonicalized in $j iterations")
#   end
#   return newH, Rs, ts
# end

function right_canonical(H; tol=1e-12, max_iter=50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  newH, Ts = block_QR_for_right_canonical(H)
  if check_convergence_right_canonical(newH, Ts; tol)
    println("Right canonicalized in 1 iterations")
    return newH, Ts
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Ts = block_QR_for_right_canonical(newH)
    cont = !check_convergence_right_canonical(newH, new_Ts; tol)
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
  return newH, Ts
end

# function compress_impo(H::InfiniteMPOMatrix; kwargs...)
#   smallH = make_block(H)
#   HL, = left_canonical(smallH)
#   HR, = right_canonical(HL)
#   HL, Rs, Ts = left_canonical(HR)
#   #At this point, we have HL[1]*Rs[1] = Rs[0] * HR[1] etc
#   if maximum(norm.(Ts)) > 1e-12
#     println(maximum(norm.(Ts)))
#     error("Ts should be 0 at this point")
#   end
#   Us = Vector{ITensor}(undef, nsites(H))
#   Ss = Vector{ITensor}(undef, nsites(H))
#   Vsd = Vector{ITensor}(undef, nsites(H))
#   for x in 1:nsites(H)
#     Us[x], Ss[x], Vsd[x] = svd(
#       Rs[x],
#       commoninds(Rs[x], HL[x][2, 2]);
#       lefttags=tags(only(commoninds(Rs[x], HL[x][2, 2]))),
#       righttags=tags(only(commoninds(Rs[x], HR[x + 1][2, 2]))),
#       kwargs...
#     )
#     #println(sum(diag(Ss[x]) .^ 2))
#   end
#   Us = CelledVector(Us, translator(H))
#   Vsd = CelledVector(Vsd, translator(H))
#   Ss = CelledVector(Ss, translator(H))
#   newHL = copy(HL.data.data)
#   newHR = copy(HR.data.data)
#   for x in 1:nsites(H)
#     ## optimizing the left canonical
#     newHL[x][2, 1] =  dag(Us[x - 1]) * newHL[x][2, 1]
#     newHL[x][3, 2] = newHL[x][3, 2] * Us[x]
#     newHL[x][2, 2] = dag(Us[x - 1]) * newHL[x][2, 2] * Us[x]
#     ## optimizing the right canonical
#     newHR[x][2, 1] = Vsd[x - 1] * newHR[x][2, 1]
#     newHR[x][3, 2] = newHR[x][3, 2] * dag(Vsd[x])
#     newHR[x][2, 2] = Vsd[x - 1] * newHR[x][2, 2] * dag(Vsd[x])
#   end
#   return InfiniteMPOMatrix(newHL, translator(H)), InfiniteMPOMatrix(newHR, translator(H))
# end


function compress_impo(H::InfiniteMPOMatrix; kwargs...)
  smallH = make_block(H)
  HL, = left_canonical(smallH)
  HR, = right_canonical(HL)
  HL, Ts = left_canonical(HR)
  #At this point, we hav<e HL[1]*Rs[1] = Rs[0] * HR[1] etc
  test_norm = maximum([norm(Ts[x][3, 2]) for x in 1:nsites(H)])
  if  test_norm> 1e-12
    error("Ts should be 0 at this point, instead it is $test_norm")
  end
  #TODO ? replace this by matrix formmulation?
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
    #println(sum(diag(Ss[x]) .^ 2))
  end
  Us = CelledVector(Us, translator(H))
  Vsd = CelledVector(Vsd, translator(H))
  Ss = CelledVector(Ss, translator(H))
  newHL = copy(HL.data.data)
  newHR = copy(HR.data.data)
  for x in 1:nsites(H)
    ## optimizing the left canonical
    newHL[x][2, 1] =  dag(Us[x - 1]) * newHL[x][2, 1]
    newHL[x][3, 2] = newHL[x][3, 2] * Us[x]
    newHL[x][2, 2] = dag(Us[x - 1]) * newHL[x][2, 2] * Us[x]
    newHL[x][1, 2] = ITensor(commoninds(newHL[x][2, 2], newHL[x][3, 2]))
    newHL[x][2, 3] = ITensor(commoninds(newHL[x][2, 2], newHL[x][2, 1]))
    ## optimizing the right canonical
    newHR[x][2, 1] = Vsd[x - 1] * newHR[x][2, 1]
    newHR[x][3, 2] = newHR[x][3, 2] * dag(Vsd[x])
    newHR[x][2, 2] = Vsd[x - 1] * newHR[x][2, 2] * dag(Vsd[x])
    newHR[x][1, 2] = ITensor(commoninds(newHR[x][2, 2], newHR[x][3, 2]) )
    newHR[x][2, 3] = ITensor(commoninds(newHR[x][2, 2], newHR[x][2, 1]) )
  end
  return InfiniteMPOMatrix(newHL, translator(H)), InfiniteMPOMatrix(newHR, translator(H))
end


function make_block(H::InfiniteMPOMatrix)
  if size(H[1]) == (3, 3)
    println("Nothing to do")
    return H
  end
  H = make_dummy_indices_explicit(H)
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


function convert_itensor_33matrix(tensor; leftdir = ITensors.Out)
  @assert order(tensor) == 4
  sit = filterinds(inds(tensor), tags="Site")
  local_sit = noprime(only(filterinds(sit, plev = 1)))
  temp = uniqueinds(tensor, sit)
  if dir(temp[1]) == leftdir
    left_ind = temp[1]
    right_ind = temp[2]
  else
    left_ind = temp[2]
    right_ind = temp[1]
  end
  len_left_ind = length(left_ind.space)
  len_right_ind = length(right_ind.space)

  blocks = Matrix{Vector{Block{4}}}(undef, 3, 3)
  for i in 1:3, j in 1:3
    blocks[i, j] = Block{4}[]
  end
  T = permute(tensor, sit..., left_ind, right_ind; allow_alias=true)
  for (n, b) in enumerate(eachnzblock(T))
    if b[3] == 1
      if b[4] == len_right_ind
        append!(blocks[1, 3], [b])
      elseif b[4] != 1
        append!(blocks[1, 2], [b])
      else
        append!(blocks[1, 1], [b])
      end
    elseif b[3] == len_left_ind
      if b[4] == 1
        append!(blocks[3, 1], [b])
      elseif b[4] != len_right_ind
        append!(blocks[3, 2], [b])
      else
        append!(blocks[3, 3], [b])
      end
    else
      if b[4] == 1
        append!(blocks[2, 1], [b])
      elseif b[4] == len_right_ind
        append!(blocks[2, 3], [b])
      else
        append!(blocks[2, 2], [b])
      end
    end
  end
  new_left_ind = Index(left_ind.space[2:end-1], dir = dir(left_ind), tags = tags(left_ind))
  new_right_ind = Index(right_ind.space[2:end-1], dir = dir(right_ind), tags = tags(right_ind))

  matrix =  fill(op("Zero", local_sit), 3, 3)
  identity = op("Id", local_sit)
  matrix[1, 1] = identity; matrix[3, 3] = identity
  if !isempty(blocks[1, 2])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{3}[Block(b[1], b[2], b[4]-1) for b in blocks[1, 2]], (sit..., new_right_ind))
    for b in blocks[1, 2]
      ITensors.blockview(temp, Block(b[1], b[2], b[4]-1)) .= T[b]
    end
    matrix[1, 2] = itensor(temp)
  end
  if !isempty(blocks[2, 1])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{3}[Block(b[1], b[2], b[3]-1) for b in blocks[2, 1]], (sit..., new_left_ind))
    for b in blocks[2, 1]
      ITensors.blockview(temp, Block(b[1], b[2], b[3]-1)) .= T[b]
    end
    matrix[2, 1] = itensor(temp)
  end
  if !isempty(blocks[2, 2])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{4}[Block(b[1], b[2], b[3]-1, b[4]-1) for b in blocks[2, 2]], (sit..., new_left_ind, new_right_ind))
    for b in blocks[2, 2]
      ITensors.blockview(temp, Block(b[1], b[2], b[3]-1, b[4]-1)) .= T[b]
    end
    matrix[2, 2] = itensor(temp)
  end
  if !isempty(blocks[2, 3])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{3}[Block(b[1], b[2], b[3]-1) for b in blocks[2, 3]], (sit..., new_left_ind))
    for b in blocks[2, 3]
      ITensors.blockview(temp, Block(b[1], b[2], b[3]-1)) .= T[b]
    end
    matrix[2, 3] = itensor(temp)
  end
  #if !isempty(blocks[3, 2])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{3}[Block(b[1], b[2], b[4]-1) for b in blocks[3, 2]], (sit..., new_right_ind))
    for b in blocks[3, 2]
      ITensors.blockview(temp, Block(b[1], b[2], b[4]-1)) .= T[b]
    end
    matrix[3, 2] = itensor(temp)
  #end
  if length(blocks[1, 3]) != 0 || length(blocks[3, 1]) != 0
    error("Terms not yet taken into account")
  end
  return matrix
end



function convert_itensor_3matrix(tensor; leftdir = ITensors.Out)
  @assert order(tensor) == 3
  sit = filterinds(inds(tensor), tags="Site")
  local_sit = noprime(only(filterinds(sit, plev = 1)))
  manip_ind = only(uniqueinds(tensor, sit))
  len_manip_ind = length(manip_ind.space)

  blocks = Vector{Vector{Block{3}}}(undef, 3)
  for i in 1:3
    blocks[i] = Block{3}[]
  end
  T = permute(tensor, sit..., manip_ind; allow_alias=true)
  for (n, b) in enumerate(eachnzblock(T))
    if b[3] == 1
      append!(blocks[1], [b])
    elseif b[3] == len_manip_ind
      append!(blocks[3], [b])
    else
      append!(blocks[2], [b])
    end
  end
  new_manip_ind = Index(manip_ind.space[2:end-1], dir = dir(manip_ind), tags = tags(manip_ind))

  if dir(manip_ind) == leftdir
    matrix =  fill(op("Zero", local_sit), 3, 1)
  else
    matrix =  fill(op("Zero", local_sit), 1, 3)
  end

  if !isempty(blocks[1])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{2}[Block(b[1], b[2]) for b in blocks[1]], (sit...,))
    for b in blocks[1]
      ITensors.blockview(temp, Block(b[1], b[2])) .= T[b]
    end
    matrix[1] = itensor(temp)
  end
  if !isempty(blocks[2])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{3}[Block(b[1], b[2], b[3]-1) for b in blocks[2]], (sit..., new_manip_ind))
    for b in blocks[2]
      ITensors.blockview(temp, Block(b[1], b[2], b[3]-1)) .= T[b]
    end
    matrix[2] = itensor(temp)
  else
    matrix[2] = ITensor(sit..., new_manip_ind)
  end
  if !isempty(blocks[3])
    temp = ITensors.BlockSparseTensor(eltype(T), undef, Block{2}[Block(b[1], b[2]) for b in blocks[3]], (sit...,))
    for b in blocks[3]
      ITensors.blockview(temp, Block(b[1], b[2])) .= T[b]
    end
    matrix[3] = itensor(temp)
  end
  return matrix
end

function convert_itensor_matrix(tensor; leftdir = ITensors.Out)
  order(tensor) == 3 && return convert_itensor_3matrix(tensor; leftdir)
  return convert_itensor_33matrix(tensor; leftdir)
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
      C[i, j] = A[i, 1] * B[1, j]
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
    C[i] = A[i, 1] * B[1]
    for k in 2:size(A, 2)
      if isempty(A[i, k]) || isempty(B[k])
        continue
      end
      C[i] = C[i] + A[i, k] * B[k]
    end
  end
  return C
end

function Base.:*(A::Vector{ITensor}, B::Matrix{ITensor})
  length(A) != size(B, 1) && error("Matrix sizes are incompatible")
  C = Vector{ITensor}(undef, size(B, 2))
  for j in 1:size(B, 2)
    C[j] = A[1] * B[1, j]
    for k in 2:size(B, 1)
      if isempty(A[k]) || isempty(B[k, j])
        continue
      end
      C[j] = C[j] + A[k] * B[k, j]
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
