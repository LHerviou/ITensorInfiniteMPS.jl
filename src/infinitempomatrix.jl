
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

#nrange(H::InfiniteMPOMatrix) = [size(H[j])[1] - 1 for j in 1:nsites(H)]
# function special_QR(H::Matrix{ITensor})
#   l1, l2 = size(H)
#   #=Check that the InfiniteMPOMatrix has compatible shape, i.e.,
#    1
#    H1  0
#    0   H2 0
#    ....
#    0 ....   0 Hn  1
#   =#
#   l1 != l2 && error("Non square matrix, not yet implemented")
#   for y in 2:l2-1
#     for x in 1:y
#       !isempty(H[x, y]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
#     end
#   end
#   for x in 1:l2-1
#     !isempty(H[x, end]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
#   end
#   for y in 1:l2-1
#     for x in y+2:l2
#       !isempty(H[x, y]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
#     end
#   end
#   #=Following https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147  (but with opposite conventions), we look for
#   0                        0                            R2
#   H2 0                     Q2  0                        0   R3
#   ....               =     ....                    x    .....     Rn
#   0 ....   0 Hn  1          0 .......     Qn  1                   c   1
#   where Hm = Qm Rm for m < n, c = <1 Hn > (operator scalar product) and Hn - 1xc = Qn Rn
#  =#
#  s =inds(H[1, 1])
#  left_tag = tags(only(uniqueinds(H[end, end-1], s)))
#  right_tag = tags(only(uniqueinds(H[2, 1], s)))
#  leftinds = [only(filterinds(H[n+1, n]; tags = left_tag))  for n in 2:l1-1]
#  Qs = ITensor[];
#  Rs = ITensor[];
#  c = dag(H[end, end])*H[end, end-1] / dim(s[1])
#  for n in 3:l1-1
#    qb, rb = qr(H[n, n-1], (s..., leftinds[n-2]); tags = right_tag, positive = true, dilatation = sqrt(dim(s[1])), dir = -dir(leftinds[n-2]))
#    append!(Qs, [qb])
#    append!(Rs, [rb])
#  end
#  qlast = H[end, end-1] - c * H[end, end]
#  rlast = sqrt.((dag(qlast) * qlast) /  dim(s[1]) )
#  qlast = qlast / rlast[1]
#  #qlast, rlast = qr(temp, (s..., leftinds[end]); tags = right_tag, positive = true, dilatation = sqrt(dim(s[1])), dir = -dir(leftinds[end]))
#  append!(Qs, [qlast])
#  append!(Rs, [rlast])
#  return Qs, Rs, -c
# end
function reversing_δ(ind)
  bs = [Block(x, x) for x in 1:length(ind.space)]
  ten = ITensors.BlockSparseTensor(Float64, undef,  bs, (dag(ind), prime(ind)))
  for x in 1:length(bs)
    temp = diagm(0=>ones(ind.space[x][2])); temp = temp[end:-1:1, :]
    ITensors.blockview(ten, bs[x]) .= temp
  end
  return itensor(ten)
end

#nrange(H::InfiniteMPOMatrix) = [size(H[j])[1] - 1 for j in 1:nsites(H)]
function block_QR_for_left_canonical(H::Matrix{ITensor})
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  l1, l2 = size(H)
  (l1 != 3 || l2 != 3) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  for y in 2:l2
    for x in 1:y-1
      !isempty(H[x, y]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
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
 t1 = tr(new_H[3, 2])/tr(H[3, 3])
 new_H[3, 2] .= new_H[3, 2] .- t1*H[3, 3]
 #From there, we can do a QR of the joint tensor [d', V]. Because we want to reseparate, we need to take add the virtual index to the left of d'
 temp_M, left_ind, right_ind = matrixITensorToITensor(new_H[2:3, 2:2], s, ITensors.In, ITensors.Out; rev = false, init_all = false, init_left_last = true)
 cL = combiner(left_ind, tags = tags(left_ind)); cR = combiner(right_ind)
 cLind = combinedind(cL); cRind = combinedind(cR)
 #For now, to do the QR, we SVD the overlap matrix.
 temp_M = cL * temp_M * cR
 overlap_mat = dag(prime(temp_M, cRind)) * temp_M / tr(H[3, 3])
 u, lambda, v = svd(overlap_mat, [dag(prime(cRind))], full = false, cutoff = 1e-12, righttags = tags(right_ind))
 #we now use the orthonormality
 temp = sqrt.(lambda); temp = 1 ./ temp
 new_V = temp_M * noprime(u) * temp
 #println("Before")
 #println((dag(prime(new_V, commoninds(new_V, v))) * new_V).tensor)
 #new_right_ind = only(commoninds(new_V, v))
 R = dag(new_V) * temp_M# / tr(H[3, 3])
 Q, R, new_right_ind = qr(R, commoninds(R, new_V), tags = tags(right_ind), dir = dir(right_ind), positive = true, dilatation = 1, full = true)
 R = R*dag(cR)/ tr(H[3, 3])
 new_V = dag(cL) * new_V * Q
 #println("After")
 #println((dag(prime(new_V, commoninds(new_V, R))) * new_V).tensor)
 #println(norm(new_V*R - dag(cL)*temp_M*dag(cR)))
 #newV, R, qR_ind = qr(cL*temp_M*cR, [s..., cLind], tags = tags(right_ind), dir = dir(right_ind), positive = false, dilatation = 1)
 #newV, R, qR_ind = qr(tr(cL*temp_M*cR), [cLind], tags = tags(right_ind), dir = dir(right_ind), positive = false, dilatation = 1)
 #newV = dag(cL)*newV; R = R*dag(cR)

 #Now, we need to split newV into the actual new V and the new d
 original_left_ind = only(uniqueinds(new_H[2, 2], new_H[3, 2]))
 T = permute(new_V, s..., new_right_ind, left_ind, allow_alias = true)
 bs_V = Block{4}[]
 bs_d = Block{3}[]; bs_for_d = Block{4}[]
 for (n, b) in enumerate(eachnzblock(T))
   if b[4] == length(left_ind.space)
     append!(bs_for_d, [b])
     append!(bs_d, [Block(b[1], b[2], b[3])])
   else
     append!(bs_V, [b])
   end
 end
 final_V = ITensors.BlockSparseTensor(eltype(T), undef,  bs_V, (s..., new_right_ind, original_left_ind))
 final_d = ITensors.BlockSparseTensor(eltype(T), undef,  bs_d, (s..., new_right_ind))
 for (n, b) in enumerate(bs_V)
     ITensors.blockview(final_V, b) .= T[b]
 end
 for (n, b) in enumerate(bs_d)
     ITensors.blockview(final_d, b) .= T[bs_for_d[n]]
 end
 new_H[3, 2] = itensor(final_d)
 new_H[2, 2] = itensor(final_V)
 #TODO decide whether I give R and t or a matrix?
 return new_H, R, t1
end

function apply_left_gauge_on_left(H::Matrix{ITensor}, R::ITensor, t::ITensor)
  new_H = copy(H)
  new_H[2, 1] = R * H[2, 1]
  new_H[3, 1] = t * H[2, 1] + H[3, 1]
  new_H[2, 2] = R * H[2, 2]
  new_H[3, 2] = t*H[2, 2] + H[3, 2]
  return new_H
end

function block_QR_for_left_canonical(H::InfiniteMPOMatrix)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = copy(H.data)
  Rs = ITensor[]
  ts = ITensor[]
  for j in 1:nsites(H)
    new_H[j], R, t = block_QR_for_left_canonical(new_H[j])
    new_H[j+1] = apply_left_gauge_on_left(new_H[j+1], R, t)
    append!(Rs, [R]); append!(ts, [t])
  end
  return InfiniteMPOMatrix(new_H), CelledVector(Rs, translator(new_H)), CelledVector(ts, translator(new_H))
end

function check_convergence_left_canonical(newH, Rs, ts; tol = 1e-12)
  for x in 1:nsites(newH)
    li, ri = inds(Rs[x])
    if li.space != ri.space
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][3, 2])) > tol
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(Rs[x] - denseblocks(δ(inds(Rs[x])...))) > tol
      return false
    end
  end
  return true
end

function left_canonical(H; tol = 1e-12, max_iter = 50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  newH, Rs, ts = block_QR_for_left_canonical(H)
  if check_convergence_left_canonical(newH, Rs, ts; tol)
    return newH, Rs, ts
  end
  j=1; cont = true
  while j <= max_iter && cont
    newH, new_Rs, new_ts = block_QR_for_left_canonical(newH)
    cont = !check_convergence_left_canonical(newH, new_Rs, new_ts; tol)
    for j in 1:nsites(newH)
      ts[j] = ts[j] + Rs[j]*new_ts[j]
      Rs[j] = new_Rs[j] * Rs[j]
    end
    j+=1
  end
  if j == max_iter + 1 && !cont
    println("Warning: reached max iterations before convergence")
  else
    println("Left canonicalized in $j iterations")
  end
  return newH, Rs, ts
end

##########################################
function block_QR_for_right_canonical(H::Matrix{ITensor})
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  l1, l2 = size(H)
  (l1 != 3 || l2 != 3) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  for y in 2:l2
    for x in 1:y-1
      !isempty(H[x, y]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
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
 t1 = tr(new_H[2, 1])/tr(H[1, 1])
 new_H[2, 1] .= new_H[2, 1] .- t1*H[1, 1]
 #From there, we can do a QR of the joint tensor [d', V]. Because we want to reseparate, we need to take add the virtual index to the left of d'
 temp_M, left_ind, right_ind = matrixITensorToITensor(new_H[2:2, 1:2], s, ITensors.In, ITensors.Out; rev = false, init_all = false, init_right_first = true)
 cL = combiner(left_ind, tags = tags(left_ind)); cR = combiner(right_ind, tags = tags(right_ind))
 cLind = combinedind(cL); cRind = combinedind(cR)
 #For now, to do the QR, we SVD the overlap matrix.
 temp_M = cL * temp_M * cR
 overlap_mat = dag(prime(temp_M, cLind)) * temp_M / tr(H[1, 1])
 u, lambda, v = svd(overlap_mat, [dag(prime(cLind))], full = false, cutoff = 1e-12, righttags = tags(left_ind))
 #we now use the orthonormality
 temp = sqrt.(lambda); temp = 1 ./ temp
 new_V = temp_M * noprime(u) * temp
 #println("Before")
 #println((dag(prime(new_V, commoninds(new_V, v))) * new_V).tensor)
 #new_right_ind = only(commoninds(new_V, v))
 R = dag(new_V) * temp_M# / tr(H[3, 3])
 temp = reversing_δ(only(uniqueinds(R, new_V)))
 Q, R, new_left_ind = qr(R*temp, commoninds(R, new_V), tags = tags(left_ind), dir = dir(left_ind), positive = true, dilatation = 1, full = true)
 R = R*dag(temp)*dag(cL)/ tr(H[1, 1])
 temp_left = reversing_δ(new_left_ind) #Is this necessary?
 new_V = noprime(dag(cR) * new_V * Q * temp_left, tags=tags(new_left_ind))
 R = noprime(dag(temp_left)*R, tags=tags(new_left_ind))
 #println("After")
 #println((dag(prime(new_V, commoninds(new_V, R))) * new_V).tensor)
 #println(norm(new_V*R - dag(cL)*temp_M*dag(cR)))
 #newV, R, qR_ind = qr(cL*temp_M*cR, [s..., cLind], tags = tags(right_ind), dir = dir(right_ind), positive = false, dilatation = 1)
 #newV, R, qR_ind = qr(tr(cL*temp_M*cR), [cLind], tags = tags(right_ind), dir = dir(right_ind), positive = false, dilatation = 1)
 #newV = dag(cL)*newV; R = R*dag(cR)

 #Now, we need to split newV into the actual new V and the new d
 original_right_ind = only(uniqueinds(new_H[2, 2], new_H[2, 1]))
 T = permute(new_V, s..., new_left_ind, right_ind, allow_alias = true)
 bs_V = Block{4}[];  bs_for_V = Block{4}[]
 bs_d = Block{3}[]; bs_for_d = Block{4}[]
 for (n, b) in enumerate(eachnzblock(T))
   if b[4] == 1
     append!(bs_for_d, [b])
     append!(bs_d, [Block(b[1], b[2], b[3])])
   else
     append!(bs_for_V, [b])
     append!(bs_V, [Block(b[1], b[2], b[3], b[4]-1)])
   end
 end
 final_V = ITensors.BlockSparseTensor(eltype(T), undef,  bs_V, (s..., new_left_ind, original_right_ind))
 final_d = ITensors.BlockSparseTensor(eltype(T), undef,  bs_d, (s..., new_left_ind))
 for (n, b) in enumerate(bs_V)
     ITensors.blockview(final_V, b) .= T[bs_for_V[n]]
 end
 for (n, b) in enumerate(bs_d)
     ITensors.blockview(final_d, b) .= T[bs_for_d[n]]
 end
 new_H[2, 1] = itensor(final_d)
 new_H[2, 2] = itensor(final_V)
 #TODO decide whether I give R and t or a matrix?
 return new_H, R, t1
end

function apply_right_gauge_on_right(H::Matrix{ITensor}, R::ITensor, t::ITensor)
  new_H = copy(H)
  new_H[3, 2] = R * H[3, 2]
  new_H[3, 1] = t * H[3, 2] + H[3, 1]
  new_H[2, 2] = R * H[2, 2]
  new_H[2, 1] = t*H[2, 2] + H[2, 1]
  return new_H
end

function block_QR_for_right_canonical(H::InfiniteMPOMatrix)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = copy(H.data)
  Rs = ITensor[]
  ts = ITensor[]
  for j in reverse(1:nsites(H))
    new_H[j], R, t = block_QR_for_right_canonical(new_H[j])
    new_H[j-1] = apply_right_gauge_on_right(new_H[j-1], R, t)
    append!(Rs, [R]); append!(ts, [t])
  end
  return InfiniteMPOMatrix(new_H), CelledVector(reverse(Rs), translator(new_H)), CelledVector(reverse(ts), translator(new_H))
end

function check_convergence_right_canonical(newH, Rs, ts; tol = 1e-12)
  for x in 1:nsites(newH)
    li, ri = inds(Rs[x])
    if li.space != ri.space
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(tr(newH[x][2, 1])) > tol
      return false
    end
  end
  for x in 1:nsites(newH)
    if norm(Rs[x] - denseblocks(δ(inds(Rs[x])...))) > tol
      return false
    end
  end
  return true
end

function right_canonical(H; tol = 1e-12, max_iter = 50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  newH, Rs, ts = block_QR_for_right_canonical(H)
  if check_convergence_right_canonical(newH, Rs, ts; tol)
    println("Right canonicalized in 1 iterations")
    return newH, Rs, ts
  end
  j=1; cont = true
  while j <= max_iter && cont
    newH, new_Rs, new_ts = block_QR_for_right_canonical(newH)
    cont = !check_convergence_right_canonical(newH, new_Rs, new_ts; tol)
    for j in 1:nsites(newH)
      ts[j] = ts[j] + Rs[j]*new_ts[j]
      Rs[j] = new_Rs[j] * Rs[j]
    end
    j+=1
  end
  if j == max_iter + 1 && cont
    println("Warning: reached max iterations before convergence")
  else
    println("Right canonicalized in $j iterations")
  end
  return newH, Rs, ts
end

function compress_impo(H::InfiniteMPOMatrix; kwargs...)
  smallH = make_block(H)
  HL, = left_canonical(smallH)
  HR, = right_canonical(HL)
  HL, Rs, Ts = left_canonical(HR)
  #At this point, we have HL[1]*Rs[1] = Rs[0] * HR[1] etc
  if maximum(norm.(Ts)) > 1e-12
    println(maximum(norm.(Ts)))
    error("Ts should be 0 at this point")
  end
  Us = Vector{ITensor}(undef, nsites(H));
  Ss = Vector{ITensor}(undef, nsites(H));
  Vsd = Vector{ITensor}(undef, nsites(H));
  for x in 1:nsites(H)
    Us[x], Ss[x], Vsd[x] = svd(Rs[x], commoninds(Rs[x], HL[x][2, 2]); lefttags = tags(only(commoninds(Rs[x], HL[x][2, 2]))), righttags = tags(only(commoninds(Rs[x], HR[x+1][2, 2]))), kwargs...)
    println(minimum(diag(Ss[x])))
    println(sum(diag(Ss[x]).^2))
  end
  Us = CelledVector(Us, translator(H)); Vsd = CelledVector(Vsd, translator(H)); Ss= CelledVector(Ss, translator(H));
  newHL = copy(HL.data)
  newHR = copy(HR.data)
  for x in 1:nsites(H)
    ## optimizing the left canonical
    newHL[x][2, 1] = dag(Us[x-1]) * newHL[x][2, 1]
    newHL[x][3, 2] = newHL[x][3, 2] * Us[x]
    newHL[x][2, 2] = dag(Us[x-1]) * newHL[x][2, 2] * Us[x]
    ## optimizing the right canonical
    newHR[x][2, 1] = Vsd[x-1] * newHR[x][2, 1]
    newHR[x][3, 2] = newHR[x][3, 2] * dag(Vsd[x])
    newHR[x][2, 2] = Vsd[x-1] * newHR[x][2, 2] * dag(Vsd[x])
  end
  return InfiniteMPOMatrix(newHL), InfiniteMPOMatrix(newHR)
end



function matrixITensorToITensor(H::Matrix{ITensor}, com_inds, left_dir, right_dir; kwargs...)
  init_all = get(kwargs, :init_all, true)
  init_left_first = get(kwargs, :init_left_first, init_all)
  init_right_first = get(kwargs, :init_right_first, init_all)
  init_left_last = get(kwargs, :init_left_last, init_all)
  init_right_last = get(kwargs, :init_right_last, init_all)
  #TODO: fix the rev
  rev_all = get(kwargs, :rev_all, false)
  rev_left = get(kwargs, :rev_left, rev_all)
  rev_right = get(kwargs, :rev_right, rev_all)
  #TODO some fixing when rev and init are mixed up

  lx, ly = size(H)
  #Generate in order the leftbasis
  left_basis = valtype(com_inds)[]
  for j in 1:lx
    for k in 1:ly
      temp_ind = filter( x->dir(x) == left_dir, uniqueinds(H[j, k], com_inds))
      if length(temp_ind) == 1
        append!(left_basis, temp_ind)
        break
      end
    end
  end
  if init_left_first
    left_basis = [Index(QN() => 1, dir = left_dir, tags = length(left_basis) > 0 ? tags(left_basis[1]) : "left_link"), left_basis...]  #Dummy index for the first line
  end
  init_left_last && append!(left_basis, [Index(QN() => 1, dir = left_dir, tags =  length(left_basis) > 0 ? tags(left_basis[1]) : "left_link")]) #Dummy index for the last line

  right_basis = valtype(com_inds)[]
  for k in 1:ly
    for j in 1:lx
      temp_ind = filter( x->dir(x) == right_dir, uniqueinds(H[j, k], com_inds))
      if length(temp_ind)==1
        append!(right_basis, temp_ind)
        break
      end
    end
  end
  if init_right_first
    right_basis = [Index(QN() => 1, dir = right_dir, tags = length(right_basis) > 0 ? tags(right_basis[1]) : "right_link"), right_basis...] #Dummy index for the first column
  end
  init_right_last && append!(right_basis, [Index(QN() => 1, dir = right_dir, tags = length(right_basis) > 0 ? tags(right_basis[1]) : "right_link")]) #Dummy index for the last column

  left_block = Vector{Pair{QN, Int64}}()
  dic_inv_left_ind = Dict{Tuple{UInt64, Int64}, Int64}()
  for index in left_basis
    for (n, qp) in enumerate(index.space)
      append!(left_block, [qp])
      dic_inv_left_ind[index.id, n] = length(left_block)
    end
  end
  new_left_index = length(left_basis) == 1 ? left_basis[1] : Index(left_block, dir = left_dir, tags = tags(left_basis[1]))

  right_block = Vector{Pair{QN, Int64}}()
  dic_inv_right_ind = Dict{Tuple{UInt64, Int64}, Int64}()
  for index in right_basis
    for (n, qp) in enumerate(index.space)
      append!(right_block, [qp])
      dic_inv_right_ind[index.id, n] = length(right_block)
    end
  end
  new_right_index = length(right_basis) == 1 ? right_basis[1] : Index(right_block, dir = right_dir, tags = tags(right_basis[1]))
  #Determine the non-zero blocks, not efficient in memory for now TODO: improve memory use
  temp_block = Block{4}[]
  elements = []
  dummy_left = ITensor(1, Index(QN() => 1)); dum_left_ind = only(inds(dummy_left))
  dummy_right = ITensor(1, Index(QN() => 1)); dum_right_ind = only(inds(dummy_right))
  for x in 1:lx
    for y in 1:ly
      isempty(H[x, y]) && continue
      tli = filter(x->dir(x) == dir(left_basis[1]), commoninds(H[x, y], left_basis))
      #This first part find which structure the local Ham has and ensure H[x, y] is properly ordered
      case = 0
      if !isempty(tli)
        li = only(tli)
        tri = filter(x->dir(x) == dir(right_basis[1]), commoninds(H[x, y], right_basis))
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y], com_inds..., li, ri, allow_alias = true)
          case = 3 #This is the default case, both legs exists
        else
          !(y == 1 || (x==lx && y ==ly )) && error("Incompatible leg")
          T = permute(H[x, y]*dummy_right, com_inds..., li, dum_right_ind, allow_alias = true)
          case = 1
        end
      else
        !(x == lx || (x==1 && y ==1 )) && error("Incompatible leg")
        tri = filter(x->dir(x) == dir(right_basis[1]), commoninds(H[x, y], right_basis))
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y]*dummy_left, com_inds..., dum_left_ind, ri, allow_alias = true)
          case = 2
        else
          !(y == 1 || (x==lx && y ==ly )) && error("Incompatible leg")
          T = permute(H[x, y]*dummy_left*dummy_right, com_inds..., dum_left_ind, dum_right_ind, allow_alias = true)
        end
      end
      for (n, b) in enumerate(eachnzblock(T))
        #TODO not completely ok for attribution to 1 and end when stuff is missing
        norm(T[b]) == 0 && continue
        if case == 0
          if x==1
            append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[1].id, 1], dic_inv_right_ind[right_basis[1].id, 1])])
          elseif x==lx
            append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[end].id, 1], dic_inv_right_ind[right_basis[end].id, 1])])
          else
            error("Something went wrong")
          end
        elseif case == 1
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[li.id, b[3]], dic_inv_right_ind[right_basis[1].id, 1])])
        elseif case == 2
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[end].id, 1], dic_inv_right_ind[ri.id, b[4]])])
        elseif case == 3 #Default case
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[li.id, b[3]], dic_inv_right_ind[ri.id, b[4]])])
        else
          println("Not treated case")
        end
        append!(elements, [T[b]])
      end
    end
  end
  Hf = ITensors.BlockSparseTensor(eltype(elements[1]), undef,  temp_block, (com_inds..., new_left_index, new_right_index))
  for (n, b) in enumerate(temp_block)
      ITensors.blockview(Hf, b) .= elements[n]
  end
  return itensor(Hf), new_left_index, new_right_index
end

function matrixITensorToITensor(H::Vector{ITensor}, com_inds; rev = false, kwargs...)
  rev && error("not yet implemented")
  init_all = get(kwargs, :init_all, true)
  init_first = get(kwargs, :init_first, init_all)
  init_last = get(kwargs, :init_last, init_all)

  lx = length(H)
  #Generate in order the leftbasis
  left_basis = valtype(com_inds)[] #Dummy index for the first line
  for j in 1:lx
    append!(left_basis, uniqueinds(H[j], com_inds))
  end
  left_dir = dir(left_basis[1])
  if init_first
    left_basis = [Index(QN() => 1, dir = left_dir), left_basis...]
  end
  init_last && append!(left_basis, [Index(QN() => 1, dir = left_dir)]) #Dummy index for the last line


  left_block = Vector{Pair{QN, Int64}}()
  dic_inv_left_ind = Dict{Tuple{UInt64, Int64}, Int64}()
  for index in left_basis
    for (n, qp) in enumerate(index.space)
      append!(left_block, [qp])
      dic_inv_left_ind[index.id, n] = length(left_block)
    end
  end
  new_left_index = Index(left_block, dir = left_dir, tags = "left_link")

  #Determine the non-zero blocks, not efficient in memory for now TODO: improve memory use
  temp_block = Block{3}[]
  elements = []
  for x in 1:lx
    isempty(H[x]) && continue
    tli = commoninds(H[x], left_basis)
    #This first part find which structure the local Ham has and ensure H[x, y] is properly ordered
    case = 0
    if !isempty(tli)
      li = only(tli)
      T = permute(H[x], com_inds..., li, allow_alias = true)
      case = 1 #This is the default case, the leg exists
    else
      !(x == lx || x==1 ) && error("Incompatible leg")
      T = permute(H[x], com_inds..., allow_alias = true)
    end
    for (n, b) in enumerate(eachnzblock(T))
      norm(T[b]) == 0 && continue
      if case == 0
        if x==1
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[1].id, 1])])
        elseif x==lx
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[end].id, 1])])
        else
          error("Something went wrong")
        end
      elseif case == 1 #Default case
        append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[li.id, b[3]])])
      else
        println("Not treated case")
      end
      append!(elements, [T[b]])
    end
  end
  Hf = ITensors.BlockSparseTensor(eltype(elements[1]), undef,  temp_block, (com_inds..., new_left_index))
  for (n, b) in enumerate(temp_block)
      ITensors.blockview(Hf, b) .= elements[n]
  end
  return itensor(Hf), new_left_index
end

function matrixITensorToITensor(H::Matrix{ITensor}; kwargs...)
  lx, ly = size(H)
  s =commoninds(H[1, 1], H[end, end])
  right_ind = uniqueinds(H[end, end], s)
  for j in reverse(1:ly-1)
    length(right_ind) == 1 && break
    right_ind = uniqueinds(H[end, j], s)
  end
  length(right_ind) != 1 && error("Not able to isolate the right index")
  left_ind = uniqueinds(H[end, end], s)
  for j in 2:lx
    left_ind = uniqueinds(H[j, 1], s)
    length(left_ind) == 1 && break
  end
  length(left_ind) != 1 && error("Not able to isolate the left index")
  dir_left_ind = dir(only(left_ind))
  dir_right_ind = dir(only(right_ind))
  #return matrixITensorToITensor(H, s, left_tag, right_tag; dir_left_ind, dir_right_ind)
  return matrixITensorToITensor(H, s, dir_left_ind, dir_right_ind; kwargs...)
end

function matrixITensorToITensor(H::Matrix{ITensor}, idleft, idright; kwargs...)
  lx, ly = size(H)
  s =commoninds(H[1, 1], H[end, end])
  right_ind = uniqueinds(H[end, end], s)
  for j in reverse(1:ly-1)
    length(right_ind) == 1 && break
    right_ind = uniqueinds(H[end, j], s)
  end
  length(right_ind) != 1 && error("Not able to isolate the right index")
  left_ind = uniqueinds(H[end, end], s)
  for j in 2:lx
    left_ind = uniqueinds(H[j, 1], s)
    length(left_ind) == 1 && break
  end
  length(left_ind) != 1 && error("Not able to isolate the left index")
  dir_left_ind = dir(only(left_ind))
  dir_right_ind = dir(only(right_ind))
  #return matrixITensorToITensor(H, s, left_tag, right_tag; dir_left_ind, dir_right_ind)
  if length(collect(idleft))==1 || length(collect(idright))==1
    return matrixITensorToITensor(H[idleft, idright], s; kwargs...)
  end
  return matrixITensorToITensor(H[idleft, idright], s, dir_left_ind, dir_right_ind; kwargs...)
end

function InfiniteMPO(H::InfiniteMPOMatrix)
  temp = matrixITensorToITensor.(H)
  new_H = CelledVector([x[1] for x in temp], translator(H))
  lis = CelledVector([x[2] for x in temp], translator(H))
  ris = CelledVector([x[3] for x in temp], translator(H))
  #retags the right_links
  s =[commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  for j in 1:nsites(H)
    newTag="Link,c=$(getcell(s[j])),n=$(getsite(s[j]))"
    temp = replacetags(ris[j], tags(ris[j]), newTag)
    new_H[j] = replaceinds(new_H[j], [ris[j]], [temp])
    ris[j] = replacetags(ris[j], tags(ris[j]), newTag)
  end
  # joining the indexes
  for j in 1:nsites(H)
    temp = δ(dag(ris[j]), dag(lis[j+1]))
    new_H[j+1] *= temp
  end
  #for j in 1
  return lis, ris, new_H
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
  temp = [matrixITensorToITensor(H[x], 2:lx-1, 2:ly-1; init_all = false) for x in 1:nsites(H)]
  new_H = CelledVector([x[1] for x in temp], translator(H))
  lis = CelledVector([x[2] for x in temp], translator(H))
  ris = CelledVector([x[3] for x in temp], translator(H))
  #for t
  temp = [matrixITensorToITensor(H[x], 2:lx-1, 1; init_all = false) for x in 1:nsites(H)]
  new_t = CelledVector([x[1] for x in temp], translator(H))
  lis_t = CelledVector([x[2] for x in temp], translator(H))
  #for b
  temp = [matrixITensorToITensor(H[x], lx, 2:ly-1; init_all = false) for x in 1:nsites(H)]
  new_b = CelledVector([x[1] for x in temp], translator(H))
  lis_b = CelledVector([x[2] for x in temp], translator(H))

  #retags the right_links
  s =[commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  for j in 1:nsites(H)
    newTag="Link,c=$(getcell(s[j])),n=$(getsite(s[j]))"
    temp = replacetags(ris[j], tags(ris[j]), newTag)
    new_H[j] = replaceinds(new_H[j], [ris[j]], [temp])
    ris[j] = replacetags(ris[j], tags(ris[j]), newTag)
  end
  # joining the indices
  for j in 1:nsites(H)
    temp = δ(dag(ris[j]), dag(lis[j+1]))
    new_H[j+1] *= temp
  end
  #Fixing the indices for b and t, recalling new_H[j] - t[j+1] and b[j]*new_H[j+1]
  for j in 1:nsites(H)
    temp = δ(dag(ris[j-1]), dag(lis_t[j]))
    new_t[j] *= temp
    temp = δ(ris[j], dag(lis_b[j]))
    new_b[j] *= temp
  end

  new_mpo =  [ fill(op("Zero", s[x]), 3, 3) for x in 1:nsites(H)]
  for j in 1:nsites(H)
    new_mpo[j][1, 1]= H[j][1, 1]
    new_mpo[j][3, 3] = H[j][end, end]
    new_mpo[j][3, 1] = H[j][end, 1]
    new_mpo[j][2, 1] = new_t[j]
    new_mpo[j][2, 2] = new_H[j]
    new_mpo[j][3, 2] = new_b[j]
  end
  return InfiniteMPOMatrix(new_mpo, translator(H))
end


function make_dummy_indices_explicit(H::InfiniteMPOMatrix)
  lx, ly = size(H[1])
  s =[commoninds(H[j][1, 1], H[j][end, end])[1] for j in 1:nsites(H)]
  new_H =  [ fill(op("Zero", s[x]), lx, ly) for x in 1:nsites(H)]
  element = eltype(H[1][1, 1])

  for n in 1:nsites(H)
    s =commoninds(H[n][1, 1], H[n][end, end])
    #First, find all right indices, per column except first and last
    right_indices = []
    for y in 2:ly-1
      for x in y:lx
        right_ind = filter( x->dir(x) == ITensors.Out, uniqueinds(H[n][x, y], s))
        if length(right_ind)==1
          append!(right_indices, right_ind)
          break
        end
      end
    end
    length(right_indices) != ly-2 && error("Some right indices missing")
    #Second, find all left indices, per line except first and last
    left_indices = []
    for x in 2:lx-1
      for y in 1:x
        left_ind = filter( x->dir(x) == ITensors.In, uniqueinds(H[n][x, y], s))
        if length(left_ind)==1
          append!(left_indices, left_ind)
          break
        end
      end
    end
    length(left_indices) != lx-2 && error("Some left indices missing")
    #Now, filling the zeros
    for j in 1:lx
      for k in 1:ly
        if isempty(H[n][j, k])
          if (j == 1 || j == lx) && (k == 1 || k == ly)
            #just site indices
            new_H[n][j, k] = ITensor(element, s...)
          elseif (j == 1 || j == lx)
            #println(inds( ITensor(element, s..., right_indices[k-1])))
            new_H[n][j, k] = ITensor(element, s..., right_indices[k-1])
          elseif (k == 1 || k == ly)
            new_H[n][j, k] = ITensor(element, s..., left_indices[j-1])
          else
            new_H[n][j, k] = ITensor(element, s..., left_indices[j-1], right_indices[k-1])
          end
        else
          new_H[n][j, k] = H[n][j, k]
        end
      end
    end
  end
  return InfiniteMPOMatrix(new_H, translator(H))
end
#
#
#   new_inds_to_concatenate = [Ind()]
#
#   centerind_space =  Vector{Pair{QN, Int64}}()
#   seen = Dict()
#   for (n, b) in enumerate(eachnzblock(T))
#       append!(centerind_space, [direction == dir(r) ? r.space[b[2]].first=>size(Rs[n], 1) : -r.space[b[2]].first=>size(Rs[n], 1)])
#       seen[b[2]] = length(seen)+1
#   end
#   centerind = Index(centerind_space, dir = direction, tags = inds(T)[2].tags)
#
#   indsQ = [inds(T)[1], centerind]
#   indsR =  [dag(centerind), inds(T)[2]]
#
#   Q = ITensors.BlockSparseTensor(ElT, undef,  [Block(b[1], seen[b[2]]) for (n, b) in enumerate(eachnzblock(T))], indsQ)
#   R = ITensors.BlockSparseTensor(ElT, undef, [Block(seen[b[2]], seen[b[2]]) for (n, b) in enumerate(eachnzblock(T))], indsR);
#   for (n, b) in enumerate(eachnzblock(T))
#       qb = Block(b[1], seen[b[2]])
#       rb = Block(seen[b[2]], seen[b[2]])
#       ITensors.blockview(Q, qb) .= Qs[n]
#       ITensors.blockview(R, rb) .= Rs[n]
#   end
#
#
#
# end
