
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
      !isempty(H[x, y]) &&
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
  #From there, we can do a QR of the joint tensor [V, d']^T. Because we want to reseparate, we need to take add the virtual index to the left of d'
  temp_M, left_ind, right_ind = matrixITensorToITensor(
    new_H[2:3, 2:2],
    s,
    ITensors.In,
    ITensors.Out;
    rev=false,
    init_all=false,
    init_left_last=true,
  )
  cL = combiner(left_ind; tags=tags(left_ind))
  cR = combiner(right_ind)
  cLind = combinedind(cL)
  cRind = combinedind(cR)

  temp_M = cL * temp_M * cR

  Q, R, new_right_ind = qr(
    temp_M,
    uniqueinds(temp_M, cRind),
    tags=tags(right_ind),
    positive=true,
    dir=dir(right_ind),
    dilatation=sqrt(tr(H[3, 3])),
    full=false,
  );
  R = R * dag(cR)
  new_V = dag(cL) *  Q

  #Now, we need to split newV into the actual new V and the new d
  original_left_ind = only(uniqueinds(new_H[2, 2], new_H[3, 2]))
  T = permute(new_V, s..., new_right_ind, left_ind; allow_alias=true)
  bs_V = Block{4}[]
  bs_d = Block{3}[]
  bs_for_d = Block{4}[]
  for (n, b) in enumerate(eachnzblock(T))
    if b[4] == length(left_ind.space)
      append!(bs_for_d, [b])
      append!(bs_d, [Block(b[1], b[2], b[3])])
    else
      append!(bs_V, [b])
    end
  end
  final_V = ITensors.BlockSparseTensor(
    eltype(T), undef, bs_V, (s..., new_right_ind, original_left_ind)
  )
  final_d = ITensors.BlockSparseTensor(eltype(T), undef, bs_d, (s..., new_right_ind))
  for (n, b) in enumerate(bs_V)
    ITensors.blockview(final_V, b) .= T[b]
  end
  for (n, b) in enumerate(bs_d)
    ITensors.blockview(final_d, b) .= T[bs_for_d[n]]
  end
  new_H[3, 2] = itensor(final_d)
  new_H[2, 2] = itensor(final_V)
  #TODO decide whether I give R and t or a matrix?
  #println(( norm(new_H[2, 2] * R - H[2, 2]), norm(new_H[3, 2] * R + new_H[3, 3] * t1 - H[3, 2])))
  return new_H, R, t1
end

function apply_left_gauge_on_left(H::Matrix{ITensor}, R::ITensor, t::ITensor)
  new_H = copy(H)
  new_H[2, 1] = R * H[2, 1]
  new_H[3, 1] = t * H[2, 1] + H[3, 1]
  new_H[2, 2] = R * H[2, 2]
  new_H[3, 2] = t * H[2, 2] + H[3, 2]
  return new_H
end

function block_QR_for_left_canonical(H::InfiniteMPOMatrix)
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
  end
  return InfiniteMPOMatrix(new_H),
  CelledVector(Rs, translator(new_H)),
  CelledVector(ts, translator(new_H))
end

function check_convergence_left_canonical(newH, Rs, ts; tol=1e-12)
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

function left_canonical(H; tol=1e-12, max_iter=50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  newH, Rs, ts = block_QR_for_left_canonical(H)
  if check_convergence_left_canonical(newH, Rs, ts; tol)
    return newH, Rs, ts
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Rs, new_ts = block_QR_for_left_canonical(newH)
    cont = !check_convergence_left_canonical(newH, new_Rs, new_ts; tol)
    for j in 1:nsites(newH)
      ts[j] = ts[j] + Rs[j] * new_ts[j]
      Rs[j] = new_Rs[j] * Rs[j]
    end
    j += 1
  end
  if j == max_iter + 1 && cont
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
  (l1 != 3 || l2 != 3) &&
    error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  for y in 2:l2
    for x in 1:(y - 1)
      !isempty(H[x, y]) &&
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
  #From there, we can do a QR of the joint tensor [d', V]. Because we want to reseparate, we need to take add the virtual index to the left of d'
  temp_M, left_ind, right_ind = matrixITensorToITensor(
    new_H[2:2, 1:2],
    s,
    ITensors.In,
    ITensors.Out;
    rev=false,
    init_all=false,
    init_right_first=true,
  )
  cL = combiner(left_ind; tags=tags(left_ind))
  cR = combiner(right_ind; tags=tags(right_ind))
  cLind = combinedind(cL)
  cRind = combinedind(cR)
  #For now, to do the QR, we SVD the overlap matrix.
  temp_M = cL * temp_M * cR


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
  new_V = dag(cR) * Q

  #Now, we need to split newV into the actual new V and the new d
  original_right_ind = only(uniqueinds(new_H[2, 2], new_H[2, 1]))
  T = permute(new_V, s..., new_left_ind, right_ind; allow_alias=true)
  bs_V = Block{4}[]
  bs_for_V = Block{4}[]
  bs_d = Block{3}[]
  bs_for_d = Block{4}[]
  for (n, b) in enumerate(eachnzblock(T))
    if b[4] == 1
      append!(bs_for_d, [b])
      append!(bs_d, [Block(b[1], b[2], b[3])])
    else
      append!(bs_for_V, [b])
      append!(bs_V, [Block(b[1], b[2], b[3], b[4] - 1)])
    end
  end
  final_V = ITensors.BlockSparseTensor(
    eltype(T), undef, bs_V, (s..., new_left_ind, original_right_ind)
  )
  final_d = ITensors.BlockSparseTensor(eltype(T), undef, bs_d, (s..., new_left_ind))
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
  new_H[2, 1] = t * H[2, 2] + H[2, 1]
  return new_H
end

function block_QR_for_right_canonical(H::InfiniteMPOMatrix)
  #We verify that H is of the form
  # 1
  # b  V
  # c  d  1
  new_H = CelledVector(copy(H.data.data), translator(H))
  Rs = ITensor[]
  ts = ITensor[]
  for j in reverse(1:nsites(H))
    new_H[j], R, t = block_QR_for_right_canonical(new_H[j])
    new_H[j - 1] = apply_right_gauge_on_right(new_H[j - 1], R, t)
    append!(Rs, [R])
    append!(ts, [t])
  end
  return InfiniteMPOMatrix(new_H),
  CelledVector(reverse(Rs), translator(new_H)),
  CelledVector(reverse(ts), translator(new_H))
end

function check_convergence_right_canonical(newH, Rs, ts; tol=1e-12)
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

function right_canonical(H; tol=1e-12, max_iter=50)
  l1, l2 = size(H[1])
  if (l1 != 3 || l2 != 3)
    H = make_block(H)
  end

  newH, Rs, ts = block_QR_for_right_canonical(H)
  if check_convergence_right_canonical(newH, Rs, ts; tol)
    println("Right canonicalized in 1 iterations")
    return newH, Rs, ts
  end
  j = 1
  cont = true
  while j <= max_iter && cont
    newH, new_Rs, new_ts = block_QR_for_right_canonical(newH)
    cont = !check_convergence_right_canonical(newH, new_Rs, new_ts; tol)
    for j in 1:nsites(newH)
      ts[j] = ts[j] + Rs[j] * new_ts[j]
      Rs[j] = new_Rs[j] * Rs[j]
    end
    j += 1
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
  Us = Vector{ITensor}(undef, nsites(H))
  Ss = Vector{ITensor}(undef, nsites(H))
  Vsd = Vector{ITensor}(undef, nsites(H))
  for x in 1:nsites(H)
    Us[x], Ss[x], Vsd[x] = svd(
      Rs[x],
      commoninds(Rs[x], HL[x][2, 2]);
      lefttags=tags(only(commoninds(Rs[x], HL[x][2, 2]))),
      righttags=tags(only(commoninds(Rs[x], HR[x + 1][2, 2]))),
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
    ## optimizing the right canonical
    newHR[x][2, 1] = Vsd[x - 1] * newHR[x][2, 1]
    newHR[x][3, 2] = newHR[x][3, 2] * dag(Vsd[x])
    newHR[x][2, 2] = Vsd[x - 1] * newHR[x][2, 2] * dag(Vsd[x])
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
