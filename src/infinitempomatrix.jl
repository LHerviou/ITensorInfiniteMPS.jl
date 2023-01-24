
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


#nrange(H::InfiniteMPOMatrix) = [size(H[j])[1] - 1 for j in 1:nsites(H)]
function left_canonical(H::Matrix{ITensor})
  l1, l2 = size(H)
  #=Check that the InfiniteMPOMatrix has compatible shape, i.e.,
   1
   H21  0
   H31   H32 0
   ....
   Hn1 ....   0 Hnn  1
  =#
  l1 != l2 && error("Non square matrix, not yet implemented")
  for y in 2:l2-1
    for x in 1:y
      !isempty(H[x, y]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
    end
  end
  for x in 1:l2-1
    !isempty(H[x, end]) && error("Format of the InfiniteMPO Matrix incompatible with current implementation")
  end

  #=Following https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147  (but with opposite conventions), so we right canonalize everything,
  we look for
  0                        0                            R2
  H2 0                     Q2  0                        0   R3
  ....               =     ....                    x    .....     Rn
  0 ....   0 Hn  1          0 .......     Qn  1                   c   1
  where Hm = Qm Rm for m < n, c = <1 Hn > (operator scalar product) and Hn - 1xc = Qn Rn
 =#
 s =inds(H[1, 1])
 left_tag = tags(only(uniqueinds(H[end, end-1], s)))
 right_tag = tags(only(uniqueinds(H[2, 1], s)))
 leftinds = [only(filterinds(H[n+1, n]; tags = left_tag))  for n in 2:l1-1]
 return matrixITensorToITensor(H, s, left_tag, right_tag; dir_left_ind = dir(leftinds[1]), dir_right_ind = -dir(leftinds[1]))
end

#build a local iMPO
#=convert                        into
 1                                      1
 H21  0                                 a W
 H31   H32 0                            b c 1
 ....
 Hn1 ....   0 Hnn  1
=#
# function matrixITensorToITensor(H::Matrix{ITensor}, com_inds, left_tag, right_tag; rev = false, dir_left_ind = In, dir_right_ind = Out)
#   rev && error("not yet implemented")
#
#   lx, ly = size(H)
#   #Generate in order the leftbasis
#   left_block = Vector{Pair{QN, Int64}}()
#   dic_inv_left_ind = Dict{Tuple{Int64, Int64}, Int64}()
#   for j in 1:lx
#     temp_left = filterinds(H[j, 1], tags = left_tag)
#     if length(temp_left) == 0
#       for k in 2:ly
#         temp_left = filterinds(H[j, k], tags = left_tag)
#         length(temp_left) != 0 && break
#       end
#     end
#     if length(temp_left) == 0 #we add a dummy index
#       append!(left_block, [QN() => 1])
#       dic_inv_left_ind[j, 1] = length(left_block)
#       continue
#     end
#     for (idx, subspace) in enumerate(temp_left[1].space)
#       append!(left_block, [subspace])
#       dic_inv_left_ind[j, idx] = length(left_block)
#     end
#   end
#   new_left_index = Index(left_block, dir = dir_left_ind, tags = left_tag)
#   #Generate in order the right basis
#   right_block = Vector{Pair{QN, Int64}}()
#   dic_inv_right_ind = Dict{Tuple{Int64, Int64}, Int64}()
#   for j in 1:ly
#     temp_right = filterinds(H[1, j], tags = right_tag)
#     if length(temp_right) == 0
#       for k in 2:lx
#         temp_right = filterinds(H[k, j], tags = right_tag)
#         length(temp_right) != 0 && break
#       end
#     end
#     if length(temp_right) == 0 #we add a dummy index
#       append!(right_block, [QN() => 1])
#       dic_inv_right_ind[j, 1] = length(right_block)
#       continue
#     end
#     for (idx, subspace) in enumerate(temp_right[1].space)
#       append!(right_block, [subspace])
#       dic_inv_right_ind[j, idx] = length(right_block)
#     end
#   end
#   new_right_index = Index(right_block, dir = dir_right_ind, tags = right_tag)
#
#   #Determine the non-zero blocks
#   temp_block = Block{4}[]
#   for x in 1:lx
#     for y in 1:ly
#       isempty(H[x, y]) && continue
#       tli = filterinds(H[x, y], tags = left_tag)
#       case = 0
#       if !isempty(tli)
#         li = only(tli)
#         tri = filterinds(H[x, y], tags = right_tag)
#         if !isempty(tri)
#           ri = only(tri)
#           T = permute(H[x, y], com_inds..., li, ri, allow_alias = true)
#           case = 3
#         else
#           T = permute(H[x, y], com_inds..., li, allow_alias = true)
#           case = 1
#         end
#       else
#         tri = filterinds(H[x, y], tags = right_tag)
#         if !isempty(tri)
#           ri = only(tri)
#           T = permute(H[x, y], com_inds..., ri, allow_alias = true)
#           case = 2
#         else
#           T = permute(H[x, y], com_inds..., allow_alias = true)
#         end
#       end
#       for (n, b) in enumerate(eachnzblock(T))
#         if case == 0
#           append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[x, 1], dic_inv_right_ind[y, 1])])
#         elseif case == 1
#           append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[x, b[3]], dic_inv_right_ind[y, 1])])
#         elseif case == 2
#           append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[x, 1], dic_inv_right_ind[y, b[3]])])
#         elseif case == 3
#           append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[x, b[3]], dic_inv_right_ind[y, b[4]])])
#         end
#       end
#     end
#   end
#
#   Hf = ITensors.BlockSparseTensor(eltype(H[1]), undef,  temp_block, (com_inds..., new_left_index, new_right_index))
#   idx_block = 1
#   for x in 1:lx
#     for y in 1:ly
#       isempty(H[x, y]) && continue
#       tli = filterinds(H[x, y], tags = left_tag)
#       if !isempty(tli)
#         li = only(tli)
#         tri = filterinds(H[x, y], tags = right_tag)
#         if !isempty(tri)
#           ri = only(tri)
#           T = permute(H[x, y], com_inds..., li, ri, allow_alias = true)
#           case = 3
#         else
#           T = permute(H[x, y], com_inds..., li, allow_alias = true)
#           case = 1
#         end
#       else
#         tri = filterinds(H[x, y], tags = right_tag)
#         if !isempty(tri)
#           ri = only(tri)
#           T = permute(H[x, y], com_inds..., ri, allow_alias = true)
#           case = 2
#         else
#           T = permute(H[x, y], com_inds..., allow_alias = true)
#         end
#       end
#       for (n, b) in enumerate(eachnzblock(T))
#           ITensors.blockview(Hf, temp_block[idx_block]) .= T[n]
#           idx_block += 1
#       end
#     end
#   end
#   return itensor(Hf), new_left_index, new_right_index
# end

function matrixITensorToITensor(H::Matrix{ITensor}, com_inds, left_dir, right_dir; rev = false, init_left = true, init_right = true)
  rev && error("not yet implemented")

  lx, ly = size(H)
  #Generate in order the leftbasis
  left_basis = init_left ? [Index(QN() => 1, dir = left_dir)] : (valtype(com_inds)[]) #Dummy index for the first index
  for k in 1:ly
    for j in 1:lx
      append!(left_basis, filter( x->dir(x) == left_dir, uniqueinds(H[j, k], com_inds)))
    end
  end
  init_right && append!(left_basis, [Index(QN() => 1, dir = left_dir)]) #Dummy index for the last index

  right_basis = init_left ? [Index(QN() => 1, dir = right_dir)] : (valtype(com_inds)[])
  for k in 1:lx
    for j in 1:ly
      append!(right_basis, filter( x->dir(x) == right_dir, uniqueinds(H[j, k], com_inds)))
    end
  end
  init_right && append!(right_basis, [Index(QN() => 1, dir = right_dir)]) #Dummy index for the last index

  left_block = Vector{Pair{QN, Int64}}()
  dic_inv_left_ind = Dict{Tuple{UInt64, Int64}, Int64}()
  for index in left_basis
    for (n, qp) in enumerate(index.space)
      append!(left_block, [qp])
      dic_inv_left_ind[index.id, n] = length(left_block)
    end
  end
  new_left_index = Index(left_block, dir = left_dir, tags = "left_link")

  right_block = Vector{Pair{QN, Int64}}()
  dic_inv_right_ind = Dict{Tuple{UInt64, Int64}, Int64}()
  for index in right_basis
    for (n, qp) in enumerate(index.space)
      append!(right_block, [qp])
      dic_inv_right_ind[index.id, n] = length(right_block)
    end
  end
  new_right_index = Index(right_block, dir = right_dir, tags = "right_link")

  #Determine the non-zero blocks
  temp_block = Block{4}[]
  elements = []
  for x in 1:lx
    for y in 1:ly
      isempty(H[x, y]) && continue
      tli = commoninds(H[x, y], left_basis)
      case = 0
      if !isempty(tli)
        li = only(tli)
        tri = commoninds(H[x, y], right_basis)
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y], com_inds..., li, ri, allow_alias = true)
          case = 3
        else
          !(y == 1 || (x==lx && y ==ly )) && error("Incompatible leg")
          T = permute(H[x, y], com_inds..., li, allow_alias = true)
          case = 1
        end
      else
        !(x == lx || (x==1 && y ==1 )) && error("Incompatible leg")
        tri = commoninds(H[x, y], right_basis)
        if !isempty(tri)
          ri = only(tri)
          T = permute(H[x, y], com_inds..., ri, allow_alias = true)
          case = 2
        else
          !(y == 1 || (x==lx && y ==ly )) && error("Incompatible leg")
          T = permute(H[x, y], com_inds..., allow_alias = true)
        end
      end
      for (n, b) in enumerate(eachnzblock(T))
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
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[left_basis[end].id, 1], dic_inv_right_ind[ri.id, b[3]])])
        elseif case == 3
          append!(temp_block, [Block(b[1], b[2], dic_inv_left_ind[li.id, b[3]], dic_inv_right_ind[ri.id, b[4]])])
        end
        append!(elements, [T[n]])
      end
    end
  end
  Hf = ITensors.BlockSparseTensor(eltype(H[1]), undef,  temp_block, (com_inds..., new_left_index, new_right_index))
  for (n, b) in enumerate(temp_block)
      ITensors.blockview(Hf, b) .= elements[n]
  end
  return itensor(Hf), new_left_index, new_right_index
end



function matrixITensorToITensor(H::Matrix{ITensor}; kwargs...)
  lx, ly = size(H)
  s =commoninds(H[1, 1], H[end, end])
  right_ind = uniqueinds(H[end, end], s)
  for j in reverse(1:ly-1)
    right_ind = uniqueinds(H[end, j], s)
    length(right_ind) == 1 && break
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
