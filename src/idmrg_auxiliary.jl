function ITensors.NDTensors.qr_positive(M::AbstractMatrix; full=false, tol = 1e-10)
  #println(typeof(M))
  sparseQR = qr(M, Val(true))
  Q = convert(Matrix, sparseQR.Q)
  #R = convert(Matrix, sparseQR.R *sparseQR.P')
  R = convert(Matrix, sparseQR.R)
  nc = size(Q, 2)
  kept_indices = Int64[]
  for c in 1:nc
    if !(norm(R[c, :]) > tol)
      continue
    end
    append!(kept_indices, c)
    if real(R[c, c]) < 0.0
      R[c, :] *= -1
      Q[:, c] *= -1
    end
  end
  return (Q[:, kept_indices], R[kept_indices, :] * sparseQR.P')
end

function LinearAlgebra.qr(A::ITensor, Linds...; kwargs...)
  tags::TagSet = get(kwargs, :tags, "Link,qr")
  left_combiner = combiner(commoninds(A, ITensors.indices(Linds)))
  right_combiner = combiner(uniqueinds(A, ITensors.indices(Linds)))

  temp_A = left_combiner * (A * right_combiner)

  QT, RT = qr(ITensors.tensor(temp_A); kwargs...)
  Q = itensor(QT); R = itensor(RT);
  q = commonind(Q, R)
  settags!(Q, tags, q)
  settags!(R, tags, q)
  q = settags(q, tags)
  println(Q); println(R);
  println(q);
  println(left_combiner);
  println(right_combiner);
  return dag(left_combiner) * Q, R * dag(right_combiner), q
end

function LinearAlgebra.qr(
  T::ITensors.NDTensors.DenseTensor{ElT,2,IndsT}; kwargs...
) where {ElT,IndsT}
  positive = get(kwargs, :positive, false)
  full = get(kwargs, :full, false)
  # TODO: just call qr on T directly (make sure
  # that is fast)
  if positive
    QM, RM = ITensors.NDTensors.qr_positive(matrix(T))
  else
    QM, RM = qr(matrix(T))
    if full
      QM = QM * I
      tRM = zeros(valtype(RM), (size(QM)[2], size(RM)[2]))
      tRM[1:size(RM)[1], 1:size(RM)[2]] = RM
      RM = tRM
    end
  end
  # Make the new indices to go onto Q and R
  #q, r = inds(T)
  #q = !full && dim(q) >= dim(r) ? sim(r) : dag(sim(q))
  q = size(QM, 2)
  Qinds = IndsT((ind(T, 1), q))
  Rinds = IndsT((dag(q), ind(T, 2)))
  Q = ITensors.NDTensors.tensor(ITensors.NDTensors.Dense(vec(Matrix(QM))), Qinds)
  R = ITensors.NDTensors.tensor(ITensors.NDTensors.Dense(vec(RM)), Rinds)
  return Q, R
end

function LinearAlgebra.qr(
  T::ITensors.NDTensors.BlockSparseMatrix{ElT}; kwargs...
) where {ElT}
  full::Bool = get(kwargs, :full, false)
  dilatation::Float64 = get(kwargs, :dilatation, 1.0)
  positive::Bool = get(kwargs, :positive, true)
  direction::ITensors.Arrow = get(kwargs, :dir, ITensors.Out)

  Qs = Vector{ITensors.DenseTensor{ElT,2}}(undef, nnzblocks(T))
  Rs = Vector{ITensors.DenseTensor{ElT,2}}(undef, nnzblocks(T))
  list_nzBlocks = Tuple{Int64, Block{2}}[]

  for (n, b) in enumerate(eachnzblock(T))
    blockT = ITensors.blockview(T, b)
    QRb = qr(blockT; full=full, positive=positive)#; kwargs...)
    if isnothing(QRb)
      println("Empty matrix")
      continue
      #return nothing
    end
    append!(list_nzBlocks, [(n, b)])
    Qb, Rb = QRb
    if dilatation == 1
      Qs[n] = Qb
      Rs[n] = Rb
    else
      Qs[n] = Qb * dilatation
      Rs[n] = Rb / dilatation
    end
  end

  if full
    q, _ = inds(T)
    centerind = direction == dir(q) ? sim(q) : dag(sim(q))
    indsQ = [inds(T)[1], centerind]
    indsR = [dag(centerind), inds(T)[2]]
    Q = ITensors.BlockSparseTensor(
      ElT, undef, Block{2}[Block(b[1], b[1]) for (n, b) in list_nzBlocks], indsQ #Something happened to blocks or collect
    )
    R = ITensors.BlockSparseTensor(
      ElT, undef, Block{2}[Block(b[1], b[2]) for (n, b) in list_nzBlocks], indsR
    )
    for (n, b) in list_nzBlocks
      qb = Block(b[1], b[1])
      rb = Block(b[1], b[2])
      ITensors.blockview(Q, qb) .= Array(Qs[n]) #I do not get why this bug when Array(ITensors.blockview(Q, qb)) .= Qs[n] does not
      ITensors.blockview(R, rb) .= Array(Rs[n])
    end
  else
    _, r = inds(T)
    centerind_space = Vector{Pair{QN,Int64}}()
    seen = Dict()
    for (n, b) in list_nzBlocks
      append!(centerind_space, [
        if direction == dir(r)
          r.space[b[2]].first => size(Rs[n], 1)
        else
          -r.space[b[2]].first => size(Rs[n], 1)
        end,
      ])
      seen[b[2]] = length(seen) + 1
    end
    # if direction != dir(r)
    #   temp = copy(seen)
    #   for (k, v) in seen
    #     temp[k] = length(seen) + 1 - v
    #   end
    # end
    centerind = Index(centerind_space; dir=direction, tags=inds(T)[2].tags)
    indsQ = [inds(T)[1], centerind]
    indsR = [dag(centerind), inds(T)[2]]

    Q = ITensors.BlockSparseTensor(
      ElT, undef, Block{2}[Block(b[1], seen[b[2]]) for (n, b) in list_nzBlocks], indsQ
    )
    R = ITensors.BlockSparseTensor(
      ElT, undef, Block{2}[Block(seen[b[2]], b[2]) for (n, b) in list_nzBlocks], indsR
    )
    for (n, b) in list_nzBlocks
      qb = Block(b[1], seen[b[2]])
      rb = Block(seen[b[2]], b[2])
      ITensors.blockview(Q, qb) .= Qs[n]
      ITensors.blockview(R, rb) .= Rs[n]
    end
  end
  return itensor(Q), itensor(R)
end
