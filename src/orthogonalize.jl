
# TODO: call as `orthogonalize(ψ, -∞)`
# TODO: could use commontags(ψ) as a default for left_tags
function right_orthogonalize(
  ψ::InfiniteMPS; left_tags=ts"Left", right_tags=ts"Right", tol::Real=1e-12
)
  # TODO: replace dag(ψ) with ψ'ᴴ?
  ψᴴ = prime(linkinds, dag(ψ))

  N = nsites(ψ)

  # The unit cell range
  # Turn into function `eachcellindex(ψ::InfiniteMPS, cell::Integer = 1)`
  cell₁ = 1:N
  # A transfer matrix made from the 1st unit cell of
  # the infinite MPS
  # TODO: make a `Cell` Integer type and call as `ψ[Cell(1)]`
  # TODO: make a TransferMatrix wrapper that automatically
  # primes and daggers, so it can be called with:
  # T = TransferMatrix(ψ[Cell(1)])
  ψ₁ = ψ[cell₁]
  ψ₁ᴴ = ψᴴ[cell₁]
  T₀₁ = ITensorMap(
    ψ₁,
    ψ₁ᴴ;
    input_inds=unioninds(commoninds(ψ[N], ψ[N + 1]), commoninds(ψᴴ[N], ψᴴ[N + 1])),
    output_inds=unioninds(commoninds(ψ[1], ψ[0]), commoninds(ψᴴ[1], ψᴴ[0])),
  )

  # TODO: make an optional initial state
  v₁ᴿᴺ = randomITensor(dag(input_inds(T₀₁)))

  # Start by getting the right eivenvector/eigenvalue of T₀₁
  # TODO: make a function `right_environments(::InfiniteMPS)` that computes
  # all of the right environments using `eigsolve` and shifting unit cells
  λ⃗₁ᴿᴺ, v⃗₁ᴿᴺ, eigsolve_info = eigsolve(T₀₁, v₁ᴿᴺ, 1, :LM; tol=tol)
  λ₁ᴿᴺ, v₁ᴿᴺ = λ⃗₁ᴿᴺ[1], v⃗₁ᴿᴺ[1]

  if imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) > 1e-15
    @show λ₁ᴿᴺ
    error(
      "Imaginary part of eigenvalue is large: imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ) = $(imag(λ₁ᴿᴺ) / norm(λ₁ᴿᴺ))",
    )
  end

  # Fix the phase of the diagonal to make Hermitian
  v₁ᴿᴺ .*= conj(sign(v₁ᴿᴺ[1, 1]))
  if !ishermitian(v₁ᴿᴺ; rtol=tol)
    @show λ₁ᴿᴺ
    @show v₁ᴿᴺ
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
    error("v₁ᴿᴺ not hermitian")
  end
  if norm(imag(v₁ᴿᴺ)) > 1e-15
    println(
      "Norm of the imaginary part $(norm(imag(v₁ᴿᴺ))) is larger than the tolerance value 1e-15. Keeping as complex.",
    )
    @show norm(v₁ᴿᴺ - swapinds(dag(v₁ᴿᴺ), reverse(Pair(inds(v₁ᴿᴺ)...))))
  else
    v₁ᴿᴺ = real(v₁ᴿᴺ)
  end

  # Initial guess for bond matrix such that:
  # ψ₁ * C₁ᴿᴺ = C₁ᴿᴺ * ψ₁ᴿ
  C₁ᴿᴺ = sqrt(v₁ᴿᴺ)
  C₁ᴿᴺ = replacetags(C₁ᴿᴺ, left_tags => right_tags; plev=1)
  C₁ᴿᴺ = noprime(C₁ᴿᴺ, right_tags)

  ## Flip the sign:
  ## @show inds(ψ[N])
  ## @show inds(C₁ᴿᴺ)
  ## @show C₁ᴿᴺ
  ## @show rᴺ = dag(uniqueind(C₁ᴿᴺ, ψ[N]))
  ## rᴺ⁻ = Index(-space(rᴺ); tags=tags(rᴺ), plev=plev(rᴺ), dir=dir(rᴺ))
  ## C₁ᴿᴺ = C₁ᴿᴺ * δ(rᴺ, rᴺ⁻)
  ## @show C₁ᴿᴺ
  ## @show inds(C₁ᴿᴺ)

  # Normalize the center matrix
  normalize!(C₁ᴿᴺ)

  Cᴿ, ψᴿ, λᴿ = right_orthogonalize_polar(
    ψ, C₁ᴿᴺ; left_tags=left_tags, right_tags=right_tags
  )
  @assert λᴿ ≈ sqrt(real(λ₁ᴿᴺ))
  return Cᴿ, ψᴿ, λᴿ
end

function right_orthogonalize_polar(
  ψ::InfiniteMPS, Cᴿᴺ::ITensor; left_tags=ts"Left", right_tags=ts"Right"
)
  N = length(ψ)
  ψᴿ = InfiniteMPS(N; reverse=ψ.reverse)
  Cᴿ = InfiniteMPS(N; reverse=ψ.reverse)
  Cᴿ[N] = Cᴿᴺ
  λ = 1.0
  for n in reverse(1:N)
    sⁿ = uniqueinds(ψ[n], ψ[n - 1], Cᴿ[n])
    lᴿⁿ = uniqueinds(Cᴿ[n], ψ[n])
    ψᴿⁿ, Cᴿⁿ⁻¹ = polar(ψ[n] * Cᴿ[n], (sⁿ..., lᴿⁿ...))
    # TODO: set the tags in polar
    ψᴿⁿ = replacetags(ψᴿⁿ, left_tags => right_tags; plev=1)
    ψᴿⁿ = noprime(ψᴿⁿ, right_tags)
    Cᴿⁿ⁻¹ = replacetags(Cᴿⁿ⁻¹, left_tags => right_tags; plev=1)
    Cᴿⁿ⁻¹ = noprime(Cᴿⁿ⁻¹, right_tags)
    ψᴿ[n] = ψᴿⁿ
    Cᴿ[n - 1] = Cᴿⁿ⁻¹
    λⁿ = norm(Cᴿ[n - 1])
    Cᴿ[n - 1] /= λⁿ
    λ *= λⁿ
    if !isapprox(ψ[n] * Cᴿ[n], λⁿ * Cᴿ[n - 1] * ψᴿ[n]; atol=1e-14)
      @show norm(ψ[n] * Cᴿ[n] - λⁿ * Cᴿ[n - 1] * ψᴿ[n])
      error("ψ[n] * Cᴿ[n] ≠ λⁿ * Cᴿ[n-1] * ψᴿ[n]")
    end
  end
  return Cᴿ, ψᴿ, λ
end

function left_orthogonalize(
  ψ::InfiniteMPS; left_tags=ts"Left", right_tags=ts"Right", tol::Real=1e-12
)
  Cᴸ, ψᴸ, λᴸ = right_orthogonalize(
    reverse(ψ); left_tags=right_tags, right_tags=left_tags, tol=tol
  )
  # Cᴸ has the unit cell shifted from what is expected
  Cᴸ = reverse(Cᴸ)
  Cᴸ_shift = copy(Cᴸ)
  for n in 1:nsites(Cᴸ)
    Cᴸ_shift[n] = Cᴸ[n + 1]
  end
  return reverse(ψᴸ), Cᴸ_shift, λᴸ
end

# TODO: rename to `orthogonalize(ψ)`? With no limit specified, it is like orthogonalizing to over point.
# Alternatively, it could be called as `orthogonalize(ψ, :)`
function mixed_canonical(
  ψ::InfiniteMPS; left_tags=ts"Left", right_tags=ts"Right", tol::Real=1e-12
)
  _, ψᴿ, _ = right_orthogonalize(ψ; left_tags=ts"", right_tags=ts"Right")
  ψᴸ, C, λ = left_orthogonalize(ψᴿ; left_tags=ts"Left", right_tags=ts"Right")
  if λ ≉ one(λ)
    error("λ should be approximately 1 after orthogonalization, instead it is $λ")
  end
  return InfiniteCanonicalMPS(ψᴸ, C, ψᴿ)
end

ITensors.orthogonalize(ψ::InfiniteMPS, ::Colon; kwargs...) = mixed_canonical(ψ; kwargs...)

#TODO put these functions somewhere else
function ortho_overlap(AC, C)
  AL, _ = polar(AC * dag(C), uniqueinds(AC, C))
  return noprime(AL)
end

function ortho_polar(AC, C)
  UAC, _ = polar(AC, uniqueinds(AC, C))
  UC, _ = polar(C, commoninds(C, AC))
  return noprime(UAC) * noprime(dag(UC))
end

function diag_ortho_polar(AC, C)
  UAC, _ = polar(AC, uniqueinds(AC, C))
  return noprime(UAC) * dag(δ(inds(C)...)  )
end


function diag_ortho_polar_both(AC, C)
  UAC, Cbis = polar(AC, uniqueinds(AC, C))
  new_ind = flip_sign(only(commoninds(Cbis, UAC)))
  return noprime(UAC * wδ(only(commoninds(Cbis, UAC)), new_ind)) , noprime(Cbis * wδ(only(commoninds(UAC, Cbis)), dag(new_ind) ))
end



function flip_sign(ind::Index{Vector{Pair{QN,Int64}}})
  space = copy(ind.space)
  for (x, sp) in enumerate(space)
    if length(sp[1][1].name) == 0 || length(sp[1][2].name) == 0
      max_flip = 1
    elseif length(sp[1][3].name) == 0
      max_flip = 2
    elseif length(sp[1][4].name) == 0
      max_flip = 3
    else
      max_flip = 4
    end
    space[x] =
      QN(
        [(qn.name, -qn.val, qn.modulus) for (y, qn) in enumerate(ind.space[x][1])][1:max_flip]...,
      ) => ind.space[x][2]
  end
  return Index(space; tags=ind.tags, dir=ind.dir, plev=ind.plev)
end

#δ, but matching the symmetries. Not sure it does not slow down some things.
function wδ(indl, indr)
  if length(indl.space) > length(indr.space)
    return wδ(indr, indl)
  end
  res = ITensor(indl, indr)
  shift_left = 1
  visited = zeros(Int64, length(indr.space))
  for spl in indl.space
    start_right = 1
    for (idxr, spr) in enumerate(indr.space)
      if spl[1]*indl.dir == -spr[1]*indr.dir && visited[idxr] == 0
        for x in 0:min(spl[2] - 1, spr[2] - 1)
          res[shift_left + x, start_right + x] = 1.0
        end
        visited[idxr] = 1
        break
      else
        start_right += spr[2]
      end
    end
    shift_left += spl[2]
  end
  return res
end
