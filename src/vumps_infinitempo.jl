# Struct for use in eigenvalue solver
struct AOL
  ψ::InfiniteCanonicalMPS
  H::InfiniteMPO
  n::Int
end

function (A::AOL)(x)
  ψ = A.ψ
  H = A.H
  n = A.n

  y = apply_left_transfer_matrix(x, H, ψ, n)
  return translatecell(translator(H), y, -1)
end

# apply the left transfer matrix at position n1 to the vector Lstart considering it at position m, adding to Ltarget
function apply_local_left_transfer_matrix(
  Lstart::ITensor,
  H::InfiniteMPO,
  ψ::InfiniteCanonicalMPS,
  n_1::Int64;
)
  return  ( (Lstart * ψ.AL[n_1]) * H[n_1] ) * dag(prime(ψ.AL[n_1]))
end

#apply the left transfer matrix n1:n1+nsites(ψ)-1
function apply_left_transfer_matrix(
  Lstart::ITensor, H::InfiniteMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  Ltarget = apply_local_left_transfer_matrix(Lstart, H, ψ, n_1)
  for j in 1:(nsites(ψ) - 1)
    Ltarget = apply_local_left_transfer_matrix(Ltarget, H, ψ, n_1 + j)
  end
  return Ltarget
end

# Also input C bond matrices to help compute the right fixed points
# of ψ (R ≈ C * dag(C))
function left_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
  N = nsites(H)
  @assert N == nsites(ψ)

  Ls = Vector{ITensor}(undef, N)
  problem_to_solve = AOL(ψ, H, 1)
  llink = only(commoninds(ψ.AL[0], ψ.AL[1]))
  llinkMPO = only(commoninds(H[0], H[1]))
  init = randomITensor(llink, prime(dag(llink)), llinkMPO)
  eₗ, temp = eigsolve(problem_to_solve, init, 1, :LM, tol = 1e-10)
  Ls[1] = temp[1]
  for n in 2:N
    Ls[n] = apply_local_left_transfer_matrix(Ls[n - 1], H, ψ, n-1)
  end
  return CelledVector(Ls, translator(ψ)), eₗ[1]
end

# Struct for use in linear system solver
struct AOR2
  ψ::InfiniteCanonicalMPS
  H::InfiniteMPO
  n::Int
end

function (A::AOR2)(x)
  ψ = A.ψ
  H = A.H
  n = A.n

  y = apply_right_transfer_matrix(x, H, ψ, n)
  return translatecell(translator(H), y, 1)
end


#apply the right transfer matrix n1:n1+nsites(ψ)-1
function apply_local_right_transfer_matrix(
  Rstart::ITensor, H::InfiniteMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  return  ( (Rstart * ψ.AR[n_1]) * H[n_1] ) * dag(prime(ψ.AR[n_1]))
end

function apply_right_transfer_matrix(
  Rstart::ITensor, H::InfiniteMPO, ψ::InfiniteCanonicalMPS, n_1::Int64
)
  RTarget = apply_local_right_transfer_matrix(Rstart, H, ψ, n_1)
  for j in 1:nsites(ψ)-1
    RTarget = apply_local_right_transfer_matrix(RTarget, H, ψ, n_1-j)
  end
  return  RTarget
end

function right_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
  N = nsites(H)
  @assert N == nsites(ψ)

  Rs = Vector{ITensor}(undef, N)
  problem_to_solve = AOR2(ψ, H, N)
  rlink = only(commoninds(ψ.AR[N+1], ψ.AR[N]))
  rlinkMPO = only(commoninds(H[N+1], H[N]))
  #Prepare init state
  rlink2 = only(commoninds(ψ.AR[N+2], ψ.AR[N+1]))
  rlinkMPO2 = only(commoninds(H[N+2], H[N+1]))
  tempR = ITensor(rlinkMPO2); tempR[1] = 1
  init = ( (ψ.AR[N+1] * δ(rlink2, dag(prime(rlink2))) ) * (H[N+1] * tempR) ) * dag(prime(ψ.AR[N+1]))

  eᵣ, temp = eigsolve(problem_to_solve, init, 4, :LM, tol = 1e-10)
  Rs[end] = temp[1]
  for n in N-1:-1:1
    Rs[n] = apply_local_right_transfer_matrix(Rs[n + 1], H, ψ, n+1)
  end
  return CelledVector(Rs, translator(ψ)), eᵣ, init, temp
end

#TODO Finish
# function vumps(H::InfiniteMPOMatrix, ψ::InfiniteMPS; kwargs...)
#   return vumps(H, orthogonalize(ψ, :); kwargs...)
# end
#
# struct H⁰
#   L::Vector{ITensor}
#   R::Vector{ITensor}
# end
#
# function (H::H⁰)(x)
#   L = H.L
#   R = H.R
#   dₕ = length(L)
#   result = L[1] * x * R[1]
#   for j in 2:dₕ
#     result += L[j] * x * R[j]
#   end
#   return noprime(result)
# end
#
# struct H¹
#   L::Vector{ITensor}
#   R::Vector{ITensor}
#   T::Matrix{ITensor}
# end
#
# function (H::H¹)(x)
#   L = H.L
#   R = H.R
#   T = H.T
#   dₕ = length(L)
#   result = ITensor(prime(inds(x)))
#   for i in 1:dₕ
#     for j in 1:dₕ
#       if !isempty(T[i, j])
#         result += L[i] * x * T[i, j] * R[j]
#       end
#     end
#   end
#   return noprime(result)
# end
#
# function tdvp_iteration_sequential(
#   solver::Function,
#   H::InfiniteMPOMatrix,
#   ψ::InfiniteCanonicalMPS;
#   (ϵᴸ!)=fill(1e-15, nsites(ψ)),
#   (ϵᴿ!)=fill(1e-15, nsites(ψ)),
#   time_step,
#   solver_tol=(x -> x / 100),
#   eager=true,
# )
#   ψ = copy(ψ)
#   ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
#   _solver_tol = solver_tol(ϵᵖʳᵉˢ)
#   N = nsites(ψ)
#
#   C̃ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#
#   eL = zeros(N)
#   eR = zeros(N)
#   for n in 1:N
#     L, eL[n] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
#     R, eR[n] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
#     if N == 1
#       # 0-site effective Hamiltonian
#       E0, C̃[n], info0 = solver(H⁰(L[1], R[2]), time_step, ψ.C[1], _solver_tol, eager)
#       # 1-site effective Hamiltonian
#       E1, Ãᶜ[n], info1 = solver(
#         H¹(L[0], R[2], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
#       )
#       Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
#       Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
#       ψ.AL[1] = Ãᴸ[1]
#       ψ.AR[1] = Ãᴿ[1]
#       ψ.C[1] = C̃[1]
#     else
#       # 0-site effective Hamiltonian
#       E0, C̃[n], info0 = solver(H⁰(L[n], R[n + 1]), time_step, ψ.C[n], _solver_tol, eager)
#       E0′, C̃[n - 1], info0′ = solver(
#         H⁰(L[n - 1], R[n]), time_step, ψ.C[n - 1], _solver_tol, eager
#       )
#       # 1-site effective Hamiltonian
#       E1, Ãᶜ[n], info1 = solver(
#         H¹(L[n - 1], R[n + 1], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
#       )
#       Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
#       Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
#       ψ.AL[n] = Ãᴸ[n]
#       ψ.AR[n] = Ãᴿ[n]
#       ψ.C[n] = C̃[n]
#       ψ.C[n - 1] = C̃[n - 1]
#     end
#   end
#   for n in 1:N
#     ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
#     ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
#   end
#   return ψ, (eL / N, eR / N)
# end
#
# function tdvp_iteration_parallel(
#   solver::Function,
#   H::InfiniteMPOMatrix,
#   ψ::InfiniteCanonicalMPS;
#   (ϵᴸ!)=fill(1e-15, nsites(ψ)),
#   (ϵᴿ!)=fill(1e-15, nsites(ψ)),
#   time_step,
#   solver_tol=(x -> x / 100),
#   eager=true,
# )
#   ψ = copy(ψ)
#   ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
#   _solver_tol = solver_tol(ϵᵖʳᵉˢ)
#   N = nsites(ψ)
#
#   C̃ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#   Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N), translator(ψ))
#
#   eL = zeros(1)
#   eR = zeros(1)
#   L, eL[1] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
#   R, eR[1] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
#   for n in 1:N
#     if N == 1
#       # 0-site effective Hamiltonian
#       E0, C̃[n], info0 = solver(H⁰(L[1], R[2]), time_step, ψ.C[1], _solver_tol, eager)
#       # 1-site effective Hamiltonian
#       E1, Ãᶜ[n], info1 = solver(
#         H¹(L[0], R[2], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
#       )
#       Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
#       Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
#       ψ.AL[1] = Ãᴸ[1]
#       ψ.AR[1] = Ãᴿ[1]
#       ψ.C[1] = C̃[1]
#     else
#       # 0-site effective Hamiltonian
#       for n in 1:N
#         E0, C̃[n], info0 = solver(H⁰(L[n], R[n + 1]), time_step, ψ.C[n], _solver_tol, eager)
#         E1, Ãᶜ[n], info1 = solver(
#           H¹(L[n - 1], R[n + 1], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
#         )
#       end
#       # 1-site effective Hamiltonian
#       for n in 1:N
#         Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
#         Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
#         ψ.AL[n] = Ãᴸ[n]
#         ψ.AR[n] = Ãᴿ[n]
#         ψ.C[n] = C̃[n]
#       end
#     end
#   end
#   for n in 1:N
#     ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
#     ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
#   end
#   return ψ, (eL / N, eR / N)
# end
