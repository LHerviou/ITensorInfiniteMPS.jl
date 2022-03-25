
mutable struct InfiniteMPOMatrix <: AbstractInfiniteMPS
  data::CelledVector{Matrix{ITensor}}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translater(mpo::InfiniteMPOMatrix) = mpo.data.translater

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

function InfiniteMPOMatrix(data::Vector{Matrix{ITensor}}, translater::Function)
  return InfiniteMPOMatrix(CelledVector(data, translater), 0, size(data)[1], false)
end



function Base.:*(A::Matrix{ITensor}, B::Matrix{ITensor})
  if size(A)[2] != size(B)[1]
    error("Matrices do not have the same dimensions")
  end
  res = Matrix{ITensor}(undef, size(A)[1], size(B)[2])
  for i in 1:size(A)[1]
    for j in 1:size(B)[1]
      @disable_warn_order res[i, j] = A[i, end]*B[end, j]
      for k in 1:size(A)[2]-1
        if !isempty(A[i, k]) && !isempty(B[k, j])
          if isempty(res[i, j])
            @disable_warn_order res[i, j] = A[i, k] * B[k, j]
          else
            @disable_warn_order res[i, j] += A[i, k] * B[k, j]
          end
        end
      end
    end
  end
  return res
end

  #nrange(H::InfiniteMPOMatrix) = [size(H[j])[1] - 1 for j in 1:nsites(H)]
