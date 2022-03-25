include("coefficient_generators.jl")

function get_perm!(lis, name)
  for j in 1:(length(lis) - 1)
    if lis[j] > lis[j + 1]
      c = lis[j]
      lis[j] = lis[j + 1]
      lis[j + 1] = c
      c = name[j]
      name[j] = name[j + 1]
      name[j + 1] = c
      sg = get_perm!(lis, name)
      return -sg
    end
  end
  return 1
end

function filter_op!(lis, name)
  x = 1
  while x <= length(lis) - 1
    if lis[x] == lis[x + 1]
      if name[x] == "Cdag" && name[x + 1] == "C"
        popat!(lis, x + 1)
        popat!(name, x + 1)
        name[x] = "N"
      else
        print("Wrong order in filter_op")
      end
    end
    x += 1
  end
end

function optimize_coefficients(coeff::Dict; prec=1e-12)
  optimized_dic = Dict()
  for (ke, v) in coeff
    if abs(v) < prec
      continue
    end
    if length(ke) == 4
      name = ["Cdag", "Cdag", "C", "C"]
    elseif length(ke) == 6
      name = ["Cdag", "Cdag", "Cdag", "C", "C", "C"]
    elseif length(ke) == 8
      name = ["Cdag", "Cdag", "Cdag", "Cdag", "C", "C", "C", "C"]
    else
      println("Optimization not implemented")
      continue
    end
    k = Base.copy(ke)
    sg = get_perm!(k, name)
    filter_op!(k, name)
    new_k = [isodd(n) ? name[n ÷ 2 + 1] : k[n ÷ 2] + 1 for n in 1:(2 * length(name))]
    optimized_dic[new_k] = sg * v
  end
  return optimized_dic
end

function generate_Hamiltonian(mpo::OpSum, coeff::Dict; global_factor=1, prec=1e-12)
  for (k, v) in coeff
    if abs(v) > prec
      add!(mpo, global_factor * v, k...)
    end
  end
  return mpo
end

function generate_Hamiltonian(coeff::Dict; global_factor=1, prec=1e-12)
  mpo = OpSum()
  return generate_Hamiltonian(mpo, coeff; global_factor=global_factor, prec=prec)
end

function generate_Hamiltonian(mpo::OpSum, coeff::Array{Float64,1}; global_factor=1)
  for (k, v) in enumerate(coeff)
    mpo += global_factor * v, "N", k
  end
  return mpo
end

function generate_Hamiltonian(coeff::Array{Float64,1}; global_factor=1)
  mpo = OpSum()
  return generate_Hamiltonian(mpo, coeff; global_factor=global_factor)
end

"""
    Assume a ordered dictionnary
"""
function split_by_range(coeff::Dict)
  sorted_coeff = Dict()
  temp = 0
  for k in keys(coeff)
    temp = max(temp, k[end] - k[2])
  end
  for n in 0:temp
    sorted_coeff[n] = Dict()
  end
  for (k, v) in coeff
    sorted_coeff[k[end] - k[2]][k] = v
  end
  return sorted_coeff
end
