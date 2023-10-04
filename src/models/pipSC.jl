function ITensors.space(::SiteType"FermCyl", pos::Int; Ly::Int64=4, conservenfparity = true, conservenf = false)
  if conservenf
    return [QN("Nf", 0) => 1, QN("Nf", 1) => 1]
  elseif conservenfparity
    return  [QN("Nf", 0, -2) => 1, QN("Nf", 1, -2) => 1]
  else
    return 2
  end
end

# Forward all op definitions to Fermion
function ITensors.op!(Op::ITensor, opname::OpName, ::SiteType"FermCyl", s::Index...)
  return ITensors.op!(Op, opname, SiteType("Fermion"), s...)
end
ITensors.has_fermion_string(::OpName"C", ::SiteType"FermCyl") = true
function ITensors.has_fermion_string(on::OpName"c", st::SiteType"FermCyl")
  return has_fermion_string(alias(on), st)
end
ITensors.has_fermion_string(::OpName"Cdag", ::SiteType"FermCyl") = true
function ITensors.has_fermion_string(on::OpName"c†", st::SiteType"FermCyl")
  return has_fermion_string(alias(on), st)
end

# The terms of the Hamiltonian in the first unit cell.
# p + ip SC. Several forms are possible
# standard is 1 2 3 ... Ly in the y direction
# zipped is 1 Ly 2 Ly-1 ...
function unit_cell_terms(::Model"pipSC"; Ly::Int64 = 4, tx = 1., ty = 1., Dx = 1., Dy = 1.0 *1im, mu = 0., order = :standard)
  opsum = OpSum{promote_type(typeof(tx), typeof(ty), typeof(Dx), typeof(Dy), typeof(mu))}()
  #Easy part: tx
  if tx != 0
    for j in 1:Ly
      opsum += -tx, "Cdag", j, "C", j+Ly
      opsum += tx, "C", j, "Cdag", j+Ly
    end
  end
  #Second easy part: Dx
  if Dx != 0
    for j in 1:Ly
      opsum += Dx, "Cdag", j, "Cdag", j+Ly
      opsum += -Dx', "C", j, "C", j+Ly
    end
  end
  if mu != 0
    for j in 1:Ly
      opsum += -mu, "N", j
    end
  end
  if order == :standard
    positions = collect(1:Ly)
  else
    positions = Int64[]
    for j in 1:Ly
      if mod(j, 2)==1
        append!(positions, (j+1)÷2)
      else
        append!(positions, Ly - (j-1)÷2)
      end
    end
  end
  inv_positions = Dict{Int64, Int64}()
  for (idx, x) in enumerate(positions)
    inv_positions[x] = idx
  end
  #Dealing with ty
  if ty !=0
    for j in 1:Ly
      j2 = mod1(j+1, Ly)
      idx1 = inv_positions[j]; idx2 = inv_positions[j2]
      if idx1 < idx2
        opsum += -ty, "Cdag", idx1, "C", idx2
        opsum += ty, "C", idx1, "Cdag", idx2
      else
        opsum += -ty, "Cdag", idx2, "C", idx1
        opsum += ty, "C", idx2, "Cdag", idx1
      end
    end
  end
  #Dealing with Dy
  if Dy != 0
    for j in 1:Ly
      j2 = mod1(j+1, Ly)
      idx1 = inv_positions[j]; idx2 = inv_positions[j2]
      if idx1 < idx2
        opsum += Dy, "Cdag", idx1, "Cdag", idx2
        opsum += -Dy', "C", idx1, "C", idx2
      else
        opsum += -Dy, "Cdag", idx2, "Cdag", idx1
        opsum += Dy', "C", idx2, "C", idx1
      end
    end
  end
  return opsum
end

"""
  We work here in k space in the momentum direction.
  Two possible ordering:
      standard K = 0 , 1, 2, ... Ly - 1 (in units of elementary moment)
      zipped  K = 0, Ly-1, 1, 2, ....
"""
function ITensors.space(::SiteType"FermCylK", pos::Int; Ly::Int64=4, conservemomentum = true, conservenfparity = true, conservenf = false, order = :standard)
  if conservemomentum
    k = mod(pos-1, Ly)
    if order != :standard
      if mod(k, 2) == 0
        k = k ÷2
      else
        k = Ly - (k+1)÷2
      end
    end
    if conservenf
      return  [QN(("Nf", 0), ("K", 0, Ly)) => 1, QN(("Nf", 1), ("K", k, Ly)) => 1]
    elseif conservenfparity
      return [QN(("Nf", 0, -2), (K, 0, Ly)) => 1, QN(("Nf", 1, -2), ("K", k, Ly)) => 1]
    else
      return [QN(K, 0, Ly) => 1, QN("K", k, Ly) => 1]
    end
  elseif conservenf
    return [QN("Nf", 0) => 1, QN("Nf", 1) => 1]
  elseif conservenfparity
    return  [QN("Nf", 0, -2) => 1, QN("Nf", 1, -2) => 1]
  else
    return 2
  end
end


# The terms of the Hamiltonian in the first unit cell.
# p + ip SC. Several forms are possible
# standard is 1 2 3 ... Ly in the y direction
# zipped is 1 Ly 2 Ly-1 ...
function unit_cell_terms(::Model"pipSCK"; Ly::Int64 = 4, tx = 1., ty = 1., Dx = 1., Dy = 1.0 *1im, mu = 0., order = :standard)
  opsum = OpSum{ComplexF64}()
  #Easy part: tx
  if tx != 0
    for j in 1:Ly
      opsum += -tx, "Cdag", j, "C", j+Ly
      opsum += tx, "C", j, "Cdag", j+Ly
    end
  end
  if order == :standard
    positions = collect(1:Ly)
  else
    positions = Int64[]
    for j in 1:Ly
      if mod(j, 2)==1
        append!(positions, (j+1)÷2)
      else
        append!(positions, Ly - (j-1)÷2)
      end
    end
  end
  inv_positions = Dict{Int64, Int64}()
  for (idx, x) in enumerate(positions)
    inv_positions[x] = idx
  end
  #Easy, doing mu and ty at the same time
  if mu != 0 || ty != 0
    for j in 1:Ly
      k = positions[j]
      opsum += -mu -2*ty*cos(k), "N", j
    end
  end
  #Less easy: Dx c^†_{x, ky} c^†_{x+1, -ky}
  if Dx != 0
    for idx1 in 1:Ly
      k = positions[idx1]
      idx2 = inv_positions[mod(-k, Ly)]
      opsum += Dx, "Cdag", idx1, "Cdag", idx2+Ly
      opsum += -Dx', "C", idx1, "C", idx2+Ly
    end
  end
  #Dealing with Dy
  if Dy != 0
    for idx1 in 1:Ly
      k = positions[idx1]
      k > Ly/2 && continue
      idx2 = inv_positions[mod(-k, Ly)]
      idx1 == idx2 && continue
      if idx1 < idx2
        opsum += 2*1im*sin(k)*Dy, "Cdag", idx1, "Cdag", idx2
        opsum += 2*1im*sin(k)*Dy', "C", idx1, "C", idx2
      else
        opsum += -2*1im*sin(k)*Dy, "Cdag", idx1, "Cdag", idx2
        opsum += -2*1im*sin(k)*Dy', "C", idx1, "C", idx2
      end
    end
  end
  return opsum
end
