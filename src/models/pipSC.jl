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


# The terms of the Hamiltonian in the first unit cell.
# p + ip SC. Several forms are possible
function unit_cell_terms(::Model"pipSC"; Ly::Int64 = 4, tx = 1., ty = 1., Dx = 1., Dy = 1., mu = 0., order = :standard)
  opsum = OpSum{ComplexF64}()
  if order == :standard
    #Easy part: tx
    if tx != 0
      for j in 1:Ly
        opsum += -tx, "Cdag", j, "C", j+Ly+1
        opsum += tx, "C", j, "Cdag", j+Ly+1
      end
    end
    #Second easy part: Dx
    if Dx != 0
      for j in Ly
        opsum += Dx, "Cdag", j, "Cdag", j+Ly+1
        opsum += -Dx, "C", j, "C", j+Ly+1
      end
    end
    #Dealing with ty
    if ty !=0
      for j in 1:Ly-1
        opsum += -ty, "Cdag", j, "C", j+1
        opsum += ty, "C", j, "Cdag", j+1
      end
      opsum += -ty, "Cdag", 1, "C", Ly
      opsum += ty, "C", 1, "Cdag", Ly
    end
    #Dealing with Dy
    if Dy != 0
      for j in 1:Ly-1
        opsum += 1im * Dy, "Cdag", j, "Cdag", j+1
        opsum += 1im * Dy, "C", j, "C", j+1
      end
      opsum += -1im * Dy, "Cdag", 1, "Cdag", Ly
      opsum += -1im * Dy, "C", 1, "C", Ly
    end
    if mu != 0
      for j in 1:Ly
        opsum += -mu, "N", j
      end
    end
  else #Here we order the sites in order [1, 2, Ly - 1, 2, Ly-2...]
    #Easy part: tx    unchanged
    if tx != 0
      for j in 1:Ly
        opsum += -tx, "Cdag", j, "C", j+Ly+1
        opsum += tx, "C", j, "Cdag", j+Ly+1
      end
    end
    #Second easy part: Dx   unchanged
    if Dx != 0
      for j in Ly
        opsum += Dx, "Cdag", j, "Cdag", j+Ly+1
        opsum += -Dx, "C", j, "C", j+Ly+1
      end
    end
    if mu != 0
      for j in 1:Ly
        opsum += -mu, "N", j
      end
    end
    ####Dealing with the ordering
    aux_1 = 2:(Ly+2)÷2
    aux_2= reverse(aux_1[end]+1:Ly)
    positions = [1]
    for j in 1:Ly-1
      if mod(j, 2) == 1
        append!(positions, aux_1[(j+1)÷2])
      else
        append!(positions, aux_2[j÷2])
      end
    end
    inv_positions = Dict{Int64, Int64}()
    for (idx, x) in positions
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
          opsum += 1im * Dy, "Cdag", idx1, "C", idx2
          opsum += 1im * Dy, "C", idx1, "Cdag", idx2
        else
          opsum += -1im * Dy, "Cdag", idx2, "Cdag", idx1
          opsum += -1im * Dy, "C", idx2, "C", idx1
        end
      end
    end
  end
  return opsum
end

function reference(::Model"ising_extended", ::Observable"energy"; J=1.0, h=1.0, J₂=0.0)
  f(k) = sqrt((J * cos(k) + J₂ * cos(2k) - h)^2 + (J * sin(k) + J₂ * sin(2k))^2)
  return -1 / 2π * ITensorInfiniteMPS.∫(k -> f(k), -π, π)
end
