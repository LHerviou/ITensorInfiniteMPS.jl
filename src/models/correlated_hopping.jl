function unit_cell_terms(::Model"correlated_hopping"; J=1.0)
  opsum = OpSum()
  opsum += -J, "Cdag", 1, "C", 2, "C", 3, "Cdag", 4
  opsum += -J, "C", 1, "Cdag", 2, "Cdag", 3, "C", 4
  return opsum
end
