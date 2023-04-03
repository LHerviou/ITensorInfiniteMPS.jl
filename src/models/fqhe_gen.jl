#Please contact Loic Herviou before using this part of the code for production
# loic.herviou@epfl.ch
###############################
function unit_cell_terms(
  ::Model"fqhe_gen"; dict_coeffs = Dict{NTuple{4, Int64}, Number}(), prec = 1e-8
)
  #we assume that dict_coeffs has already been filtered, and is ready to be used
  opt = optimize_coefficients(dict_coeffs; prec=prec)
  return generate_Hamiltonian(opt)
end
