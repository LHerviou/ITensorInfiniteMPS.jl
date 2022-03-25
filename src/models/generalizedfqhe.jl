function ITensors.OpSum(
  ::Model"generalized_fqhe", n1, n2; Ly::Float64=10., Vs_1b::Array{Float64,1}=Float64[], Vs_2b::Float64=0., Vs_3b::Float64=0., prec::Float64 = 1e-6
  )
  opt = Dict()
  if length(Vs_2b) != 0
    coeff_2b = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs_2b)
    opt = merge(opt, optimize_coefficients(coeff_2b; prec=prec))
  end
  if Vs_3b != 0
    coeff_3b = build_three_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_3b)
    opt = merge(opt, optimize_coefficients(coeff_3b; prec=prec))
  end
  if Vs_4b != 0
    coeff_4b = build_four_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_4b)
    opt = merge(opt, optimize_coefficients(coeff_4b; prec=prec))
  end
  opt = filter_optimized_Hamiltonian_by_first_site(opt)
  range_model = check_max_range_optimized_Hamiltonian(opt)
  while range_model >= rough_N
    rough_N = rough_N + 2
    if length(Vs_2b) != 0
      coeff_2b = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs_2b)
      opt = merge(opt, optimize_coefficients(coeff_2b; prec=prec))
    end
    if Vs_3b != 0
      coeff_3b = build_three_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_3b)
      opt = merge(opt, optimize_coefficients(coeff_3b; prec=prec))
    end
    if Vs_4b != 0
      coeff_4b = build_four_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_4b)
      opt = merge(opt, optimize_coefficients(coeff_4b; prec=prec))
    end
    opt = filter_optimized_Hamiltonian_by_first_site(opt)
    range_model = check_max_range_optimized_Hamiltonian(opt)
  end
  return generate_Hamiltonian(opt)
end

function ITensors.MPO(::Model"generalized_fqhe", s::CelledVector, n::Int64;  Ly::Float64=10., Vs_2b::Array{Float64,1}=Float64[], Vs_3b::Float64=0., Vs_4b::Float64=0., prec::Float64 = 1e-6
  )
  rough_N = round(Int64, 2 * Ly)
  opt = Dict()
  if length(Vs_2b) != 0
    coeff_2b = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs_2b)
    opt = merge(opt, optimize_coefficients(coeff_2b; prec=prec))
  end
  if Vs_3b != 0
    coeff_3b = build_three_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_3b)
    opt = merge(opt, optimize_coefficients(coeff_3b; prec=prec))
  end
  if Vs_4b != 0
    coeff_4b = build_four_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_4b)
    opt = merge(opt, optimize_coefficients(coeff_4b; prec=prec))
  end
  opt = filter_optimized_Hamiltonian_by_first_site(opt)
  range_model = check_max_range_optimized_Hamiltonian(opt)
  while range_model >= rough_N
    rough_N = rough_N + 2
    if length(Vs_2b) != 0
      coeff_2b = build_two_body_coefficient_pseudopotential(; N_phi=rough_N, Ly=Ly, Vs=Vs_2b)
      opt = merge(opt, optimize_coefficients(coeff_2b; prec=prec))
    end
    if Vs_3b != 0
      coeff_3b = build_three_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_3b)
      opt = merge(opt, optimize_coefficients(coeff_3b; prec=prec))
    end
    if Vs_4b != 0
      coeff_4b = build_four_body_pseudopotentials(; N_phi=rough_N, Ly=Ly, global_sign = Vs_4b)
      opt = merge(opt, optimize_coefficients(coeff_4b; prec=prec))
    end
    opt = filter_optimized_Hamiltonian_by_first_site(opt)
    range_model = check_max_range_optimized_Hamiltonian(opt)
  end
  opsum = generate_Hamiltonian(opt)

  return MPO(opsum, [s[x] for x in n:(n + range_model)])
end
