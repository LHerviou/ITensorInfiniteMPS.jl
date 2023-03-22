##############################################################################
#     SU(3) definitions
##############################################################################
"""
    space(::SiteType"SU3";
          conserve_qns = false)
Create the Hilbert space for a site of type "SU3".
Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(::SiteType"SU3"; conserve_qns=false)
  if conserve_qns
    return [
      QN(("C1", 1), ("C2", 1)) => 1,
      QN(("C1", -1), ("C2", 1)) => 1,
      QN(("C1", 0), ("C2", -2)) => 1,
    ]
  end
  return 3
end

ITensors.val(::ValName"1", ::SiteType"SU3") = 1
ITensors.val(::ValName"2", ::SiteType"SU3") = 2
ITensors.val(::ValName"3", ::SiteType"SU3") = 3
ITensors.state(::StateName"1", ::SiteType"SU3") = [1, 0, 0]
ITensors.state(::StateName"2", ::SiteType"SU3") = [0, 1, 0]
ITensors.state(::StateName"3", ::SiteType"SU3") = [0, 0, 1]

function ITensors.op!(Op::ITensor, ::OpName"P11", ::SiteType"SU3", s::Index)
  return Op[s' => 1, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P12", ::SiteType"SU3", s::Index)
  return Op[s' => 1, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P13", ::SiteType"SU3", s::Index)
  return Op[s' => 1, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P21", ::SiteType"SU3", s::Index)
  return Op[s' => 2, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P22", ::SiteType"SU3", s::Index)
  return Op[s' => 2, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P23", ::SiteType"SU3", s::Index)
  return Op[s' => 2, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P31", ::SiteType"SU3", s::Index)
  return Op[s' => 3, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P32", ::SiteType"SU3", s::Index)
  return Op[s' => 3, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P33", ::SiteType"SU3", s::Index)
  return Op[s' => 3, s => 3] = +1
end

function ITensorInfiniteMPS.unit_cell_terms(::Model"su3_heisenberg"; J1=1.0, J2=0.0)
  opsum = OpSum()
  if J1 != 0
    opsum += J1, "P11", 1, "P11", 2
    opsum += J1, "P12", 1, "P21", 2
    opsum += J1, "P13", 1, "P31", 2
    opsum += J1, "P21", 1, "P12", 2
    opsum += J1, "P22", 1, "P22", 2
    opsum += J1, "P23", 1, "P32", 2
    opsum += J1, "P31", 1, "P13", 2
    opsum += J1, "P32", 1, "P23", 2
    opsum += J1, "P33", 1, "P33", 2
  end
  if J2 != 0
    opsum += J2, "P11", 1, "P11", 3
    opsum += J2, "P12", 1, "P21", 3
    opsum += J2, "P13", 1, "P31", 3
    opsum += J2, "P21", 1, "P12", 3
    opsum += J2, "P22", 1, "P22", 3
    opsum += J2, "P23", 1, "P32", 3
    opsum += J2, "P31", 1, "P13", 3
    opsum += J2, "P32", 1, "P23", 3
    opsum += J2, "P33", 1, "P33", 3
  end
  return opsum
end

##############################################################################
#     SU(4) definitions
##############################################################################

"""
    space(::SiteType"SU4";
          conserve_qns = false)
Create the Hilbert space for a site of type "SU4".
Optionally specify the conserved symmetries and their quantum number labels.
"""
function ITensors.space(::SiteType"SU4"; conserve_qns=false)
  if conserve_qns
    return [
      QN(("C1", 1), ("C2", 1), ("C3", 1)) => 1,
      QN(("C1", -1), ("C2", 1), ("C3", 1)) => 1,
      QN(("C1", 0), ("C2", -2), ("C3", 1)) => 1,
      QN(("C1", 0), ("C2", 0), ("C3", -3)) => 1,
    ]
  end
  return 4
end

ITensors.val(::ValName"1", ::SiteType"SU4") = 1
ITensors.val(::ValName"2", ::SiteType"SU4") = 2
ITensors.val(::ValName"3", ::SiteType"SU4") = 3
ITensors.val(::ValName"4", ::SiteType"SU4") = 4
ITensors.state(::StateName"1", ::SiteType"SU4") = [1, 0, 0, 0]
ITensors.state(::StateName"2", ::SiteType"SU4") = [0, 1, 0, 0]
ITensors.state(::StateName"3", ::SiteType"SU4") = [0, 0, 1, 0]
ITensors.state(::StateName"4", ::SiteType"SU4") = [0, 0, 0, 1]

function ITensors.op!(Op::ITensor, ::OpName"P11", ::SiteType"SU4", s::Index)
  return Op[s' => 1, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P12", ::SiteType"SU4", s::Index)
  return Op[s' => 1, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P13", ::SiteType"SU4", s::Index)
  return Op[s' => 1, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P14", ::SiteType"SU4", s::Index)
  return Op[s' => 1, s => 4] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P21", ::SiteType"SU4", s::Index)
  return Op[s' => 2, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P22", ::SiteType"SU4", s::Index)
  return Op[s' => 2, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P23", ::SiteType"SU4", s::Index)
  return Op[s' => 2, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P24", ::SiteType"SU4", s::Index)
  return Op[s' => 2, s => 4] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P31", ::SiteType"SU4", s::Index)
  return Op[s' => 3, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P32", ::SiteType"SU4", s::Index)
  return Op[s' => 3, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P33", ::SiteType"SU4", s::Index)
  return Op[s' => 3, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P34", ::SiteType"SU4", s::Index)
  return Op[s' => 3, s => 4] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P41", ::SiteType"SU4", s::Index)
  return Op[s' => 4, s => 1] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P42", ::SiteType"SU4", s::Index)
  return Op[s' => 4, s => 2] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P43", ::SiteType"SU4", s::Index)
  return Op[s' => 4, s => 3] = +1
end

function ITensors.op!(Op::ITensor, ::OpName"P44", ::SiteType"SU4", s::Index)
  return Op[s' => 4, s => 4] = +1
end

function ITensorInfiniteMPS.unit_cell_terms(::Model"su4_heisenberg"; J1=1.0, J2=0.0)
  opsum = OpSum()
  if J1 != 0
    opsum += J1, "P11", 1, "P11", 2
    opsum += J1, "P12", 1, "P21", 2
    opsum += J1, "P13", 1, "P31", 2
    opsum += J1, "P14", 1, "P41", 2
    opsum += J1, "P21", 1, "P12", 2
    opsum += J1, "P22", 1, "P22", 2
    opsum += J1, "P23", 1, "P32", 2
    opsum += J1, "P24", 1, "P42", 2
    opsum += J1, "P31", 1, "P13", 2
    opsum += J1, "P32", 1, "P23", 2
    opsum += J1, "P33", 1, "P33", 2
    opsum += J1, "P34", 1, "P43", 2
    opsum += J1, "P41", 1, "P14", 2
    opsum += J1, "P42", 1, "P24", 2
    opsum += J1, "P43", 1, "P34", 2
    opsum += J1, "P44", 1, "P44", 2
  end
  if J2 != 0
    opsum += J2, "P11", 1, "P11", 3
    opsum += J2, "P12", 1, "P21", 3
    opsum += J2, "P13", 1, "P31", 3
    opsum += J2, "P14", 1, "P41", 3
    opsum += J2, "P21", 1, "P12", 3
    opsum += J2, "P22", 1, "P22", 3
    opsum += J2, "P23", 1, "P32", 3
    opsum += J2, "P24", 1, "P42", 3
    opsum += J2, "P31", 1, "P13", 3
    opsum += J2, "P32", 1, "P23", 3
    opsum += J2, "P33", 1, "P33", 3
    opsum += J2, "P34", 1, "P43", 3
    opsum += J2, "P41", 1, "P14", 3
    opsum += J2, "P42", 1, "P24", 3
    opsum += J2, "P43", 1, "P34", 3
    opsum += J2, "P44", 1, "P44", 3
  end
  return opsum
end
