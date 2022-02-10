using SpecialFunctions: besselix
using HCubature

include("generic.jl")

"""
    coeff_Coulomb_cylinder_integrand_LL0(x::Float64; dpL::Int64=1, N::Int64=0, dn::Int64=0)

Auxiliary function integrated in build_coefficient_Coulomb_cylinder_LL0
Inputs: x the coordinate in the aperiodic direction, `dpL = 2pi/Ly`, `N = abs(n_1 - m_1)` and `dn = abs(m_1 - n_2)`
Outputs: a float value
cf Misguitch et al
"""
function coeff_Coulomb_cylinder_integrand_LL0(x; dpL=1, N=0, dn=0)
    return besselix.(N, x.^2 * 2/dpL^2 ).*exp.( -dn^2*dpL^2*x.^2 ./ (2*x.^2 .+1) )./sqrt.(2*x.^2 .+1)
end


"""
    build_coefficient_Coulomb_cylinder_LL0(Ly::Float64, N_phi::Int64)

Build the dictionnary of the two body coefficients of the Coulomb interaction in the 0th LL
Inputs: Ly the dimension of the cylinder in its periodic direction, `N_phi` the number of orbitals
Outputs: a Dictionnary of the coefficients: keys are (m_1, m_2, n_2, n_1)
"""
function build_coefficient_Coulomb_cylinder_LL0(Ly::Float64, N_phi::Int64)
    precomputation = Dict{Tuple{Int64, Int64}, Float64}()
    dpL = 2*pi/Ly
    coefficients=Dict{Array{Int64, 1}, Float64}()
    for m_1 = 0:N_phi-1
        for m_2 = m_1+1:N_phi-1
            for n_1 = max(0, m_1+m_2 - N_phi+1):min(N_phi-1, m_1+m_2)
                n_2 = m_1 + m_2 - n_1
                if n_1 >= n_2
                    continue
                end
                #cd_m_1, cd_m_2, c_n_2, c_n_1
                N = abs(n_1 - m_1)
                dn = abs(m_1 - n_2)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp = precomputation[(N, dn)]

                #cd_m_1, cd_m_2, c_n_1, c_n_2
                N = abs(n_2 - m_1)
                dn = abs(m_1 - n_1)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp -= precomputation[(N, dn)]

                #cd_m_2, cd_m_1, c_n_2, c_n_1
                N = abs(n_1 - m_2)
                dn = abs(m_2 - n_2)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp -= precomputation[(N, dn)]

                #cd_m_2, cd_m_1, c_n_1, c_n_2
                N = abs(n_2 - m_2)
                dn = abs(m_2 - n_1)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp += precomputation[(N, dn)]
                coefficients[[m_1, m_2, n_2, n_1]] = temp
            end
        end
    end
    return coefficients
end

#TOFINISH
function coeff_Coulomb_cylinder_integrand_LLn(x; dpL=1, N=0, dn=0, n=0)
    a = x[1]; qx = x[2]
    return besselix.(N, x.^2 * 2/dpL^2 ).*exp.( -dn^2*dpL^2*x.^2 ./ (2*x.^2 .+1) )./sqrt.(2*x.^2 .+1)
end

#TOFINISH
function build_coefficient_Coulomb_cylinder_LLn(Ly, N_phi)
    precomputation = Dict{Tuple{Int64, Int64}, Float64}()
    dpL = 2*pi/Ly
    coefficients=Dict{Array{Int64, 1}, Float64}()
    for m_1 = 0:N_phi-1
        for m_2 = m_1+1:N_phi-1
            for n_1 = max(0, m_1+m_2 - N_phi+1):min(N_phi-1, m_1+m_2)
                n_2 = m_1 + m_2 - n_1
                if n_1 >= n_2
                    continue
                end
                #cd_m_1, cd_m_2, c_n_2, c_n_1
                N = abs(n_1 - m_1)
                dn = abs(m_1 - n_2)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp = precomputation[(N, dn)]

                #cd_m_1, cd_m_2, c_n_1, c_n_2
                N = abs(n_2 - m_1)
                dn = abs(m_1 - n_1)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp -= precomputation[(N, dn)]

                #cd_m_2, cd_m_1, c_n_2, c_n_1
                N = abs(n_1 - m_2)
                dn = abs(m_2 - n_2)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp -= precomputation[(N, dn)]

                #cd_m_2, cd_m_1, c_n_1, c_n_2
                N = abs(n_2 - m_2)
                dn = abs(m_2 - n_1)
                if !haskey(precomputation, (N, dn))
                    precomputation[(N, dn)] =1/sqrt(pi)*exp(-N^2*dpL^2/2)*hcubature(x-> coeff_Coulomb_cylinder_integrand_LL0(x, dpL=dpL, N=N, dn=dn), [0.], [10*Ly])[1][1]
                end
                temp += precomputation[(N, dn)]
                coefficients[[m_1, m_2, n_2, n_1]] = temp
            end
        end
    end
    return coefficients
end

#TOFINISH
function build_coefficient_Coulomb_cylinder(Ly, N_phi; n=0, prec=1e-12)
    if n>0
        println("Not implemented, defaulting to 0")
    end
    if n == 0
        return build_coefficient_Coulomb_cylinder_LL0(Ly, N_phi)
    end
    return build_coefficient_Coulomb_cylinder_LLn(Ly, N_phi, n)
end



function build_coefficient_Coulomb(; r::Float64 = 1., Lx::Float64 = -1., Ly::Float64 = -1., N_phi::Int64 = 10, n::Int64=0, prec=1e-12)
    if Lx != -1
        println(string("Generating Coulomb coefficients n=", n," from Lx"))
        Ly = 2*pi*N_phi/Lx
        r = Lx/Ly
    elseif Ly !=-1
        println(string("Generating Coulomb coefficients n=", n," from Ly"))
        Lx = 2*pi*N_phi/Ly
        r = Lx/Ly
    else
        println(string("Generating Coulomb coefficients n=", n," from r"))
        Lx = sqrt(2*pi*N_phi*r)
        Ly = sqrt(2*pi*N_phi/r)
    end
    println(string("Parameters are N_phi=", N_phi, ", r=", round(r, digits = 3), ", Lx =", round(Lx, digits = 3), " and Ly =", round(Ly, digits = 3)))
    return build_coefficient_Coulomb_cylinder(Ly, N_phi, n=n, prec=prec)
end



function extract_diagonal_term(dic, N_phi)
    if length(collect(keys(dic))[1]) == 4
        coeff = zeros(typeof(dic[[0, 1, 1, 0]]), N_phi)
        for y in 0:N_phi-2
            for z in y+1:N_phi-1
                if haskey(dic, [y, z, z, y])
                    v = dic[[y, z, z, y]]
                    coeff[y+1] += v
                    coeff[z+1] += v
                else
                    println(string("Configuration ", (y, z, z, y), " not found"))
                end
            end
        end
    elseif length(collect(keys(dic))[1]) == 6
        coeff = zeros(typeof(dic[[0, 1, 2, 2, 1, 0]]), N_phi)
        for x in 0:N_phi-3
            for y in x+1:N_phi-2
                for z in y+1:N_phi-1
                    if haskey(dic, [x, y, z, z, y, x])
                        v = dic[[x, y, z, z, y, x]]
                        coeff[x+1] += v
                        coeff[y+1] += v
                        coeff[z+1] += v
                    else
                        println(string("Configuration ", (x, y, z, z, y, x), " not found"))
                    end
                end
            end
        end
    elseif length(collect(keys(dic))[1]) == 8
        coeff = zeros(typeof(dic[[0, 1, 2, 3, 3, 2, 1, 0]]), N_phi)
        for w in 0:N_phi-4
            for x in w+1:N_phi-3
                for y in x+1:N_phi-2
                    for z in y+1:N_phi-1
                        if haskey(dic, [w, x, y, z, z, y, x, w])
                            v = dic[[w, x, y, z, z, y, x, w]]
                            coeff[w+1] += v
                            coeff[x+1] += v
                            coeff[y+1] += v
                            coeff[z+1] += v
                        else
                            println(string("Configuration ", (w, x, y, z, z, y, x, w), " not found"))
                        end
                    end
                end
            end
        end
    end
    return coeff
end


function extract_twobody_diagonal_term(dic, N_phi)
    if length(collect(keys(dic))[1]) == 4
        coeff = Dict()
        println("Interaction is too small")
    elseif length(collect(keys(dic))[1]) == 6
        coeff = Dict()
        for m1 in 0:N_phi-2
            for m2 in m1+1:N_phi-1
                for n1 in 0:N_phi-2
                    n2 = m1 + m2 - n1
                    if n2 <= n1
                        continue
                    else
                        temp_coeff = 0
                        for x in 0:N_phi-1
                            if x==m1 || x==m2 || x==n1 || x==n2
                                continue
                            end
                            ca = sort([m1, m2, x])
                            cb = reverse(sort([n1, n2, x]))
                            key = [ca..., cb...]
                            if haskey(dic, key)
                                temp_coeff += dic[key]
                            end
                        end
                        if abs(temp_coeff)> 1e-12
                            coeff[[m1, m2, n2, n1]] = temp_coeff
                        end
                    end
                end
            end
        end
    elseif length(collect(keys(dic))[1]) == 8
        coeff = Dict()
        for m1 in 0:N_phi-2
            for m2 in m1+1:N_phi-1
                for n1 in 0:N_phi-2
                    n2 = m1 + m2 - n1
                    if n2 <= n1
                        continue
                    else
                        temp_coeff = 0
                        for x in 0:N_phi-2
                            if x==m1 || x==m2 || x==n1 || x==n2
                                continue
                            end
                            for y in x+1_N_phi-1
                                if y==m1 || y==m2 || y==n1 || y==n2
                                    continue
                                end
                                ca = sort([m1, m2, x, y])
                                cb = reverse(sort([n1, n2, x, y]))
                                key = [ca..., cb...]
                                if haskey(dic, key)
                                    temp_coeff += 2*dic[key]
                                end
                            end
                        end
                        if abs(temp_coeff)> 1e-12
                            coeff[[m1, m2, n2, n1]] = temp_coeff
                        end
                    end
                end
            end
        end
    end
    return coeff
end


###############################
###Pseudpotentials
###############################

"""
    two_body_factorized_pseudomomentum_auxiliary(x, Ly, m)

Auxiliary function that computes the value of the factorized form of the pseudocoefficient
Inputs: x momentum `2*pi*n/Ly`, Ly the length in the periodic direction and m the moment of the pseudo coefficient
Outputs: a floating value
"""
function two_body_factorized_pseudomomentum_auxiliary(x, Ly, m::Int64)
    return 2^(0.75) * sqrt(2*pi/Ly) * exp(-x^2) * Hermite_polynomial(2*x, n=m)/sqrtfactorial(m)/pi^0.25
end


"""
    build_two_body_coefficient_pseudopotential_factorized_cylinder(Lx, Ly, N_phi, maximalMoment::Int64; prec=1e-12, fermions = true)

Build the dictionary of the factorized form of the two body pseudopotentials on the cylinder
Inputs: Lx and Ly are the dimensions on the torus (Ly the orbital basis), N_phi the number of orbitals, and m the maximal moment (starting at 0)
Outputs: a dictionary of coefficients with keys the shift of the center of mass, values are either Float64
"""
function build_two_body_coefficient_pseudopotential_factorized_cylinder(Lx, Ly, N_phi::Int64, maximalMoment::Int64; prec=1e-12, fermions = true)
    if fermions
        sg_ph = -1
        admissible_m = 1:2:maximalMoment
    else
        sg_ph = 1
        admissible_m = 0:2:maximalMoment
    end
    coefficients = Dict{Float64, Array{Float64, 1}}()
    for k =0.5:0.5:N_phi/2-0.5
        coefficients[k] = zeros(Float64, length(admissible_m))
        for (idx_m, m) in enumerate(admissible_m)
            coefficients[k][idx_m] += (two_body_factorized_pseudomomentum_auxiliary(2*pi*k/Ly, Ly, m) + sg_ph * two_body_factorized_pseudomomentum_auxiliary(-2*pi*k/Ly, Ly, m))/2
        end
    end
    return coefficients
end


function build_two_body_coefficient_pseudopotential_cylinder(coeff, Vs, N_phi)
    full_coeff = Dict()
    for j=0.5:0.5:N_phi-1.5
        for l=mod1(j, 1):1:min(j, N_phi-1-j)
            sg_1 = 1
            n_1 = mod(round(Int64, j+l), N_phi)
            n_2 = mod(round(Int64, j-l), N_phi)
            if n_1 > n_2
                sg_1 = -1
                temp = n_2
                n_2 = n_1
                n_1 = temp
            elseif n_1 == n_2
                continue
            end
            for k = mod1(j, 1):1:min(j,  N_phi-1-j)
                sg_2 = 1
                m_1 = mod(round(Int64, j+k), N_phi)
                m_2 = mod(round(Int64, j-k), N_phi)
                if m_1 > m_2
                    sg_2 = -1
                    temp = m_2
                    m_2 = m_1
                    m_1 = temp
                elseif m_1 == m_2
                    continue
                end
                if !haskey(full_coeff, (m_1, m_2, n_2, n_1))
                    full_coeff[[m_1, m_2, n_2, n_1]] = sg_1*sg_2*sum([coeff[l][x]*Vs[x]*coeff[k][x] for x in 1:length(Vs)])
                else
                    full_coeff[[m_1, m_2, n_2, n_1]] += sg_1*sg_2*sum([coeff[l][x]*Vs[x]*coeff[k][x] for x in 1:length(Vs)])
                end
            end
        end
    end
    return full_coeff
end


function build_two_body_coefficient_pseudopotential(; r::Float64 = 1., Lx::Float64 = -1., Ly::Float64 = -1., N_phi::Int64 = 10, Vs::Array{Float64, 1}=[0], prec=1e-12)
    if Lx != -1
        println("Generating pseudopotential coefficients from Lx")
        Ly = 2*pi*N_phi/Lx
        r = Lx/Ly
    elseif Ly !=-1
        println("Generating pseudopotential coefficients from Ly")
        Lx = 2*pi*N_phi/Ly
        r = Lx/Ly
    else
        println("Generating pseudopotential coefficients from r")
        Lx = sqrt(2*pi*N_phi*r)
        Ly = sqrt(2*pi*N_phi/r)
    end
    println(string("Parameters are N_phi=", N_phi, ", r=", round(r, digits = 3), ", Lx =", round(Lx, digits = 3), "and Ly =", round(Ly, digits = 3)))
    maximalMoment = 2*length(Vs)-1
    temp_coeff = build_two_body_coefficient_pseudopotential_factorized_cylinder(Lx, Ly, N_phi, maximalMoment, prec=prec, fermions = true)
    return build_two_body_coefficient_pseudopotential_cylinder(temp_coeff, Vs, N_phi)
end




"""
    build_two_body_coefficient_pseudopotential_factorized_cylinder_TI(Lx, Ly, N_phi, maximalMoment::Int64; prec=1e-12, fermions = true)

Build the dictionary of the translation invariant factorized form of the two body pseudopotentials on the cylinder
Inputs: Lx and Ly are the dimensions on the torus (Ly the orbital basis), N_phi the number of orbitals, and m the maximal moment (starting at 0)
Outputs: a dictionary of coefficients with keys the shift of the center of mass, values are either Float64
"""
function build_two_body_coefficient_pseudopotential_cylinder_TI(coeff, Vs, N_phi)
    full_coeff = Dict()
    for j=0.5:0.5:N_phi-1.5
        for l=mod1(j, 1):1:min(j, N_phi-1-j)
            sg_1 = 1
            n_1 = mod(round(Int64, j+l), N_phi)
            n_2 = mod(round(Int64, j-l), N_phi)
            if n_1 > n_2
                sg_1 = -1
                temp = n_2
                n_2 = n_1
                n_1 = temp
            elseif n_1 == n_2
                continue
            end
            for k = mod1(j, 1):1:min(j,  N_phi-1-j)
                sg_2 = 1
                m_1 = mod(round(Int64, j+k), N_phi)
                m_2 = mod(round(Int64, j-k), N_phi)
                if m_1 > m_2
                    sg_2 = -1
                    temp = m_2
                    m_2 = m_1
                    m_1 = temp
                elseif m_1 == m_2
                    continue
                end
                if n_1 * m_1 != 0
                    continue
                end
                if !haskey(full_coeff, (m_1, m_2, n_2, n_1))
                    full_coeff[[m_1, m_2, n_2, n_1]] = sg_1*sg_2*sum([coeff[l][x]*Vs[x]*coeff[k][x] for x in 1:length(Vs)])
                else
                    full_coeff[[m_1, m_2, n_2, n_1]] += sg_1*sg_2*sum([coeff[l][x]*Vs[x]*coeff[k][x] for x in 1:length(Vs)])
                end
            end
        end
    end
    return full_coeff
end


function build_two_body_coefficient_pseudopotential_TI(; r::Float64 = 1., Lx::Float64 = -1., Ly::Float64 = -1., N_phi::Int64 = 10, Vs::Array{Float64, 1}=[0], prec=1e-12)
    if Lx != -1
        println("Generating pseudopotential coefficients from Lx")
        Ly = 2*pi*N_phi/Lx
        r = Lx/Ly
    elseif Ly !=-1
        println("Generating pseudopotential coefficients from Ly")
        Lx = 2*pi*N_phi/Ly
        r = Lx/Ly
    else
        println("Generating pseudopotential coefficients from r")
        Lx = sqrt(2*pi*N_phi*r)
        Ly = sqrt(2*pi*N_phi/r)
    end
    println(string("Parameters are N_phi=", N_phi, ", r=", round(r, digits = 3), ", Lx =", round(Lx, digits = 3), "and Ly =", round(Ly, digits = 3)))
    maximalMoment = 2*length(Vs)-1
    temp_coeff = build_two_body_coefficient_pseudopotential_factorized_cylinder(Lx, Ly, N_phi, maximalMoment, prec=prec, fermions = true)
    return build_two_body_coefficient_pseudopotential_cylinder_TI(temp_coeff, Vs, N_phi)
end



#############################
###Three-body pseudpotentials
#############################
function build_three_body_coefficient_factorized_cylinder(Lx, Ly, N_phi; prec=1e-12)
    coefficients = Dict{Tuple{Int64, Int64}, Float64}()
    for g = 0:2
        for (k, q) = Iterators.product(-3N_phi-g:3:3N_phi-1, -3N_phi-g:3:3N_phi-1)
            coefficients[(k, q)] = sqrt(Lx/Ly)/sqrt(N_phi)*(2*pi/Ly)^3 * W_polynomial(k/3, q/3, -k/3-q/3)*exp(-2*pi^2/Ly^2*( (k/3)^2 + (q/3)^2 + (-k/3 - q/3)^2 ) )
        end
    end
    return coefficients
end

function streamline_three_body_dictionnary_cylinder(dic, N_phi)
    coefficients = Dict{Tuple{Int64, Int64, Int64}, Float64}()
    for n_1 in 0:N_phi-3
        for n_2 in n_1+1:N_phi-2
            for n_3 in n_2+1:N_phi-1
                R3 = n_1 + n_2 + n_3
                coefficients[(n_1, n_2, n_3)]= dic[(R3 - 3n_1, R3 - 3n_2)]
            end
        end
    end
    return coefficients
end

function build_hamiltonian_from_three_body_factorized_streamlined_dictionary(coeff, N_phi; global_sign = 1, prec = 1e-12)
    full_coeff = Dict{Array{Int64, 1}, Float64}()
    for R3 =3:3N_phi-3  #Barycenter
        for n_1 in max(0, R3 - 2N_phi + 3):min(N_phi-3, R3÷3 - 1)
            for n_2 in max(n_1+1, R3 - n_1 - N_phi + 1):min(N_phi-2, (R3 - n_1 - 1)÷2)
                n_3 = R3 - n_1 - n_2
                for m_1 in max(0, R3 - 2N_phi + 3):min(N_phi-3, R3÷3 - 1)
                    for m_2 in max(m_1+1, R3 - m_1 - N_phi + 1):min(N_phi-2, (R3 - m_1 - 1)÷2)
                        m_3 = mod(R3 - m_1 - m_2, N_phi)
                        temp = global_sign * (coeff[n_1, n_2, n_3]'*coeff[m_1, m_2, m_3])
                        if abs(temp) > prec
                            full_coeff[[m_1, m_2, m_3, n_3, n_2, n_1]] = temp
                        end
                    end
                end
            end
        end
    end
    return full_coeff
end

function build_three_body_pseudopotentials(; r::Float64 = 1., Lx::Float64 = -1., Ly::Float64 = -1., N_phi::Int64 = 10, prec=1e-12, global_sign = 1)
    if Lx != -1
        println("Generating 3body pseudopotential coefficients from Lx")
        Ly = 2*pi*N_phi/Lx
        r = Lx/Ly
    elseif Ly !=-1
        println("Generating 3body pseudopotential coefficients from Ly")
        Lx = 2*pi*N_phi/Ly
        r = Lx/Ly
    else
        println("Generating 3body pseudopotential coefficients from r")
        Lx = sqrt(2*pi*N_phi*r)
        Ly = sqrt(2*pi*N_phi/r)
    end
    println(string("Parameters are N_phi=", N_phi, ", r=", round(r, digits = 3), ", Lx =", round(Lx, digits = 3), "and Ly =", round(Ly, digits = 3)))
    coeff = build_three_body_coefficient_factorized_cylinder(Lx, Ly, N_phi, prec=prec)
    coeff = streamline_three_body_dictionnary_cylinder(coeff, N_phi)
    return build_hamiltonian_from_three_body_factorized_streamlined_dictionary(coeff, N_phi, global_sign = global_sign, prec = prec)
end



##############################
###Four-body pseudopotentials
##############################
function build_four_body_coefficient_factorized_cylinder(Lx, Ly, N_phi; prec=1e-12)
    prefactor = sqrt(Lx/Ly)/sqrt(N_phi)*(2*pi/Ly)^4
    coefficients = Dict{Tuple{Int64, Int64, Int64}, Float64}()
    for g = 0:3
        expFactor = zeros(length(collect(-4N_phi-g:4:4N_phi-1)))
        for (xk, k) in enumerate(-4N_phi-g:4:4N_phi-1)
            expFactor[xk] = exp(-2*pi^2/Ly^2*(k/4)^2)
        end
        for (xk, k) in enumerate(-4N_phi-g:4:4N_phi-1)
            for (xq, q) in enumerate(-4N_phi-g:4:4N_phi-1)
                if k==q
                    continue
                end
                for (xr, r) in enumerate(-4N_phi-g:4:4N_phi-1)
                    if k==r || q==r
                        continue
                    end
                    coefficients[(k, q, r)]= prefactor*W_polynomial(k/4, q/4, r/4, -k/4-q/4-r/4)*expFactor[xk]*expFactor[xq]*
                                    expFactor[xr]*exp(-2*pi^2/Ly^2*(k/4+ q/4 + r/4)^2)
                end
            end
        end
    end
    return coefficients
end


function streamline_four_body_dictionnary_cylinder(dic, N_phi)
    coefficients = Dict{Tuple{Int64, Int64, Int64, Int64}, Float64}()
    for n_1 in 0:N_phi-4
        for n_2 in n_1+1:N_phi-3
            for n_3 in n_2+1:N_phi-2
                for n_4 in n_3+1:N_phi-1
                    R3 = n_1 + n_2 + n_3 + n_4
                    coefficients[(n_1, n_2, n_3, n_4)]= dic[(R3 - 4n_1, R3 - 4n_2, R3 - 4n_3)]
                end
            end
        end
    end
    return coefficients
end


function build_hamiltonian_from_four_body_factorized_streamlined_dictionary(coeff, N_phi; global_sign = 1, prec = 1e-12)
    full_coeff = Dict{Array{Int64, 1}, Float64}()
    for R4 =6:4N_phi-10  #Barycenter
        for n_1 in max(0, R4 - 3N_phi + 6):min(N_phi-4, (R4-6)÷4)
            for n_2 in max(n_1+1, R4 - n_1 - 2N_phi + 3):min(N_phi-3, (R4 - n_1 - 3)÷3)
                for n_3 in max(n_2+1, R4 - n_1 - n_2 - N_phi + 1):min(N_phi-2, (R4 -n_1 - n_2 -1)÷2)
                    n_4 = R4 - n_1 - n_2 - n_3
                    for m_1 in max(0, R4 - 3N_phi + 6):min(N_phi-4, (R4-6)÷4)
                        for m_2 in max(m_1+1, R4 - m_1 - 2N_phi + 3):min(N_phi-3, (R4 - m_1 - 3)÷3)
                            for m_3 in max(m_2+1, R4 - m_1 - m_2 - N_phi + 1):min(N_phi-2, (R4 -m_1 - m_2 -1)÷2)
                                m_4 = R4 - m_1 - m_2 - m_3
                                temp = global_sign * (coeff[n_1, n_2, n_3, n_4]'*coeff[m_1, m_2, m_3, m_4])
                                if abs(temp) > prec
                                    full_coeff[[m_1, m_2, m_3, m_4, n_4, n_3, n_2, n_1]] = temp
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return full_coeff
end

function build_four_body_pseudopotentials(; r::Float64 = 1., Lx::Float64 = -1., Ly::Float64 = -1., N_phi::Int64 = 10, prec=1e-12, global_sign = 1)
    if Lx != -1
        println("Generating 4body pseudopotential coefficients from Lx")
        Ly = 2*pi*N_phi/Lx
        r = Lx/Ly
    elseif Ly !=-1
        println("Generating 4body pseudopotential coefficients from Ly")
        Lx = 2*pi*N_phi/Ly
        r = Lx/Ly
    else
        println("Generating 4body pseudopotential coefficients from r")
        Lx = sqrt(2*pi*N_phi*r)
        Ly = sqrt(2*pi*N_phi/r)
    end
    println(string("Parameters are N_phi=", N_phi, ", r=", round(r, digits = 3), ", Lx =", round(Lx, digits = 3), "and Ly =", round(Ly, digits = 3)))
    coeff = build_four_body_coefficient_factorized_cylinder(Lx, Ly, N_phi, prec=prec)
    coeff = streamline_four_body_dictionnary_cylinder(coeff, N_phi)
    return build_hamiltonian_from_four_body_factorized_streamlined_dictionary(coeff, N_phi, global_sign = global_sign, prec = prec)
end
