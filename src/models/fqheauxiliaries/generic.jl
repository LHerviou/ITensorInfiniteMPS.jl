
"""
    sqrtfactorial(m)

Return `sqrt(m!)`
Inputs: `m` a number
"""
function sqrtfactorial(m)
    res = 1
    for j=1:m
        res*=sqrt(j)
    end
    return res
end



"""
    Laguerre_polynomial(x; n=0)

Return the Laguerre polynomial of arbitrary degre
Inputs: `x` the input and `n` the degree of the Laguerre polynomial
Output: the value of the Laguerre polynomial
"""
function Laguerre_polynomial(x; n=0)
    if n==0
        return 1
    elseif n==1
        return 1-x
    elseif n==2
        return x^2/2 -2x + 1
    end
    Lm2 = 1
    Lm1 = 1-x
    Lm = x^2/2 -2x + 1
    for j = 3:n
        Lm2 = Lm1
        Lm1 = Lm
        Lm = ((2j-1-x)*Lm1 - (j-1)*Lm2)/j
    end
    return Lm
end


"""
    Hermite_polynomial(x; n=0)

Return the Hermite polynomial of arbitrary degre
Inputs: `x` the input and `n` the degree of the Hermite polynomial (probabilist notation)
Output: the value of the Laguerre polynomial
"""
function Hermite_polynomial(x; n=0)
    if n==0
        return 1
    elseif n==1
        return x
    elseif n==2
        return x^2-1
    end
    Lm2 = 1
    Lm1 = x
    Lm = x^2-1
    for j = 3:n
        Lm2 = Lm1
        Lm1 = Lm
        Lm = x*Lm1 - (j-1)*Lm2
    end
    return Lm
end




###Implementing required polynomials for 3 body pseudo potentials:
function W_polynomial(ns...)
    N = length(ns)
    if N==2
        return ns[1] - ns[2]
    end
    if N==1
        return 0
    end
    res = 1
    for j=2:N
        res*=(ns[1]-ns[j])
    end
    return res*W_polynomial(ns[2:end]...)
end


function W_polynomial(n1, n2)
    return (n1-n2)
end
function W_polynomial(n1, n2, n3)
    return (n1-n2)*(n1-n3)*(n2-n3)
end

function W_polynomial(n1, n2, n3, n4)
    return (n1-n2)*(n1-n3)*(n1-n4)*(n2-n3)*(n2-n4)*(n3-n4)
end



function check_max_range_optimized_Hamiltonian(coeff::Dict)
    temp = 0
    for k in keys(coeff)
        temp = max(temp, k[end] - k[2])
    end
    return temp
end


function filter_optimized_Hamiltonian_by_site(coeff::Dict; n = 1)
    res = Dict()
    for (k, v) in coeff
        for x in 2:2:length(k)
            if k[x] == n
                res[k] = v
            elseif k[x] > n
                continue
            end
        end
    end
    return res
end

function filter_optimized_Hamiltonian_by_first_site(coeff::Dict; n = 1)
    res = Dict()
    for (k, v) in coeff
        if k[2] == n
            res[k] = v
        end
    end
    return res
end


function sort_by_configuration(coeff::Dict)
    res = Dict()
    for (k, v) in coeff
        if !haskey(res, k[1:2:end])
            res[k[1:2:end]] = Dict()
        end
        res[k[1:2:end]][k] = v
    end
    return res
end
