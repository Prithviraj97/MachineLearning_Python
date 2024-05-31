using Statistics, Distributions, Combinatorics, Random, SpecialFunctions, PyCall, Pkg, Conda
using Combinatorics, LinearAlgebra, CSV, DataFrames, Serialization, IterTools, Distributed, .Threads
scipy_integrate = pyimport("scipy.integrate")
#Pkg.add("Conda")
#Pkg.add("PyCall")
#Conda.add("scipy")

Threads.nthreads()=4
println("Number of threads:", Threads.nthreads())

mu_00=mu_11=1 #optimal payoff distribution
mu_01=mu_10=-1 #suboptimal payoff distribution
sigma_00=sigma_01=sigma_10=sigma_11=1 #in Efferson 2016, they use sigma = 1,5,9


function N(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_00, mean_10)
    """
    Calculates the numerator of the choice rule which represents the social information indicating that Z=0.
    """

    n1 = phi_hat * (1-gamma_hat) * (1-q_hat)^i * q_hat^(K-i)
    n2 = (1-phi_hat) * gamma_hat * q_hat^i * (1-q_hat)^(K-i)
    n3 = (1-phi_hat) * (1-gamma_hat) * (1-q_hat)^i * q_hat^(K-i)
    n4 = phi_hat * gamma_hat * q_hat^i * (1-q_hat)^(K-i)

    if A == 1 && J == 1
        if i == 0
            result = 0.0
        elseif 0 < i < K
            result = n1 * h_10(K, i) + n2 * h_11(K, i)
        else  # i == K
            result = n1 + n2
        end
    elseif A == 1 && J == 0
        if i == 0
            result = n1 + n2
        elseif 0 < i < K
            result = n1 * h_00(K, i) + n2 * h_01(K, i)
        else  # i == K
            result = 0.0
        end
    elseif A == 0
        if J == 1
            if i == 0
                result = 0.0
            elseif 0 < i < K
                result = n3 * h_10(K, i) + n4 * h_11(K, i)
            else  # i == K
                result = n3 + n4
            end
        else  # J == 0
            if i == 0
                result = n3 + n4
            elseif 0 < i < K
                result = n3 * h_00(K, i) + n4 * h_01(K, i)
            else  # i == K
                result = 0.0
            end
        end
    end
    return result * (mean_00 - mean_10)
end

function B(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_01, mean_11)
    """
    Calculates the denominator of the choice rule which represents the social information indicating that Z=1.
    """

    b1 = (1 - phi_hat) * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b2 = phi_hat * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)
    b3 = phi_hat * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b4 = (1 - phi_hat) * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)

    if A == 1 && J == 1
        if i == 0
            result = 0.0
        elseif 0 < i < K
            result = (b1 * h_10(K, i) + b2 * h_11(K, i)) 
        else  # i == K
            result = (b1 + b2) 
        end
    elseif A == 1 && J == 0
        if i == 0
            result = (b1 + b2)
        elseif 0 < i < K
            result = (b1 * h_00(K, i) + b2 * h_01(K, i)) 
        else  # i == K
            result = 0.0
        end
    elseif A == 0
        if J == 1
            if i == 0
                result = 0.0
            elseif 0 < i < K
                result = (b3 * h_10(K, i) + b4 * h_11(K, i)) 
            else  # i == K
                result = (b3 + b4) 
            end
        else  # J == 0
            if i == 0
                result = (b3 + b4) 
            elseif 0 < i < K
                result = (b3 * h_00(K, i) + b4 * h_01(K, i)) 
            else  # i == K
                result = 0.0
            end
        end
    end
    return result * (mean_11 - mean_01)
end


#pdf and cdf for the optimal and suboptimal distributions:
f_opt(x) = pdf(Normal(mu_00, sigma_00), x)
F_opt(x) = cdf(Normal(mu_00, sigma_00), x)

f_sub(x) = pdf(Normal(mu_01, sigma_01), x)
F_sub(x) = cdf(Normal(mu_01, sigma_01), x)

g_opt(v) = pdf(Normal(mu_11, sigma_11), v)
G_opt(v) = cdf(Normal(mu_11, sigma_11), v)

g_sub(v) = pdf(Normal(mu_10, sigma_10), v)
G_sub(v) = cdf(Normal(mu_10, sigma_10), v)

#order statistics:
f_X_opt(K, i, x) = (K - i) * F_opt(x)^(K - i - 1) * f_opt(x)
f_X_sub(K, i, x) = (K - i) * F_sub(x)^(K - i - 1) * f_sub(x)

g_V_opt(K, i, v) = i * G_opt(v)^(i - 1) * g_opt(v)
g_V_sub(K, i, v) = i * G_sub(v)^(i - 1) * g_sub(v)

#conditional probabilities:
function h_10(K, i)
    f = (v, x) -> f_X_opt(K, i, x) * g_V_sub(K, i, v)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (x) -> -Inf, (x) -> x)
    return 1 - result
end

function h_11(K, i)
    f = (v, x) -> f_X_sub(K, i, x) * g_V_opt(K, i, v)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (x) -> -Inf, (x) -> x)
    return 1 - result
end

function h_00(K, i)
    f = (x, v) -> f_X_opt(K, i, x) * g_V_sub(K, i, v)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (v) -> -Inf, (v) -> v)
    return 1 - result
end

function h_01(K, i)
    f = (x, v) -> f_X_sub(K, i, x) * g_V_opt(K, i, v)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (v) -> -Inf, (v) -> v)
    return 1 - result
end

#private environmental signal cognitive rep:
function signal_ratio(s, rho_hat)
    if s == 1
        return rho_hat / (1 - rho_hat)
    else
        return (1 - rho_hat) / rho_hat
    end
end
#signal_ratio.(s, rho_hat) = ifelse.(s .== 1, rho_hat ./ (1 .- rho_hat), (1 .- rho_hat) ./ rho_hat)

#decision-making rule:
function choice_1(signal_ratio, N, B)
    if B == 0
        return 0
    elseif signal_ratio > N / B
        return 1
    else
        return 0
    end
end
#choice_1.(signal_ratio, N, B) = ifelse.(B .== 0, 0, ifelse.(signal_ratio .> N ./ B, 1, 0))







function main_computation()
    K = 3
    mu_00=mu_11=1 #optimal payoff distribution
    mu_01=mu_10=-1 #suboptimal payoff distribution
    sigma_00=sigma_01=sigma_10=sigma_11=1 #in Efferson 2016, they use sigma = 1,5,9

    #create matrix for Omega - the set of possible observations:
    i_values = 0:K
    s_values = [0, 1]
    a_values = [0, 1]
    j_values = [0, 1]
    
    combinations = []
    
    for i in i_values
        if i == 0
            for (s, a) in Iterators.product(s_values, a_values)
                push!(combinations, [i, s, a, 0]) #the 0 is j=0, if i=0 j must equal 0
            end
        elseif i == K
            for (s, a) in Iterators.product(s_values, a_values)
                push!(combinations, [i, s, a, 1]) #the 1 is j=1, if i=K j must equal 1
            end
        else
            for (j, s, a) in Iterators.product(j_values, s_values, a_values)
                push!(combinations, [i, s, a, j])
            end
        end
    end
    
    Omega_matrix = hcat(combinations...)
    println("Shape of the Omega matrix: ", size(Omega_matrix))
    #Shape of the Omega matrix: (4, 24)
    
    #create matrix for the genotype space:
    rho_hat_values = LinRange(0.5001, 0.9999, 6) 
    q_hat_values = LinRange(0.0001, 0.9999, 11) 
    gamma_hat_values = LinRange(0.0001, 0.9999, 11) 
    phi_hat_values = LinRange(0.0001, 0.9999, 11) 
    #the above implies a genotype space of 7,986 genotypes
    
    combinations_gene = collect(Iterators.product(rho_hat_values, q_hat_values, gamma_hat_values, phi_hat_values))
    genotype_matrix = hcat(map(collect, combinations_gene)...)
    println("Shape of the genotype matrix: ", size(genotype_matrix))
    #Shape of the genotype matrix: (4, 7986)

    choice_matrix = zeros(size(genotype_matrix, 2), size(Omega_matrix, 2))

    @threads for g_idx in axes(genotype_matrix, 2)
        rho_hat, q_hat, gamma_hat, phi_hat = genotype_matrix[:, g_idx]
        for o_idx in axes(Omega_matrix, 2)
            i, s, a, j = Omega_matrix[:, o_idx]

            N_value = N(i, q_hat, gamma_hat, phi_hat, a, j, K, mu_00, mu_10)
            B_value = B(i, q_hat, gamma_hat, phi_hat, a, j, K, mu_01, mu_11)
            signal_ratio_value = signal_ratio(s, rho_hat)
            C_value = choice_1(signal_ratio_value, N_value, B_value)
            choice_matrix[g_idx, o_idx] = C_value
        end
    end

    @info "Computation completed"
    return choice_matrix
end

function save_data(data, filename)
    open(filename, "w") do f
        serialize(f, data)
    end
end

function load_data(filename)
    open(filename, "r") do f
        return deserialize(f)
    end
end

@time choice_matrix = main_computation()
println("First 5 rows of the choice matrix:")
println(choice_matrix[1:5, :])

choice_file = "/Users/kbonner/Documents/PhD/Projects/strategy_space/Results/test_vectorized/choice_7986.jls" #saves choice results while looping through calculations
save_data(choice_matrix, choice_file)

##
#load and check results:
choice_matrix = load_data(choice_file)

if choice_matrix !== nothing
    println("Choice matrix loaded successfully.")
    println("Shape of the matrix: ", size(choice_matrix))
else
    println("choice matrix empty.")
end

#reduce the matrix so that only unique rows remain:
reduced_matrix = unique(choice_matrix, dims=1)  #finds unique rows

println("Shape of the reduced result matrix: ", size(reduced_matrix))
#Shape of the reduced result matrix: () <- result from genotype matrix with shape (4, 7986)