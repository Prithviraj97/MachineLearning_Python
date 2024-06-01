using Statistics, Distributions, Combinatorics, Random, SpecialFunctions
using Combinatorics, LinearAlgebra, CSV, DataFrames, Serialization, IterTools, Distributed, .Threads
using PyCall

scipy_integrate = pyimport("scipy.integrate")
# ThreadTools.nthreads() = 4
println("Number of threads:", Threads.nthreads())

# Constants
mu_00 = mu_11 = 1 # optimal payoff distribution
mu_01 = mu_10 = -1 # suboptimal payoff distribution
sigma_00 = sigma_01 = sigma_10 = sigma_11 = 1 # sigma values

# Define PDF and CDF functions using Distributions.jl
f_opt(x) = pdf(Normal(mu_00, sigma_00), x)
F_opt(x) = cdf(Normal(mu_00, sigma_00), x)
f_sub(x) = pdf(Normal(mu_01, sigma_01), x)
F_sub(x) = cdf(Normal(mu_01, sigma_01), x)
g_opt(v) = pdf(Normal(mu_11, sigma_11), v)
G_opt(v) = cdf(Normal(mu_11, sigma_11), v)
g_sub(v) = pdf(Normal(mu_10, sigma_10), v)
G_sub(v) = cdf(Normal(mu_10, sigma_10), v)

# Define function to handle integration using scipy
function integrate_function(f)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (x) -> -Inf, (x) -> x)
    return 1 - result
end

# Conditional probabilities (order statistics)
function h_00(K, i)
    f = (x, v) -> f_opt(x) * (K - i) * F_opt(x)^(K - i - 1) * g_sub(v) * i * G_sub(v)^(i - 1)
    integrate_function(f)
end

function h_01(K, i)
    f = (x, v) -> f_sub(x) * (K - i) * F_sub(x)^(K - i - 1) * g_opt(v) * i * G_opt(v)^(i - 1)
    integrate_function(f)
end

# Signal ratio function
function signal_ratio(s, rho_hat)
    return ifelse(s == 1, rho_hat / (1 - rho_hat), (1 - rho_hat) / rho_hat)
end

# Decision-making rule
function choice_1(signal_ratio, N, B)
    return ifelse(B == 0, 0, ifelse(signal_ratio > N / B, 1, 0))
end
# Define functions N and B
function N(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_00, mean_10)
    n1 = phi_hat * (1 - gamma_hat) * (1 - q_hat)^i * q_hat^(K - i)
    n2 = (1 - phi_hat) * gamma_hat * q_hat^i * (1 - q_hat)^(K - i)
    n3 = (1 - phi_hat) * (1 - gamma_hat) * (1 - q_hat)^i * q_hat^(K - i)
    n4 = phi_hat * gamma_hat * q_hat^i * (1 - q_hat)^(K - i)

    if A == 1 && J == 1
        if i == 0
            result = 0.0
        elseif 0 < i < K
            result = n1 * h_10(K, i) + n2 * h_11(K, i)
        else
            result = n1 + n2
        end
    elseif A == 1 && J == 0
        if i == 0
            result = n1 + n2
        elseif 0 < i < K
            result = n1 * h_00(K, i) + n2 * h_01(K, i)
        else
            result = 0.0
        end
    elseif A == 0
        if J == 1
            if i == 0
                result = 0.0
            elseif 0 < i < K
                result = n3 * h_10(K, i) + n4 * h_11(K, i)
            else
                result = n3 + n4
            end
        else
            if i == 0
                result = n3 + n4
            elseif 0 < i < K
                result = n3 * h_00(K, i) + n4 * h_01(K, i)
            else
                result = 0.0
            end
        end
    end
    return result * (mean_00 - mean_10)
end

function B(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_01, mean_11)
    b1 = (1 - phi_hat) * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b2 = phi_hat * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)
    b3 = phi_hat * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b4 = (1 - phi_hat) * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)

    if A == 1 && J == 1
        if i == 0
            result = 0.0
        elseif 0 < i < K
            result = (b1 * h_10(K, i) + b2 * h_11(K, i))
        else
            result = (b1 + b2)
        end
    elseif A == 1 && J == 0
        if i == 0
            result = (b1 + b2)
        elseif 0 < i < K
            result = (b1 * h_00(K, i) + b2 * h_01(K, i))
        else
            result = 0.0
        end
    elseif A == 0
        if J == 1
            if i == 0
                result = 0.0
            elseif 0 < i < K
                result = (b3 * h_10(K, i) + b4 * h_11(K, i))
            else
                result = (b3 + b4)
            end
        else
            if i == 0
                result = (b3 + b4)
            elseif 0 < i < K
                result = (b3 * h_00(K, i) + b4 * h_01(K, i))
            else
                result = 0.0
            end
        end
    end
    return result * (mean_11 - mean_01)
end

# Main computation function optimized for threading
function main_computation()
    K = 3
    mu_00 = mu_11 = 1
    mu_01 = mu_10 = -1
    sigma_00 = sigma_01 = sigma_10 = sigma_11 = 1

    # Create matrix for Omega - the set of possible observations:
    i_values = 0:K
    s_values = [0, 1]
    a_values = [0, 1]
    j_values = [0, 1]
    combinations = collect(Iterators.product(i_values, s_values, a_values, j_values))
    
    Omega_matrix = hcat(map(collect, combinations)...)
    println("Shape of the Omega matrix: ", size(Omega_matrix))
    
    # Generate genotypes
    rho_hat_values = LinRange(0.5001, 0.9999, 6)
    q_hat_values = LinRange(0.0001, 0.9999, 11)
    gamma_hat_values = LinRange(0.0001, 0.9999, 11)
    phi_hat_values = LinRange(0.0001, 0.9999, 11)
    combinations_gene = collect(Iterators.product(rho_hat_values, q_hat_values, gamma_hat_values, phi_hat_values))
    genotype_matrix = hcat(map(collect, combinations_gene)...)
    println("Shape of the genotype matrix: ", size(genotype_matrix))
    
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

@time choice_matrix = main_computation()
println("First 5 rows of the choice matrix:")
println(choice_matrix[1:5, :])

# Save and load results
choice_file = "choice_matrix.jls"
open(choice_file, "w") do

 f
    serialize(f, choice_matrix)
end

loaded_matrix = open(choice_file, "r") do f
    deserialize(f)
end

if loaded_matrix !== nothing
    println("Choice matrix loaded successfully.")
    println("Shape of the matrix: ", size(loaded_matrix))
else
    println("Choice matrix empty.")
end

# Reduce matrix to unique rows
reduced_matrix = unique(loaded_matrix, dims=1)
println("Shape of the reduced result matrix: ", size(reduced_matrix))
