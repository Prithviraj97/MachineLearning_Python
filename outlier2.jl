using Statistics, Distributions, Combinatorics, Random, SpecialFunctions
using PyCall
using LinearAlgebra, DataFrames, Serialization, IterTools, ThreadTools

ThreadTools.nthreads() = 1
# Ensure correct number of threads
println("Number of threads: ", ThreadTools.nthreads())

# Define distributions using parameters
mu_00 = mu_11 = 1
mu_01 = mu_10 = -1
sigma_00 = sigma_01 = sigma_10 = sigma_11 = 1

# Statistical functions
f_opt(x) = pdf(Normal(mu_00, sigma_00), x)
F_opt(x) = cdf(Normal(mu_00, sigma_00), x)
f_sub(x) = pdf(Normal(mu_01, sigma_01), x)
F_sub(x) = cdf(Normal(mu_01, sigma_01), x)
g_opt(v) = pdf(Normal(mu_11, sigma_11), v)
G_opt(v) = cdf(Normal(mu_11, sigma_11), v)
g_sub(v) = pdf(Normal(mu_10, sigma_10), v)
G_sub(v) = cdf(Normal(mu_10, sigma_10), v)

# Use PyCall for scipy integration
scipy_integrate = pyimport("scipy.integrate")

# Precompute integrals for h functions if they are often reused
h_cache = Dict()
function h_00(K, i)
    key = ("h_00", K, i)
    if haskey(h_cache, key)
        return h_cache[key]
    end
    f = (x, v) -> f_opt(x) * (K - i) * F_opt(x)^(K - i - 1) * g_sub(v) * i * G_sub(v)^(i - 1)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (v) -> -Inf, (v) -> v)
    h_cache[key] = 1 - result
    return h_cache[key]
end

function h_01(K, i)
    key = ("h_01", K, i)
    if haskey(h_cache, key)
        return h_cache[key]
    end
    f = (x, v) -> f_sub(x) * (K - i) * F_sub(x)^(K - i - 1) * g_opt(v) * i * G_opt(v)^(i - 1)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (v) -> -Inf, (v) -> v)
    h_cache[key] = 1 - result
    return h_cache[key]
end

function h_10(K, i)
    key = ("h_10", K, i)
    if haskey(h_cache, key)
        return h_cache[key]
    end
    f = (v, x) -> g_sub(v) * i * G_sub(v)^(i - 1) * f_opt(x) * (K - i) * F_opt(x)^(K - i - 1)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (x) -> -Inf, (x) -> x)
    h_cache[key] = 1 - result
    return h_cache[key]
end

function h_11(K, i)
    key = ("h_11", K, i)
    if haskey(h_cache, key)
        return h_cache[key]
    end
    f = (v, x) -> g_opt(v) * i * G_opt(v)^(i - 1) * f_sub(x) * (K - i) * F_sub(x)^(K - i - 1)
    result, _ = scipy_integrate.dblquad(f, -Inf, Inf, (x) -> -Inf, (x) -> x)
    h_cache[key] = 1 - result
    return h_cache[key]
end

# N and B functions as provided
function N(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_00, mean_10)
    n1 = phi_hat * (1 - gamma_hat) * (1 - q_hat)^i * q_hat^(K - i)
    n2 = (1 - phi_hat) * gamma_hat * q_hat^i * (1 - q_hat)^(K - i)
    n3 = (1 - phi_hat) * (1 - gamma_hat) * (1 - q_hat)^i * q_hat^(K - i)
    n4 = phi_hat * gamma_hat * q_hat^i * (1 - q_hat)^(K - i)
    result = 0.0

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
    b1 = (1 - phi_hat) * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b2 = phi_hat * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)
    b3 = phi_hat * gamma_hat * (1 - q_hat)^i * q_hat^(K - i)
    b4 = (1 - phi_hat) * (1 - gamma_hat) * q_hat^i * (1 - q_hat)^(K - i)
    result = 0.0

    if A == 1 && J == 1
        if i == 0
            result = 0.0
        elseif 0 < i < K
            result = b1 * h_10(K, i) + b2 * h_11(K, i)
        else  # i == K
            result = b1 + b2
        end
    elseif A == 1 && J == 0
        if i == 0
            result = b1 + b2
        elseif 0 < i < K
            result = b1 * h_00(K, i) + b2 * h_01(K, i)
        else  # i == K
            result = 0.0
        end
    elseif A == 0
        if J == 1
            if i == 0
                result = 0.0
            elseif 0 < i < K
                result = b3 * h_10(K, i) + b4 * h_11(K, i)
            else  # i == K
                result = b3 + b4
            end
        else  # J == 0
            if i == 0
                result = b3 + b4
            elseif 0 < i < K
                result = b3 * h_00(K, i) + b4 * h_01(K, i)
            else  # i == K
                result = 0.0
            end
        end
    end
    return result * (mean_11 - mean_01)
end

# Signal ratio function
function signal_ratio(s, rho_hat)
    return ifelse(s == 1, rho_hat / (1 - rho_hat), (1 - rho_hat) / rho_hat)
end

# Decision-making rule
function choice_1(signal_ratio, N, B)
    return ifelse(B == 0, 0, ifelse(signal_ratio > N / B, 1, 0))
end

# Main computation function with threading
function main_computation()
    K = 3
    i_values = 0:K
    s_values = [0, 1]
    a_values = [0, 1]
    j_values = [0, 1]

    combinations = []
    for i in i_values
        if i == 0
            for (s, a) in Iterators.product(s_values, a_values)
                push!(combinations, [i, s, a, 0]) # j=0 if i=0
            end
        elseif i == K
            for (s, a) in Iterators.product(s_values, a_values)
                push!(combinations, [i, s, a, 1]) # j=1 if i=K
            end
        else
            for (j, s, a) in Iterators.product(j_values, s_values, a_values)
                push!(combinations, [i, s, a, j])
            end
        end
    end

    Omega_matrix = hcat(combinations...)
    println("Shape of the Omega matrix: ", size(Omega_matrix))

    rho_hat_values = LinRange(0.5001, 0.9999, 6)
    q_hat_values = LinRange(0.0001, 0.9999, 11)
    gamma_hat_values = LinRange(0.0001, 0.9999, 11)
    phi_hat_values = LinRange(0.0001, 0.9999, 11)

    combinations_gene = collect(Iterators.product(rho_hat_values, q_hat_values, gamma_hat_values, phi_hat_values))
    genotype_matrix = hcat(map(collect, combinations_gene)...)
    println("Shape of the genotype matrix: ", size(genotype_matrix))

    choice_matrix = zeros(size(genotype_matrix, 2), size(Omega_matrix, 2))

    Threads.@threads for g_idx in axes(genotype_matrix, 2)
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

    return choice_matrix
end

@time choice_matrix = main_computation()
println("First 5 rows of the choice matrix:")
println(choice_matrix[1:5, :])

# Save and load functionality
function save_data(data, filename)
    open(filename, "w") do f
        serialize(f, data)
    end
end

function load_data(filename)
    open(filename, "r") do f
        deserialize(f)
    end
end

# Example usage
choice_file = "choice_matrix.jls"
save_data(choice_matrix, choice_file)
loaded_matrix = load_data(choice_file)

if loaded_matrix !== nothing
    println("Choice matrix loaded successfully.")
    println("Shape of the matrix: ", size(loaded_matrix))
else
    println("Choice matrix empty.")
end

reduced_matrix = unique(loaded_matrix, dims=1)
println("Shape of the reduced result matrix: ", size(reduced_matrix))
