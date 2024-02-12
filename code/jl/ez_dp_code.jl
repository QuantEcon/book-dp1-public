include("ez_model.jl")

"The policy operator for the original model."
function T_σ(v::Matrix, σ, model)
    w_n, e_n = size(v)
    w_idx, e_idx = 1:w_n, 1:e_n
    v_new = similar(v)
    for (i, j) in product(w_idx, e_idx)
        v_new[i, j] = B(i, j, σ[i, j], v, model) 
    end
    return v_new
end

"The policy operator for the subordinate model."
function T_σ(h::Vector, σ, model)
    w_n = length(h)
    h_new = similar(h)
    for i in 1:w_n
        h_new[i] = B(i, σ[i], h, model) 
    end
    return h_new
end

"Compute a greedy policy for the original model."
function get_greedy(v::Matrix, model)
    w_n, e_n = size(v)
    w_idx, e_idx = 1:w_n, 1:e_n
    σ = Matrix{Int32}(undef, w_n, e_n)
    for (i, j) in product(w_idx, e_idx)
        _, σ[i, j] = findmax(B(i, j, k, v, model) for k in w_idx)
    end
    return σ
end

"Compute a greedy policy for the subordinate model."
function get_greedy(h::Vector, model)
    w_n = length(h)
    σ = Array{Int32}(undef, w_n)
    for i in 1:w_n
        _, σ[i] = findmax(B(i, k, h, model) for k in 1:w_n)
    end
    return σ
end


"Approximate lifetime value of policy σ."
function get_value(v_init, σ, m, model)
    v = v_init
    for i in 1:m
        v = T_σ(v, σ, model)
    end
    return v
end

"Optimistic policy iteration routine."
function optimistic_policy_iteration(v_init, 
                                     model; 
                                     tolerance=1e-9, 
                                     max_iter=1_000,
                                     m=100)
    v = v_init
    error = tolerance + 1
    k = 1
    while error > tolerance && k < max_iter
        last_v = v
        σ = get_greedy(v, model)
        v = get_value(v, σ, m, model)
        error = maximum(abs.(v - last_v))
        println("Completed iteration $k with error $error.")
        k += 1
    end
    return v, get_greedy(v, model)
end

