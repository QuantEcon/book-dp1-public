function create_laborer_model(; α=0.3, β=0.2)
    return (; α, β)
end

function laborer_update(x, model)  # update X from t to t+1
    (; α, β) = model
    if x == 1    
        x′ = rand() < α ? 2 : 1
    else 
        x′ = rand() < β ? 1 : 2
    end
    return x′
end

function sim_chain(k, p, model)
    X = Array{Int32}(undef, k)
    X[1] = rand() < p ? 1 : 2
    for t in 1:(k-1)
        X[t+1] = laborer_update(X[t], model)
    end
    return X
end

function test_convergence(; k=10_000_000, p=0.5)
    model = create_laborer_model()
    (; α, β) = model
    ψ_star = (1/(α + β)) * [β α]

    X = sim_chain(k, p, model)
    ψ_e = (1/k) * [sum(X .== 1)  sum(X .== 2)]
    error = maximum(abs.(ψ_star - ψ_e))
    approx_equal = isapprox(ψ_star, ψ_e, rtol=0.01)
    println("Sup norm deviation is $error")
    println("Approximate equality is $approx_equal")
 end

