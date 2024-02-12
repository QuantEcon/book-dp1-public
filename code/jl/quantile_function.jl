import Distributions.quantile, Distributions.DiscreteNonParametric

"Compute the τ-th quantile of v(X) when X ∼ ϕ and v = sort(v)."
function quantile(τ, v, ϕ)
    for (i, v_value) in enumerate(v)
        p = sum(ϕ[1:i])  # sum all ϕ[j] s.t. v[j] ≤ v_value
        if p ≥ τ         # exit and return v_value if prob ≥ τ
            return v_value
        end
    end

end

"For each i, compute the τ-th quantile of v(Y) when Y ∼ P(i, ⋅)"
function R(τ, v, P)
    return [quantile(τ, v, P[i, :]) for i in eachindex(v)]
end


function quantile_test(τ)
    ϕ = [0.1, 0.2, 0.7]
    v = [10, 20, 30]

    d = DiscreteNonParametric(v, ϕ)
    return quantile(τ, v, ϕ), quantile(d, τ)
end
    


