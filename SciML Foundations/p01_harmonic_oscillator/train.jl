function mse_loss(y_hat, y)
    loss = (y_hat - y)^2
    d_out = 2(y_hat - y)
    
    return(loss, d_out)
end

mutable struct AdamState{T<:AbstractFloat}
    ms_w::Vector{Matrix{T}}
    vs_w::Vector{Matrix{T}}
    ms_b::Vector{Vector{T}}
    vs_b::Vector{Vector{T}}
    t::Int
end

function init_adam(params)
    ms_w = [zeros(size(w)) for w in params.weights]
    vs_w = [zeros(size(w)) for w in params.weights]
    
    ms_b = [zeros(size(b)) for b in params.biases]
    vs_b = [zeros(size(b)) for b in params.biases]

    return AdamState(ms_w, vs_w, ms_b, vs_b, 0)
end

function adam_update!(params, grads, adam_state, lr)
    β₁ = 0.9
    β₂ = 0.999
    ε = 1e-8

    adam_state.t += 1
    t = adam_state.t

    for l in 1:length(params.weights)
        adam_state.ms_w[l] .= β₁ .* adam_state.ms_w[l] .+ (1 - β₁) .* grads.weights[l]
        adam_state.vs_w[l] .= β₂ .* adam_state.vs_w[l] .+ (1 - β₂) .* (grads.weights[l] .^ 2)
        
        m̂_w = adam_state.ms_w[l] ./ (1 - β₁^t)
        v̂_w = adam_state.vs_w[l] ./ (1 - β₂^t)
        
        params.weights[l] .-= lr .* m̂_w ./ (sqrt.(v̂_w) .+ ε)

        adam_state.ms_b[l] .= β₁ .* adam_state.ms_b[l] .+ (1 - β₁) .* grads.biases[l]        
        adam_state.vs_b[l] .= β₂ .* adam_state.vs_b[l] .+ (1 - β₂) .* (grads.biases[l] .^ 2)
        
        m̂_b = adam_state.ms_b[l] ./ (1 - β₁^t)
        v̂_b = adam_state.vs_b[l] ./ (1 - β₂^t)
        
        params.biases[l] .-= lr .* m̂_b ./ (sqrt.(v̂_b) .+ ε)
    end
end

function train!(params, X, Y, lr, n_epochs)
    adam_state = init_adam(params)
    epoch_losses = Float64[]
    n_samples = length(X)

    for epoch in 1:n_epochs
        total_lose = 0.0

        for i in 1:n_samples
            x = X[i]
            y = Y[i]

            y_hat, cache = forward(x, params)
            loss, d_out = mse_loss(y_hat, y)
            total_lose += loss

            grads = backward(d_out, cache, params)
            adam_update!(params, grads, adam_state, lr)
        end
        mean_loss = total_lose / n_samples
        push!(epoch_losses, mean_loss)

        if epoch % 500 == 0
            println("Epoch $epoch / $n_epochs | Mean Loss: $(round(mean_loss, digits=6))")
        end
    end
    return epoch_losses
end