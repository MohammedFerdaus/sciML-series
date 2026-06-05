using LinearAlgebra
using Statistics
using Random

struct MLPParams{T<:AbstractFloat}
    weights::Vector{Matrix{T}}
    biases::Vector{Vector{T}}
end

struct ForwardCache{T<:AbstractFloat}
    zs::Vector{Vector{T}}
    as_::Vector{Vector{T}}
    das_::Vector{Vector{T}}
end

function init_params(rng, layer_sizes, ::Type{T}=Float64) where {T<:AbstractFloat}
    weights = Vector{Matrix{T}}()
    biases  = Vector{Vector{T}}()
    
    for i in 1:length(layer_sizes)-1
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        
        a = sqrt(6.0 / (fan_in + fan_out))
    
        weight_matrix = rand(rng, fan_out, fan_in) * 2a .- a
        bias_vector = zeros(fan_out)
        
        push!(weights, weight_matrix)
        push!(biases, bias_vector)
    end
    
    return MLPParams(weights, biases)
end

function tanh_and_grad(z)
    a = tanh.(z)
    da = 1 .- a.^2
    return (a,da)
end

function forward(x, params)
    as_ = [x]
    zs = Vector{Vector{eltype(x)}}() 
    das_ = Vector{Vector{eltype(x)}}()

    L = length(params.weights)
    for i in 1:L-1
        W = params.weights[i]
        b = params.biases[i]
        a_prev = as_[end]

        z = W * a_prev .+ b
        a, da = tanh_and_grad(z)

        push!(zs, z)
        push!(das_, da)
        push!(as_, a)

    end
    W = params.weights[L]
    b = params.biases[L]
    a_prev = as_[end]
    z = W * a_prev .+ b
    
    push!(zs, z)
    push!(as_, z)
    push!(das_, zeros(eltype(z), length(z)))

    output_scalar = first(zs[end])
    
    return (output_scalar, ForwardCache(zs, as_, das_))
end

function backward(d_out, cache, params)
    dWs = Vector{Matrix{typeof(d_out)}}()
    dbs = Vector{Vector{typeof(d_out)}}()

    δ = [d_out]
    L = length(params.weights)

    for l in L:-1:1
        a_prev = cache.as_[l] 

        dW = δ * a_prev' 
        db = δ 

        push!(dWs, dW)
        push!(dbs, db)
        
        if l > 1
            da_prev = cache.das_[l-1]
            W = params.weights[l]
            δ_projected = W' * δ
            δ = δ_projected .* da_prev 
        end
    end
    reverse!(dWs)
    reverse!(dbs)

    return MLPParams(dWs, dbs)
end