using Random
using CairoMakie

include("mlp.jl")
include("train.jl")

function finite_difference_check(params, x, y, eps=1e-5)
    # Compute analytic gradients
    y_hat, cache = forward(x, params)
    _, d_out = mse_loss(y_hat, y)
    grads = backward(d_out, cache, params)
    
    max_relative_error = 0.0

    for l in 1:length(params.weights)
        # Check weight gradients
        for i in 1:size(params.weights[l], 1)
            for j in 1:size(params.weights[l], 2)
                orig = params.weights[l][i, j]

                params.weights[l][i, j] = orig + eps
                loss_plus, _ = mse_loss(forward(x, params)[1], y)

                params.weights[l][i, j] = orig - eps
                loss_minus, _ = mse_loss(forward(x, params)[1], y)

                params.weights[l][i, j] = orig

                numerical  = (loss_plus - loss_minus) / (2eps)
                analytic   = grads.weights[l][i, j]
                rel_error  = abs(analytic - numerical) / (abs(numerical) + 1e-8)
                max_relative_error = max(max_relative_error, rel_error)
            end
        end

        # Check bias gradients
        for i in 1:length(params.biases[l])
            orig = params.biases[l][i]

            params.biases[l][i] = orig + eps
            loss_plus, _ = mse_loss(forward(x, params)[1], y)

            params.biases[l][i] = orig - eps
            loss_minus, _ = mse_loss(forward(x, params)[1], y)

            params.biases[l][i] = orig

            numerical  = (loss_plus - loss_minus) / (2eps)
            analytic   = grads.biases[l][i]
            rel_error  = abs(analytic - numerical) / (abs(numerical) + 1e-8)
            max_relative_error = max(max_relative_error, rel_error)
        end
    end

    println("Max relative error: $max_relative_error")
    max_relative_error > 1e-3 && error("Gradient check failed — bug in backward pass.")
    println("Gradient check passed.")
    return max_relative_error
end

function test_sin_regression()
    rng = MersenneTwister(42)
    raw_x = rand(rng, 100) .* 2π .- π
    X = [[x] for x in raw_x]
    Y = sin.(raw_x)

    params = init_params(rng, [1, 64, 64, 1], Float64)
    losses = train!(params, X, Y, 1e-3, 5000)

    println("Final loss: $(losses[end])")
    losses[end] < 1e-3 || @warn "Loss did not converge — check init and Adam."

    # Evaluate on fine grid
    grid = collect(range(-π, π, length=500))
    predicted = [forward([t], params)[1] for t in grid]

    fig = Figure(size = (800, 600))
    ax  = Axis(fig[1, 1], title="Sin Regression", xlabel="x", ylabel="y")
    lines!(ax, grid, sin.(grid),  color=:blue, linewidth=2.5, label="True sin(x)")
    lines!(ax, grid, predicted,   color=:red,  linewidth=2.5, linestyle=:dash, label="NN Prediction")
    axislegend(ax, position=:rt)
    save("sin_regression_results.png", fig)
    println("Plot saved.")
    return fig
end

function run_tests()
    rng = MersenneTwister(123)
    params = init_params(rng, [1, 4, 4, 1], Float64)

    println("Gradient Check")
    finite_difference_check(params, [0.5], sin(0.5))

    println("Sin Regression")
    test_sin_regression()
end

run_tests()