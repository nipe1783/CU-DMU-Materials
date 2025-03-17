using Plots
using Flux
using Random: randperm
using StaticArrays

# Generate data points

f(x) = (1 - x) * sin(20 * log(x + 0.2))

n = 300
dx = rand(Float32, n)
dy = convert.(Float32, f.(dx) + 0.05 * randn(n))

# Visualize the data
scatter(dx, dy, label="data", title="Sine Function Approximation")

# Define the neural network model
m = Chain(
    Dense(1 => 50, σ),
    Dense(50 => 50, σ),
    Dense(50 => 1)
)

function loss(x, y)
    pred = m(reshape(x, 1, :))
    return sum((pred .- y) .^ 2)
end

# Prepare data - avoid using SVectors directly in the training loop
data = [(reshape([dx[i]], 1, 1), [dy[i]]) for i in 1:length(dx)]
Loss = []
# Training loop
for i in 1:20000
    Flux.train!(loss, Flux.params(m), data, Flux.Optimise.Adam())
    if i % 500 == 0
        sorted_x = sort(dx)
        predictions = []
        for x in sorted_x
            pred = m(reshape([x], 1, 1))[1]
            push!(predictions, pred)
        end

        p = plot(sorted_x, x -> f(x), label="Exact Function", lw=2)
        scatter!(p, sorted_x, predictions, label="NN approximation", lw=2)
        # scatter!(p, dx, dy, label="data", alpha=0.6)

        epoch_loss = sum([loss(reshape([x], 1, 1), [f(x)]) for x in dx])
        println("Loss: ", epoch_loss)
        push!(Loss, epoch_loss)

        display(i)
        display(p)

        p2 = plot(Loss, label="Loss", title="Loss over epochs", xlabel="Epochs", ylabel="Loss")
        display(p2)
    end
end