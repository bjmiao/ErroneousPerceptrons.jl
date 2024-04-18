using JLD;
include("../src/perceptrons.jl")

# Example usage
# n_samples = 800
n_features = 500
kappa = 0.1
lr = 1.0/n_features
max_epochs = 2e5

for n_samples in [700, 750, 780, 820, 850, 900]
    # for type in ["float", "binary"]
    for type in ["float"]
        println(type)
        if type == "binary"
            X, y = generate_linearly_separable_data(n_samples, n_features, "b-b")
        elseif type == "float"
            X, y = generate_linearly_separable_data(n_samples, n_features, "f-b")
        end
        println(X[1:10, 1])
        perceptron = init_perceptron(n_features)  # Perceptron with n input features
        margin_all = []
        for epochs in 1:max_epochs
            acc = train_perceptron(perceptron, X, y, epochs=1,
                kappa = kappa, learning_rate = lr)
            margin = get_margin(perceptron, X, y)
            push!(margin_all, margin)
            # scatter!(repeat([epochs], n_samples), get_margin(perceptron_hack, X, y), color="black", markersize=1, label = "")
        end
        margin_mat = hcat(margin_all...)'
        println(size(margin_mat))

        save("data/binary_float/$(type)/margin_$(kappa)_$(n_samples)_$(n_features).jld", "data", margin_mat)
    end
end