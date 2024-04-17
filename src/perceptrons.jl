# Perceptron structure
mutable struct Perceptron
    weights::Vector{Float64}
    bias::Float64
end

# Initialize perceptron with random weights and bias
function init_perceptron(n_features::Int)
    weights = rand(n_features) .- 0.5
    # normalize perceptron weights
    normalize_factor = sqrt(n_features / sum(weights .^ 2))
    weights = weights * normalize_factor
    bias = 0
    # bias = rand() - 0.5
    return Perceptron(weights, bias)
end

##
function generate_float_inputs(n_samples::Int, n_features::Int)
    X = rand(Float64, n_samples, n_features) .* 4.0
    return X
end

# Generate random binary input vectors
function generate_binary_inputs(n_samples::Int, n_features::Int)
    X = abs.(rand(Int, n_samples, n_features) .% 2 ) .* 2.0 .- 1.0
    return X
end

function generate_binary_labels(n_samples::Int)
    return abs.(rand(Int, n_samples) .% 2 ) .* 2.0 .- 1.0
end

# Generate linearly separable data
function generate_linearly_separable_data(n_samples::Int, n_features::Int, mode::String = "b-b")
    if mode == "b-b" # binary input, binary labels
        X = generate_binary_inputs(n_samples, n_features)
        y = generate_binary_labels(n_samples)
    elseif mode == "f-b" # float input, binary labels
            X = generate_float_inputs(n_samples, n_features)
            y = generate_binary_labels(n_samples)
    end
        
    return X, y
end

# Activation function (unit step function)
activation(x) = x >= 0 ? 1.0 : -1.0

# Predict output for a single input
function predict_output(perceptron::Perceptron, inputs::Vector{Float64})
    activation(sum(perceptron.weights .* inputs) + perceptron.bias)
end

# Get the margin. If positive, means on the right direction; otherwise, on the wrong direction
function get_margin(perceptron::Perceptron, inputs::Matrix{Float64}, outputs::Vector{Float64})
    N = length(outputs)
    margin = outputs .* (inputs * perceptron.weights .+ perceptron.bias) .* (N ^ -0.5)
    margin
end

# Train the perceptron using the Perceptron learning rule
function train_perceptron(perceptron::Perceptron, X::Matrix{Float64}, y::Vector{Float64}; epochs::Int=10, kappa::Float64=0.0, learning_rate::Float64=0.1)
    accuracy_all = []
    N = size(X, 2)
    for epoch in 1:epochs
        # normalize perceptron weights
        normalize_factor = sqrt(N / sum(perceptron.weights .^ 2))
        perceptron.weights = perceptron.weights * normalize_factor
        # perceptron.bias = perceptron.bias * normalize_factor

        # We get margin for all the input patterns
        margin = get_margin(perceptron, X, y)
        # update
        for i in 1:length(margin)
            if margin[i] < kappa
                error = y[i] # we just need the error direction
                # perceptron.weights += learning_rate * (0.9 + rand() / 10.) * error .* X[i, :]
                perceptron.weights += learning_rate * error .* X[i, :]
                # perceptron.bias += learning_rate * error
            end
        end

        margin = get_margin(perceptron, X, y)
        accuracy = sum(margin .> kappa) / length(y)
        push!(accuracy_all, accuracy)
        # early end
        if accuracy > (1.0 - 1.0 / length(margin))
            for _epoch in epoch+1:epochs
                push!(accuracy_all, accuracy)
            end
            break
        end
    end
    accuracy_all
end


function run_training_sessions(n_samples, n_features;
    max_epochs::Int=10, kappa::Float64=0.0, learning_rate::Float64=0.1,
    n_trials::Int = 1, train_perceptron_func = train_perceptron
)
	# return a matrix with (ntrials, nepochs)
	all_accuracy_list = []
	for _ in 1:n_trials
		perceptron = init_perceptron(n_features)	
		X, y = generate_linearly_separable_data(n_samples, n_features)
		accuracy_list = train_perceptron_func(perceptron, X, y, epochs = max_epochs, kappa = kappa, learning_rate = learning_rate)
		# return accuracy - epoch array
		push!(all_accuracy_list, accuracy_list)
	end
	all_accuracy_mat = hcat(all_accuracy_list...)'
	return all_accuracy_mat
end