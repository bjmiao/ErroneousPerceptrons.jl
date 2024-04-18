# Perceptron structure
mutable struct Perceptron
    weights::Vector{Float64}
    bias::Float64
    stored_variable::Vector{Float64}
end

# Initialize perceptron with random weights and bias
function init_perceptron(n_features::Int)
    weights = rand(n_features) .- 0.5
    # normalize perceptron weights
    normalize_factor = sqrt(n_features / sum(weights .^ 2))
    weights = weights * normalize_factor
    bias = 0
    # bias = rand() - 0.5
    stored_variable = zeros(n_features * 3)
    return Perceptron(weights, bias, stored_variable)
end

##
function generate_float_inputs(n_samples::Int, n_features::Int)
    X = rand(Float64, n_samples, n_features) .* 10.0 .- 5.0
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


function update_normal(perceptron, X, y, kappa, learning_rate)
    # We get margin for all the input patterns
    k = length(y)
    query_count = 0
    update_count = 0
    for i in 1:k
        margin = get_margin(perceptron, X, y)
        query_count += 1
        if margin[i] < kappa
            update_count += 1
            error = y[i] # we just need the error direction
            # perceptron.weights += learning_rate * (0.9 + rand() / 10.) * error .* X[i, :]
            perceptron.weights += learning_rate * error .* X[i, :]
            # perceptron.bias += learning_rate * error
        end
    end
    query_count, update_count
end

function update_random(perceptron, X, y, kappa, learning_rate)
    k = length(y)
    # update max N times so that we can match the max epoch
    query_count = 0
    update_count = 0
    for i in 1:k
        margin = get_margin(perceptron, X, y)
        query_count += 1
        need_update_elements = filter(x -> margin[x] < kappa, eachindex(margin) )
        # println(margin)
        # println(need_update_elements)

        if length(need_update_elements) == 0
            break
        end
        update_element = rand(need_update_elements)
        # update_element = need_update_elements[1]
        update_count += 1
        error = y[update_element] # we just need the error direction
        # perceptron.weights += learning_rate * (0.9 + rand() / 10.) * error .* X[i, :]
        perceptron.weights += learning_rate * error .* X[update_element, :]
    end
    query_count, update_count
end

function update_unsign(perceptron, X, y, kappa, learning_rate)
    k = length(y)
    # update max N times so that we can match the max epoch
    query_count = 0
    update_count = 0
    for i in 1:k
        margin = get_margin(perceptron, X, y)
        query_count += 1
        need_update_elements = filter(x -> margin[x] < kappa, eachindex(margin) )
        # println(margin)
        # println(need_update_elements)

        if length(need_update_elements) == 0
            break
        end
        update_element = rand(need_update_elements)
        # update_element = need_update_elements[1]
        update_count += 1
        error = abs(y[update_element]) # we just need the error direction
        # perceptron.weights += learning_rate * (0.9 + rand() / 10.) * error .* X[i, :]
        perceptron.weights += learning_rate * perceptron.stored_variable[update_element] * error .* X[update_element, :]
        
        margin = get_margin(perceptron, X, y)
        if margin[update_element] < kappa
            new_var = perceptron.stored_variable[update_element]
            if new_var < 0
                new_var = - new_var + 1
            else
                new_var = - new_var - 1
            end
            perceptron.stored_variable[update_element] = new_var
        else
            perceptron.stored_variable[update_element] = 1
        end

    end
    query_count, update_count
end

# Train the perceptron using the Perceptron learning rule
function train_perceptron_stat_steps(
    perceptron::Perceptron,
    X::Matrix{Float64}, y::Vector{Float64};
    epochs::Int=10, kappa::Float64=0.0, learning_rate::Float64=0.1,
    update_func = update_normal
)
    N = size(X, 2)
    query_count = 0
    update_count = 0
    for epoch in 1:epochs
        # normalize perceptron weights
        normalize_factor = sqrt(N / sum(perceptron.weights .^ 2))
        perceptron.weights = perceptron.weights * normalize_factor
        # perceptron.bias = perceptron.bias * normalize_factor

        new_query_count, new_update_count = update_func(perceptron, X, y, kappa, learning_rate)
        query_count += new_query_count
        update_count += new_update_count

        margin = get_margin(perceptron, X, y)
        # println(margin)
        accuracy = sum(margin .> kappa) / length(y)
        # early end
        if accuracy > (1.0 - 1.0 / length(margin))
        #     for _epoch in epoch+1:epochs
        #         push!(accuracy_all, accuracy)
        #     end
            break
        end
    end
    query_count, update_count
end
