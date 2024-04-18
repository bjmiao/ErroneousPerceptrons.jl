
using JLD;
include("../src/perceptrons.jl")

function get_success_rate(all_accuracy_mat)
	sum(all_accuracy_mat[:, end] .> (1. - 1. / size(all_accuracy_mat, 2))) / size(all_accuracy_mat, 1)
end

max_epochs = 200000
# for kappa in [0., 0.1, 0.2, 0.5]
for kappa in [0.]
    if kappa == 0.
        # p_list = [1.5, 1.7, 1.8, 1.9, 1.95, 1.98, 2]
        p_list = [1.0]
    elseif kappa == 0.1
        p_list = [1.3, 1.5, 1.7, 1.8, 1.85, 1.9]
    elseif kappa == 0.2
        p_list = [1.1, 1.3, 1.45, 1.48, 1.5, 1.6]
    elseif kappa == 0.5
        p_list = [1.0, 1.1, 1.2, 1.25, 1.3, 1.4]
    end
    # Threads.@threads for p in p_list
    for p in p_list
        for n_features in [100]
        # for n_features in [1000]
            n_samples = Int(p * n_features)
            println(n_samples, n_features)
            all_accuracy_mat = run_training_sessions(
                n_samples, n_features, 
                max_epochs = max_epochs, kappa = kappa, learning_rate = 1.0/n_features, n_trials=12
            )
            println(get_success_rate(all_accuracy_mat))
            save("data/kappa_$(kappa)/data_$(n_samples)_$(n_features).jld", "data", all_accuracy_mat)
        end
    end
end
