
using JLD;
using CSV;
using DataFrames;
using Random;
using Base.Threads;
include("../src/perceptrons.jl")

function get_success_rate(all_accuracy_mat)
	sum(all_accuracy_mat[:, end] .> (1. - 1. / size(all_accuracy_mat, 2))) / size(all_accuracy_mat, 1)
end

num_trials = 1
max_epochs = 20000
all_records = []
# for kappa in [0.]
for kappa in [0., 0.1, 0.2, 0.5]
    if kappa == 0.
        # p_list = [1.5, 1.6, 1.7, 1.75]
        p_list = [1.8, 1.9, 1.92, 1.95]
    elseif kappa == 0.1
        # p_list = [1.3, 1.35, 1.4, 1.45]
        p_list = [1.5, 1.55, 1.6, 1.65]
    elseif kappa == 0.2
        p_list = [1.1, 1.15, 1.2, 1.25]
        # p_list = [1.25, 1.28, 1.3, 1.32]
    elseif kappa == 0.5
        p_list = [0.7, 0.74, 0.78, 0.8]
        # p_list = [0.88, 0.9, 0.92, 0.95]
    end
    lk = ReentrantLock()
    Threads.@threads for p in p_list
    # for p in p_list
        for n_features in [200]
        # for n_features in [1000]
            n_samples = Int(round(p * n_features))
            println(n_samples, n_features)
            for _ in 1:num_trials
                X, y = generate_linearly_separable_data(n_samples, n_features)

                perceptron = init_perceptron(n_features)	
                normal_query_count, normal_update_count = train_perceptron_stat_steps(
                    perceptron, X, y, epochs = max_epochs, kappa = kappa,
                    learning_rate = 1.0/n_features,
                    update_func = update_normal,
                )
                margin = get_margin(perceptron, X, y)
                success_rate_normal = length(margin[margin .> kappa]) / n_samples

                perceptron = init_perceptron(n_features)
                random1_query_count, random1_update_count = train_perceptron_stat_steps(
                    perceptron, X, y, epochs = max_epochs, kappa = kappa,
                    learning_rate = 1.0/n_features,
                    update_func = update_random,
                )
                perceptron = init_perceptron(n_features)
                random2_query_count, random2_update_count = train_perceptron_stat_steps(
                    perceptron, X, y, epochs = max_epochs, kappa = kappa,
                    learning_rate = 1.0/n_features,
                    update_func = update_random,
                )
                perceptron = init_perceptron(n_features)
                random3_query_count, random3_update_count = train_perceptron_stat_steps(
                    perceptron, X, y, epochs = max_epochs, kappa = kappa,
                    learning_rate = 1.0/n_features,
                    update_func = update_random,
                )

                perceptron = init_perceptron(n_features)
                perceptron.stored_variable = ones(n_samples)
                unsign_query_count, unsign_update_count = train_perceptron_stat_steps(
                    perceptron, X, y, epochs = max_epochs, kappa = kappa,
                    learning_rate = 1.0/n_features,
                    update_func = update_unsign,
                )
                margin = get_margin(perceptron, X, y)
                success_rate_unsign = length(margin[margin .> kappa]) / n_samples

                println(normal_update_count, " ", random1_update_count, " ", random2_update_count," ", random3_update_count, " ", unsign_update_count)
                lock(lk) do
                    push!(all_records, Dict(
                        "n_samples" => n_samples,
                        "n_features" => n_features,
                        "kappa" => kappa,
                        "normal_update_count" => normal_update_count,
                        "random1_update_count" => random1_update_count,
                        "random2_update_count" => random2_update_count,
                        "random3_update_count" => random3_update_count,
                        "unsign_update_count" => unsign_update_count,
                        "success_rate_normal" => success_rate_normal,
                        "success_rate_unsign" => success_rate_unsign,
                    ))
                end
            end
            # println(get_success_rate(all_accuracy_mat))
            # save("data/kappa_$(kappa)/data_$(n_samples)_$(n_features).jld", "data", all_accuracy_mat)
        end
    end
end

df = DataFrame(all_records)
CSV.write("a.csv", df)