%clear;
load('aaai.mat');

appliance_presence = [ %(appliance,house)
     1 1 0 0 1 1;
     1 1 1 0 0 0;
     1 0 1 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 1];

%appliance params
fridge_state_means = [2 180 160];
fridge_state_covs = [5 100 100];
fridge_trans = [0.95 0.03 0.02; 0 0 1; 0.2 0 0.8];

micro_state_means = [4 1700];
micro_state_covs = [100 1000];
micro_trans = [0.99 0.01; 1 0];

wd_state_means = [0 5000];
wd_state_covs = [100 5000];
wd_trans = [0.9 0.1; 0.5 0.5];

dw_state_means = [0 1400];
dw_state_covs = [100 1300];
dw_trans = [0.9999 0.0001; 0.3 0.7];

ac_state_means = [4 2300];
ac_state_covs = [100 2300];
ac_trans = [0.99 0.01; 0.9 0.1];

transition_matrix = {fridge_trans, micro_trans, wd_trans, dw_trans, dw_trans, ac_trans};
state_means = {fridge_state_means, micro_state_means, wd_state_means, dw_state_means, dw_trans, ac_state_means};
state_covs = {fridge_state_covs, micro_state_covs, wd_state_covs, dw_state_covs, dw_trans, ac_state_covs};
always_on = {0, 3, 0, 0, 0, 4};
window_length = {200, 6, 20, 30, 0, 20};
starting_point = {1, 7000, 2000, 1, 1, 5000};
training_length = {3000, 4000, 5000, 5000, 0, 2500};
num_of_windows = {3, 4, 3, 2, 0, 4};
lik_thres = {0.01, 0.00001, 0.001, 0.001, 0, 0.001};

training_type = 2; %1 = no training, 2 = agg training, 3 = submetered training

for house=1:6
    
    residue = loads{house}(:,1);
    inferred_power{house} = zeros(size(loads{house}(:,2:end)));
    for appliance=find(appliance_presence(:,house))'
        appliance_idx = appliance+1;

        % calculate diff model params
        init = ones(1, length(state_means{appliance})) / length(state_means{appliance});
        num_states = length(init);
        idx = repmat(1:num_states,num_states,1);
        emit_mean = state_means{appliance}(idx) - state_means{appliance}(idx');
        emit_cov = state_covs{appliance}(idx) + state_covs{appliance}(idx');

        % make difference hmm
        prior_dhmm_bnet = make_dhmm(init, ...
                        state_means{appliance}, ...
                        state_covs{appliance}, ...
                        emit_mean, ...
                        emit_cov, ...
                        transition_matrix{appliance});

        prior_hmm_bnet = make_hmm(init, ...
                        state_means{appliance}, ...
                        state_covs{appliance}, ...
                        transition_matrix{appliance});

        % REDD evidence
        observed_power = residue;
        diffs = [0 diff(observed_power)'];
        evidence = {};
        evidence(2,:) = num2cell(diffs);
        T = length(evidence);

        %create engine
        engine = smoother_engine(jtree_2TBN_inf_engine(prior_dhmm_bnet));

        %select training data
        if training_type == 1
            
        elseif training_type == 2
            if appliance == 6
                training_data = residue(7200:7450) - min(residue(7200:7450));
            else
                training_data = find_training_ranges_generic(...
                            residue(starting_point{appliance}:starting_point{appliance}+training_length{appliance}-1), ...
                            window_length{appliance}, ...
                            prior_dhmm_bnet, ...
                            num_of_windows{appliance});
            end
               
        elseif training_type == 3
            ha_idx = {};
            ha_idx{1}{1} = 250:700;
            ha_idx{2}{1} = 2600:2900;
            ha_idx{5}{1} = 730:1000;
            ha_idx{6}{1} = 400:750;
            
            ha_idx{1}{2} = 1:150;
            ha_idx{2}{2} = 9650:9900;
            ha_idx{3}{2} = 200:1200;
            
            ha_idx{1}{3} = 5940:6050;
            ha_idx{3}{3} = 2420:2470;
            
            ha_idx{6}{6} = 7200:7500;
            
            training_data = loads{house}(ha_idx{house}{appliance}, appliance_idx);
            
            training_data = awgn(training_data,25,'measured');
        end

        %train models
        if training_type == 1
            trained_hmm_bnet = prior_hmm_bnet;
            trained_dhmm_bnet = prior_dhmm_bnet;
            hmm_emissions = struct(trained_hmm_bnet.CPD{2});
        elseif training_type == 2 || training_type == 3
            [trained_dhmm_bnet, loglik] = learn_params_generic(prior_dhmm_bnet, diff(training_data));
            try
                [trained_hmm_bnet, hmm_loglik] = learn_params_generic(prior_hmm_bnet, training_data);
            catch err
                trained_hmm_bnet = prior_hmm_bnet;
            end
            hmm_emissions = struct(trained_hmm_bnet.CPD{2});
            trained_dhmm_bnet.CPD{1} = tabular_CPD(trained_dhmm_bnet, 1, 'CPT', ones(1, length(init)) / length(init));
            trained_dhmm_bnet.CPD{2} = trained_hmm_bnet.CPD{2};
            trained_dhmm_bnet.CPD{3} = prior_dhmm_bnet.CPD{3};
        end
        
        if training_type == 3
            trained_dhmm_bnet.CPD{1} = tabular_CPD(trained_dhmm_bnet, 1, 'CPT', ones(1, length(init)) / length(init));
            trained_dhmm_bnet.CPD{3} = prior_dhmm_bnet.CPD{3};
        end

        %infer power (viterbi)
        evidence(3,:) = num2cell(observed_power);
        [mpe, ll, max_prob ignored_obs] = my_viterbi_diff2(trained_dhmm_bnet, evidence, 1, lik_thres{appliance});
        
        %subract from aggregate
        residue = residue - inferred_power{house}(:,appliance);
        
        %calculate performance
        %error = abs(actual_power - inferred_power{house}(:,appliance));
        %norm_average_error = sum(error) / sum(actual_power)
        %rms_error = sqrt(mean(error.^2))
    end

    %norm_disag_error = sqrt(sum(error .^ 2) / sum(actual_power .^ 2))
    %mean_square_norm_error = sqrt((sum(norm_error .^ 2)) / T)

end


