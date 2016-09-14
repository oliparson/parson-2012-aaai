function [bnet2, loglik] = learn_params_generic(bnet, observations)

%num_states = length(state_means);
%init = ones(1,num_states) / num_states;
%bnet = make_hmm(init, state_means, state_covs, trans_matrix);

engine = smoother_engine(jtree_2TBN_inf_engine(bnet));
ev(2,:) = num2cell([0; observations])';
%ev(2,1) = cell(1);
[bnet2, LLtrace] = my_learn_params_dbn_em(engine, {ev}, 'max_iter', 10, 'verbose', 0);
loglik = LLtrace(end);

% appliance_parameters.initial = struct(bnet2.CPD{1});
% appliance_parameters.initial.CPT = appliance_parameters.initial.CPT';
% appliance_parameters.emission = struct(bnet2.CPD{2});
% appliance_parameters.transition = struct(bnet2.CPD{3});

% print learnt params
% appliance_parameters.initial.CPT
% appliance_parameters.emission.mean
% appliance_parameters.emission.cov(:,:)
% appliance_parameters.transition.CPT
