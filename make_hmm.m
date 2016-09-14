function [ bnet ] = make_hmm( init, emit_mean, emit_cov, trans )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

%generic hmm
intra = [0 1; 0 0];
inter = [1 0; 0 0];
hstates = size(init,2); % discrete states
ostates = 1; % continuous
node_sizes = [hstates ostates];
observed_nodes = [2]; % per slice. all others are assumed hidden
discrete_nodes = [1]; % per slice. all others are assumed continuous
eclass1 = [1 2];
eclass2 = [3 2];
eclass = [eclass1 eclass2];
bnet = mk_dbn(intra, inter, node_sizes, ...
'discrete', discrete_nodes, 'observed', observed_nodes, ... 
'eclass1', eclass1, 'eclass2', eclass2);

%model params
bnet.CPD{1} = tabular_CPD(bnet, 1, 'CPT', init);
bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', emit_mean, 'cov', emit_cov);
bnet.CPD{3} = tabular_CPD(bnet, 3, 'CPT', trans);

end

