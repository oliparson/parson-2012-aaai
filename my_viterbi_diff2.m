function [mpe, ll, max_prob ignored_obs] = my_viterbi_diff2(bnet, evidence2, ignore_obs, lik_thres)

evidence = evidence2(1:2,:);
aggregate_power = evidence2(3,:);
observed = ~isemptycell(evidence);

T = length(evidence);
mpe = evidence;
%bnet = bnet_from_engine(engine);
d_states = bnet.node_sizes(bnet.dnodes_slice);
max_prob = zeros(d_states,T);
edges = zeros(d_states,T);
ignored_obs = zeros(1,T);

initial = struct(bnet.CPD{1});
emission2 = struct(bnet.CPD{2});
transition = struct(bnet.CPD{3});
emission = struct(bnet.CPD{4});

max_lik = zeros(T,1);
% max product forward pass
for t=1:T,
    
    % prob of states at t-1
    if t>1
        chain = max_prob(:,t-1);
    end
    
    % prob of transitions
    trans = transition.CPT;
    
    % prob of diff emission
    emit = normpdf(evidence{2,t}, permute(emission.mean, [2,3,1]), sqrt(permute(emission.cov, [3,4,1,2])));
    
    % prob of absolute emission
    emit2 = normcdf(aggregate_power{t}, emission2.mean(:)', emission2.cov(:)');
    emit2 = emit2 / sum(emit2(:));
    
    % ignore observations of low probability
    threshold = lik_thres;
    if ignore_obs && sum(emit(:)) < threshold
        ignored_obs(t) = 1;
        
        if t==1
            max_prob(:,t) = log(initial.CPT);
        else
            % product
            product = repmat(chain,[1,d_states]) + log(trans) + log(repmat(emit2,[d_states,1]));
            % max
            [max_prob(:,t) edges(:,t)] = max(product);
        end
    else
        if t==1
            max_prob(:,t) = log(initial.CPT);
        else
            %normalise emission probabilites
            emit = emit / sum(emit(:));
            % product
            product = repmat(chain,[1,d_states]) + log(trans) + log(emit) + log(repmat(emit2,[d_states,1]));
            % max
            [max_prob(:,t) edges(:,t)] = max(product);
        end
    end
    
    %normalise
    %average_log_prob = sum(max_prob(:,t)) / length(max_prob(:,t));
    max_prob(:,t) = max_prob(:,t) - max(max_prob(:,t));
    
    a = max_prob(:,t);
    max_prob((isinf(a)),t) = min(a(~isinf(a))) - 100;
    
    if any(isinf(max_prob(:,t)))
        1
    end
    
    % if observed then prob = 1
    if observed(1,t)
        max_prob(:,t) = 0;
        max_prob(evidence{1,t},t) = 1;
    end
end

% figure(3);
% plot(max_lik);

% max_prob

% backward pass
[ll mpe{1,T}] = max(max_prob(:,T));
for t=T:-1:2,
    mpe{1,t-1} = edges(mpe{1,t},t);
end
