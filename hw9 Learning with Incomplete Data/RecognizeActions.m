% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

    % INPUTS
    % datasetTrain: dataset for training models, see PA for details
    % datasetTest: dataset for testing models, see PA for details
    % G: graph parameterization as explained in PA decription
    % maxIter: max number of iterations to run for EM

    % OUTPUTS
    % accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
    % predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances

    % Train a model for each action
    % Note that all actions share the same graph parameterization and number of max iterations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    K = length(datasetTrain);
    P = repmat(struct('c', 0, 'clg', 0, 'transMatrix', 0), K, 1);
    for i = 1:K
        D = datasetTrain(i);
        [P(i) _ _ _] = EM_HMM(D.actionData, D.poseData, G, D.InitialClassProb, D.InitialPairProb, maxIter);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Classify each of the instances in datasetTrain
    % Compute and return the predicted labels and accuracy
    % Accuracy is defined as (#correctly classified examples / #total examples)
    % Note that all actions share the same graph parameterization

    accuracy = 0;
    predicted_labels = [];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % YOUR CODE HERE
    poseData = datasetTest.poseData;
    n_action = length(datasetTest.actionData);
    predicted_labels = zeros(n_action, 1);
    for l = 1:n_action
        action = datasetTest.actionData(l);
        ll_k = zeros(1, K); % log likelihood for each classifier class
        action_length = length(action.marg_ind);
        for k = 1:K % loop through all classifier class
            F = repmat(struct('var', 0, 'card', 0, 'val', []), 2 * action_length, 1);
            for t = 1:action_length
                pose_idx = action.marg_ind(t);
                % initial state prob
                if t == 1
                    F(1) = struct('var', [1], 'card', [K], 'val', P(k).c);
                end
                % emission prob
                F(2 * t) = struct('var', [t], 'card', [K], 'val', EmissionLogProb(P, k, pose_idx, poseData, G));
                % transition prob
                if t < action_length
                    F(2 * t + 1) = struct('var', [t, t + 1], 'card', [K, K], 'val', reshape(P(k).transMatrix, 1, K * K));
                end
            end
            [M, PCalibrated] = ComputeExactMarginalsHMM(F);
            ll_k(k) = logsumexp(PCalibrated.cliqueList(1).val);
        end
        [_, predicted_labels(l)] = max(ll_k);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    try
        accuracy = sum(predicted_labels == datasetTest.labels) / n_action;
    catch
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function ll = EmissionLogProb(P, k, n, poseData, G)
    n_parts = size(poseData, 2);
    K = length(P(k).c);
    ll = zeros(1, K);
    for c = 1:K
        llc = 0;
        for i = 1:n_parts
            y = squeeze(poseData(n, i, 1));
            x = squeeze(poseData(n, i, 2));
            alpha = squeeze(poseData(n, i, 3));
            sigma_y = P(k).clg(i).sigma_y(c);
            sigma_x = P(k).clg(i).sigma_x(c);
            sigma_angle = P(k).clg(i).sigma_angle(c);

            if G(i, 1) == 0
                % only class as parent
                llc = llc + lognormpdf(y, P(k).clg(i).mu_y(c), sigma_y);
                llc = llc + lognormpdf(x, P(k).clg(i).mu_x(c), sigma_x);
                llc = llc + lognormpdf(alpha, P(k).clg(i).mu_angle(c), sigma_angle);
            else
                parent_node = G(i, 2);
                parent_val = squeeze(poseData(n, parent_node, :))';
                theta = P(k).clg(i).theta(c, :);
                llc = llc + lognormpdf(y, dot(theta(1:4), [1 parent_val]), sigma_y);
                llc = llc + lognormpdf(x, dot(theta(5:8), [1 parent_val]), sigma_x);
                llc = llc + lognormpdf(alpha, dot(theta(9:12), [1 parent_val]), sigma_angle);
            end

        end
        ll(c) = llc;
    end
end