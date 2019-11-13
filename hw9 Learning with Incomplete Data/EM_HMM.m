% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

    % INPUTS
    % actionData: structure holding the actions as described in the PA
    % poseData: N x 10 x 3 matrix, where N is number of poses in all actions
    % G: graph parameterization as explained in PA description
    % InitialClassProb: N x K matrix, initial allocation of the N poses to the K
    %   states. InitialClassProb(i,j) is the probability that example i belongs
    %   to state j.
    %   This is described in more detail in the PA.
    % InitialPairProb: V x K^2 matrix, where V is the total number of pose
    %   transitions in all HMM action models, and K is the number of states.
    %   This is described in more detail in the PA.
    % maxIter: max number of iterations to run EM

    % OUTPUTS
    % P: structure holding the learned parameters as described in the PA
    % loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
    %   each iteration
    % ClassProb: N x K matrix of the conditional class probability of the N examples to the
    %   K states in the final iteration. ClassProb(i,j) is the probability that
    %   example i belongs to state j. This is described in more detail in the PA.
    % PairProb: V x K^2 matrix, where V is the total number of pose transitions
    %   in all HMM action models, and K is the number of states. This is
    %   described in more detail in the PA.

    % Initialize variables
    N = size(poseData, 1);
    K = size(InitialClassProb, 2);
    L = size(actionData, 2); % number of actions
    V = size(InitialPairProb, 1);
    n_parts = size(poseData, 2);
    n_theta = size(poseData, 3) * (size(poseData, 3) + 1);

    ClassProb = log(InitialClassProb);
    PairProb = log(InitialPairProb);

    loglikelihood = zeros(maxIter, 1);

    P.c = [];
    P.clg.sigma_x = [];
    P.clg.sigma_y = [];
    P.clg.sigma_angle = [];

    % EM algorithm
    for iter = 1:maxIter

        % M-STEP to estimate parameters for Gaussians
        % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
        % Fill in P.clg for each body part and each class
        % Make sure to choose the right parameterization based on G(i,1)
        % Hint: This part should be similar to your work from PA8 and EM_cluster.m

        P.c = zeros(1, K);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        agg_class_prob = zeros(L, K);
        for l = 1:L
            agg_class_prob(l, :) = ClassProb(actionData(l).marg_ind(1), :);
        end
        P.c = logsumexp(agg_class_prob')';

        P.c = P.c - logsumexp(P.c);

        for c = 1:K
            W = exp(ClassProb(:, c));
    
            for i = 1:n_parts

                if c == 1 && iter == 1
                    P.clg(i).sigma_y = zeros(1, K);
                    P.clg(i).sigma_x = zeros(1, K);
                    P.clg(i).sigma_angle = zeros(1, K);
                end

                D_y = squeeze(poseData(:, i, 1)); % n x 1
                D_x = squeeze(poseData(:, i, 2));
                D_angle = squeeze(poseData(:, i, 3));

                if G(i, 1) == 0

                    if c == 1 && iter == 0
                        P.clg(i).mu_y = zeros(1, K);
                        P.clg(i).mu_x = zeros(1, K);
                        P.clg(i).mu_angle = zeros(1, K);
                    end

                    % only class as parent
                    [P.clg(i).mu_y(c), P.clg(i).sigma_y(c)] = FitG(D_y, W);
                    [P.clg(i).mu_x(c), P.clg(i).sigma_x(c)] = FitG(D_x, W);
                    [P.clg(i).mu_angle(c), P.clg(i).sigma_angle(c)] = FitG(D_angle, W);
                else

                    if c == 1 && iter == 1
                        P.clg(i).theta = zeros(K, n_theta);
                    end

                    parent_node = G(i, 2);
                    D_p = squeeze(poseData(:, parent_node, :));
                    [Beta_y, sigma_y] = FitLG(D_y, D_p, W);
                    [Beta_x, sigma_x] = FitLG(D_x, D_p, W);
                    [Beta_angle, sigma_angle] = FitLG(D_angle, D_p, W);
                    P.clg(i).sigma_y(c) = sigma_y;
                    P.clg(i).sigma_x(c) = sigma_x;
                    P.clg(i).sigma_angle(c) = sigma_angle;
                    P.clg(i).theta(c, :) = [[Beta_y(end); Beta_y(1:end - 1)]' [Beta_x(end); Beta_x(1:end - 1)]' [Beta_angle(end); Beta_angle(1:end - 1)]'];
                end

            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % M-STEP to estimate parameters for transition matrix
        % Fill in P.transMatrix, the transition matrix for states
        % P.transMatrix(i,j) is the probability of transitioning from state i to state j
        P.transMatrix = zeros(K, K); % [i, j] conditional P(j | i)

        % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
        P.transMatrix = P.transMatrix + size(PairProb, 1) * .05;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        cond_pair_prob = logsumexp(PairProb'); % K*K x 1
        sum(exp(cond_pair_prob))
        cond_pair_prob = logsumexp([cond_pair_prob log(size(PairProb, 1) * .05) * ones(K*K, 1)]); % K*K x 1
        cond_pair_prob = reshape(cond_pair_prob, K, K); % K x K
        P.transMatrix = cond_pair_prob;
        P.transMatrix = P.transMatrix - logsumexp(P.transMatrix);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each
        % of the poses in all actions = log( P(Pose | State) )
        % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m

        logEmissionProb = zeros(N, K);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for n = 1:N
            for c = 1:K
                llc = 0;
                for i = 1:n_parts
                    y = squeeze(poseData(n, i, 1));
                    x = squeeze(poseData(n, i, 2));
                    alpha = squeeze(poseData(n, i, 3));
                    sigma_y = P.clg(i).sigma_y(c);
                    sigma_x = P.clg(i).sigma_x(c);
                    sigma_angle = P.clg(i).sigma_angle(c);

                    if G(i, 1) == 0
                        % only class as parent
                        llc = llc + lognormpdf(y, P.clg(i).mu_y(c), sigma_y);
                        llc = llc + lognormpdf(x, P.clg(i).mu_x(c), sigma_x);
                        llc = llc + lognormpdf(alpha, P.clg(i).mu_angle(c), sigma_angle);
                    else
                        parent_node = G(i, 2);
                        parent_val = squeeze(poseData(n, parent_node, :))';
                        theta = P.clg(i).theta(c, :);
                        llc = llc + lognormpdf(y, dot(theta(1:4), [1 parent_val]), sigma_y);
                        llc = llc + lognormpdf(x, dot(theta(5:8), [1 parent_val]), sigma_x);
                        llc = llc + lognormpdf(alpha, dot(theta(9:12), [1 parent_val]), sigma_angle);
                    end

                end
                logEmissionProb(n, c) = llc;
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % E-STEP to compute expected sufficient statistics
        % ClassProb contains the conditional class probabilities for each pose in all actions
        % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
        % Also compute log likelihood of dataset for this iteration
        % You should do inference and compute everything in log space, only converting to probability space at the end
        % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues

        ClassProb = zeros(N, K);
        PairProb = zeros(V, K^2);
        loglikelihood(iter) = 0;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for l = 1:L
            action = actionData(l);
            % add factors
            action_length = length(action.marg_ind);
            F = repmat(struct('var', 0, 'card', 0, 'val', []), 2 * action_length, 1);

            for t = 1:action_length
                pose_idx = action.marg_ind(t);
                % initial state prob
                if t == 1
                    F(1) = struct('var', [1], 'card', [K], 'val', P.c);
                end
                % emission prob
                F(2 * t) = struct('var', [t], 'card', [K], 'val', logEmissionProb(pose_idx, :));
                % transition prob
                if t < action_length
                    F(2 * t + 1) = struct('var', [t, t + 1], 'card', [K, K], 'val', reshape(P.transMatrix, 1, K * K));
                end

            end
            [M, PCalibrated] = ComputeExactMarginalsHMM(F);
            % populate class prob
            for j = 1:length(M)
                ClassProb(action.marg_ind(M(j).var), :) = M(j).val;
            end
            % populate pair prob
            for j = 1:length(PCalibrated.cliqueList)
                PairProb(action.pair_ind(j), :) = PCalibrated.cliqueList(j).val - logsumexp(PCalibrated.cliqueList(j).val);
            end
            loglikelihood(iter) = loglikelihood(iter) + logsumexp(PCalibrated.cliqueList(1).val);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Print out loglikelihood
        disp(sprintf('EM iteration %d: log likelihood: %f', ...
            iter, loglikelihood(iter)));

        if exist('OCTAVE_VERSION')
            fflush(stdout);
        end

        % Check for overfitting by decreasing loglikelihood
        if iter > 1

            if loglikelihood(iter) < loglikelihood(iter - 1)
                break;
            end

        end

    end
    ClassProb = exp(ClassProb);
    PairProb = exp(PairProb);
    P.c = exp(P.c);
    P.transMatrix = exp(P.transMatrix);
    % Remove iterations if we exited early
    loglikelihood = loglikelihood(1:iter);
