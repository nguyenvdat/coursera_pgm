% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

    % INPUTS
    % poseData: N x 10 x 3 matrix, where N is number of poses;
    %   poseData(i,:,:) yields the 10x3 matrix for pose i.
    % G: graph parameterization as explained in PA8
    % InitialClassProb: N x K, initial allocation of the N poses to the K
    %   classes. InitialClassProb(i,j) is the probability that example i belongs
    %   to class j
    % maxIter: max number of iterations to run EM

    % OUTPUTS
    % P: structure holding the learned parameters as described in the PA
    % loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
    %   each iteration
    % ClassProb: N x K, conditional class probability of the N examples to the
    %   K classes in the final iteration. ClassProb(i,j) is the probability that
    %   example i belongs to class j

    % Initialize variables
    N = size(poseData, 1);
    K = size(InitialClassProb, 2);

    ClassProb = InitialClassProb;

    loglikelihood = zeros(maxIter, 1);

    P.c = [];
    P.clg.sigma_x = [];
    P.clg.sigma_y = [];
    P.clg.sigma_angle = [];
    n_parts = size(poseData, 2);
    n_theta = size(poseData, 3) * (size(poseData, 3) + 1);

    % EM algorithm
    for iter = 1:maxIter

        % M-STEP to estimate parameters for Gaussians
        %
        % Fill in P.c with the estimates for prior class probabilities
        % Fill in P.clg for each body part and each class
        % Make sure to choose the right parameterization based on G(i,1)
        %
        % Hint: This part should be similar to your work from PA8

        P.c = zeros(1, K);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for c = 1:K
            W = ClassProb(:, c);
            P.c(c) = sum(W) / N;

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

        % E-STEP to re-estimate ClassProb using the new parameters
        %
        % Update ClassProb with the new conditional class probabilities.
        % Recall that ClassProb(i,j) is the probability that example i belongs to
        % class j.
        %
        % You should compute everything in log space, and only convert to
        % probability space at the end.
        %
        % Tip: To make things faster, try to reduce the number of calls to
        % lognormpdf, and inline the function (i.e., copy the lognormpdf code
        % into this file)
        %
        % Hint: You should use the logsumexp() function here to do
        % probability normalization in log space to avoid numerical issues

        ClassProb = zeros(N, K);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        logpdf = inline('-log(sigma*sqrt(2*pi))-(x-mu).^2 ./ (2*sigma.^2)', 'x', 'mu', 'sigma');

        for c = 1:K

            for n = 1:N
                ll = log(P.c(c));

                for i = 1:n_parts
                    y = poseData(n, i, 1);
                    x = poseData(n, i, 2);
                    alpha = poseData(n, i, 3);
                    sigma_y = P.clg(i).sigma_y(c);
                    sigma_x = P.clg(i).sigma_x(c);
                    sigma_angle = P.clg(i).sigma_angle(c);

                    if G(i, 1) == 0
                        % only class as parent
                        ll = ll + logpdf(y, P.clg(i).mu_y(c), sigma_y);
                        ll = ll + logpdf(x, P.clg(i).mu_x(c), sigma_x);
                        ll = ll + logpdf(alpha, P.clg(i).mu_angle(c), sigma_angle);
                    else
                        parent_node = G(i, 2);
                        parent_val = squeeze(poseData(n, parent_node, :))';
                        theta = P.clg(i).theta(c, :);
                        ll = ll + logpdf(y, dot(theta(1:4), [1 parent_val]), sigma_y);
                        ll = ll + logpdf(x, dot(theta(5:8), [1 parent_val]), sigma_x);
                        ll = ll + logpdf(alpha, dot(theta(9:12), [1 parent_val]), sigma_angle);
                    end

                end

                ClassProb(n, c) = ll;
            end

        end

        loglikelihood(iter) = sum(logsumexp(ClassProb));
        ClassProb = exp(ClassProb);
        ClassProb = ClassProb ./ sum(ClassProb, 2);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Compute log likelihood of dataset for this iteration
        % Hint: You should use the logsumexp() function here
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % YOUR CODE HERE
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Print out loglikelihood
        disp(sprintf('EM iteration %d: log likelihood: %f', ...
            iter, loglikelihood(iter)));

        if exist('OCTAVE_VERSION')
            fflush(stdout);
        end

        % Check for overfitting: when loglikelihood decreases
        if iter > 1

            if loglikelihood(iter) < loglikelihood(iter - 1)
                break;
            end

        end

    end

    % Remove iterations if we exited early
    loglikelihood = loglikelihood(1:iter);
