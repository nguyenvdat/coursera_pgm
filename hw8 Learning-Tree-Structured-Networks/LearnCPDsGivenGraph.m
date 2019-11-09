function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.

P.clg.sigma_x = 0;
P.clg.sigma_y = 0;
P.clg.sigma_angle = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_parts = size(dataset, 2);
n_theta = size(dataset, 3) * (size(dataset, 3) + 1);
for c = 1:K
    D_c = dataset(labels(:, c) == 1, :, :);
    P.c(c) = length(D_c) / N;
    for i = 1:n_parts
        if c == 1
            P.clg(i).sigma_y = zeros(1,K);
            P.clg(i).sigma_x = zeros(1,K);
            P.clg(i).sigma_angle = zeros(1,K);
        end
        D_y = squeeze(D_c(:, i, 1)); % n x 1
        D_x = squeeze(D_c(:, i, 2));
        D_angle = squeeze(D_c(:, i, 3));
        if G(i, 1) == 0
            if c == 1
                P.clg(i).mu_y = zeros(1,K);
                P.clg(i).mu_x = zeros(1,K);
                P.clg(i).mu_angle = zeros(1,K);
            end
            % only class as parent
            [P.clg(i).mu_y(c), P.clg(i).sigma_y(c)] = FitGaussianParameters(D_y);
            [P.clg(i).mu_x(c), P.clg(i).sigma_x(c)] = FitGaussianParameters(D_x);
            [P.clg(i).mu_angle(c), P.clg(i).sigma_angle(c)] = FitGaussianParameters(D_angle);
        else
            if c == 1
                P.clg(i).theta = zeros(K, n_theta);
            end
            parent_node = G(i, 2);
            D_p = squeeze(D_c(:, parent_node, :));
            [Beta_y, sigma_y] = FitLinearGaussianParameters(D_y, D_p);
            [Beta_x, sigma_x] = FitLinearGaussianParameters(D_x, D_p);
            [Beta_angle, sigma_angle] = FitLinearGaussianParameters(D_angle, D_p);
            P.clg(i).sigma_y(c) = sigma_y;
            P.clg(i).sigma_x(c) = sigma_x;
            P.clg(i).sigma_angle(c) = sigma_angle;
            P.clg(i).theta(c, :) = [[Beta_y(end); Beta_y(1:end-1)]' [Beta_x(end); Beta_x(1:end-1)]' [Beta_angle(end); Beta_angle(1:end-1)]'];
        end
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);

