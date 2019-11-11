function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    D_c = dataset(labels(:, k) == 1, :, :);
    [A _] = LearnGraphStructure(D_c);
    G(:, :, k) = ConvertAtoG(A);
    %%%%%%%%%%%%%%%%%%%%%%%%%
end

% estimate parameters

P.c = zeros(1,K);
% compute P.c
% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

% These are dummy lines added so that submit.m will run even if you 
% have not started coding. Please delete them.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_parts = size(dataset, 2);
n_theta = size(dataset, 3) * (size(dataset, 3) + 1);
for c = 1:K
    correctG = G(:, :, c);
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
        if correctG(i, 1) == 0
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
            parent_node = correctG(i, 2);
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
end