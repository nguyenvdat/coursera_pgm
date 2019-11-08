function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
for i = 1:N
    loglikelihood = loglikelihood + loglikelihoodInstance(P, G, squeeze(dataset(i, :, :)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function ll = loglikelihoodInstance(P, G, D)
    K = 2;
    ll = 0;
    N = size(D, 1);
    for c = 1:K
        if length(size(G)) == 3
            correctG = G(:,:,c);
        else
            correctG = G;
        end
        llc = log(P.c(c));
        for i = 1:N
            y = D(i, 1);
            x = D(i, 2);
            alpha = D(i, 3);
            sigma_y = P.clg(i).sigma_y(c);
            sigma_x = P.clg(i).sigma_x(c);
            sigma_angle = P.clg(i).sigma_angle(c);
            if correctG(i, 1) == 0
                % only class as parent
                llc = llc + lognormpdf(y, P.clg(i).mu_y(c), sigma_y);
                llc = llc + lognormpdf(x, P.clg(i).mu_x(c), sigma_x);
                llc = llc + lognormpdf(alpha, P.clg(i).mu_angle(c), sigma_angle);
            else
                parent_node = correctG(i, 2);
                parent_val = D(parent_node, :);
                theta = P.clg(i).theta(c, :);
                llc = llc + lognormpdf(y, dot(theta(1:4), [1 parent_val]), sigma_y);
                llc = llc + lognormpdf(x, dot(theta(5:8), [1 parent_val]), sigma_x);
                llc = llc + lognormpdf(alpha, dot(theta(9:12), [1 parent_val]), sigma_angle);
            end
        end
        ll = ll + exp(llc);
    end
    ll = log(ll);
end