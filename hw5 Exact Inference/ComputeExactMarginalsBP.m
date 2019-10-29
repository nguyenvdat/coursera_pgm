%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
nVar = length(unique([F.var]));
M = repmat(struct('var', [], 'card', [], 'val', []), nVar, 1);
P = CreateCliqueTree(F, E);
P = CliqueTreeCalibrate(P, isMax);
N = length(P.cliqueList);
marginalizedCount = 0;
isDone = false;
for i = 1:N
    if isDone
        break
    end
    for j = 1:length(P.cliqueList(i).var)
        if isempty(M(P.cliqueList(i).var(j)).var)
            M(P.cliqueList(i).var(j)) = FactorMarginalization(P.cliqueList(i), setdiff(P.cliqueList(i).var, P.cliqueList(i).var(j)));
            M(P.cliqueList(i).var(j)).val = M(P.cliqueList(i).var(j)).val / sum(M(P.cliqueList(i).var(j)).val);
            marginalizedCount = marginalizedCount + 1;
        end
        if marginalizedCount == nVar
            isDone = true;
            break
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
