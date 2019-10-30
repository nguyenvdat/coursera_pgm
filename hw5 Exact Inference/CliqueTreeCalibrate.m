%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 if isMax
    for i = 1:N
        P.cliqueList(i).val = log(P.cliqueList(i).val);
    end
 end

while true
    [i, j] = GetNextCliques(P, MESSAGES);
    if i == 0 && j == 0
        break
    end
    neighborNodes = find(P.edges(:, i));
    MESSAGES(i, j) = P.cliqueList(i);
    for idx = 1:length(neighborNodes)
        if neighborNodes(idx) ~= j
            if isMax
                MESSAGES(i, j) = FactorSum(MESSAGES(i, j), MESSAGES(neighborNodes(idx), i));
            else
                MESSAGES(i, j) = FactorProduct(MESSAGES(i, j), MESSAGES(neighborNodes(idx), i));
            end
        end
    end
    marginNodes = setdiff(P.cliqueList(i).var, P.cliqueList(j).var);
    if isMax
        MESSAGES(i, j) = FactorMaxMarginalization(MESSAGES(i, j), marginNodes);
    else
        MESSAGES(i, j) = FactorMarginalization(MESSAGES(i, j), marginNodes);
        MESSAGES(i, j).val = MESSAGES(i, j).val / sum(MESSAGES(i, j).val);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N
    neighborNodes = find(P.edges(:, i));
    for j = 1:length(neighborNodes)
        if isMax
            P.cliqueList(i) = FactorSum(P.cliqueList(i), MESSAGES(neighborNodes(j), i));
        else
            P.cliqueList(i) = FactorProduct(P.cliqueList(i), MESSAGES(neighborNodes(j), i));
        end
    end
end
return
