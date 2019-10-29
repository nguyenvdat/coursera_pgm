%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P.edges = C.edges;
M = length(C.factorList)
% Assign factor to clique
factorAssignment = zeros(M, N);
var2Card = zeros(100,1);
for i = 1:M
    f = C.factorList(i);
    for j = 1:length(f.var)
        var2Card(f.var(j)) = f.card(j);
    end
    for j = 1:N
        if all(ismember(f.var, C.nodes{j}))
            P.cliqueList(j) = FactorProduct(P.cliqueList(j), f);
            factorAssignment(i,j) = 1;
            break
        end
    end
end
% Clique with no factor
for i =1:N
    if sum(factorAssignment(:, i)) == 0
        P.cliqueList(i).var = C.nodes{i};
        card = [];
        for j = 1:length(P.cliqueList(i).var)
            card = [card var2Card(P.cliqueList(i).var(j))];
        end
        P.cliqueList(i).card = card;
        P.cliqueList(i).val = ones(1, prod(P.cliqueList(i).card));
    end
end
% Clique with not enough var
for i =1:N
    if sum(factorAssignment(:, i)) == 0
        continue 
    end
    nVars = [];
    idx = find(factorAssignment(:, i));
    for j = 1:length(idx)
        nVars = union(nVars, C.factorList(idx(j)).var);
    end
    newVars = setdiff(C.nodes{i}, nVars);
    if ~isempty(newVars)
        card = [];
        for j = 1:length(newVars)
            card = [card var2Card(newVars(j))];
        end
        P.cliqueList(i).card = [card P.cliqueList(i).card];
        P.cliqueList(i).var = [newVars P.cliqueList(i).var];
        P.cliqueList(i).val = repmat(P.cliqueList(i).val, prod(card),1);
    end
end
% Sort the factor
for j = 1:N
    oldCard = P.cliqueList(j).card;
    [var, idx] = sort(P.cliqueList(j).var);
    P.cliqueList(j).var = var;
    P.cliqueList(j).card = P.cliqueList(j).card(idx);
    assignment = IndexToAssignment(1:prod(P.cliqueList(j).card), P.cliqueList(j).card);
    [v, back_idx] = sort(idx);
    assignment = assignment(:, back_idx);
    index = AssignmentToIndex(assignment, oldCard);
    P.cliqueList(j).val = P.cliqueList(j).val(index);
end
end

