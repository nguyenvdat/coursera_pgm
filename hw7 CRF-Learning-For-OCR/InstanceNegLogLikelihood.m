% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    nChar = length(y);
    P = cliqueTreeFromFeatures(featureSet, theta, nChar);
    [P, logZ] = CliqueTreeCalibrate(P, 0);
    weightedFeatureCount = computeWeightedFeatureCount(featureSet, y, theta, nChar);
    nll = logZ - weightedFeatureCount + modelParams.lambda / 2 * sum(theta .* theta);
    grad = gradModelFeatureCount(P, featureSet, grad, nChar);
    grad = gradDataFeatureCount(P, featureSet, y, grad, nChar);
    grad = gradRegularization(grad, theta, modelParams.lambda);
end

function grad = gradModelFeatureCount(P, featureSet, grad, nChar)
    % Calculate grad model expected feature count 
    K = 26;
    for i = 1:nChar
        if i < nChar
            % singleton feature
            B = FactorMarginalization(P.cliqueList(i), [i + 1]);
            for y_i = 1:K
                singletonIdx = getFeatureThetaIndex(featureSet, [i], [y_i], nChar);
                grad(singletonIdx) = grad(singletonIdx) + B.val(y_i);
            end
            % pairwise feature
            for y_i = 1:K
                for y_next = 1:K
                    pairwiseIdx = getFeatureThetaIndex(featureSet, [i, i + 1], [y_i, y_next], nChar);
                    grad(pairwiseIdx) = grad(pairwiseIdx) + P.cliqueList(i).val(AssignmentToIndex([y_i, y_next], [K, K]));
                end
            end
        else
            % singleton feature
            B = FactorMarginalization(P.cliqueList(i - 1), [i - 1]);
            for y_i = 1:K
                singletonIdx = getFeatureThetaIndex(featureSet, [i], [y_i], nChar);
                grad(singletonIdx) = grad(singletonIdx) + B.val(y_i);
            end
        end
    end
end

function grad = gradDataFeatureCount(P, featureSet, y, grad, nChar)
    % Calculate grad data feature count
    K = 26;
    for i = 1:nChar
        singletonIdx = getFeatureThetaIndex(featureSet, [i], [y(i)], nChar);
        grad(singletonIdx) = grad(singletonIdx) - 1;
        if i < nChar
            pairwiseIdx = getFeatureThetaIndex(featureSet, [i, i + 1], [y(i), y(i + 1)], nChar);
            grad(pairwiseIdx) = grad(pairwiseIdx) - 1;
        end
    end
end

function grad = gradRegularization(grad, theta, lambda)
    grad = grad + lambda * theta;
end

function weightedFeatureCount = computeWeightedFeatureCount(featureSet, y, theta, nChar)
    weightedFeatureCount = 0;
    for i = 1:nChar
        if i < nChar
            pairwiseTheta = theta(getFeatureThetaIndex(featureSet, [i, i+1], [y(i), y(i + 1)], nChar));
            weightedFeatureCount = weightedFeatureCount + pairwiseTheta;
        end
        singletonTheta = theta(getFeatureThetaIndex(featureSet, [i], [y(i)], nChar));
        weightedFeatureCount = weightedFeatureCount + sum(singletonTheta);
    end
end

function thetaIdx = getFeatureThetaIndex(featureSet, vars, assignment, nChar)
    % Return the index of theta for this feature
    K = 26;
    singletonCount = K * nChar * 33;
    features = featureSet.features;
    if length(vars) == 2
        i = assignment(1);
        j = assignment(2);
        x = vars(1);
        featureIdx = singletonCount + (i - 1) * K * (nChar - 1) + (j - 1) * (nChar - 1) + x;
        thetaIdx = features(featureIdx).paramIdx;
    else
        featureIdx = ([features(1:singletonCount).var] == vars) & ([features(1:singletonCount).assignment] == assignment);
        thetaIdx = [features.paramIdx](featureIdx);
    end
end

function P = cliqueTreeFromFeatures(featureSet, theta, nChar)
    K = 26;
    cliqueList = repmat(struct('var', [], 'card', [K, K], 'val', zeros(1, K * K)), nChar - 1, 1);
    edges = zeros(nChar - 1, nChar - 1);
    for i = 1:nChar - 1
        cliqueList(i).var = [i, i + 1];
        if i < nChar - 1
            edges(i, i + 1) = 1;
            edges(i + 1, i) = 1;
        end
    end
    for i = 1:length(featureSet.features)
        f = featureSet.features(i);
        vars = f.var;
        cliqueIndex = vars(1);
        if length(vars) == 1 % singleton feature
            val = f.assignment(1);
            if cliqueIndex == nChar
                cliqueIndex = cliqueIndex - 1;
                assignment = [(1:K)' ones(K, 1) * val];
            else
                assignment = [ones(K, 1) * val (1:K)'];
            end
        else
            assignment = f.assignment;
        end
        index = AssignmentToIndex(assignment, [K, K]);
        cliqueList(cliqueIndex).val(index(cliqueList(cliqueIndex).val(index) == 0)) = 1;
        cliqueList(cliqueIndex).val(index) = cliqueList(cliqueIndex).val(index) * exp(theta(f.paramIdx));
    end
    P.cliqueList = cliqueList;
    P.edges = edges;
end