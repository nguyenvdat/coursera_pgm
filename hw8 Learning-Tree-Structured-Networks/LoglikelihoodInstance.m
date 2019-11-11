function ll = LoglikelihoodInstance(P, G, D)
    K = 2;
    ll = 0;
    N = size(D, 1);
    ll = zeros(K, 1);
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
        ll(c) = llc;
        % ll = ll + exp(llc);
    end
    % ll = log(ll);
end