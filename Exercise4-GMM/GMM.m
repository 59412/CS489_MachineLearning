function [rho,mu,S] = GMM(X, K)
% X is data, K is initialization for model

[data_total,data_dim] = size(X);
max_round = 500;
TOL = 10^(-5);

% likelihood result container
log_like = zeros(1,max_round); % 1x500

% initialize rho mu and diagonal S
rho = rand([1,K]);

% normalize
rho = rho./sum(rho);   
mu = randn([K,data_dim]); 

% Only keep diagonal values, no use to declear 3 dim
S = rand([K,data_dim]);   

%%%%%%%%%%%%%%%%%%%%%%%%%%% iteration phase%%%%%%%%%%%%%%%%%%%%%
for iter = 1:max_round
    r = zeros(data_total,K); 
    log_total = sum(log(S),2); 
    
    S = S.^(-1);        % Get S's inverse, viable since diagonal
    
    for i = 1:data_total
        for k = 1:K
            % for log likelihood
            r(i,k) = log(rho(1,k)) - 0.5*log_total(k,1);
            r(i,k) = r(i,k) - 0.5*dot(S(k,:),(X(i,:)-mu(k,:)).^2); 
        end
    end
    
    % calculate negative log_likelihood
    log_sum_exp = log(sum(exp(r)));
    log_like(1,iter) = -1 * sum(log_sum_exp);  
    
    likeli_diff = log_like(1,iter)-log_like(1,iter-1);
    
    % break when smaller than predefined threashold 10^-5
    if (iter > 1) && (likeli_diff <= TOL * abs(log_like(1,iter)))
        break;
    end
    
    % normalize with exponential trick 
    K_max = max(r,[],2); 
    for i= 1:data_total
        r(i,:) = r(i,:) - K_max(i,1);
    end
    
    % retrieve actual value from log
    exp_r = exp(r);
    exp_r_total = sum(exp_r,2);
    
    for i = 1:data_total
        for k = 1:K
            % re-normalize
            r(i,:) = exp_r(i,:)./exp_r_total(i,1);
        end
    end
    
    % Use eps to avoid divide by 0 case
    safe_div = sum(r) + ones(1,K)*eps;
    
    % clear for result keeping 
    rho = safe_div./data_total;
    mu = zeros([K,data_dim]); 
    S = zeros([K,data_dim]);
    
    for k = 1:K
        for i = 1:data_total    
            mu(k,:) = mu(k,:) + r(i,k) * X(i,:);
            S(k,:) = S(k,:) + r(i,k) * (X(i,:).^2);
        end
        
        %calculate current mu S
        mu(k,:) = mu(k,:)./safe_div(1,k);
        S(k,:) = S(k,:) - mu(k,:).^2;
    end
    
    % Use eps to avoid divide by 0 case
    S = S + ones(K,data_dim) * eps;
    
end

end