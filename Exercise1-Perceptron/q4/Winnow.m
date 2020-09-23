function [w,b,mistake] = Winnow(X, y, max_round)

[sz1 , sz2] = size(X); 
w = ones(1,sz2);
w = w/(sz2+1);

b = 1/(sz2+1);
[M,~] = max(X(:));
[N,~] = min(X(:));
max_value = max(abs(M),abs(N));
step_size = 1/max_value;
%X = X / max_value;
mistake = zeros(1,max_round);

for t = 1:max_round
    for i = 1:sz1
        if (y(i) * (dot(w,X(i,:)) + b)) <= 0
            w = w.*exp(step_size*y(i)*X(i,:));
            b = b * exp(step_size*y(i));
            temp_sum = 0;
            for j = 1:sz2
                temp_sum = temp_sum + w(j);
            end
            s = b + temp_sum;
            w = w / s;
            b = b / s;
            mistake(t) = mistake(t) + 1;
        end
    end
end

p = linspace(1,max_round,max_round);
plot(p,mistake,'-o')

            
    