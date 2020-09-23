function [w,b,mistake] = Perceptron2(X, y, max_round)

[sz1 , sz2] = size(X); 
w = zeros(1,sz2);
b = 0;
mistake = zeros(1,max_round);

for t = 1:max_round
    for i = 1:sz1
        w = w + y(i) * X(i,1:sz2);
        b = b + y(i);
        if (y(i) * (dot(w,X(i,:)) + b)) <= 0
            mistake(t) = mistake(t) + 1;
        end
    end
end

p = linspace(1,max_round,max_round);
plot(p,mistake,'-o')

            
    