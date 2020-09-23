function [w,b,mistake] = Q6P1(X, y, max_round)

[~ , sz2] = size(X); 
w = zeros(1,sz2);
b = 0;
mistake = zeros(1,max_round);

extra1 = rand(100,sz2);
extra2 = rand(100,1);

%for l = 1:1000
%    if extra2(l) >= 0.5
%        extra2(l) = 0;
%    else
%        extra2(l) = 1;
%    end
%end

%aa = ones(1000,sz2);
%bb = ones(1000,1);
%extra1 = aa- 2 * extra1;
%extra2 = bb- 2 * extra2;

X = [X;extra1];
y = [y;extra2];

[sz1 , sz2] = size(X); 

for t = 1:max_round
    for i = 1:sz1
        if (y(i) * (dot(w,X(i,:)) + b)) <= 0
            w = w + y(i) * X(i,1:sz2);
            b = b + y(i);
            mistake(t) = mistake(t) + 1;
        end
    end
end

p = linspace(1,max_round,max_round);
plot(p,mistake,'-o')