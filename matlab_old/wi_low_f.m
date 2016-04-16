function [wi_m_low] = wi_low_f(i)

% Calculate the low wage

% i = index of the shock

global  discount Markov lambda1 M N Si_m W_low_star zi_m_grid

%initialization
wi_m_low = zeros(M,1);

for m = 1:M
    
    %calculate the sum
    sum = 0;
    
    for j=1:N
        if Si_m(m,j)> 0
        sum = sum + Markov(i,j)*((lambda1*Si_m(m,j)) + (1 - lambda1)*W_low_star(j,i,m));
        end
    end
    
    wi_m_low(m,1) = zi_m_grid(m,i) - discount*(sum);
end

end

