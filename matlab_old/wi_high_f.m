function [wi_m_high] = wi_high_f(i)

% calculates the high wages
global  discount Markov lambda1 M N Si_m W_high_star zi_m_grid

%initialization
wi_m_high = zeros(M,1);

for m = 1:M
    
    %calculate the sum
    sum = 0;
    
    for j=1:N
        if Si_m(m,j)> 0
        sum = sum + Markov(i,j)*((lambda1*Si_m(m,j)) + (1 - lambda1)*W_high_star(j,i,m));
        end
    end
    
    wi_m_high(m,1) = Si_m(m,i) + zi_m_grid(m,i) - discount*(sum);
end

end
