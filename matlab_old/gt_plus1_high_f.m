function [gt_plus1] = gt_plus1_high_f(ut_m,gt_wi_low,gt_wi_high,i)

%function that calculates the measure of workers at high wages
global lm delta lambda1 M W_low W_high Si_m

gt_plus1 = zeros(M,1);

for m =1:M
    
    
    if Si_m(m,i)> 0
        
    %calculate the sum in the last paranthesis:
    sum = 0;
    if W_low(i,i,m) > Si_m(m,i)
        sum = sum + gt_wi_low(m,1);
    end
    
    if W_high(i,i,m) > Si_m(m,i)
        sum = sum + gt_wi_high(m,1);
    end
    
    
    gt_plus1(m,1) = (1 - delta)*(lambda1*(1 - ut_m(m,1))*lm(m,1) + (1 - lambda1)*((gt_wi_high(m,1) + sum)));
    end
    
end

end
