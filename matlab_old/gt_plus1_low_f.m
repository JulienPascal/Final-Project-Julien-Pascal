function [gt_plus1] = gt_plus1_low_f(ut_m,gt_wi_low,gt_wi_high,i)

%function that calculates the measure of workers at low wages
global lm lambda0 delta lambda1 M W_low W_high Si_m

gt_plus1 = zeros(M,1);

for m =1:M
    
    %calculate the sum in the last paranthesis:
    if Si_m(m,i)> 0
        
    sum = 0;
    if W_low(i,i,m) < 0
        sum = sum + gt_wi_low(m,1);
    end
    
    if W_high(i,i,m) < 0
        sum = sum + gt_wi_high(m,1);
    end
    
    
    gt_plus1(m,1) = lambda0*ut_m(m,1)*lm(m,1) + (1 - delta)*(1 - lambda1)*(gt_wi_low(m,1) + sum);
    end
    
end

end
