function [ut_plus1_m] = ut_plus1_f(St_m,ut_m)

%Calculates the next period's unemployment rate
global delta lambda0 M

%initialization:
ut_plus1_m = zeros(M,1);

for i=1:M
    
    % if the surplus is negative, unemployment is 1:
    if St_m(i,1) <= 0
    ut_plus1_m(i,1) = 1;
    else
    ut_plus1_m(i,1) = ut_m(i,1) + delta*(1 - ut_m(i,1)) - lambda0*ut_m(i,1);
    end
    
end

end

