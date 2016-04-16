function [qt] = qt_f(St_m,ut_m,ut)

%function that calculates the job-to-job mobility
global tau lambda1 delta lm M 

qt =0;

for i=1:M
    
    %calculate the sum
    if St_m(i,1)> 0
    qt = qt + (1 - ut_m(i,1))*lm(i,1);
    end
    
end

%finish the calculation:
qt = tau*lambda1*(1 - delta)*qt/(1 - ut);
end
