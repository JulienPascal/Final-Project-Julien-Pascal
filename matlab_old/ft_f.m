function [ft] = ft_f(St_m,ut_m,ut)

% St_m a vector of dimension M times 1 
global lm lambda0 M

ft = 0;

for i=1:M
    
    % calculate the sum
    if St_m(i,1)> 0
    ft = ft + ut_m(i,1)*lm(i,1);
    end
    
end

%multiply by lamba0 and divide by the overall level of unemployment ut:
ft = lambda0*ft/ut;


end
