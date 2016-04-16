function [st] = st_f(St_m,ut_m,ut)

%function that calculates the job destruction rate
global lm delta M

 st = 0;
for i=1:M
   
    %calculate the integral
    if St_m(i,1) <= 0
    st = st + (1 - ut_m(i,1))*lm(i,1);
    end
    
end

%finish the calculation:
st = delta + (1 - delta)*st/(1 - ut);
end

