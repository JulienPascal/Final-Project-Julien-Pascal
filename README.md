# Final-Project-Julien-Pascal

This repository contains a replication of the Paper from Jean Marc Robin (2011) "On the Dynamics of Unemployment and Wage Distribution" built from scratch. Originally, the code was for Matlab.

The folder "Julia" contains a Julia version of it. To run the code, execute the file "main2.m" in the Julia folder. 

Update 1:
- I have included a version that runs with Julia
- Run the file "main2.jl" that is in the folder "Julia"
- The value function iteration for the wages is slow (slightly more than 20 mn)

Update 2, 21/02/2016:
- I have included two files that perform the estimation of the parameters by the Simulated Method of Moments: "estim_params.jl" and "objective_function.jl". To estimate the parameters, run "estim_params.jl"
