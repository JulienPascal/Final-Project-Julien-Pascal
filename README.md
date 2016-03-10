# Computational Economics
## Final-Project-Julien-Pascal


This repository contains a replication of the Paper from Jean Marc Robin (2011) "On the Dynamics of Unemployment and Wage Distribution" Econometrica, 79: 1327–1355. doi: 10.3982/ECTA9070. http://onlinelibrary.wiley.com/doi/10.3982/ECTA9070/abstract
Originally, I coded it for Matlab. The folder "Julia" contains a Julia version of it. To run the code, execute the file "main2.m" in the Julia folder. 

## Main elements of the model:
This is a search-and-matching model with heterogeneous agents that differ in their "ability". Firms are assumed to be identical. Search is random, and worker-firm pairs are formed only when the match surplus is positive. Productivity shocks occur according to a Markov process. Following a shock in productivity, worker-firm pairs with negative surplus are destroyed. On-the-job search is permitted, and firms can make counteroffers to retain their workers. 

Wages are endogeneously determined according to a sequential auction model: unemployed workers are offered their reservation wage, while employed workers accepting an outside are offered receive all the match surplus. 

Calibrated on the US labor market, the model is able to replicated fairly well the variation in unemployment and the volatitily in wages. It explains why low wages and high wages are more procycle than intermediate wages. 

Update 1:
- I have included a version that runs with Julia
- Run the file "main2.jl" that is in the folder "Julia"
- The value function iteration for the wages is slow (slightly more than 20 mn)

Update 2, 21/02/2016:
- I have included two files that perform the estimation of the parameters by the Simulated Method of Moments: "estim_params.jl" and "objective_function.jl". To estimate the parameters, run "estim_params.jl"

Update 3, 10/03/2016:
-Include two files in python in the folder "Data" that calculate interesting moments for the model:
  - calculate_turnover_BLS.py HP filters the data and calculates the moments the model intends to match
  - calculate_volatility_wages.py calculate the volatility of the wage deciles
-The data necessary for the estimations is also in the folder "Data"

