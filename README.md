# Computational Economics
## Final-Project-Julien-Pascal


This repository contains a replication of the Paper from Jean Marc Robin (2011) "On the Dynamics of Unemployment and Wage Distribution" Econometrica, 79: 1327–1355. doi: 10.3982/ECTA9070. http://onlinelibrary.wiley.com/doi/10.3982/ECTA9070/abstract. Originally, I coded it for Matlab. The Matlab files are at the root of the repository. The folder "Julia" contains a Julia version of it, which includes corrections and improvements compared to the MatLab version. To run the code, execute the file "main2.m" in the Julia folder. The folder "data" contains
data on the U.S. labor market, mainly from the Bureau of Labor Statistics. It also includes two files in python that produce the graphs presented below and that calculate some interesting statistics of the U.S. Labor Market. 

## Main elements of the model:
This is a search-and-matching model with heterogeneous agents that differ in their "ability". Firms are assumed to be identical. Search is random, and worker-firm pairs are formed only when the match surplus is positive. Productivity shocks occur according to a Markov process. Following a shock in productivity, worker-firm pairs with negative surplus are destroyed. On-the-job search is permitted, and firms can make counteroffers to retain their workers. 

Wages are endogeneously determined according to a sequential auction model: unemployed workers are offered their reservation wage, while employed workers accepting an outside are offered receive all the match surplus. 

Calibrated on the US labor market, the model is able to replicated fairly well the variation in unemployment and the volatitily in wages. It explains why low wages and high wages are more procycle than intermediate wages. 

## Stylized facts the U.S. Labor Market

This section offers a visual presentation of the facts that the model aims at explaining.

###Monthly unemployment rate 1947 - 2015
![myimage-alt-tag](https://github.com/JulianPasc/Final-Project-Julien-Pascal/blob/master/Data/Unemployment_1948_2016.png)
Source:Author's calculations based on data from The Bureau of Labor Statistics. Series LNS12000000 and LNS13000000

###Unemployment rate by educational attainment
![myimage-alt-tag](https://github.com/JulianPasc/Final-Project-Julien-Pascal/blob/master/Data/Overall_vs_group_edu_u_rate.png)

###Business cycle co-movements
![myimage-alt-tag](https://github.com/JulianPasc/Final-Project-Julien-Pascal/blob/master/Data/Cycle_unemployment_output.png)
Source:Author's calculations based on data from the Bureau of Labor Statistics and the National Bureau of Economic Research. Series LNS12000000 and LNS13000000 for unemployment rate; PRS85006043 for seasonally adjusted real output in the non-farm business sector; USRECM for recession periods. Notes: The data was successively log-transformed, HP-filtered with a smoothing parameter equal to 2.5\times10^{5}, detrended and exponentiated. The shaded areas represent the NBER-defined recessions.

###Detrended wage deciles
![myimage-alt-tag](https://github.com/JulianPasc/Final-Project-Julien-Pascal/blob/master/Data/delinearized_wage_deciles.png)

###Table of moments
![myimage-alt-tag](https://github.com/JulianPasc/Final-Project-Julien-Pascal/blob/master/Data/moments_table.png)

## Updates:
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

Update 4, 11/03/2016:
- Added some graphs that are produced by the files calculate_turnover_BLS.py and calculate_volatility_wages.py

