########################################
# Estimation of the parameters of the model
########################################
# II. In this file, the code that performs the optimization
# This file loads "objective_function.jl"
# Coded by Julien Pascal

# Using Ubuntu LTS 14.04 and I need to do the following for PyPlot to load without an error:
Libdl.dlopen("/usr/lib/liblapack.so.3", Libdl.RTLD_GLOBAL)

path = "/home/julien/Documents/COURS/5A/MASTER THESIS/Labor Market/Julia/version 4" #the path to my directory
cd(path) #locate in the correct directory
    
    using PyCall
    using PyPlot
    using Distributions
    using StatsFuns
    using JLD, HDF5 #load and save data
    #using JuMP
    using NLopt
    #using Optim
    #using BlackBoxOptim
    using DataFrames

include("Copulas.jl") #From Florian Oswald: https://github.com/floswald/Copulas.jl/blob/master/main.jl
include("objective_function.jl")


# Values in the original paper:
#z0 = 0.77 ,  sigma = 0.023, rho = 0.94, lambda0 = 0.99 eta = 2.00 mu = 5.56
params_name = ["z0", "sigma", "rho", "lambda0", "eta", "mu"]
params_opt = [0.77, 0.023, 0.94, 0.99, 2.00, 5.56]
params0 = [0.70, 0.020, 0.9, 0.9, 2.00, 5.0] #starting values for the optimization

# Try several optimization routines:
#COBYLA (Constrained Optimization BY Linear Approximations) algorithm for derivative-free optimization
#=
opt = Opt(:LN_COBYLA, 6)
lower_bounds!(opt, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
xtol_rel!(opt,1e-1)
min_objective!(opt, objective_function)
@time (minf,minx,ret) = optimize(opt, params0)
objective_function(minx,minx)

#Using the Nelder-Mead Simplex
opt = Opt(:LN_NELDERMEAD, 6)
maxtime!(opt, 100)
lower_bounds!(opt, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
xtol_rel!(opt,1e-1)
min_objective!(opt, objective_function)
@time (minf,minx,ret) = optimize(opt, params0)
objective_function(minx,minx)
=#

# Tom Rowan's "Subplex" algorithm:
opt = Opt(:LN_SBPLX, 6)
lower_bounds!(opt, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
xtol_rel!(opt,1e-1)
min_objective!(opt, objective_function)
@time (minf,minx,ret) = optimize(opt, params0)

# Save the result in a csv file:
df = DataFrame()
df[:Paramater] = params_name 
df[:Value] = minx
df

writetable("estimated_parameters.csv", df)
