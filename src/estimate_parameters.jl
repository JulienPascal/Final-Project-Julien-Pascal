################################
# Coded by Julien Pascal
# Last modification : 16/04/2016
# Replication of the the following paper:

####################################################################################
# "On the Dynamics of Unemployment and Wage Distributions"
# Paper from Jean Marc Robin (2011)
# Available here: https://sites.google.com/site/jmarcrobin/research/publications
####################################################################################

#######################################################################################
# Model that estimates the parameters of the model by the method of simulated moments
module estimate_parameters


    using PyPlot, Distributions, StatsFuns, JLD, HDF5, NLopt, DataFrames, Copulas
    include("objective_function.jl")

    export MSM

    function MSM()
    	tic()

		params_name = ["z0", "sigma", "rho", "lambda0", "eta", "mu"]
		params_opt = [0.77, 0.023, 0.94, 0.99, 2.00, 5.56] #optimal parameters in the paper
		params0 = [0.77, 0.023, 0.94, 0.99, 2.00, 5.5] #starting values for the optimization

		
		########################################
		# First find a global mimimum and then
		# refine the global mimimum using a local
		# minimization routine
		#######################################


		########################
		# 1. Global Optimization

		println("1. Global minimization")

		#=
		#DIRECT-L: J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT algorithm," J. Global Optimization, vol. 21 (1), p. 27-37 (2001). 
		opt = Opt(:GN_DIRECT_L, 6)
		lower_bounds!(opt, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
		upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
		xtol_rel!(opt,1e-1)
		min_objective!(opt, objective_function)
		@time (minf,minx,ret) = optimize(opt, params0)
		=#

		#Controlled Random Search (CRS) with local mutation:
		# take 1012.905254 seconds to run on my computer
		println("Controlled Random Search (CRS) with local mutation:") 
		opt1 = Opt(:GN_CRS2_LM, 6)
		lower_bounds!(opt1, [0.7, 0.01, 0.8, 0.8, 1.5, 5.0])
		upper_bounds!(opt1, [0.99, 0.99, 0.99, 0.99, 2.5, 6.0]) 
		xtol_rel!(opt1,1e-1)
		min_objective!(opt1, objective_function)
		@time (minf1,minx1,ret1) = optimize(opt1, params0) #start at params0
		#objective_function(minx1,minx1)
		#objective_function(params_opt,params_opt)


		# Save the result in a csv file:
		df_global = DataFrame()
		df_global[:Paramater] = params_name 
		df_global[:Value] = minx1
		df_global[:Minf] = minf1
		println("Global Optimization Results")
		println(df_global)

		writetable("tables/estimated_parameters_step1.csv", df_global)

		#ISRES (Improved Stochastic Ranking Evolution Strategy)
		#=
		opt = Opt(:GN_ISRES, 6)
		lower_bounds!(opt, [0.7, 0.01, 0.8, 0.8, 1.5, 5.0])
		upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 2.5, 6.0]) 
		xtol_rel!(opt,1e-1)
		min_objective!(opt, objective_function)
		@time (minf,minx,ret) = optimize(opt, params_opt)
		objective_function(minx,minx)
		objective_function(params_opt,params_opt)
		=#

		#ESCH (evolutionary algorithm)
		#=
		opt = Opt(:GN_ESCH, 6)
		lower_bounds!(opt, [0.7, 0.01, 0.8, 0.8, 1.5, 5.0])
		upper_bounds!(opt, [0.99, 0.4, 0.99, 0.99, 2.5, 6.0]) 
		xtol_rel!(opt,1e-1)
		min_objective!(opt, objective_function)
		@time (minf,minx,ret) = optimize(opt, params_opt)
		objective_function(minx,minx)
		objective_function(params_opt,params_opt)
		=#


		########################
		# 2. Local optimization
		#######################
		# Refine the global minimum
		println("2. Local minimization")

		#COBYLA (Constrained Optimization BY Linear Approximations) algorithm for derivative-free optimization
		# Takes 157.536096016 to run on my computer
		println("Constrained Optimization BY Linear Approximations") 
		opt2 = Opt(:LN_COBYLA, 6)
		lower_bounds!(opt2, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
		upper_bounds!(opt2, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
		xtol_rel!(opt2,1e-2)
		min_objective!(opt2, objective_function)
		@time (minf2,minx2,ret2) = optimize(opt2, minx1) #starts with the value of the global minimization
		#objective_function(minx,minx)
		#objective_function(params_opt,params_opt)


		#=
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
		#=
		opt = Opt(:LN_SBPLX, 6)
		lower_bounds!(opt, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
		upper_bounds!(opt, [0.99, 0.99, 0.99, 0.99, 3.00, 6.00]) 
		xtol_rel!(opt,1e-1)
		min_objective!(opt, objective_function)
		@time (minf,minx,ret) = optimize(opt, params0)
		objective_function(minx,minx)
		objective_function(params_opt,params_opt)
		=#

		# Save the result in a csv file:
		df = DataFrame()
		df[:Paramater] = params_name 
		df[:Value] = minx2
		df[:Minf] = minf2
		println("MSM results:")
		println(df)

		writetable("tables/estimated_parameters.csv", df)
		toc()
	end
end