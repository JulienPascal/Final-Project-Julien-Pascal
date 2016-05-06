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

	path_main = "/home/julien/Final-Project-Julien-Pascal"
	path_table = string(path_main,"/tables/")  #the path tables
	path_surpluses = string(path_main,"/surpluses/")#path to precalculated objects, to speed up the process


    using PyPlot, Distributions, StatsFuns, JLD, HDF5, DataFrames, Copulas, NLopt, BlackBoxOptim
   	include("objective_function.jl") #To use package NLopt
   	include("objective_function_2.jl") #To use with BlackBoxOptim and the simulated annhealing procedure
   	include("objective_function_3.jl") #To compute the optimal weighting matrix
   	include("simulated_annealing/samin.jl")#simulated annealing module

	if isfile(string(path_surpluses,"random_numbers.jld")) #load it if exists
	        println(string(path_surpluses,"random_numbers.jld"," found"))
	        println("loading it")
	        file_random_numbers =  jldopen(string(path_surpluses,"random_numbers.jld"), "r")
		    random_numbers = read(file_random_numbers, "random_numbers")
	else 
	    println("creating a series of random draws to be held fixed during the optimization")
	    n_years = 2500; # years to simulate
		number_periods = n_years*4; # one period = one quarter
	    random_numbers = zeros(number_periods)#column vector is length number_periods with the numers coming from the random generator. Type Array{Float64,1}

	    for t=1:number_periods
	    	random_numbers[t] = rand()
	    end

	    #Save the results in a JLD file:
		jldopen(string(path_surpluses,"random_numbers.jld"), "w") do file
		    write(file, "random_numbers", random_numbers) 
		end
	end


    export MSM

    function MSM(routine = "NLopt", steps = "1Step")
    	tic()

    	params_name =["z0", "sigma", "lambda0", "eta", "mu", "x_lower_bar"] #parameters to be estimated
		number_parameters = 6 #number of parameters

    	#params_name =["z0", "sigma", "lambda0", "eta", "mu", "x_lower_bar", "alpha", "delta"] #parameters to be estimated
		#number_parameters = 8 #number of parameters

		#params_name = ["z0", "sigma", "rho", "lambda0", "eta", "mu", "x_lower_bar", "delta"] #parameters to be estimated
		#number_parameters = 8 #number of parameters
		
		#params_name = ["z0", "sigma", "rho", "lambda0", "eta", "mu", "x_lower_bar"] #parameters to be estimated
		#number_parameters = 7 #number of parameters

		#Paper and initial values:
		#params_opt = [0.77, 0.023, 0.94, 0.99, 2.00, 5.56, 0.73, 0.042] #optimal parameters in the paper
		#params0 = [0.77, 0.023, 0.94, 0.99, 2.00, 5.5, 0.70, 0.046] #starting values for the optimization
		#params0 = [0.408512, 0.0678116, 0.674442, 0.726062, 2.6711, 1.61011, 0.73, 0.046]
		#params0 =    [0.77, 0.067, 0.90, 2.67, 1.6, 0.73]

		params0 = [ 0.9573874974174598, 0.02376049264005116, 0.9090148634236118, 5.628756473043127, 3.8906895043507066, 0.6827425005901246]
		params_opt = [0.77, 0.023, 0.99, 2.00, 5.56, 0.73]


		#Bounds of the parameters space:
		#lb = [0.70, 0.01, 0.70, 1.00, 1.00, 0.60, 0.60] #lower bound
		#ub = [0.99, 0.10, 0.99, 4.00, 6.00, 0.90, 0.80] #upper bound
		lb = [0.01, 0.01, 0.01, 1.00, 1.00, 0.01] #lower bound
		ub = [0.99, 0.99, 0.999, 6.00, 6.00, 0.99] #upper bound

		# Eta and sigma and delta may be binding

		# Empirical moments to match:
		# mean productivity, std productivity, mean unemployment rate, std unemployment rate, mean job finding rate, std job finding rate, mean job separation rate, std separation rate
		# One which worked well with DE:
		#b0 = [1.0; 0.023;  0.059;  0.210; 0.761; 0.085; 0.046; 0.156]
		#number_moments = 8

		b0 = [1.0; 0.023;  0.059;  0.210; 0.761; 0.085; 0.046]
		number_moments = 7

		#Checks:
			#check dimensions:
		if size(lb,1)!= number_parameters
			error("Dimension lower bound misspecified")
		elseif size(ub,1)!= number_parameters
			error("Dimension upper bound misspecified")
		elseif size(params0,1)!= number_parameters
			error("Dimension parameter starting values misspecified")
		elseif size(params_name,1)!= number_parameters
			error("Dimension parameter names misspecified")
		elseif size(params_name,1)!= number_parameters
			error("Dimension the vector of parameters misspecified")
		elseif size(b0,1)!= number_moments
			error("Dimension the vector of empirical moments misspecified")
		end
			#check whether the starting value is within the bounds:
		for i=1:number_parameters
			if params0[i] > ub[i]
				println(i)
				error("Starting value above the upper bound")
			end
			if params0[i] < lb[i]
				println(i)
				error("Starting value below the lower bound")
			end
		end

		#Create the search range for the package DEoptim:
		#creates the bounds:
		Search_range = Array{Tuple{Float64,Float64},1}(number_parameters) #Type for Deoptim
		for i=1:number_parameters
			Search_range[i] = (lb[i], ub[i])
		end
		println("Search range")
		println(Search_range)

		if routine == "NLopt"
			println("NLopt")

			if steps == "1Step"
				##########################################
				# 1-Step procedure using the package NLopt
				##########################################
				println("1-Step procedure")

				###################################
				# A. Global Optimization Algorithms
				###################################

				#Controlled Random Search (CRS) with local mutation:
				# take 636.5 seconds seconds to run on my computer
				# 3689.061901 seconds on the last run
				println("Controlled Random Search (CRS) with local mutation:") 
				opt1 = Opt(:GN_CRS2_LM, number_parameters) #
				lower_bounds!(opt1, lb)
				upper_bounds!(opt1, ub) 
				xtol_rel!(opt1,1e-1)
				maxtime!(opt1, 3600)
				Weighting_matrix = eye(number_moments,number_moments)./(b0.*b0) #Use the Identity as a weighting matrix
				min_objective!(opt1, (x,g)->objective_function(x,g,Weighting_matrix, random_numbers, b0))
				@time (minf1,minx1,ret1) = optimize(opt1, params0) #start at params0

				df_global = DataFrame()
				df_global[:Paramater] = params_name 
				df_global[:Value] = minx1
				df_global[:Minf] = minf1
				println("Step 1.1. Global Optimization Results with W = Identity")
				println(df_global)
				writetable("tables/estimated_parameters_1step_CRS.csv", df_global)
				writetable("tables/estimated_parameters.csv", df_global)

			elseif steps == "2Step"
				#############################################
				# 2-Step procedure using the package NLopt
				#############################################
				println("2-Step procedure")

				###################################
				# 1st step: use the identity matrix
				# as the identity
				##################################
				println("STEP 1. Global minimization with W = I")

				#Controlled Random Search (CRS) with local mutation:
				# take 636.5 seconds seconds to run on my computer
				# 3689.061901 seconds on the last run
				println("Controlled Random Search (CRS) with local mutation:") 
				opt1 = Opt(:GN_CRS2_LM, number_parameters) 
				lower_bounds!(opt1, lb)
				upper_bounds!(opt1, ub) 
				xtol_rel!(opt1,1e-1)
				maxtime!(opt1, 3600)
				Weighting_matrix = eye(number_moments,number_moments) #Use the Identity as a weighting matrix
				min_objective!(opt1, (x,g)->objective_function(x,g,Weighting_matrix, random_numbers, b0))
				@time (minf1,minx1,ret1) = optimize(opt1, params0) #start at params0

				df_step1 = DataFrame()
				df_step1[:Paramater] = params_name 
				df_step1[:Value] = minx1
				df_step1[:Minf] = minf1
				println("Step 1.1. Global Optimization Results with W = Identity")
				println(df_global)
				writetable("tables/estimated_parameters_step1_CRS.csv", df_step1)
				writetable("tables/estimated_parameters_step1.csv", df_step1)

				############################################
				# 2nd step: use the optimal weighting matrix
				############################################
				println("Step 2. Global minimization with W = Optimal weighting matrix")
				println("Step 2.0 Calculating the Optimal weighting matrix")
				intermediate_step = objective_function_3(minx1) #returns a dictionary with 3 elements: ("St"=> St, "inverse_St" => inverse_St, "simulated_moments"=> simulated_moments)
				Optimal_weighting_matrix = intermediate_step["inverse_St"]

				println("Optimal weighting matrix:")
				println(intermediate_step["inverse_St"])
				writedlm("tables/optimal_weighting_matrix.txt") #Save the optimal weighting matrix into a txt file

				println("Step 2.1. Global Optimization Results with W = Optimal weighting matrix")
				println("Controlled Random Search (CRS) with local mutation:")  #Takes 2277.229133 seconds
				opt2 = Opt(:GN_CRS2_LM, number_parameters)
				lower_bounds!(opt2, lb)
				upper_bounds!(opt2, ub) 
				xtol_rel!(opt2,1e-1)
				maxtime!(opt2, 3600)
				min_objective!(opt2, (x,g)->objective_function(x,g,Optimal_weighting_matrix, random_numbers, b0)) #Optimal weighting matrix as an input
				@time (minf2,minx2,ret2) = optimize(opt2, minx1) #start with the mimimum of step 1

				df_step2 = DataFrame()
				df_step2[:Paramater] = params_name 
				df_step2[:Value] = minx2
				df_step2[:Minf] = minf2
				println("MSM results:")
				println(df)
				writetable("tables/estimated_parameters_step2_CRS.csv", df_step2)
				writetable("tables/estimated_parameters.csv", df_step2)

			else
				error("Incorrect name for the second argument of MSM()") 
			end
		elseif routine == "DE"
		########################
		# Differential Evolution
		########################
		println("Differential Evolution")

			if steps == "1Step"
				##################################################
				# 1-Step procedure using the package BlackBoxOptim
				##################################################
				println("1-Step procedure")

				Weighting_matrix = eye(number_moments,number_moments)./(b0.*b0) #Square of percentage deviations. Could use the identity matrix as well. 
				#set the time limit to be = 3600 seconds
				#Tolerance = 0.01

				opt_DE = bbsetup(x->objective_function_2(x, Weighting_matrix, random_numbers, b0); Method=:dxnes, SearchRange = Search_range, NumDimensions = number_parameters, ϵ=0.01, MaxTime =  3600)

				res_DE = bboptimize(opt_DE)
				bs = best_candidate(res_DE) #best "candidate" = vector the minimizes the objective function
				bf = best_fitness(res_DE) #value of

				df= DataFrame()
				df[:Paramater] = params_name 
				df[:Value] = bs 
				df[:Minf] = bf
				println("Step 1. MSM results:")
				println(df)
				writetable("tables/estimated_parameters_step1_DE.csv", df)
				writetable("tables/estimated_parameters.csv", df)

			elseif steps == "2Step"
				##################################################
				# 2-Step procedure using the package BlackBoxOptim
				##################################################
				println("2-Step procedure")

				###################################
				# 1st step: use the identity matrix
				# as the identity
				##################################
				println("STEP 1. Global minimization with W = I")

	
				Weighting_matrix = eye(number_moments,number_moments) #Use the Identity as a weighting matrix
				#set the time limit to be = 3600 seconds
				#Tolerance = 0.01
				opt_DE = bbsetup(x->objective_function_2(x, Weighting_matrix, random_numbers, b0); Method=:xnes, SearchRange = Search_range, NumDimensions = number_parameters, ϵ=0.01, MaxTime =  3600)

				res_DE = bboptimize(opt_DE)
				bs = best_candidate(res_DE) #best "candidate" = vector the minimizes the objective function
				bf = best_fitness(res_DE) #value of

				df_step1 = DataFrame()
				df_step1[:Paramater] = params_name 
				df_step1[:Value] = bs 
				df_step1[:Minf] = bf
				println("Step 1. MSM results:")
				println(df)
				writetable("tables/estimated_parameters_step1_DE.csv", df_step1)
				writetable("tables/estimated_parameters_step1.csv", df_step1)

				############################################
				# 2nd step: use the optimal weighting matrix
				############################################
				println("Step 2. Global minimization with W = Optimal weighting matrix")
				println("Step 2.0 Calculating the Optimal weighting matrix")
				intermediate_step = objective_function_3(bs) #returns a dictionary with 3 elements: ("St"=> St, "inverse_St" => inverse_St, "simulated_moments"=> simulated_moments)
				Optimal_weighting_matrix = intermediate_step["inverse_St"]

				println("Optimal weighting matrix:")
				println(intermediate_step["inverse_St"])

				#7241.939754208 seconds
				println("Step 2.1 Global Optimization Results with W = Optimal weighting matrix")
				opt_DE2 = bbsetup(x->objective_function_2(x, Optimal_weighting_matrix, random_numbers, b0); Method=:xnes, SearchRange = Search_range, NumDimensions = number_parameters, ϵ=0.01, MaxTime =  3600)

				res_DE2 = bboptimize(opt_DE2)
				bs2 = best_candidate(res_DE2) #best "candidate" = vector the minimizes the objective function
				bf2 = best_fitness(res_DE2) #value of

				df_step2 = DataFrame()
				df_step2[:Paramater] = params_name 
				df_step2[:Value] = bs2
				df_step2[:Minf] = bf2
				println("Step 2. MSM results:")
				println(df_step2)
				writetable("tables/estimated_parameters.csv", df_step2)
				writetable("tables/estimated_parameters_step2_DE.csv", df_step2)
			else
				error("Incorrect name for the second argument of MSM()") 
			end

		elseif routine == "SA" 
		######################
		# Simulated Annealing
		######################
		println("Simulated Annealing")

		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		#!!! THIS PART HAS TO BE CHANGED
		#!!! NOT COHERENT ANYMORE with obective_function_2

			#Boundaries of the problem and starting value:
			lb = [0.7, 0.01, 0.8, 0.8, 1.5, 5.0]
			initial=[0.70, 0.01, 0.9, 0.90, 2.0, 5.3]
			ub = [0.99, 0.99, 0.99, 0.99, 2.5, 6.0]

			xopt = samin(objective_function_2, initial, lb, ub, maxevals = 300, verbosity=2) 
			#1537.63419783 seconds for 300 iterations

			# Save the result in a csv file:
			df = DataFrame()
			df[:Paramater] = params_name 
			df[:Value] = xopt[1]
			df[:Minf] = objective_function_2(xopt[1])
			println("MSM results:")
			println(df)

			writetable("tables/estimated_parameters.csv", df)
		elseif routine == "LOCAL" #To refine a Global equilibrium
			
			println("LOCAL OPTIMIZATION: COBYLA") 
			opt1 = Opt(:LN_COBYLA, number_parameters) 
			lower_bounds!(opt1, lb)
			upper_bounds!(opt1, ub) 
			xtol_rel!(opt1,1e-1)
			maxtime!(opt1, 3600)
			Weighting_matrix = eye(number_moments,number_moments)./(b0.*b0) #Use the Identity as a weighting matrix
			min_objective!(opt1, (x,g)->objective_function(x,g,Weighting_matrix, random_numbers, b0))
			@time (minf1,minx1,ret1) = optimize(opt1, params0) #start at params0

			df_global = DataFrame()
			df_global[:Paramater] = params_name 
			df_global[:Value] = minx1
			df_global[:Minf] = minf1
			println("LOCAL OPTIMIZATION COBYLA")
			println(df_global)
			writetable("tables/estimated_parameters_COBYLA.csv", df_global)
			writetable("tables/estimated_parameters.csv", df_global)

			#compare the global and the local min:
			println("Global min")
			min0 = objective_function(params0, params0, Weighting_matrix, random_numbers, b0)
			println(min0)

			println("Local min")
			println(minf1)


		else
			error("Incorrect name for the first argument of MSM()") 
			println("Either NLopt, DE, SA, or LOCAL")
		end

	toc() 
	end
end


#########################
# Other NLopt Algorithms
#########################

##################################
# B. Gloabl optimization algorithms
##################################

#=
#DIRECT-L: J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT algorithm," J. Global Optimization, vol. 21 (1), p. 27-37 (2001). 
:GN_DIRECT_L
=#

#ISRES (Improved Stochastic Ranking Evolution Strategy)
#=
:GN_ISRES
=#

#ESCH (evolutionary algorithm)
#=
:GN_ESCH
=#

##################################
# B. Local optimization algorithms
##################################

#=
#Using the Nelder-Mead Simplex
:LN_NELDERMEAD
=#

# Tom Rowan's "Subplex" algorithm:
#=
:LN_SBPLX
=#