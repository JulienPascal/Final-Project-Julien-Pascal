################
# Julien  Pascal

# Move to my folder: 
home = ENV["HOME"]
path_main = "/home/julien/Final-Project-Julien-Pascal"
cd(path_main)	

# Check for the existence of crucial files:

# A. Check that the empirical moments have been calculated
# Otherwise has to launch the python code "Calculate_Moments.py"
if isfile("tables/table_moments.csv") == false
	println("tables/table_moments.csv is missing")
	error("Execute Calculate_Moments.py to create it")
end

# B. Check that the parameters of the model have been estimated
# Otherwise estimate them by the method of simulated moments
if isfile("tables/estimated_parameters.csv") == false
	println("tables/estimated_parameters.csv is missing")
	println("launching the simulated method of moments procedure")
	println("this may take a while")

	include("src/estimate_parameters.jl")

	using estimate_parameters
	#= first argument: minimization algorithm to use
		- "NLopt": use Controlled Random Search (CRS) with local mutation 
		- "DE": Differential Evolution
		- "SA": Simulated Annealing
	# second argument: 1-step or 2-step procedure (using the optimal weighting matrix)
		- "1Step": use the Identity matrix as the weighting matrix
		- "2Step": use the optimal weighting matrix
	# by default: "NLopt", "1Step"
	=#
	MSM("LOCAL", "1Step")
end

# C. Are the value functions for the wages there?
# If not, calculate them by value function iteration
if isfile("surpluses/W_high.jld") == false  || isfile("surpluses/W_high_star.jld") == false  || isfile("surpluses/W_low_star.jld") == false  || isfile("surpluses/W_low.jld") == false 
	println("Wages surpluses are missing")
	println("Calculating them by value function iteration")
	println("this may take a while")

	include("src/calculate_wages.jl")
	using calculate_wages
	Wages() #2380 seconds to run
end

################################################
# D. If the file are there, simulate one economy
################################################
simulate = "true" #set to "true" if want to simulate one economy
if simulate == "true"
	include("src/simulation_economy.jl")
	using simulation_economy

	#In two steps:
	#Simulate the model and store the results:
	model_ouput = simulation_economy.execute_simulation()

	#Analyse the economy:
	analysis_output = simulation_economy.analyse_economy(model_ouput, 10)
end

#Can be done in one step:
#Simulate_and_Analyse_Economy()

######################################
# E. Test what is driving inequalities
######################################
bargaining = "false" #set to "true" if want to test the bargaining power theory
if bargaining == "true"
	include("src/Bargaining_power_test.jl")
	using Bargaining_power_test

	Bargaining_power_test.create_and_analyse_bargaining_power()
end

