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
	using 	estimate_parameters
	MSM()
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


# If the file are there, simulate:
# include code: 
include("src/simulation_economy.jl")
using simulation_economy

model_ouput = simulation_economy.execute_simulation()

simulation_economy.analyse_economy(model_ouput, 10)

#Simulate_and_Analyse_Economy()
