########
# TESTS
########
# To be completed

#################################
# PATHS
# change "path_main" if necessary
path_main = "/home/julien/Final-Project-Julien-Pascal"
path_table = string(path_main,"/tables/")  #the path tables
path_surpluses = string(path_main,"/surpluses/")#path to precalculated objects, to speed up the process

file_random_numbers =  jldopen(string(path_surpluses,"random_numbers.jld"), "r")
random_numbers = read(file_random_numbers, "random_numbers")

include(string(path_main,"/src/objective_function.jl"))
include(string(path_main,"/src/objective_function_2.jl"))


number_parameters = 6 #number of parameters
# Read the estimated paramaters
if isfile(string(path_table, "estimated_parameters.csv"))
	println("loading estimated parameters") 
	coeff = readtable(string(path_table, "estimated_parameters.csv"))
else #if does not exist, use the parameters from the original paper
	parameters = [0.77, 0.023, 0.99, 2.00, 5.56, 0.73]
end

z0 = coeff[:Value][1];
sigma = coeff[:Value][2];
lambda0 = coeff[:Value][3];
eta =  coeff[:Value][4];
mu =  coeff[:Value][5];
x_lower_bar= coeff[:Value][6] #lower bound of the ability level


parameters = [z0, sigma, lambda0, eta, mu, x_lower_bar] #parameters to be estimated

b0 = [1.0; 0.023;  0.059;  0.210; 0.761; 0.085; 0.046] #moments to match
number_moments = 7

Weighting_matrix = eye(number_moments,number_moments)./(b0.*b0)

# 1. Make sure the objective functions used by NLopt and DE coincide:
context("Test the objective functions") do 

	facts("Test objective = objective 2") do

	result_objective = objective_function(parameters,parameters, Weighting_matrix, random_numbers, b0)
	result_objective2 = objective_function_2(parameters, Weighting_matrix, random_numbers, b0)
	difference = result_objective - result_objective2

		#the difference should be close to zero
		@fact difference --> roughly(0; atol=0.001)

	end
end

# Look if the estimated parameters perform well compared the original ones:
context("Test estimated parameters vs origianl paper's parameters") do 

	facts("estimated parameters vs original paper's parameters") do

	parameters_original = [0.77, 0.023, 0.99, 2.00, 5.56, 0.73]

	result_original = objective_function(parameters_original ,parameters_original , Weighting_matrix, random_numbers, b0)

	result_estimated = objective_function_2(parameters, Weighting_matrix, random_numbers, b0)

	difference = result_estimated - result_original 

		# if difference is negative, then the estimated parameters are better wrt to the criteria used:
		@fact difference --> less_than_or_equal(0)

	end
end

# 2. With the estimated coefficient, make sure the system behave properly
# e.g. the unemployment rate cannot be negative

# 3. Make sure the wages are finite

# 4. check the accuracy of the minimum of the objective function
# Use a local mininmization algorithm (example: COBYLA) with the estimated parameters
# as an initial condition. 
# A. If the local minimization find a better point, then the global
# maximum is not one. Use this new local minimum instead. 
# B. Else, keep the global max