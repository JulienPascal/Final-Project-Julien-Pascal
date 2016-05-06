################################
# Coded by Julien Pascal
# Last modification : 16/04/2016
# Replication of the the following paper:

####################################################################################
# "On the Dynamics of Unemployment and Wage Distributions"
# Paper from Jean Marc Robin (2011)
# Available here: https://sites.google.com/site/jmarcrobin/research/publications
####################################################################################

####################################################
# Model that performs the simulation of the economy
module Bargaining_power_test

	using PyPlot, Distributions, DataFrames, StatsFuns, JLD, Copulas, GLM
	#to install the package "Copulas": Pkg.clone("https://github.com/floswald/Copulas.jl.git")
	#package from Florian Oswald

	export Test_bargaining_power

	#change path_main if necessary:
	path_main = "/home/julien/Final-Project-Julien-Pascal"

	path_table = string(path_main,"/tables/")  #the path tables
	path_surpluses = string(path_main,"/surpluses/")#path to precalculated objects, to speed up the process
	path_figures = string(path_main,"/figures/") #where to save the figures

	#Application from [0,1] intp [1,M]
	#To find the index corresponding to the percentile
	#
	#Use the fact that the Beta distribution is on [0,1]
	#and the function was discretized on an equidistant grid
	#
	# p = vector of pecentiles
	# M = integer, number of points
	function percentile_to_index(p::Float64,M::Int)

		index = 0 #initialization
		index = round(Int,(M-1)*p +1) #computation

		return index
	end

    ##################
    # Function that returns the inverse of lognormal distribution
    function logninv(p,mu,sigma)

        #output: scalar x 
        logx0 = -sqrt(2)*erfcinv(2*p);
        x = exp(sigma*logx0 + mu);

        return x
    end
    #################

    ##########################
    #define match productivity:
    function yi_m(y_index,x_index, ygrid, xgrid)
        match = ygrid[y_index,1]*xgrid[x_index,1]
        return match
    end
    ##########################

    ##########################
    # define opportunity cost of employment:
    function zi_m(y_index,x_index, ygrid, xgrid, z0, alpha) 

        op_cost = z0 + alpha*(yi_m(y_index,x_index, ygrid, xgrid) - z0) #call the function defined above

        return op_cost
    end

    #################################
    # Exit from unemployment rate: ft
    function ft_f(St_m,ut_m,ut,lm,lambda0,M)
    #St_m a vector of dimension M times 1 

        ft = 0;
        for i=1:M
            #calculate the sum
            if St_m[1,i]> 0
                ft = ft + ut_m[1,i]*lm[i,1];
            end
        end
        #multiply by lamba0 and divide by the overall level of unemployment ut:
        ft = lambda0*ft/ut;
        return ft
    end

    ##################################################
    #function that calculates the job destruction rate
    function st_f(St_m,ut_m,ut,lm, delta, M)

        st = 0;
        for i=1:M
        #calculate the integral
            if St_m[1,i] <= 0
                st = st + (1 - ut_m[1,i])*lm[i,1];
            end
        end

        #finish the calculation:
        st = delta + (1 - delta)*st/(1 - ut);
        return st
    end

    #############################################################
    # function that calculate the measure of workers at hiw wages
    function gt_plus1_high_f(ut_m, gt_wi_low, gt_wi_high, i, lm, delta, lambda1, M, W_low, W_high, Si_m)

        gt_plus1 = zeros(1,M);

        for m = 1:M
            if Si_m[i,m]> 0
                #calculate the sum in the last paranthesis:
                sum = 0;
                if W_low[i,i,m] > Si_m[i,m]
                    sum = sum + gt_wi_low[1,m];
                end
                
                if W_high[i,i,m] > Si_m[i,m]
                    sum = sum + gt_wi_high[1,m];
                end
                
                gt_plus1[1,m] = (1 - delta)*(lambda1*(1 - ut_m[1, m])*lm[m,1] + (1 - lambda1)*((gt_wi_high[1,m] + sum)));
            end
        end
        return gt_plus1 # output = 1 times M vector
    end

    #############################################################
    #function that calculates the measure of workers at low wages
    function gt_plus1_low_f(ut_m, gt_wi_low, gt_wi_high, i, lm, lambda0, delta, lambda1, M, W_low, W_high, Si_m)
        #Inputs:
        # ut_m = 1 times M vector 
        # gt_wi_low =  1 times M vector
        # gt_wi_high = 1 times M vector
        # i : integer, state of the economy

        gt_plus1 = zeros(1,M);

        for m =1:M
            #calculate the sum in the last paranthesis:
            if Si_m[i,m]> 0
                sum = 0;
                if W_low[i,i,m] < 0
                    sum = sum + gt_wi_low[1,m];
                end
                
                if W_high[i,i,m] < 0
                    sum = sum + gt_wi_high[1,m];
                end
                
                gt_plus1[1,m] = lambda0*ut_m[1,m]*lm[m,1] + (1 - delta)*(1 - lambda1)*(gt_wi_low[1,m] + sum);
            end
        end
        return gt_plus1 #Outuput is 1 times M vector
    end

    ##############################################
    #Calculates the next period's unemployment rate
    function ut_plus1_f(St_m, ut_m, delta, lambda0, M)

        #initialization:
        ut_plus1_m = zeros(1,M);

        for i=1:M
            #if the surplus is negative, unemployment is 1:
            if St_m[1,i] <= 0
                ut_plus1_m[1,i]= 1;
            else
                ut_plus1_m[1,i]= ut_m[1,i] + delta*(1 - ut_m[1,i]) - lambda0*ut_m[1,i];
            end
        end
        return ut_plus1_m 
    end

    ########################################
    #function that calculates the high wages
    function wi_high_f(i, discount, Markov, lambda1, M, N, Si_m, W_high_star, zi_m_grid)

        #initialization
        wi_m_high = zeros(1,M);

        for m = 1:M
            #calculate the sum
            sum = 0;
            for j=1:N
                if Si_m[j,m]> 0
                sum = sum + Markov[i,j]*((lambda1*Si_m[j,m]) + (1 - lambda1)*W_high_star[j,i,m]);
                end
            end
            wi_m_high[1,m] = Si_m[i,m] + zi_m_grid[i,m]- discount*(sum);
        end
        return wi_m_high
    end

    ########################
    #Calculate the low wages:
    function wi_low_f(i, discount, Markov, lambda1, M, N, Si_m, W_low_star, zi_m_grid)

        #input: i = integer, state of the economy
        
        #initialization
        wi_m_low = zeros(1,M);
        for m = 1:M
            #calculate the sum
            sum = 0;
            for j=1:N
                if Si_m[j,m]> 0
                sum = sum + Markov[i,j]*((lambda1*Si_m[j,m]) + (1 - lambda1)*W_low_star[j,i,m]);
                end
            end
            wi_m_low[1,m] = zi_m_grid[i,m] - discount*(sum);
        end
        return wi_m_low
    end


    ############
    # Parameters
    ############

    #############################################
    # Values of the parameters:
    # A. Values from the 2011 paper
    # return a dictionary:
    function parameters_papers()
	    z0 = 0.77
	    sigma = 0.023
	    pho = 0.94
	    lambda0 = 0.99
	    k = 0.12
	    lambda1  = k*lambda0
	    s = 0.42
	    x_lower_bound = 0.73
	    eta = 2.00
	    mu = 5.56
	    alpha = 0.64
	    tau = 0.5
		delta = 0.042 # "4.2# exogenous layoff rate"
		r = 0.05/4 # quarterly interest rate
		discount = (1 - delta)/(1 + r)
		epsilon = 0.002
		N = 100 #number of states
		M = 500 #number of ability levels

	    return Dict("z0"=> z0, "sigma"=> sigma,"pho"=>pho,"lambda0"=>lambda0,"k"=>k, "lambda1"=>lambda1, "s"=>s, "x_lower_bound"=>x_lower_bound, "eta"=>eta, "mu"=>mu,"alpha"=>alpha, "tau" => tau, "delta" => delta, "r" =>r , "discount" =>discount, "epsilon" =>epsilon, "N"=>N, "M"=>M) 
 	end

    #########################################
    # B. Estimated values for the parameters:
    # Read the coefficients estimated by the method of simulated moments:
    # Check the table exists
    function parameters_estimated()

	    # call it read estimated_parameters
	    if isfile(string(path_table, "estimated_parameters.csv"))
	    	println("loading estimated parameters") 
			coeff = readtable(string(path_table, "estimated_parameters.csv"))
			"""
			z0 = coeff[:Value][1]
		    sigma = coeff[:Value][2]
		    pho = coeff[:Value][3]     
		    lambda0 = coeff[:Value][4] 
		    eta = coeff[:Value][5]
		    mu = coeff[:Value][6]
		    x_lower_bound = coeff[:Value][7]
		    delta = coeff[:Value][8] #exogenous layoff rate

		    k = 0.12
		    lambda1  = k*lambda0
		    #x_lower_bound = 0.73
		    alpha = 0.64
		    ##############################################""

		    tau = 0.5
		    #delta = 0.046
		    r = 0.05/4 # quarterly interest rate
		    discount = (1 - delta)/(1 + r)
		    epsilon = 0.002
		    """

		    """
		    z0 = coeff[:Value][1];
		    sigma = coeff[:Value][2];
		    lambda0 = coeff[:Value][3];
		    eta =  coeff[:Value][4];
		    mu =  coeff[:Value][5];
		    x_lower_bound = coeff[:Value][6] #lower bound of the ability level
		    alpha = coeff[:Value][7]
		    delta = coeff[:Value][8]

		    pho = 0.94
		    k = 0.12;
		    lambda1  = k*lambda0;
		    #x_lower_bound = 0.73
		    #alpha = 0.64
		    tau = 0.5
		    #delta = 0.043 # exogenous layoff rate

		    r = 0.05/4 # interest rate
		    discount = (1 - delta)/(1 + r)
		    epsilon = 0.002
		    """

			z0 = coeff[:Value][1];
		    sigma = coeff[:Value][2];
		    lambda0 = coeff[:Value][3];
		    eta =  coeff[:Value][4];
		    mu =  coeff[:Value][5];
		    x_lower_bound = coeff[:Value][6] #lower bound of the ability level

		    pho = 0.94
		    k = 0.12;
		    lambda1  = k*lambda0;
		    #x_lower_bound = 0.73
		    alpha = 0.64
		    tau = 0.5
		    delta = 0.042 # exogenous layoff rate

		    r = 0.05/4 # interest rate
		    discount = (1 - delta)/(1 + r)
		    epsilon = 0.002


		    N = 100 #number of states
			M = 500 #number of ability levels

			return Dict("z0"=> z0, "sigma"=> sigma,"pho"=>pho,"lambda0"=>lambda0,"k"=>k, "lambda1"=>lambda1, "x_lower_bound"=>x_lower_bound, "eta"=>eta, "mu"=>mu,"alpha"=>alpha, "tau" => tau, "delta" => delta, "r" =>r , "discount" =>discount, "epsilon" =>epsilon, "N"=>N, "M"=>M) 
 
		else
			println(string(path_table, "estimated_parameters.csv"," not found"))
			println(string("parameters from the orginal paper were used instead"))
			output_dictionary = parameters_papers()
			return output_dictionary 
		end

	end

	###########################################
	# function that generates the state space:
	# Return a dictionary with the different grids:
	# Elements: N, M, xgrid, agrid, ygrid, Markov, cop, b lm, zi_m_grid
	# Return the dictionary "parameters" as well
	function setup_grids()

		#Load parameters
		params = parameters_estimated()
		x_lower_bound = params["x_lower_bound"]
		epsilon = params["epsilon"]
		sigma = params["sigma"]
		pho= params["pho"]
		eta = params["eta"]
		mu = params["mu"]
		z0 = params["z0"]
		alpha = params["alpha"]
	    N = params["N"] #number of states
	    M = params["M"] #number of ability levels


	    # Define the grid for ability x
	    xgrid_space = linspace(x_lower_bound,x_lower_bound+1,M); # column vector with M rows

	    xgrid = Array{Float64}(M,1);
	    for i=1:M
	    	xgrid[i,1] = xgrid_space[i];
	    end

	    # Define the grid for match ability yi:
	    # 1. a:
	    agrid = linspace((0+epsilon),(1-epsilon),N);

	    # 2. yi:
	    #intialization:
	    ygrid = Array{Float64}(N,1);
	    for i=1:N
	    	ygrid[i] = logninv(agrid[i],0,sigma); #calculate the inverse of the lognormal
	    end

	    #Markov transition matrix:
	    Markov = zeros(N,N);

	    # number of dimension
	    # AR1 parameter
	    cop = Copulas.NormalCopula(2,pho)
	    #Gaussian copula pdf:
	    for j=1:N #move along the column:
	        for i=1:N #move along the lines:
	            Markov[i,j] = Copulas.dnormCopula([agrid[i] agrid[j]], cop)[];
	        end   
	    end

	    #Normalize so that each row sum to 1:
	    for i = 1:N
	        Markov[i,:] = Markov[i,:]./sum(Markov[i,:]);
	    end

	    # distribution of workers on the grid x 
	    lm = Array{Float64}(M,1);

	    # load the beta pdf
	    b = Beta(eta, mu) 

	    for i=1:M
	        lm[i,1] = pdf(b, xgrid[i]-x_lower_bound); #evaluation of the beta pdf at a given point
	    end

	    #normalization so that mass of workers sum up to 1:
	    lm[:,1] = lm[:,1]./sum(lm[:,1]);

	    # Pre calculate the opportunity cost along the grid of ability:
	    # N times M matrix
	    # rows = aggregate staes
	    # columns = ability level
	    zi_m_grid = zeros(N,M);

	    for i=1:N
	        for m=1:M
	            zi_m_grid[i,m] = zi_m(i,m, ygrid, xgrid, z0, alpha);
	        end
	    end

		return Dict("xgrid"=>xgrid,"agrid"=>agrid,"ygrid"=>ygrid, "Markov"=>Markov, "cop"=>cop,
		"lm"=>lm, "b"=>b, "zi_m_grid"=>zi_m_grid, "params"=> params) 
	end


    ###############################################
    # Load functions if already in the folder
    # Otherwise create them through value function iteration
    # Return a dictionary
    # Return the dictionaries "parameters" and "setup_grid" as well
    function create_surpluses_functions()
	
	    grids = setup_grids()
	    params = grids["params"]
	    N = params["N"] #number of states
	    M = params["M"] #number of ability levels
	    discount = params["discount"]

	    zi_m_grid = grids["zi_m_grid"]
	    xgrid = grids["xgrid"]
	    ygrid = grids["ygrid"]
	    Markov = grids["Markov"]

	    # Pre calculate values:
		G = zeros(N,M); #intialization
		p = zeros(N,M);
		for i = 1:N
			for m = 1:M
				G[i,m] = yi_m(i,m, ygrid, xgrid) - zi_m_grid[i,m];
				p[i,m] = yi_m(i,m, ygrid, xgrid);
			end
		end

    	# A. The match surplus
    	#=
	    if isfile(string(path_surpluses,"match_surplus.jld")) #load it if exists
	        println(string(path_surpluses,"match_surplus.jld"," found"))
	        println("loading it")
	        #Si_m = jldopen(string(path_surpluses,"match_surplus.jld"), "r") do file
	        #    read(file, "Si_m")
	        #end
	        file0 =  jldopen(string(path_surpluses,"match_surplus.jld"), "r")
		    Si_m = read(file0, "Si_m")
		    #close(file0)
	    else #otherwise create it by vale function iteration   
	    =#                                

		    ##########################################################
		    # Calculate the match surplus by value function iteration:
		    # N times M matrices
		    # N = aggregate level
		    # M = ability level
		    println("Calculating the match surplus:")
	        Si_m = zeros(N,M);

	        #tol = 0.0001;
	        tol = 0.01;
	        maxits = 300;
	        dif = tol+tol;
	        its = 1;

	        #initialization:
	        up = ones(N,M);
	        up_plus1 = zeros(N,M);
	        compare = zeros(N,M);

		        while dif>tol 
		            up_plus1 =  G + discount*Markov*max(up,compare)
		            dif = norm(up_plus1 - up)          
		            up = up_plus1
		            its = its + 1
		            print(its)
		                    
		            if its > maxits
		                break
		            end
		        end

	        Si_m = up
	        #save the match surplus:
	       	#jldopen(string(path_surpluses,"/match_surplus.jld"), "w") do file
	        #	write(file, "Si_m", Si_m) 
	        #end
	        #println(string(path_surpluses,"/match_surplus.jld")," created")
	    #end

	    # B. the wages:
		if isfile(string(path_surpluses,"W_low_star.jld")) 
			println(string(path_surpluses,"W_low_star.jld", " found"))
			println("loading it")
		    #W_low_star = jldopen(string(path_surpluses,"W_low_star.jld"), "r") do file
		    #       read(file, "W_low_star")
		    #end

		    file1 =  jldopen(string(path_surpluses,"W_low_star.jld"), "r")
		    W_low_star = read(file1, "W_low_star")
		    #close(file1)

		else
			println(string(path_surpluses,"W_low_star.jld", " is missing"))
		    error("Run the function Wages() of the module calculate_wages")
		end

		if isfile(string(path_surpluses,"W_low.jld")) 
			println(string(path_surpluses,"W_low.jld", " found"))
			println("loading it")
		    #W_low = jldopen(string(path_surpluses,"W_low.jld"), "r") do file
		    #       read(file, "W_low")
		    #end
		   	file2 =  jldopen(string(path_surpluses,"W_low.jld"), "r")
		    W_low = read(file2, "W_low")
		    #close(file2)

		else
			println(string(path_surpluses,"W_low.jld", " is missing"))
		    error("Run the function Wages() of the module calculate_wages")
		end

		if isfile(string(path_surpluses,"W_high_star.jld"))
			println("loading it")
			println(string(path_surpluses,"W_high_star.jld", " found")) 
		    #W_high_star = jldopen(string(path_surpluses,"W_high_star.jld"), "r") do file
		    #        read(file, "W_high_star")
		    #end

		    file3 =  jldopen(string(path_surpluses,"W_high_star.jld"), "r")
		    W_high_star = read(file3, "W_high_star")
		    #close(file3)
		    
		else
			println(string(path_surpluses,"W_high_star.jld", " is missing"))
		    error("Run the function Wages() of the module calculate_wages")
		end

		if isfile(string(path_surpluses,"W_high.jld")) 
			println("loading it")
			println(string(path_surpluses,"W_high.jld", " found"))
		    #W_high = jldopen(string(path_surpluses,"W_high.jld"), "r") do file
		    #       read(file, "W_high")
		    #end

		    file4 =  jldopen(string(path_surpluses,"W_high.jld"), "r")
		    W_high = read(file4, "W_high")
		    #close(file4)

		else
			println(string(path_surpluses,"W_high.jld", " is missing"))
		    error("Run the function Wages() of the module calculate_wages")
		end

		return Dict("Si_m"=> Si_m ,"G"=>G,"p"=>p, "W_low_star"=>W_low_star, "W_low"=>W_low,
		"W_high_star"=>W_high_star, "W_high"=>W_high, "params" => params, "grids" => grids) 
	end

	#############################################
	# Function that actually simulate the economy
	#############################################
	# returns a dictionary 
	function execute_simulation_2(n_years = 2500)

		println("Simulate the economy")
		#Load the parameters, the grids and the value functions:
		Surpluses = create_surpluses_functions()

		params = Surpluses["params"]
		x_lower_bound = params["x_lower_bound"]
		epsilon = params["epsilon"]
		sigma = params["sigma"]
		pho= params["pho"]
		eta = params["eta"]
		mu = params["mu"]
		z0 = params["z0"]
		alpha = params["alpha"]
	    N = params["N"] #number of states
	    M = params["M"] #number of ability levels
	    lambda0 = params["lambda0"] 
	    lambda1 = params["lambda1"] 
	    delta = params["delta"] 
	    discount = params["discount"] 

	    grids = Surpluses["grids"]
	    zi_m_grid = grids["zi_m_grid"]
	    xgrid = grids["xgrid"]
	    ygrid = grids["ygrid"]
	    Markov = grids["Markov"]
	    lm = grids["lm"]
	    b = grids["b"] #beta distribution

	    Si_m = Surpluses["Si_m"]
	    G = Surpluses["G"]
	    p = Surpluses["p"]
	    W_low_star = Surpluses["W_low_star"]
	    W_low = Surpluses["W_low"]
	    W_high_star = Surpluses["W_high_star"]
	    W_high = Surpluses["W_high"]

	    # Load the randomly generated numbers:
		if isfile(string(path_surpluses,"random_numbers.jld")) #load it if exists
	        println(string(path_surpluses,"random_numbers.jld"," found"))
	        println("loading it")
	        file_random_numbers =  jldopen(string(path_surpluses,"random_numbers.jld"), "r")
		    random_numbers = read(file_random_numbers, "random_numbers")
	    else 
	    	error("random numbers not generated")
	    end

		#n_years = 2500; # years to simulate
		number_periods = n_years*4; # one period = one quarter

		y_index_r = Array{Int16}(number_periods,1); #store the indexes of the shock 
		y_r = Array{Float64}(number_periods,1) #store value of the shock

		#initial shock:
		y_index_r[1,1] = 50; #initial shock index
		y_r[1,1] = ygrid[y_index_r[1,1], 1]; #initial shock value

		#Initialization:
		ut_m_r = ones(number_periods,M); #row for the period, column for ability level
		St_m_r = Array{Float64}(number_periods,M); #row for the period, column for ability

		ut_r = Array{Float64}(number_periods,1);

		ft_r = Array{Float64}(number_periods,1);
		qt_r = Array{Float64}(number_periods,1);
		st_r = Array{Float64}(number_periods,1);

		wi_m_low_r = zeros(number_periods,M); #row for the period, column for ability;
		wi_m_high_r = zeros(number_periods,M); #row for the period, column for ability

		gt_wi_low_r = zeros(number_periods,M); #row for the period, column for ability;
		gt_wi_high_r = zeros(number_periods,M); #row for the period, column for ability;

		wi_m_r = zeros(number_periods,M);#row for the period, column for ability;
		wi_r = zeros(number_periods,1); #mean wage
		    
		#measure of unemployed workers:
		m_unemployed_r = zeros(number_periods,M); #0 unemployed at t = 0

				#Indexes to find the percentiles in the distribution of workers:
		p_10 = percentile_to_index(quantile(b, 0.10), M) # index of the 10th percentile on the grid [1,M]
		p_20 = percentile_to_index(quantile(b, 0.20), M) # index of the 20th percentile on the grid [1,M]
		p_25 = percentile_to_index(quantile(b, 0.25), M) # etc.
		p_30 = percentile_to_index(quantile(b, 0.30), M)
		p_40 = percentile_to_index(quantile(b, 0.40 ),M)
		p_50 = percentile_to_index(quantile(b, 0.50), M)
		p_60 = percentile_to_index(quantile(b, 0.60), M)
		p_70 = percentile_to_index(quantile(b, 0.70), M)
		p_80 = percentile_to_index(quantile(b, 0.80), M)
		p_90 = percentile_to_index(quantile(b, 0.90), M)

		#unemployment by education:
		#25th percentile:
		u_25_p = zeros(number_periods,1);
		#50th percentile:
		u_50_p = zeros(number_periods,1);
		#90th percentile
		u_90_p = zeros(number_periods,1);

		#At t=1, half on the people at the starting salary and half of the people at the promotion salary
		gt_wi_low_r[1,:] = transpose(lm)/2
		gt_wi_high_r[1,:] = transpose(lm)/2

		########################
		# Loop over the economy:
		for t=1:(number_periods-1) 
		    
		    #println(string("quarter #:", t))
		    # Calculate the aggregate unemployment
		    ut_r[t,1] = dot(ut_m_r[t,:],lm[:,1]); #dot product. 

		    # Calculate the surplus given the actual value of the shock
		    St_m_r[t,:] = Si_m[y_index_r[t,1],:];
		    
		    #Remark: everytime I select a row, Julia treats it as a column vector
		    #That is why I have to transpose my inputs in the next function
		    # Exit rate from unemployment:
		    ft_r[t,1] = ft_f(St_m_r[t,:]', ut_m_r[t,:]', ut_r[t,1], lm, lambda0, M);

		    #Job destruction rate:
		    st_r[t,1] = st_f(St_m_r[t,:]', ut_m_r[t,:]', ut_r[t,1], lm, delta, M);
		    
		    # Law of motion of unemployment:
		    ut_m_r[t+1,:] = ut_plus1_f(St_m_r[t,:]', ut_m_r[t,:]', delta, lambda0, M);

		    #Calculate group unemployment rate:
		    #25th percentile:
		    u_25_p[t+1,1] = dot(lm[1:p_25,1], ut_m_r[t+1,1:p_25])/sum(lm[1:p_25,1]);

		    #50th percentile:
		    u_50_p[t+1,1] = dot(lm[1:p_50,1], ut_m_r[t+1,1:p_50])/sum(lm[1:p_50,1]);
		    
		    #90th percentile
		    u_90_p[t+1,1]= dot(lm[1:p_90,1], ut_m_r[t+1,1:p_90])/sum(lm[1:p_90,1]);
		    
		    #Calculate the wages:
		    # 1. starting wages
		    wi_m_low_r[t,:] = wi_low_f(y_index_r[t,1], discount, Markov, lambda1, M, N, Si_m, W_low_star, zi_m_grid); # input: the index state of the economy
		    
		    #2. promotion wages
		    wi_m_high_r[t,:] = wi_high_f(y_index_r[t,1], discount, Markov, lambda1, M, N, Si_m, W_high_star, zi_m_grid); #input: the index state of the economy
		    
		    #measure of workers of ability m employed at low wage at the end of
		    #period t
		    gt_wi_low_r[t+1,:] = gt_plus1_low_f(ut_m_r[t,:]',gt_wi_low_r[t,:]',gt_wi_high_r[t,:]',y_index_r[t,1], lm, lambda0, delta, lambda1, M, W_low, W_high, Si_m);

		    #measure of workers of ability m employed at high wage at the end of
		    #period t
		    gt_wi_high_r[t+1,:] = gt_plus1_high_f(ut_m_r[t,:]', gt_wi_low_r[t,:]', gt_wi_high_r[t,:]', y_index_r[t,1], lm, delta, lambda1, M, W_low, W_high, Si_m);

		    #measure of unemployed workers at the end of period t:
		    # = mesure of workers with ability m minus employed people
		    m_unemployed_r[t+1,:] = (transpose(lm[:,1]) - (gt_wi_low_r[t+1,:]' + gt_wi_high_r[t+1,:]'));

		    #calculate the average wage by worker type:
		    #weight by the measure of workers with low/high wages:
		    for i=1:M #loop over ability level
		    	mass = gt_wi_low_r[t+1,i] + gt_wi_high_r[t+1,i]
		    	if (mass!= 0) & (isnan(mass)==false) #cannot divide by zero and don't want to deal with NaN
					wi_m_r[t,i] = (wi_m_low_r[t,i]*gt_wi_low_r[t+1,i] + wi_m_high_r[t,i]*gt_wi_high_r[t+1,i])/(gt_wi_low_r[t+1,i] + gt_wi_high_r[t+1,i])
				else
			   		wi_m_r[t,i] = NaN 
			   	end
			end
		    ############################
		    # mean wage denoted by wi_r
		    #
		    # calculate recursively the mean wage:
			wi_r[t,1]= 0; #initialization
			csum = 0; #initialisation

			for i=1:M #loop over ability
				mass = gt_wi_low_r[t+1,i] + gt_wi_high_r[t+1,i]
				if (mass!= 0) & (isnan(mass)==false) & (isnan(wi_m_r[t,i])==false) #cannot divide by zero and don't want to deal with NaN
					wi_r[t,1] = wi_r[t,1] + wi_m_r[t,i]*mass; #weight by the measure of workers at the given wage
					csum = csum + mass; 
				end
			end

			if csum != 0
				wi_r[t,1] = wi_r[t,1]/csum; 
			else
				wi_r[t,1] = NaN #no one has a wage. 
			end
		    ######################

		    # New shock from the markov transition matrix:
		    r = rand(); #draw a number from the uniform distribution on 0, 1
		    #r = random_numbers[t]
		    
		    # I use the Markov transition matrix previously defined:
		    prob = Markov[y_index_r[t,1],:];
		    
		    #I stock the index in the Markov matrix, as well as the value of the
		    #shock y
		    y_index_r[t+1,1] = sum(r .>= cumsum([0; prob]));

		    y_r[t+1,1] = ygrid[y_index_r[t+1,1], 1];
		end
		
		#Surpluses: contains the surplus value functions, the parameters, the parameter grids
		#n_years: number years the simulation runs
		#number_periods: number of periods the simulation runs (one period = one quarter) 
		#y_r: deviation of productivity from its long time tredn

		return Dict("Surpluses"=> Surpluses ,"n_years"=>n_years, "number_periods"=>number_periods, "y_r"=>y_r , "ut_m_r"=>ut_m_r,
		"ut_r"=>ut_r, "ft_r"=>ft_r, "st_r" => st_r, "wi_r" => wi_r, "wi_m_r" =>wi_m_r, "u_25_p"=>u_25_p, "u_50_p"=>u_50_p, "u_90_p"=>u_90_p,
		"gt_wi_low_r"=>gt_wi_low_r, "gt_wi_high_r"=>gt_wi_high_r) 
	end

	############################
	# Analysis of the simulation
	############################
	# model: has to be the output of the function execute_simulation()
	# get_rid_of: to discard the impact of initial condition, get rid of a given percentage of first observations
	function analyse_economy_2(model, get_rid_of = 10)

		println("Analyse the economy")
		Surpluses = model["Surpluses"]

		params = Surpluses["params"]
		x_lower_bound = params["x_lower_bound"]
		epsilon = params["epsilon"]
		sigma = params["sigma"]
		pho= params["pho"]
		eta = params["eta"]
		mu = params["mu"]
		z0 = params["z0"]
		alpha = params["alpha"]
	    N = params["N"] #number of states
	    M = params["M"] #number of ability levels
	    lambda0 = params["lambda0"] 
	    lambda1 = params["lambda1"] 
	    delta = params["delta"] 
	    discount = params["discount"] 

	    grids = Surpluses["grids"]
	    zi_m_grid = grids["zi_m_grid"]
	    xgrid = grids["xgrid"]
	    ygrid = grids["ygrid"]
	    Markov = grids["Markov"]
	    lm = grids["lm"]

	    Si_m = Surpluses["Si_m"]
	    G = Surpluses["G"]
	    p = Surpluses["p"]
	    W_low_star = Surpluses["W_low_star"]
	    W_low = Surpluses["W_low"]
	    W_high_star = Surpluses["W_high_star"]
	    W_high = Surpluses["W_high"]

	    n_years = model["n_years"]
	   	number_periods = model["number_periods"]
	   	y_r =  model["y_r"]
	   	ut_m_r = model["ut_m_r"]
	   	ut_r = model["ut_r"]
	   	ft_r = model["ft_r"]
	   	st_r = model["st_r"]
	   	wi_r = model["wi_r"]
	   	wi_m_r =  model["wi_m_r"]

	   	u_25_p = model["u_25_p"]
	   	u_50_p = model["u_50_p"]
	   	u_90_p = model["u_90_p"]

	   	gt_wi_low_r = model["gt_wi_low_r"]
	   	gt_wi_high_r =model["gt_wi_high_r"]

		discard = round(Int,floor(number_periods/get_rid_of)); #get rid of the first "get_rid_of"th observations

		# wage per decile:
		    # rows = time period
		    # columns = percentile. ex: first row = 10th decile, 2nd row= 20th decile
		w_p = zeros(number_periods,9); #initialization
		
		#########################
		#Dynamics of wage decile:
		#########################

		###############################################
		# Calculate wage deciles and interdecile ratios 

		trim = floor(Int16,1*discard); #to gain time, calculate for fewer periods

		D5_D1 = zeros(number_periods,1);
		D9_D1 = zeros(number_periods,1);
		D9_D5 = zeros(number_periods,1);

		for t = (discard):(number_periods-1)

		    per = 0; #intialization

		    for i=1:9 #10th, 20th, ..., 90th deciles

		        #choose the decile of interest
		        per = per + 0.1; #10th, 20th, ..., 90th
		        a = 0; #intialization
		        sump = 0;
		        
		        while ((sump<per) & (a<500))
		                
					#initialize values:
					a = a+1;
					sump=0;
					csum=0;

		            #calculate a sum
		            for m = 1:a #loop over ability levels
						if (wi_m_r[t,m] < wi_m_r[t,a]) & (isnan(wi_m_r[t,m]) == false) & (wi_m_r[t,m]!=0) #take into account only employed workers
							sump = sump + (gt_wi_low_r[t+1, m] + gt_wi_high_r[t+1, m]); #weight by the distribution of workers
						end
		            end
					csum = sum(gt_wi_low_r[t+1,:]+ gt_wi_high_r[t+1,:]);

					if csum !=0 
						sump = sump/csum;
					else
						sump = 0;  
					end
		        end
		        
		        #store the value of the wage decile:
		        w_p[t,i] = wi_m_r[t,a];
		    end
		    
		    #calculate interdecile ratio
		    D5_D1[t,1] = w_p[t,5]/w_p[t,1];
		    D9_D1[t,1] = w_p[t,9]/w_p[t,1];
		    D9_D5[t,1] = w_p[t,9]/w_p[t,5];
		end


		#Share of workers in the starting salary:
		starting_wage_share = 0;
		starting_wage_share = sum(gt_wi_low_r[discard:(number_periods-1),:])/(sum(gt_wi_high_r[discard:(number_periods-1),:]) + sum(gt_wi_low_r[discard:(number_periods-1),:])) 

		#Summary statistics:
		mean_simulation = zeros(5)
		mean_simulation[1] = sum(mean(y_r[discard:(number_periods-1),1]))
		mean_simulation[2] = sum(mean(ut_r[discard:(number_periods-1),1]))
		mean_simulation[3] = sum(mean(ft_r[discard:(number_periods-1),1]))
		mean_simulation[4] = sum(mean(st_r[discard:(number_periods-1),1]))
		mean_simulation[5] = sum(mean(wi_r[discard:(number_periods-1),1]))

		std_simulation = zeros(5)
		std_simulation[1] = sum(std(log(y_r[discard:(number_periods-1),1])))
		std_simulation[2] = sum(std(log(ut_r[discard:(number_periods-1),1])))
		std_simulation[3] = sum(std(log(ft_r[discard:(number_periods-1),1])))
		std_simulation[4] = sum(std(log(st_r[discard:(number_periods-1),1])))
		std_simulation[5] = sum(std(log(wi_r[discard:(number_periods-1),1])))

		skewness_simulation = zeros(5)
		skewness_simulation[1] = sum(skewness(log(y_r[discard:(number_periods-1),1])))
		skewness_simulation[2] = sum(skewness(log(ut_r[discard:(number_periods-1),1])))
		skewness_simulation[3] = sum(skewness(log(ft_r[discard:(number_periods-1),1])))
		skewness_simulation[4] = sum(skewness(log(st_r[discard:(number_periods-1),1])))
		skewness_simulation[5] = sum(skewness(log(wi_r[discard:(number_periods-1),1])))
		
		kurtosis_simulation = zeros(5)
		kurtosis_simulation[1] = sum(kurtosis(log(y_r[discard:(number_periods-1),1])))
		kurtosis_simulation[2] = sum(kurtosis(log(ut_r[discard:(number_periods-1),1])))
		kurtosis_simulation[3] = sum(kurtosis(log(ft_r[discard:(number_periods-1),1])))
		kurtosis_simulation[4] = sum(kurtosis(log(st_r[discard:(number_periods-1),1])))
		kurtosis_simulation[5] = sum(kurtosis(log(wi_r[discard:(number_periods-1),1])))

		mean_inequalities = zeros(3) 
		mean_inequalities[1] = mean(D5_D1[discard:(number_periods-1),1])
		mean_inequalities[2] = mean(D9_D1[discard:(number_periods-1),1])
		mean_inequalities[3] = mean(D9_D5[discard:(number_periods-1),1])

		wage_deciles = zeros(9) #store the mean of each decile: 10th, 20th, etc. 
		for i=1:9 #the "i" argument select the decile: 10th, 20th, etc. 
			wage_deciles[i] = mean(w_p[discard:(number_periods-1),i])
		end

		params_name = ["y", "u", "ft", "st", "w"]
		df = DataFrame()
		df[:Parameters] = params_name 
		df[:Mean] = round(mean_simulation, 4)
		df[:Std] = round(std_simulation, 4)
		df[:Skewness] = round(skewness_simulation, 4)
		df[:Kurtosis] = round(kurtosis_simulation, 4)
		
		params_name_2 = ["D5_D1", "D9_D1", "D9_D5"]
		df_inequalities = DataFrame()
		df_inequalities[:Parameters] = params_name_2
		df_inequalities[:Value] = mean_inequalities 

		params_name_3 = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
		df_wage = DataFrame()
		df_wage[:Parameters] = params_name_3
		df_wage[:Value] = wage_deciles

		return Dict("Moments" => df, "Inequality_ratios"=> df_inequalities, "wage_deciles"=> df_wage, "starting_wage_share" => starting_wage_share)
	end

	#Function that simulates the economy based on 
	#the estimated parameters:
	#
	# By default:
	# number_repetitions = 100 #number of economies to simulate
	# years_each_economy = 60 #100 years
	# get_rid_of_the_first_x_percent = 10 #discard the first 10th observations
	function create_sample_bargaining_power(number_repetitions = 100, years_each_economy = 60, get_rid_of_the_first_x_percent = 10)

		#want to store the D9_D1 ratio and the unemployment rate
		mean_D9_D1_vector = zeros(number_repetitions)
		mean_D5_D1_vector = zeros(number_repetitions)
		mean_D9_D5_vector = zeros(number_repetitions)

		mean_D1_vector = zeros(number_repetitions)
		mean_D5_vector = zeros(number_repetitions)
		mean_D9_vector = zeros(number_repetitions)

		mean_unemployment_rate_vector = zeros(number_repetitions)
		mean_productivity_vector = zeros(number_repetitions)
		mean_wage_vector = zeros(number_repetitions)

		starting_wage_share = zeros(number_repetitions)

		for i=1:number_repetitions 

			println(i)
			model_ouput = execute_simulation_2(years_each_economy) 
			
			analysis_output = analyse_economy_2(model_ouput, get_rid_of_the_first_x_percent) #get rid of the first 10th observations

			mean_D5_D1_vector[i] = analysis_output["Inequality_ratios"][:Value][1]
			mean_D9_D1_vector[i] = analysis_output["Inequality_ratios"][:Value][2]
			mean_D9_D5_vector[i] = analysis_output["Inequality_ratios"][:Value][3]

			mean_D1_vector[i]= analysis_output["wage_deciles"][:Value][1]
			mean_D5_vector[i]= analysis_output["wage_deciles"][:Value][5]
			mean_D9_vector[i]= analysis_output["wage_deciles"][:Value][9]

			mean_productivity_vector[i] = analysis_output["Moments"][:Mean][1] #mean productivity
			mean_unemployment_rate_vector[i] = analysis_output["Moments"][:Mean][2] # mean unemployment rate
			mean_wage_vector[i] = analysis_output["Moments"][:Mean][5] # mean wage

			starting_wage_share[i] = analysis_output["starting_wage_share"] # mean wage

		end

		#Save the results:
		params_name_test = ["D5_D1", "D9_D1", "D9_D5", "mean_productivity", "mean_unemployment_rate", "mean_wage", "starting_wage_share"]

		df_test = DataFrame()
		df_test[:D5_D1] = mean_D5_D1_vector
		df_test[:D9_D1] = mean_D9_D1_vector
		df_test[:D9_D5] = mean_D9_D5_vector

		df_test[:D1] = mean_D1_vector
		df_test[:D5] = mean_D5_vector
		df_test[:D9] = mean_D9_vector

		df_test[:mean_productivity] = mean_productivity_vector
		df_test[:mean_unemployment_rate] = mean_unemployment_rate_vector
		df_test[:mean_wage] = mean_wage_vector
		df_test[:starting_wage_share] = starting_wage_share
		writetable("tables/bargaining_power_test.csv", df_test)

		return df_test
	end

	#Input should be a dataframe returned by the function "create_sample_bargaining_power"
	function analyse_bargaining_power(df_test)

		#############
		#Create logs:
		df_test[:log_mean_productivity] = log(df_test[:mean_productivity] )
		df_test[:log_D9_D1] = log(df_test[:D9_D1])
		df_test[:log_mean_unemployment_rate] = log(df_test[:mean_unemployment_rate])
		df_test[:log_mean_wage] = log(df_test[:mean_wage])
		df_test[:log_starting_wage_share] = log(df_test[:starting_wage_share])

		##############
		#Scatterplots:
		figure("bargaining power test 1",figsize=(10,10))
		scatter(df_test[:mean_productivity], df_test[:D9_D1])
		OLS = glm(D9_D1 ~ mean_productivity,df_test,Normal(),IdentityLink()) #ordinary OLC
		fit = predict(OLS)#fitted OLS line
		xlabel("Mean Productivity")
		ylabel("Mean D9/D1")
		plot(df_test[:mean_productivity],fit[:])
		savefig("figures/D9_D1_Mean_Productivity.png")

		figure("bargaining power test 2",figsize=(10,10))
		scatter(df_test[:mean_unemployment_rate], df_test[:D9_D1])
		OLS = glm(D9_D1 ~ mean_unemployment_rate,df_test,Normal(),IdentityLink()) #ordinary OLC
		fit = predict(OLS)#fitted OLS line
		xlabel("Mean Unemployment Rate")
		ylabel("Mean D9/D1")
		plot(df_test[:mean_unemployment_rate],fit[:])
		savefig("figures/D9_D1_Mean_Unemployment_Rate.png")

		figure("bargaining power test 3",figsize=(10,10))
		scatter(df_test[:mean_wage], df_test[:D9_D1])
		xlabel("Mean Wage")
		ylabel("Mean D9/D1")
		OLS = glm(D9_D1 ~ mean_wage,df_test,Normal(),IdentityLink()) #ordinary OLC
		fit = predict(OLS)#fitted OLS line
		plot(df_test[:mean_wage],fit[:])
		savefig("figures/D9_D1_Mean_Wage.png")


		figure("bargaining power test 4",figsize=(10,10))
		scatter(df_test[:starting_wage_share], df_test[:D9_D1])
		xlabel("Share of employed workers in the starting wage")
		ylabel("Mean D9/D1")
		OLS = glm(D9_D1 ~ starting_wage_share, df_test,Normal(),IdentityLink()) #ordinary OLC
		fit = predict(OLS)#fitted OLS line
		plot(df_test[:starting_wage_share],fit[:])
		savefig("figures/D9_D1_Mean_share_employed_wokers_starting_wage.png")

		figure("bargaining power test 5",figsize=(10,10))
		scatter(df_test[:mean_unemployment_rate], df_test[:starting_wage_share])
		xlabel("Mean unemployment rate")
		ylabel("Share of employed workers in the starting wage")
		OLS = glm(starting_wage_share ~ mean_unemployment_rate, df_test,Normal(),IdentityLink()) #ordinary OLC
		fit = predict(OLS)#fitted OLS line
		plot(df_test[:mean_unemployment_rate],fit[:])
		savefig("figures/Mean_share_employed_wokers_starting_wage_Mean_unemployment_rate.png")

		#############
		#Regressions:
		#############
		# productivity:
		OLS = glm(log_D9_D1 ~ log_mean_productivity,df_test,Normal(),IdentityLink())
		println(OLS)
		# unemployment:
		OLS = glm(log_D9_D1 ~ log_mean_unemployment_rate,df_test,Normal(),IdentityLink())
		println(OLS)
		# wages:
		OLS = glm(log_D9_D1 ~ log_mean_wage,df_test,Normal(),IdentityLink())
		println(OLS)
		# share of employed workers in the starting wage:
		OLS = glm(log_D9_D1 ~ log_starting_wage_share,df_test,Normal(),IdentityLink())
		println(OLS)
		# More unemployed, more employed workers in the starting wage
		OLS = glm(log_starting_wage_share ~ log_mean_unemployment_rate,df_test,Normal(),IdentityLink())
		println(OLS)
	end

	#Function that looks whether the file "tables/bargaining_power_test.csv" exists.
	#If no, create it
	#Then anlyse the output of the N simulated economies to see the determinants of 
	#wage inqualities in the model
	function create_and_analyse_bargaining_power()

		#look whether the file "tables/bargaining_power_test.csv" exists:
		if isfile(string(path_table, "bargaining_power_test.csv")) == false
			println("Simulating N economies")
			df_test = Test_bargaining_power() #use the default values
		else
			df_test = readtable(string(path_table, "bargaining_power_test.csv"))
		end

		#Analyse the determinants of wage inequalities:
		println("Analyse N economies")
		analyse_bargaining_power(df_test)

	end

end