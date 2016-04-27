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


    using PyPlot, Distributions, StatsFuns, JLD, HDF5, DataFrames, Copulas, NLopt
   	include("objective_function.jl") #To use package NLopt
   	include("simulated_annealing/samin.jl")#simulated annealing module

    export MSM

    function MSM(routine = "NLopt")
    	tic()

		params_name = ["z0", "sigma", "rho", "lambda0", "eta", "mu"]
		params_opt = [0.77, 0.023, 0.94, 0.99, 2.00, 5.56] #optimal parameters in the paper
		params0 = [0.77, 0.023, 0.94, 0.99, 2.00, 5.5] #starting values for the optimization

		if routine == "NLopt"
		########################################
		# Two Steps procedure:
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

		elseif routine == "SA"
		######################
		# Simulated Annealing
		######################

			######################################################
			#Apparently, I have to redefine the objective function 
			#here. Otherwise the simulated annealing will not work

			function objective_function_SA(parameters)
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
			    ##########################
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

			    function ut_plus1_f(St_m, ut_m, delta, lambda0, M)
			    #Calculates the next period's unemployment rate

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

			    ############
			    # Parameters
			    ############
			   
			    #parameters to estimate:
			    z0 = parameters[1];
			    sigma = parameters[2];
			    pho = parameters[3];
			    lambda0 = parameters[4];
			    eta =  parameters[5];
			    mu =  parameters[6];

			    k = 0.12;
			    lambda1  = k*lambda0;
			    s = 0.42
			    x_lower_bound = 0.73
			    alpha = 0.64
			    tau = 0.5
			    delta = 0.042 # "4.2# exogenous layoff rate"
			    r = 0.05/4 # interest rate
			    discount = (1 - delta)/(1 + r)
			    epsilon = 0.002

			    # grid parameters
			    N = 100 #number of states
			    M = 500 #number of ability levels

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
			    # AR1 parameter:

			    cop = Copulas.NormalCopula(2,pho)


			    #Gaussian copula pdf:
			    for j=1:N #move along the column:
			        for i=1:N #move along the lines:
			            Markov[i,j] = Copulas.dnormCopula([agrid[i] agrid[j]], cop)[];
			        end   
			    end

			    #normalize so that each row sum to 1:
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
			                                                
			    ##########################################################
			    # Calculate the match surplus by value function iteration:
			    # N times M matrices
			    # N = aggregate level
			    # M = ability level

			    G = zeros(N,M); #intialization
			    p = zeros(N,M);

			    # Pre calculate values:
			    for i = 1:N
			        for m = 1:M
			            G[i,m] = yi_m(i,m, ygrid, xgrid) - zi_m_grid[i,m];
			            p[i,m] = yi_m(i,m, ygrid, xgrid);
			        end
			    end

			    # N times M matrix
			    # column index indicates the workers' type
			    # row index indicates the shock index 

			        Si_m = zeros(N,M);

			        tol = 0.0001;
			        maxits = 300;
			        dif = tol+tol;
			        its = 1;

			        #initialization:
			        up = ones(N,M);
			        up_plus1 = zeros(N,M);
			        compare = zeros(N,M);


			        while dif>tol 

			            up_plus1 =  G + discount*Markov*max(up,compare)
			        
			            dif = norm(up_plus1 - up);      

			            up = up_plus1;

			            its = its + 1;
			                    
			            if its > maxits
			                break
			            end
			        end

			        Si_m = up;
			        
			###########################
			# Simulation of the economy
			###########################
			n_years = 2500; # years to simulate
			number_periods = n_years*4; # one period = one quarter

			discard = round(Int,floor(number_periods/10)); #get rid of the first 10th observations
			 
			y_index_r = Array{Int16}(number_periods,1); #store the indexes of the shock 
			y_r = Array{Float64}(number_periods,1) #store value of the shock

			#initial shock:
			y_index_r[1,1] = 50; #initial shock index
			y_r[1,1] = ygrid[y_index_r[1,1], 1]; #initial shock value

			#Initialization:
			ut_m_r = ones(number_periods,M); #row for the period, column for ability
			St_m_r = Array{Float64}(number_periods,M); #row for the period, column for ability

			ut_r = Array{Float64}(number_periods,1);

			ft_r = Array{Float64}(number_periods,1);
			qt_r = Array{Float64}(number_periods,1);
			st_r = Array{Float64}(number_periods,1);

			######################
			# Loop over the economy:
			for t=1:(number_periods-1) 
			    
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

			    # New shock from the markov transition matrix:
			    r = rand(); #draw a number from the uniform distribution on 0, 1
			    
			    # I use the Markov transition matrix previously defined:
			    prob = Markov[y_index_r[t,1],:];
			    
			    #I stock the index in the Markov matrix, as well as the value of the
			    #shock y
			    y_index_r[t+1,1] = sum(r .>= cumsum([0; prob]));

			    y_r[t+1,1] = ygrid[y_index_r[t+1,1], 1];
			end

			#Take into account mean productivity,

			    #Check for odd values:
			    for i = discard:(number_periods-1)
			        if (isless(ut_r[i,1],0))
			            return Inf
			        end
			    end 

			#Moments to match:
			mean_prod = mean(y_r[discard:(number_periods-1),1]) # Production
			std_prod = std(log(y_r[discard:(number_periods-1),1]))

			mean_unemployment = mean(ut_r[discard:(number_periods-1),1]) #Unemployment
			std_unemployment = std(log(ut_r[discard:(number_periods-1),1]))
			kurtosis_unemployment = kurtosis(log(ut_r[discard:(number_periods-1),1])) + 3 #returns excess kurtosis, so I add 3

			mean_exit_rate = mean(ft_r[discard:(number_periods-1),1]) #exit rate from unemployment

			b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; kurtosis_unemployment; mean_exit_rate]; #simulated moments
			b0 = [ 1; 0.0226;  0.058;  0.22; 2.65; 0.78]; #moments to match

			Weighting_matrix = eye(6,6)

			#db = (b1-b0)./b0;
			#distance = dot(db,db);
			db = (b1-b0) 
			distance = sum(transpose(db)*Weighting_matrix*db) #use the function "sum" because the output has to be a scalar, not a 1 times 1 matrix. 

			#calculate the 
			return distance
			end

			lb = [0.7, 0.01, 0.8, 0.8, 1.5, 5.0]

			initial=[0.70, 0.01, 0.9, 0.90, 2.0, 5.3]

			ub = [0.99, 0.99, 0.99, 0.99, 2.5, 6.0]

			xopt = samin(objective_function_SA, initial, lb, ub, maxevals = 300, verbosity=2) 
			#1537.63419783 seconds for 300 iterations

			# Save the result in a csv file:
			df = DataFrame()
			df[:Paramater] = params_name 
			df[:Value] = xopt[1]
			df[:Minf] = objective_function_SA(xopt[1])
			println("MSM results:")
			println(df)

			writetable("tables/estimated_parameters.csv", df)

		end

	toc() 
	end
end