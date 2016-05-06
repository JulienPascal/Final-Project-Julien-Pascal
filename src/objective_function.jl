################################
# Coded by Julien Pascal
# Last modification : 16/04/2016
# Replication of the the following paper:

####################################################################################
# "On the Dynamics of Unemployment and Wage Distributions"
# Paper from Jean Marc Robin (2011)
# Available here: https://sites.google.com/site/jmarcrobin/research/publications
####################################################################################

##########################################################################################
# Objective function that is called by the function MSM in the file estimate_parameters.jl
# To be used by the package NLopt
function objective_function(parameters::Vector, grad::Vector, Weighting_matrix::Matrix, random_numbers::Vector, b0::Vector)


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



    ############
    # Parameters
    ############
    
    #parameters to estimate:
    """
    z0 = parameters[1];
    sigma = parameters[2];
    pho = parameters[3];
    lambda0 = parameters[4];
    eta =  parameters[5];
    mu =  parameters[6];
    x_lower_bound = parameters[7] #lower bound of the ability level
    delta = parameters[8] #exogenous separation rate
    """

    """
    z0 = parameters[1];
    sigma = parameters[2];
    lambda0 = parameters[3];
    eta =  parameters[4];
    mu =  parameters[5];
    x_lower_bound = parameters[6] #lower bound of the ability level
    alpha = parameters[7]
    delta = parameters[8]

    pho = 0.94
    k = 0.12;
    lambda1  = k*lambda0;
    #x_lower_bound = 0.73
    #alpha = 0.64
    tau = 0.5
    #delta = 0.042 # exogenous layoff rate

    r = 0.05/4 # interest rate
    discount = (1 - delta)/(1 + r)
    epsilon = 0.002
    """

    z0 = parameters[1];
    sigma = parameters[2];
    lambda0 = parameters[3];
    eta =  parameters[4];
    mu =  parameters[5];
    x_lower_bound = parameters[6] #lower bound of the ability level

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

        # tol = 0.0001
        tol = 0.01; #increasing the tolerance to this one speed up the evaluation of the objective function by 10%
        maxits = 300;
        dif = tol+tol;
        its = 1;

        #initialization:
        up = ones(N,M);
        up_plus1 = zeros(N,M);
        compare = zeros(N,M);


        while dif>tol 

            @fastmath up_plus1 =  G + discount*Markov*max(up,compare)
        
            @fastmath dif = norm(up_plus1 - up);      

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
    #r = rand(); #draw a number from the uniform distribution on 0, 1
    r = random_numbers[t]
    
    # I use the Markov transition matrix previously defined:
    prob = Markov[y_index_r[t,1],:];
    
    #I stock the index in the Markov matrix, as well as the value of the
    #shock y
    y_index_r[t+1,1] = sum(r .>= cumsum([0; prob]));

    y_r[t+1,1] = ygrid[y_index_r[t+1,1], 1];
end


#Check for odd values. These entities cannot be negative or zero:
for i = discard:(number_periods-1)
    if ut_r[i,1]<=0
        return Inf #+infinity of type Float64
    elseif ft_r[i,1]<=0
        return Inf
    elseif st_r[i,1]<=0
        return Inf
    end
end 

# Calculate the moments from the simulation: 
mean_prod = mean(y_r[discard:(number_periods-1),1]) # Production
std_prod = std(log(y_r[discard:(number_periods-1),1]))

mean_unemployment = mean(ut_r[discard:(number_periods-1),1]) #Unemployment
std_unemployment = std(log(ut_r[discard:(number_periods-1),1]))

mean_exit_rate = mean(ft_r[discard:(number_periods-1),1]) #job finding rate ft
std_exit_rate = std(log(ft_r[discard:(number_periods-1),1]))

mean_job_destruction_rate = mean(st_r[discard:(number_periods-1),1]) #job separation rate st
std_job_destruction_rate = std(log(st_r[discard:(number_periods-1),1]))


#Check for odd values. Don't want an economy with constant values:
if std_prod == 0
    return Inf
elseif std_unemployment == 0
    return Inf
elseif std_exit_rate == 0
    return Inf
elseif std_job_destruction_rate == 0
    return Inf
end


#kurtosis_unemployment = kurtosis(log(ut_r[discard:(number_periods-1),1])) + 3 #returns excess kurtosis, so I add 3

# Classic
#b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; kurtosis_unemployment; mean_exit_rate]; #simulated moments
#b0 = [ 1; 0.0226;  0.058;  0.22; 2.65; 0.78]; #moments to match from the original paper
#b0 = [ 1; 0.023;  0.059;  0.218; 2.405; 0.762]; #moments to match

# New one
#b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; mean_exit_rate; std_exit_rate]; #simulated moments
#b0 = [ 1; 0.023;  0.059;  0.218; 0.762; 0.085] #new one

# New one 2:
#b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; kurtosis_unemployment; mean_exit_rate; std_exit_rate] #simulated moments
#b0 = [ 1.0; 0.023;  0.059;  0.210; 2.374; 0.761; 0.085]; #moments to match

# New one 3: replace the kurtosis of unemployment rate by the std of the job destruction rate
# !!! Have to adjust other function:
#b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; mean_job_destruction_rate; std_job_destruction_rate ; mean_exit_rate; std_exit_rate] #simulated moments
#b0 = [ 1.0; 0.023;  0.059;  0.210; 0.156; 0.761; 0.085]; #moments to match

# mean productivity, std productivity, mean unemployment rate, std unemployment rate, mean job finding rate, std job finding rate, mean job separation rate, std separation rate
#b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; mean_exit_rate; std_exit_rate; mean_job_destruction_rate; std_job_destruction_rate] #simulated moments

b1 = [mean_prod; std_prod; mean_unemployment; std_unemployment; mean_exit_rate; std_exit_rate; mean_job_destruction_rate] #simulated moments
#b0 = [1.0; 0.023;  0.059;  0.210; 0.761; 0.085; 0.046; 0.156]

db = (b1-b0) 
distance = sum(transpose(db)*Weighting_matrix*db) #use the function "sum" because the output has to be a scalar, not a 1 times 1 matrix. 

return distance
end

"""
using PyPlot, Distributions, StatsFuns, JLD, HDF5, DataFrames, Copulas, NLopt, BlackBoxOptim
path_main = "/home/julien/Final-Project-Julien-Pascal"
path_table = string(path_main,"/tables/")  #the path tables
path_surpluses = string(path_main,"/surpluses/")#path to precalculated objects, to speed up the process
file_random_numbers =  jldopen(string(path_surpluses,"random_numbers.jld"), "r")
random_numbers = read(file_random_numbers, "random_numbers")
Weighting_matrix = eye(8,8)
#params_opt = [0.77, 0.023, 0.94, 0.99, 2.00, 5.56, 0.73, 0.046]

#["z0", "sigma", "lambda0", "eta", "mu", "x_lower_bar", "alpha"]
params_opt = [0.77, 0.023, 0.99, 2.00, 5.56, 0.73, 0.64]

b0 = [1.0; 0.023;  0.059;  0.210; 0.761; 0.085; 0.046; 0.156]


objective_function(params_opt,params_opt,Weighting_matrix,random_numbers,b0)
"""
