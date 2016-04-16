    ####################################
    # "On the Dynamics of Unemployment and Wage Distributions"
    # Paper from Jean Marc Robin (2011)
    # Available here: https://dl.dropboxusercontent.com/u/8875503/Research/ecta9070.pdf
    ####################################
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

    include("Copulas.jl") #From Florian Oswald: https://github.com/floswald/Copulas.jl/blob/master/main.jl


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

    ############
    # Parameters
    ############
    # Define global variables that I pass to functions

    # Values of the parameters:
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
    # AR1 parameter
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

    G = zeros(N,M) #intialization
    p = zeros(N,M)

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
        
            dif = norm(up_plus1 - up)          

            up = up_plus1

            its = its + 1
            print(its)
                    
            if its > maxits
                break
            end

        end

        Si_m = up;
        
        ##############################################
        # Create the wages by value function iteration
        ##############################################
        # 1. Wage low

        W_low = zeros(N,N,M); #Use mutlidimensional arrays
        W_low_star = zeros(N,N,M);

        up_ini = zeros(N,N,M);
        down_ini = zeros(N,N,M);

        ###########################
        # Pre caculates some values:

        Z = zeros(N,N,M);

            for m=1:M
                for i = 1:N
                    for k = 1:N
                    
                    Z[k,i,m] = zi_m_grid[i,m] - zi_m_grid[k,m]
                
                    #initialization
                    up_ini[k,i,m] = Z[k,i,m];

                        a = max(up_ini[k,i,m],0)
                        down_ini[k,i,m] = min(a,Si_m[i,m])

                    end
                end
            end


function calculate_W_low(m, N, M, discount, Markov, up_ini, down_ini, Z, Si_m)

        #########################
        tol = 0.1;
        maxits = 100;
        compare = zeros(N,1);
        a = 0

        #move along workers' types
            
            its = 1
            dif = tol+tol
          
            up = up_ini #initial value
            up_plus1 = zeros(N,N)

            down = down_ini;
            down_star = down_ini;
 

            while dif>tol 
                
                #move along the columns
                for i=1:N
                    for k = 1:N

                        a = max(up[k,i],0)
                        down[k,i] = min(a,Si_m[i,m])
                     
                        sum = 0

                        for j = 1:N
                            if Si_m[j,m] > 0
                                sum = sum + (Markov[k,j] - Markov[i,j])*(lambda1*Si_m[j,m]+(1-lambda1)*(down[j,i]))
                            end
                        end
                        
                        up_plus1[k,i] = (Z[k,i,m])  + discount*sum

                        a = max(up_plus1[k,i],0)
                        down_star[k,i] = min(a, Si_m[i,m])  
                    end
                end
           
                dif = vecnorm(up_plus1[:,:]- up[:,:],2)

                up[:,:] = up_plus1[:,:]
                        
                its = its + 1 #show the iteration number
                        
                if its > maxits
                    break
                end
            end

            #W_low[:,:,m] = up[:,:,m] 

    #println(dif) #show the current worker type
    return [up; down_star] # concatenate vertically and return the result
end

##############################################
# Create the wages by value function iteration
##############################################
# 2. Wage high
up_high_ini = zeros(N,N,M);
down2_ini = zeros(N,N,M);

# Initialization:
W_high = zeros(N,N,M);
W_high_star = zeros(N,N,M);

###########################
# Pre caculates some values:
Q = zeros(N,N,M);

    for m=1:M
        for i = 1:N
            for k = 1:N     
                Q[k,i,m] = Z[k,i,m] + Si_m[i,m]
                
                #initialization
                up_high_ini[k,i,m] = Q[k,i,m];

                    a = max(up_high_ini[k,i,m],0);
                    down2_ini[k,i,m] = min(a, Si_m[i,m]);
            end
        end
    end


function calculate_W_high(m, N, M, discount, Markov, up_high_ini, down2_ini, Q, Si_m)

    #initialization:
    tol = 0.1;
    maxits = 100;
    its = 1;
    dif = tol+tol;
      
    up_high = up_high_ini;
    up_plus1_high = zeros(N,N);

    down = down2_ini;
    down_star = down2_ini;
    a = 0

    while dif>tol 
            
        #move along the columns
        for i=1:N
            for k=1:N

                a = max(up_high[k,i],0);
                down[k,i] = min(a, Si_m[i,m]);
                   
                sum = 0;
                for j = 1:N
                    if Si_m[j,m] > 0
                     sum = sum + (Markov[k,j] - Markov[i,j])*(lambda1*Si_m[j,m]+(1-lambda1)*(down[j,i])) ;
                    end
                 end
                
                up_plus1_high[k,i] =  Q[k,i,m] + discount*sum;

                a = max(up_plus1_high[k,i], 0);

                down_star[k,i] = min(a, Si_m[i,m]);
            end
        end
       
            dif = vecnorm(up_plus1_high[:,:] - up_high[:,:],2)
                    
            up_high[:,:] = up_plus1_high[:,:];
                    
            its = its + 1

            if its > maxits
                break
            end    
    end

    return [up_high; down_star] # concatenate vertically and return the result
end
