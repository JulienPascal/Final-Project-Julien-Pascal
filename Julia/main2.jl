    ####################################
    # "On the Dynamics of Unemployment and Wage Distributions"
    # Paper from Jean Marc Robin (2011)
    # Available here: https://dl.dropboxusercontent.com/u/8875503/Research/ecta9070.pdf
    ####################################
    # Coded by Julien Pascal
  

    tic()
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

    #########################
    function gt_plus1_high_f(ut_m,gt_wi_low,gt_wi_high,i, lm, delta, lambda1, M, W_low, W_high, Si_m)

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

    #function that calculates the measure of workers at low wages
    function gt_plus1_low_f(ut_m,gt_wi_low,gt_wi_high,i, lm, lambda0, delta, lambda1, M, W_low, W_high, Si_m)
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

    #Calculate the low wage
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
                                                    

    ###############################################
    # Load functions if already in the folder
    # Otherwise create them through value function 
    # iteration

    create_surplus_function = 0
    create_worker_surpluses_low = 0
    create_worker_surpluses_high = 0
    create_unemployment_value = 0


    if isfile(string(path,"/match_surplus.jld")) 
        #Si_m = load(string(path,"/match_surplus.jld"))
        Si_m = jldopen(string(path,"/match_surplus.jld"), "r") do file
            read(file, "Si_m")
        end
    else                                          
        create_surplus_function = 1
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
    if create_surplus_function == 1
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

        Si_m = up
        
        #save
        jldopen(string(path,"/match_surplus.jld"), "w") do file
           write(file, "Si_m", Si_m) 
        end
    end

if isfile(string(path,"/W_low_star.jld")) 
    W_low_star = jldopen(string(path,"/W_low_star.jld"), "r") do file
            read(file, "W_low_star")
    end
else
    error("Run the file multi_processors.jl to built the wage surpluses")
end

if isfile(string(path,"/W_low.jld")) 
    W_low = jldopen(string(path,"/W_low.jld"), "r") do file
            read(file, "W_low")
    end
else
    error("Run the file multi_processors.jl to built the wage surpluses")
end

if isfile(string(path,"/W_high_star.jld")) 
    W_high_star = jldopen(string(path,"/W_high_star.jld"), "r") do file
            read(file, "W_high_star")
    end
else
    error("Run the file multi_processors.jl to built the wage surpluses")
end

if isfile(string(path,"/W_high.jld")) 
     W_high = jldopen(string(path,"/W_high.jld"), "r") do file
            read(file, "W_high")
    end
else
    error("Run the file multi_processors.jl to built the wage surpluses")
end


###########################
# Simulation of the economy
###########################
n_years = 2500; # years to simulate
number_periods = n_years*4; # one period = one quarter

discard = floor(number_periods/10); #get rid of the first 10th observations
 
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

wi_m_low_r = zeros(number_periods,M); #row for the period, column for ability;
wi_m_high_r = zeros(number_periods,M); #row for the period, column for ability

gt_wi_low_r = zeros(number_periods,M); #row for the period, column for ability;
gt_wi_high_r = zeros(number_periods,M); #row for the period, column for ability;

wi_m_r = zeros(number_periods,M);#row for the period, column for ability;
wi_r = zeros(number_periods,1); #mean wage

# wage per decile:
    # rows = time period
    # columns = percentile. ex: first row = 10th decile, 2nd row= 20th decile
w_p = zeros(number_periods,9); #initialization
    
#measure of unemployed workers:
m_unemployed_r = zeros(number_periods,M); #0 unemployed at t = 0

#Indexes to find the percentiles in the distribution of workers
p_10 = 43;
p_20 = 65;
p_25 = 75;
p_30 = 84;
p_40 = 103;
p_50 = 122;
p_60 = 143;
p_70 = 167;
p_80 = 197;
p_90 = 239;

#unemployment by education:
#25th percentile:
u_25_p = zeros(number_periods,1);
#50th percentile:
u_50_p = zeros(number_periods,1);
#90th percentile
u_90_p = zeros(number_periods,1);


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
    gt_wi_high_r[t+1,:] = gt_plus1_high_f(ut_m_r[t,:]',gt_wi_low_r[t,:]',gt_wi_high_r[t,:]',y_index_r[t,1], lm, delta, lambda1, M, W_low, W_high, Si_m);

    #measure of unemployed workers at the end of period t:
    # = mesure of workers with ability m minus employed people
    m_unemployed_r[t+1,:] = (transpose(lm[:,1]) - (gt_wi_low_r[t+1,:]' + gt_wi_high_r[t+1,:]'));

    #calculate the average wage by worker type:
    #weight by the measure of workers with low/high wages:
    wi_m_r[t,:] = transpose((wi_m_low_r[t,:].*gt_wi_low_r[t+1,:] + wi_m_high_r[t,:].*gt_wi_high_r[t+1,:])./(gt_wi_low_r[t+1,:]+gt_wi_high_r[t+1,:]));
    
    ####################
    # mean wage
    # calculate recursively the mean wage:
        wi_r[t,1]= 0; #initialization
        csum = 0; #initialisation
        for i=1:M
            if (isnan(wi_m_r[t,i])==false) #do not take into account unemployed workers
            wi_r[t,1] = wi_r[t,1] + wi_m_r[t,i]*(gt_wi_low_r[t+1,i]+gt_wi_high_r[t+1,i]); #weight by the measure of workers at the given wage
            csum = csum + (gt_wi_low_r[t+1,i]+gt_wi_high_r[t+1,i]); 
            end
        end
        wi_r[t,1] = wi_r[t,1]/csum; 
    ######################

    # New shock from the markov transition matrix:
    r = rand(); #draw a number from the uniform distribution on 0, 1
    
    # I use the Markov transition matrix previously defined:
    prob = Markov[y_index_r[t,1],:];
    
    #I stock the index in the Markov matrix, as well as the value of the
    #shock y
    y_index_r[t+1,1] = sum(r .>= cumsum([0; prob]));

    y_r[t+1,1] = ygrid[y_index_r[t+1,1], 1];
end

############################
# Analysis of the simulation
############################

#Plot the surplus function:
        #=
        xgrid_plot = repmat(xgrid',N,1)
        ygrid_plot = repmat(ygrid,1,M)

        fig = figure("pyplot_surfaceplot",figsize=(10,10))
        ax = fig[:add_subplot](2,1,1, projection = "3d") 
        ax[:plot_surface](xgrid_plot, ygrid_plot, Si_m, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25) 
        xlabel("X") 
        ylabel("Y")
        title("Surface Plot")

        subplot(212) 
        ax = fig[:add_subplot](2,1,2) 
        cp = ax[:contour](xgrid_plot, ygrid_plot, Si_m, colors="black", linewidth=2.0) 
        ax[:clabel](cp, inline=1, fontsize=10) 
        xlabel("X") 
        ylabel("Y")
        title("Contour Plot")
        tight_layout()
        =#

# Plot productivity match:
figure("1",figsize=(10,10))
for m=1:10:M
    plot(ygrid,p[:,m])
end
xlabel("Aggregate State") 
ylabel("Productivity")
title("Match Productivity by Ability")

figure("2",figsize=(10,10))
time_plot = discard:(number_periods-1);
# Plot the shock:
plot(time_plot ,y_r[discard:(number_periods-1),1])
xlabel("Periods") 
ylabel("Aggregate shock")
title("Aggregate shock over time")

figure("3",figsize=(10,10))
#Plot aggregate unemployment:
plot(time_plot,ut_r[discard:(number_periods-1),1])
xlabel("Periods") 
ylabel("Unemployment rate")
title("Aggregate Unemployment rate over time")

figure("4",figsize=(10,10))
#Plot exit rate from unemployment:
plot(time_plot,ft_r[discard:(number_periods-1),1])
xlabel("Periods") 
ylabel("ft")
title("Exit rate from unemployment")

figure("5",figsize=(10,10))
#Plot job destruction rate:
plot(time_plot, st_r[discard:(number_periods-1),1])
xlabel("Periods") 
ylabel("st")
title("Job destruction rate")

#plot the average wage
figure("6",figsize=(10,10))
plot(time_plot, wi_r[discard:(number_periods-1),1]);
title("Mean Wage")
xlabel("Periods")
ylabel("wage")


#Unemployment rate by education level
figure("7",figsize=(10,10))
# 25% low skills
scatter(ut_r[discard:(number_periods-1),1], u_25_p[discard:(number_periods-1),1], alpha=0.4, color="green") 
# 50% low skills
scatter(ut_r[discard:(number_periods-1),1], u_50_p[discard:(number_periods-1),1], alpha=0.7, color="red")
# 90% low skills
scatter(ut_r[discard:(number_periods-1),1], u_90_p[discard:(number_periods-1),1], alpha=1)
title("Unemployment Rate Among Various Skill Groups")
xlabel("overall unemployment rate")
ylabel("unemployment rate by skill")

#########################
#Dynamics of wage decile:
#########################

###############################################
# Calculate wage deciles and interdecile ratios 

trim = floor(Int16,0.5*discard); #to gain time, calculate for fewer periods

D5_D1 = zeros(number_periods,1);
D9_D1 = zeros(number_periods,1);
D9_D5 = zeros(number_periods,1);

for t = (number_periods-trim):(number_periods-1)
    per = 0; #intialization

    for i=1:9

        #choose the decile of interest
        per = per + 0.1;
        a = 0; #intialization
        sump = 0;
        
        while ((sump<per) & (a<500))
                
              #initialize values:
              a = a+1;
              sump=0;
              csum=0;
              
              #calculate a sum
              for m = 1:a #loop over ability levels
                      if (((wi_m_r[t,m] < wi_m_r[t,a]) & (isnan(wi_m_r[t,m]) == false))) #take into account only employed workers
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

time_plot2 = (number_periods-trim):(number_periods-1);
#Plot wage deciles:
figure("8",figsize=(10,10))
for m = 1:9 #loop over the deciles
    plot(time_plot2, w_p[(number_periods-trim):(number_periods-1),m])
end

xlabel("Periods")
ylabel("wage")
title("Dynamics of Wage Deciles")

#Plot Interdecile ratios:
figure("9",figsize=(10,10))
plot(time_plot2, D5_D1[(number_periods-trim):(number_periods-1),1], color="green")
plot(time_plot2, D9_D1[(number_periods-trim):(number_periods-1),1], color="red")
plot(time_plot2, D9_D5[(number_periods-trim):(number_periods-1),1])
xlabel("Periods")
ylabel("Inter-decile ratios")
title("Wage Inequalities")


#Summary statistics:
mean(y_r[discard:(number_periods-1),1])
mean(ut_r[discard:(number_periods-1),1])
mean(ft_r[discard:(number_periods-1),1])
mean(st_r[discard:(number_periods-1),1])
mean(wi_r[discard:(number_periods-1),1])


std(log(y_r[discard:(number_periods-1),1]))
std(log(ut_r[discard:(number_periods-1),1]))
std(log(ft_r[discard:(number_periods-1),1]))
std(log(st_r[discard:(number_periods-1),1]))
std(log(wi_r[discard:(number_periods-1),1]))

skewness(log(y_r[discard:(number_periods-1),1]))
skewness(log(ut_r[discard:(number_periods-1),1]))
skewness(log(ft_r[discard:(number_periods-1),1]))
skewness(log(st_r[discard:(number_periods-1),1]))
skewness(log(wi_r[discard:(number_periods-1),1]))

kurtosis(log(y_r[discard:(number_periods-1),1]))
kurtosis(log(ut_r[discard:(number_periods-1),1]))
kurtosis(log(ft_r[discard:(number_periods-1),1]))
kurtosis(log(st_r[discard:(number_periods-1),1]))
kurtosis(log(wi_r[discard:(number_periods-1),1]))

toc()


