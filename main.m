%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "On the Dynamics of Unemployment and Wage Distributions"
% Paper from Jean Marc Robin (2011)
% Available here: https://dl.dropboxusercontent.com/u/8875503/Research/ecta9070.pdf
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Coded by Julien Pascal
% Last modification : 22/01/2016

% clear the memory and all the graphs
clear all
close all

tic %start time watcher

%%%%%%%%%%%%
% Parameters
%%%%%%%%%%%%
% Define global variables that I pass to functions
global Si_m lm lambda0 lambda1 M N Markov
global epsilon z0 sigma pho s eta mu alpha
global x_lower_bound xgrid agrid ygrid zi_m_grid
global delta r tau discount
global W_low W_low_star W_high W_high_star Ui_m

z0 = 0.77;
sigma = 0.023;
pho = 0.94;
lambda0 = 0.99;
lambda1  = 0.12*lambda0;
s = 0.42;
x_lower_bound = 0.73;
eta = 2.00 ;
mu = 5.56;
alpha = 0.64;

tau = 0.5;

delta = 0.042; % "4.2% exogenous layoff rate"
r = 0.05; % interest rate
discount = (1 - delta)/(1 + r);

epsilon = 0.002;

% grid parameters
N = 100;
M = 500;

% Define the grid for ability x
xgrid = transpose(linspace(x_lower_bound,x_lower_bound+1,M)); %create a grid

% Define the grid for match ability yi:
    % 1. a:
    agrid = transpose(linspace((0+epsilon),(1-epsilon),N));

    % 2. yi:
    %intialization:
    ygrid = zeros(N,1);
    for i=1:N
        ygrid(i,1) = logninv(agrid(i), 0, sigma);
    end

%Markov transition matrix:
Markov = NaN(N,N);

%Gaussian copula pdf:
for j=1:N %move along the column:
for i=1:N %move along the lines:
    Markov(i,j) = copulapdf('Gaussian',[agrid(i) agrid(j)], [pho]);
end   
end

%normalize so that each row sum to 1:
for i = 1:N
Markov(i,:) = Markov(i,:)./sum(Markov(i,:));
end

% distribution of workers on the grid x 
lm = zeros(M,1);

for i=1:M
    lm(i,1) = betapdf(xgrid(i,1)-x_lower_bound,eta,mu);
end

%normalization so that mass of workers sum up to 1:
lm(:,1) = lm(:,1)./sum(lm(:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%
%define match productivity:
yi_m = @(x_index,y_index) (ygrid(y_index)*xgrid(x_index));
%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%
% define opportunity cost of employment:
zi_m = @(x_index,y_index) (z0 + alpha*(yi_m(x_index,y_index) - z0));
%%%%%%%%%%%%%%%%%%%%%%%%%%

% calculate the opportunity cost along the grid of ability:
for i=1:N
    for m=1:M
        zi_m_grid(m,i) = zi_m(m,i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load functions if already in the folder
% Otherwise create them through value function 
% iteration
create_surplus_function = 0;
create_worker_surpluses_low = 0;
create_worker_surpluses_high = 0;
create_unemployment_value = 0;

if exist('match_surplus.mat', 'file') == 2
    Si_load = load('match_surplus.mat');
    Si_m = Si_load.Si_m;
    clear Si_load
else
    create_surplus_function  = 1;
end

if exist('W_low.mat', 'file') == 2 
    W_low_load = load('W_low.mat');
    W_low = W_low_load.W_low;
    clear W_low_load
    
else
    create_worker_surpluses_low = 1;
end

if exist('W_low_star.mat', 'file') == 2 
    W_low_star_load = load('W_low_star.mat');
    W_low_star = W_low_star_load.W_low_star;
    clear W_low_star_load
   
else
    create_worker_surpluses_low = 1;
end


if exist('W_high.mat', 'file') == 2
    W_high_load = load('W_high.mat');
    W_high = W_high_load.W_high;
    clear W_high_load
else
    create_worker_surpluses_high = 1;
end

if exist('W_high_star.mat', 'file') == 2 
    W_high_star_load = load('W_high_star.mat');
    W_high_star = W_high_star_load.W_high_star;
    clear W_high_star_load

else
    create_worker_surpluses_high = 1;
end

if exist('unemployment_surplus.mat', 'file') == 2 
    Ui_m_load = load('unemployment_surplus.mat');
    Ui_m = Ui_m_load.Ui_m;
    clear Ui_m_load

else
    create_unemployment_value = 1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the match surplus by value function iteration:

G = zeros(M,N); %intialization
p = zeros(M,N);

for m=1:M
    for i = 1:N
        G(m,i) = yi_m(m,i) - zi_m(m,i);
        p(m,i) = yi_m(m,i);
    end
end

% row index indicates the workers' type
% column index indicates the shock index 
if create_surplus_function == 1
Si_m = zeros(M,N);

tol = 0.01;
maxits = 100;
dif = tol+tol;
its = 1;

%initialization:
  up = ones(M,N);
  up_plus1 = zeros(M,N);
  compare = zeros(M,N);


while dif>tol 
    
   
       for i = 1:N
            
                
            sum = zeros(M,1);
            sum = sum + max(up(:,:),compare)*transpose(Markov(i,:));
            
            
        up_plus1(:,i) =  G(:,i) + discount*sum;
       
        end
    
    
    dif = norm(up_plus1(:,:) - up(:,:))
                
    up(:,:) = up_plus1(:,:);
                
    its = its + 1
                
                if its > maxits
                      break
                end
    
    
end

Si_m(:,:) = up(:,:);

figure
contour(Si_m(:,:),'ShowText','on') %contour plot of the surplus
xlabel('x')
ylabel('y')

save('match_surplus.mat', 'Si_m')
end


if create_worker_surpluses_low == 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the wages by value function iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Wage low
%
% Initialization:
W_low = zeros(N,N,M);
W_low_star = zeros(N,N,M);

create_z = 0;

if exist('Z.mat', 'file') == 2
    Z_load = load('Z.mat')
    Z = Z_load.Z;
    clear Z_load
else
    create_z  = 1;
end

up = zeros(N,N,M);
up_plus1 = zeros(N,N,M);
%%%%%%%%%%%%%%%%%%%%
if create_z == 1
Z = zeros(N,N,M);

for m=1:M
    for i = 1:N
        for k = 1:N
            
        Z(k,i,m) = zi_m(m,i)  - zi_m(m,k);
        
        %initialization
        up(k,i,m) = Z(k,i,m);
        end
    end
end

save('Z.mat', 'Z');
else
    
   for m=1:M
       up(:,:,m) = Z(:,:,m);
   end
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%
tol = 0.1;
maxits = 100;
compare = zeros(N,1);

%move along workers' types
for m = 1:M
    
    its = 1;
    dif = tol+tol;
  
    down = zeros(N,N);
    while dif>tol 
        
        %move along the columns
        for i=1:N
            
            
            for k = 1:N
            down(k,i) = min(max(up(k,i,m),0),Si_m(m,i));
               
            sum = 0;
            
            for j = 1:N
            
                if Si_m(m,j) > 0
                sum = sum + (Markov(k,j) - Markov(i,j))*(lambda1*Si_m(m,j)+(1-lambda1)*(down(j,i))) ;
                end
            
            end
           
            
            up_plus1(k,i,m) = transpose(Z(k,i,m))  + discount*sum;
            W_low_star(k,i,m) = min(max(up_plus1(k,i,m) ,0),Si_m(m,i));
            end
        end
   
   m %show the current worker type
   dif = norm(up_plus1(:,:,m) - up(:,:,m))
                
   up(:,:,m) = up_plus1(:,:,m);
                
   its = its + 1 %show the iteration number
                
                if its > maxits
                break
                end
        
    end
    W_low(:,:,m) = up(:,:,m);
    
end

save('W_low.mat', 'W_low');
save('W_low_star.mat', 'W_low_star');

for m = 1:30:M
figure
contour(W_low(:,:,m),'ShowText','on')
xlabel('x')
ylabel('y')
end

for m = 1:30:M
figure
contour(W_low_star(:,:,m),'ShowText','on')
xlabel('x')
ylabel('y')
end

end

if create_worker_surpluses_high == 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the wages by value function iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Wage high
%
% Initialization:
W_high = zeros(N,N,M);
W_high_star = zeros(N,N,M);

create_z = 0;

if exist('Z.mat', 'file') == 2
    Z_load = load('Z.mat')
    Z = Z_load.Z;
    clear Z_load
else
    create_z  = 1;
end

up = zeros(N,N,M);
up_plus1 = zeros(N,N,M);
%%%%%%%%%%%%%%%%%%%%
if create_z == 1
Z = zeros(N,N,M);

for m=1:M
    for i = 1:N
        for k = 1:N
            
        Z(k,i,m) = zi_m(m,i)  - zi_m(m,k);
        
        %initialization
        up(k,i,m) = Z(k,i,m) + Si_m(m,i);
        end
    end
end

save('Z.mat', 'Z');
else
    
   for m=1:M
       for i = 1:N
       up(:,:,m) = Z(:,:,m) + Si_m(m,i);
       end
   end
   
end

%initialization:
tol = 0.1;
maxits = 100;

compare = zeros(N,1);
%move along workers' types
for m = 1:M
    
    its = 1;
    dif = tol+tol;
  
    down = zeros(N,N);
    while dif>tol 
        
        %move along the columns
        for i=1:N
            
            for k=1:N
            down(k,i) = min(max(up(k,i,m),0),Si_m(m,i));
               
            sum = 0;
            for j = 1:N

                if Si_m(m,j) > 0
                sum = sum + (Markov(k,j) - Markov(i,j))*(lambda1*Si_m(m,j)+(1-lambda1)*(down(j,i))) ;
                end
            
            end
           
            
            up_plus1(k,i,m) = Si_m(m,i) + Z(k,i,m)  + discount*sum;
            W_high_star(k,i,m) = min(max(up_plus1(k,i,m) ,0),Si_m(m,i));
            end
        end
   
   m
   dif = norm(up_plus1(:,:,m) - up(:,:,m))
                
   up(:,:,m) = up_plus1(:,:,m);
                
   its = its + 1
                
                if its > maxits
                break
                end
        

        
    end
    W_high(:,:,m) = up(:,:,m);
    
end

save('W_high.mat', 'W_high');
save('W_high_star.mat', 'W_high_star');

for m = 1:30:M
figure
contour(W_high(:,:,m),'ShowText','on')
xlabel('x')
ylabel('y')
end

for m = 1:30:M
figure
contour(W_high_star(:,:,m),'ShowText','on')
xlabel('x')
ylabel('y')
end

end



colorVec = hsv(M);
figure
a = 0;
hold on

for m=1:10:M
    a = a+1;
    plot(ygrid,p(m,:),'Color',colorVec(m,:))
    str_legend_workers{a} = sprintf('x=%g',xgrid(m)); %prepare the legend
end

title('Match Productivity by Ability')
xlabel('aggregate shock')
ylabel('match productivity')
legend(str_legend_workers);
hold off;
clear str_legend_workers




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create Value of unemployment
% Not necessary
%
if create_unemployment_value == 1
Ui_m = zeros(M,N);

tol = 0.01;
maxits = 300;
dif = tol+tol;
its = 1;

%initialization:
  up = ones(M,N);
  up_plus1 = zeros(M,N);
  compare = zeros(M,N);


while dif>tol 
    
   
       for i = 1:N
            
        sum = up(:,:)*transpose(Markov(i,:));
            
        up_plus1(:,i) =  zi_m_grid(:,i)  + (1/(1+r))*sum;
       
        end
    
    
    dif = norm(up_plus1(:,:) - up(:,:))
                
    up(:,:) = up_plus1(:,:);
                
    its = its + 1
                
                if its > maxits
                      break
                end
    
    
end

clear sum;

Ui_m(:,:) = up(:,:);

figure
contour(Ui_m(:,:),'ShowText','on')
title('Unemployment surplus')
xlabel('x')
ylabel('y')

save('unemployment_surplus.mat', 'Ui_m')

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation of the economy
%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_years = 2500; % years to simulate
number_periods = n_years*4; % one period = one quarter

discard = floor(number_periods/10); %get rid of the first 10th observations

%Calculate group unemployment rate:
% indexes of deciles of the population distribution:
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

%initilization:
u_10_p = zeros(number_periods,1);
u_20_p = zeros(number_periods,1);
u_25_p  = zeros(number_periods,1);
u_30_p = zeros(number_periods,1);
u_40_p = zeros(number_periods,1);
u_50_p = zeros(number_periods,1);
u_60_p = zeros(number_periods,1);
u_70_p = zeros(number_periods,1);
u_80_p = zeros(number_periods,1);
u_90_p = zeros(number_periods,1);

% wage per decile:
    % rows = percentile. ex: first row = 10th percentile, 2nd row= 20th
    % percentile
    % columns = time period
w_p = zeros(9,number_periods); %initialization
    
y_index_r = zeros(number_periods,1);
y_r = zeros(number_periods,1);

%initial shock:
y_index_r(1,1) = 50; %initial shock index
y_r(1,1) = ygrid(y_index_r(1,1) ,1); %initial shock value

%Initialization:
ut_m_r = zeros(M,number_periods);
St_m_r = zeros(M,number_periods);

wi_m_low_r = zeros(M,number_periods);
wi_m_high_r = zeros(M,number_periods);

wi_m_r = zeros(M,number_periods);

wi_r = zeros(number_periods,1); %mean wage

gt_wi_low_r = zeros(M,number_periods);
gt_wi_high_r = zeros(M,number_periods);


ut_r = zeros(number_periods,1);

ft_r = zeros(number_periods,1);
qt_r = zeros(number_periods,1);
st_r = zeros(number_periods,1);

%measure of unemployed workers:
m_unemployed_r = zeros(M,number_periods); % 0 unemployed at t = 0

%%%%%%%%%%%%%%%%%%%%%%
% Loop over the economy:
for t=1:number_periods
    
    % Calculate the aggregate unemployment
    ut_r(t,1) = dot(ut_m_r(:,t),lm);
    
    % Calculate the surplus given the actual value of the shock
    St_m_r(:,t) = Si_m(:,y_index_r(t,1));
    
    % Exit rate from unemployment:
    ft_r(t,1) = ft_f(St_m_r(:,t), ut_m_r(:,t) ,ut_r(t,1));
    
    % Job destruction rate:
    st_r(t,1) = st_f(St_m_r(:,t), ut_m_r(:,t) ,ut_r(t,1));
    
    % Law of motion of unemployment:
    ut_m_r(:,t+1) = ut_plus1_f(St_m_r(:,t),ut_m_r(:,t));
    
    % Calculate group unemployment rate:

    % 25 percentile:
     u_25_p(t+1,1) = transpose(lm(1:p_25,1))*ut_m_r(1:p_25,t+1)/sum(lm(1:p_25,1));

    % 50 percentile:
    u_50_p(t+1,1) = transpose(lm(1:p_50,1))*ut_m_r(1:p_50,t+1)/sum(lm(1:p_50,1));
    
    % 90 percentile
    u_90_p(t+1,1) = transpose(lm(1:p_90,1))*ut_m_r(1:p_90,t+1)/sum(lm(1:p_90,1));
    
    % Calculate the wages:
    wi_m_low_r(:,t) = wi_low_f(y_index_r(t,1)); % former-unemplpyed wage
    
    wi_m_high_r(:,t) = wi_high_f(y_index_r(t,1)); % promotion wage
    
    
    % measure of workers of ability m employed at low wage at the end of
    % period t
    gt_wi_low_r(:,t+1) = gt_plus1_low_f(ut_m_r(:,t),gt_wi_low_r(:,t),gt_wi_high_r(:,t),y_index_r(t,1));
    % measure of workers of ability m employed at high wage at the end of
    % period t
    gt_wi_high_r(:,t+1) = gt_plus1_high_f(ut_m_r(:,t),gt_wi_low_r(:,t),gt_wi_high_r(:,t),y_index_r(t,1));
    
    % measure of unemployed workers at the end of period t:
    % = mesure of workers with ability m minus employed people
    m_unemployed_r(:,t+1) = (lm(:,1) - (gt_wi_low_r(:,t+1) + gt_wi_high_r(:,t+1)));
    
    % calculate the average wage by worker type:
    % weight by the measure of workers with low/high wages:
    wi_m_r(:,t) = (wi_m_low_r(:,t).*gt_wi_low_r(:,t+1) + wi_m_high_r(:,t).*gt_wi_high_r(:,t+1))./(gt_wi_low_r(:,t+1)+gt_wi_high_r(:,t+1));
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % mean wage:
    wi_r(t,1) = 0;
    csum = 0;
    for i=1:M
        if (isnan(wi_m_r(i,t))==0)
        wi_r(t,1) = wi_r(t,1) + wi_m_r(i,t)*(gt_wi_low_r(i,t+1)+gt_wi_high_r(i,t+1));
        csum = csum + (gt_wi_low_r(i,t+1)+gt_wi_high_r(i,t+1));
        end
    end
    wi_r(t,1) = wi_r(t,1)/csum;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % New shock from the markov transition matrix:
    r = rand;
    
    % I use the Markov transition matrix previously defined:
    %prob = transpose(Markov(:,y_index_r(t,1)));
    prob = Markov(y_index_r(t,1),:);
    
    %I stock the index in the Markov matrix, as well as the value of the
    %shock y
    y_index_r(t+1,1) = sum(r >= cumsum([0, prob]));
    y_r(t+1,1) = ygrid(y_index_r(t+1,1) ,1);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis of the simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the aggregate shock value
figure
plot(y_r(discard:number_periods-1,1));
title('Aggregate Shock')
xlabel('time')
ylabel('aggregate shock')
print('Aggregate_Shock_Value','-dpng');

% plot the exit rate
figure
plot(ft_r(discard:number_periods-1,1));
title('Unemployment Exit Rate ft')
xlabel('time')
ylabel('ft')
print('Exit_Rate','-dpng');

% plot the job destruction rate
figure
plot(st_r(discard:number_periods-1,1));
title('Job Destruction Rate st')
xlabel('time')
ylabel('st')
print('Job_Destruction_Rate','-dpng');

% plot the aggregate unemployment level
figure
plot(ut_r(discard:number_periods-1,1));
title('Aggregate Unemployment Rate')
xlabel('time')
ylabel('unemployment rate')
print('Aggregate_Unemployment_rate','-dpng');

% plot the average wage
figure
plot(wi_r(discard:number_periods-1,1));
title('Mean Wage')
xlabel('time')
ylabel('wage')
print('Mean_aggregate_wage','-dpng');

% scatter plot unemployment rate/aggregate shock:
figure
scatter(y_r(discard:number_periods-1,1),ut_r(discard:number_periods-1,1))
title('Unemployment Rate and Aggregate Shock')
xlabel('aggregate shock')
ylabel('unemployment rate')
print('U_rate_and_shock','-dpng');

%Unemployment rate by education level
figure
hold on
scatter(ut_r(discard:number_periods-1,1),u_25_p(discard:number_periods-1,1))
scatter(ut_r(discard:number_periods-1,1),u_50_p(discard:number_periods-1,1))
scatter(ut_r(discard:number_periods-1,1),u_90_p(discard:number_periods-1,1))
scatter(ut_r(discard:number_periods-1,1),ut_r(discard:number_periods-1,1))
title('Unemployment Rate Among Various Skill Groups')
xlabel('overall unemployment rate')
ylabel('unemployment rate by skill')
legend('25% low','50% low', '90% low', '45 degree line')
hold off
print('U_by_edu_and_Overall_U','-dpng');

%%%%%%%%%%%%%%%%%%%%%%%%%
%Dynamics of wage decile:
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate wage deciles and interdecile ratios 
D5_D1 = zeros(number_periods,1);
D9_D1 = zeros(number_periods,1);
D9_D5 = zeros(number_periods,1);

trim = floor(2*discard); %to gain time, calculate for the last periods
for t = (number_periods-trim):(number_periods-1)
    p = 0; %intialization

    for i=1:9

        %choose the percentile of interest
        p = p + 0.1;
        a = 0; %intialization
        sump = 0;
        
         while (sump<p) & (a<500)
            
          %initialize values:
          a = a+1;
          sump=0;
          csum=0;
          
          %calculate a sum
          for m = 1:a
              
                  if ((wi_m_r(m,t) < wi_m_r(a,t)) & (isnan(wi_m_r(m,t)) == 0))
                   sump = sump + (gt_wi_low_r(m,t+1)+gt_wi_high_r(m,t+1)); %weight by the distribution of workers
                  end
                  
          end
          csum = sum(gt_wi_low_r(:,t+1) + gt_wi_high_r(:,t+1));
          
          if csum ~=0 
          sump = sump/csum;
          else
          sump = 0;  
          end
          
         end
        
        %store the value
        w_p(i,t) = wi_m_r(a,t);

    end
    
    %calculate interdecile ratio
    D5_D1(t,1) = w_p(5,t)/w_p(1,t);
    D9_D1(t,1) = w_p(9,t)/w_p(1,t);
    D9_D5(t,1) = w_p(9,t)/w_p(5,t);

end

% calculate the volatility of wages deciles:
    %initialization:
    std_w_p = zeros(9,1);

    %loop over the wages deciles:
    for i=1:9
        std_w_p(i,1) = std(w_p(i,(number_periods-trim):(number_periods-1)));
    end

% create a table with the volatility of wages deciles:
Title = {'Volatility (Actual)', 'Volatility (Simulated)'};
Type = {'P10'; 'P20';'P30';'P40';'P5O';'P60';'P70';'P80';'P90'};

%source: Heathcote, Perri and Violance, quoted in Robin 2011
Actual = [0.032; 0.023; 0.020; 0.018; 0.015; 0.014; 0.013; 0.011; 0.013];

input1.data = [Actual std_w_p(:,1)];
input1.tableColLabels = Title;
input1.tableRowLabels = Type;
input1.transposeTable = 0;
input1.dataFormatMode = 'column';
input1.dataFormat = {'%.4f',2}; %two digits precision
input1.dataNanString = '-';
input1.tableColumnAlignment = 'c';
% Switch table borders on/off:
input1.tableBorders = 1;
% LaTex table caption:
input1.tableCaption = 'Volatility of Wages Deciles';
% LaTex table label:
input1.tableLabel = 'Table 1';
% Switch to generate a complete LaTex document or just a table:
input1.makeCompleteLatexDocument = 1;
% call latexTable:
latex1 = latexTable(input1);
    
% Plot wage deciles:
colorVec = hsv(9);
figure
hold on
for i = 1:9
plot(w_p(i,(number_periods-trim):(number_periods-1)),'Color',colorVec(i,:))
end
xlabel('time')
ylabel('wage')
legend('D1','D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9')
title('Dynamics of Wage Deciles')
hold off
print('Dynamics_w_deciles','-dpng');

% Plot Interdecile ratios:
figure
hold on
plot(D5_D1((number_periods-trim):(number_periods-1),1),'Color',colorVec(1,:))
plot(D9_D1((number_periods-trim):(number_periods-1),1),'Color',colorVec(2,:))
plot(D9_D5((number_periods-trim):(number_periods-1),1),'Color',colorVec(3,:))
xlabel('time')
ylabel('Inter-decile ratios')
legend('D5/D1','D9/D1', 'D9/D5')
title('Wage Inequalities')
hold off
print('Dynamics_inter_d_ratios','-dpng');

%scatter plot unemployment rate by ability:
mean_unemployment_ability = zeros(M,1);
for m=1:M
    mean_unemployment_ability(m,1) = mean(ut_m_r(m,discard:number_periods-1));
end

figure
hold on

[ax,p1,p2] = plotyy(xgrid(:,1),lm(:,1),xgrid(:,1),mean_unemployment_ability(:,1),'semilogy','plot');
xlabel(ax(1),'ability') % label x-axis
ylabel(ax(1),'distribution of workers') % label left y-axis
ylabel(ax(2),'unemployment rate by ability') % label right y-axis
axis auto
axis([ax, p1, p2 ],[0.73 1.7 0 0.007 0 1])
title('Ability and Unemployment Rate')
print('Ability_and_U_rate','-dpng');


% Plot the low wage by ability
colorVec = hsv(M);
figure
hold on;
a=0;
clear str_legend_workers
for i=1:50:M
a = a+1;
plot(discard:number_periods-1, wi_m_low_r(i,discard:number_periods-1) ,'Color',colorVec(i,:))
str_legend_workers{a} = sprintf('x=%g',xgrid(i)); %prepare the legend

end
title('Starting Wage')
xlabel('time')
ylabel('starting wage')
legend(str_legend_workers);
hold off;
print('Low_wage','-dpng');

% Plot the high wage by ability
figure
hold on;
a=0;
clear str_legend_workers 
for i=1:50:M
a = a +1;
plot(discard:number_periods-1, wi_m_high_r(i,discard:number_periods-1) ,'Color',colorVec(i,:))
str_legend_workers{a} = sprintf('x=%g',xgrid(i)); %prepare the legend
end
title('Promotion Wage')
xlabel('time')
ylabel('promotion wage')
legend(str_legend_workers);
hold off;
print('High_wage','-dpng');

% Plot the mean wage by worker type
figure
hold on;
a=0;
clear str_legend_workers
for i=1:50:M
a=a+1;
plot(discard:number_periods-1, wi_m_r(i,discard:number_periods-1) ,'Color',colorVec(i,:))
str_legend_workers{a} = sprintf('x=%g',xgrid(i)); %prepare the legend
end
title('Mean Wage by Ability')
xlabel('time')
ylabel('mean wage')
legend(str_legend_workers);
hold off;

% plot the measure of workers at the high wage
figure
hold on;
a=0;
clear str_legend_workers
for i=1:10:M
a=a+1;
plot(discard:number_periods-1, gt_wi_high_r(i,discard:number_periods-1) ,'Color',colorVec(i,:))
str_legend_workers{a} = sprintf('x=%g',xgrid(i)); %prepare the legend
end
title('Measure of Workers at Promotion Wages')
xlabel('time')
legend(str_legend_workers);
hold off;
print('Measure_high_wage','-dpng');

% plot the measure of workers at low wage
figure
hold on;
a=0;
clear str_legend_workers
for i=1:10:M
a=a+1;
plot(discard:number_periods-1, gt_wi_low_r(i,discard:number_periods-1) ,'Color',colorVec(i,:))
str_legend_workers{a} = sprintf('x=%g',xgrid(i)); %prepare the legend
end

title('Measure  of Workers at Starting Wages')
xlabel('time')
legend(str_legend_workers);
hold off;
print('Measure_low_wage','-dpng');

%Display the moments:
mean_y = mean(y_r(discard:number_periods-1,1));
mean_u = mean(ut_r(discard:number_periods-1,1));
mean_ft = mean(ft_r(discard:number_periods-1,1));
mean_st = mean(st_r(discard:number_periods-1,1));
mean_w = mean(wi_r(discard:number_periods-1,1));

% std of the logs:
std_y = std(log(y_r(discard:number_periods-1,1)));
std_u = std(log(ut_r(discard:number_periods-1,1)));
std_ft = std(log(ft_r(discard:number_periods-1,1)));
std_st = std(log(st_r(discard:number_periods-1,1)));
std_w = std(log(wi_r(discard:number_periods-1,1)));

X = sprintf('Mean productivity =%d. \n Mean unemployment rate = %d. \n Mean exit rate = %d. \n Mean job destruction rate = %d. \n Mean wage = %d.',mean_y, mean_u, mean_ft, mean_st, mean_w);
disp(X)

Y = sprintf('Std productivity =%d. \n Std unemployment rate = %d. \n Std exit rate = %d. \n Std job destruction rate = %d \n  Std wage = %d. \n(log of the variables) ',std_y, std_u, std_ft, std_st, std_w);
disp(Y)


% Create a Latek table for the moments:
Title = {'Productivity'; 'Unempl. Rate';'Unempl. Exit Rate';'Job Destruction Rate';'Wage'};
Type = {'Mean'; 'Std'};

Col0 = [mean_y;std_y];
Col1 = [mean_u;std_u];
Col2 = [mean_ft;std_ft];
Col3 = [mean_st;std_st];
Col4 = [mean_w;std_w];

input2.data=[Col0 Col1 Col2 Col3 Col4];
input2.tableColLabels = Title;
input2.tableRowLabels = Type;
input2.transposeTable = 0;
input2.dataFormatMode = 'column';
input2.dataFormat = {'%.4f',5}; %two digits precision
input2.dataNanString = '-';
input2.tableColumnAlignment = 'c';
% Switch table borders on/off:
input2.tableBorders = 1;
% LaTex table caption:
input2.tableCaption = 'Fit of Employment and Turnover Moments';
% LaTex table label:
input2.tableLabel = 'Table2';
% Switch to generate a complete LaTex document or just a table:
input2.makeCompleteLatexDocument = 1;
% call latexTable:
latex2 = latexTable(input2);

%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the Match Surplus 
% 1. Contour Plot 
figure
contour(ygrid(:,1), xgrid(:,1), Si_m(:,:), 'ShowText','on')
xlabel('Shock value y')
ylabel('Ability value x')
title('Contour Plot Surplus Function')
print('Surplus_function','-dpng');

% 2. Plot for Specific Values
colorVec = hsv(M);
figure
a = 0;
hold on
for m=1:10:M
    a = a+1;
    plot(ygrid,Si_m(m,:),'Color',colorVec(m,:))
    str_legend_workers{a} = sprintf('x=%g',xgrid(m)); %prepare the legend
end
title('Match Surplus by Ability')
xlabel('aggregate shock')
ylabel('match productivity')
legend(str_legend_workers);
hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Promotion Wages at Median Ability
figure
contour(ygrid(:,1), ygrid(:,1),W_high(:,:,p_50),'ShowText','on')
ylabel('Current period y')
xlabel('Next period y')
title('Contour Plot Promotion Wages at the Median Ability Level')
print('Wage_High_contourplot','-dpng');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Starting Wages at Median Ability
figure
contour(ygrid(:,1), ygrid(:,1),W_low(:,:,p_50),'ShowText','on')
ylabel('Current period y')
xlabel('Next period y')
title('Contour Plot Starting Wages at the Median Ability Level')
print('Wage_Low_contourplot','-dpng');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table with the parameters
% Create a Latek table for the moments:
Title3 = {'Parameter Value'};
Type3 = {'tau'; 'z0';'sigma';'rho';'lambda0';'lambda1';'s';'\underline{x}';'eta';'mu';'alpha'};

input3.data = [tau; z0; alpha; pho; lambda0; lambda1; s; x_lower_bound; eta; mu; alpha]; 
input3.tableColLabels = Title3;
input3.tableRowLabels = Type3;
input3.transposeTable = 0;
input3.dataFormatMode = 'row';
input3.dataFormat = {'%.4f', 11}; %two digits precision
input3.dataNanString = '-';
input3.tableColumnAlignment = 'c';
% Switch table borders on/off:
input3.tableBorders = 1;
% LaTex table caption:
input3.tableCaption = 'Structural Parameters';
% LaTex table label:
input3.tableLabel = 'Table3';
% Switch to generate a complete LaTex document or just a table:
input3.makeCompleteLatexDocument = 1;
% call latexTable:
latex3 = latexTable(input3);

toc
