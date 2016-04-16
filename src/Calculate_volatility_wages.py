##########################################################
# Calculate the volatility of the wages deciles in the US
##########################################################
# Use data from Jonathan Heathcote, Fabrizio Perri, Giovanni L. Violante
# Unequal We Stand: An Empirical Analysis of Economic Inequality in the United States, 1967-2006
# http://www.nber.org/papers/w15483

# I follow the procedure described by Robin 2011 in 
# http://onlinelibrary.wiley.com/doi/10.3982/ECTA9070/abstract

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from more_itertools import unique_everseen
import pylab
import matplotlib.dates as mdates
from math import exp, log
import math
from datetime import date, timedelta as td
import statsmodels.api as sm
import matplotlib.cm as cm
from tabulate import tabulate
import csv

path = '/home/julien/master-s_thesis/data' #path to the data folder
path_figure = '/home/julien/master-s_thesis/figures/' #where to save the figures
path_table = '/home/julien/master-s_thesis/tables/' #where to save the tables
os.chdir(path) #locate in the data folder

######################
# I. Data 2000 to 2015
######################

df = pd.read_csv("LEU.csv")

starting_year = 2000 #first quarter 1951
starting_month = 1
starting_date = date(starting_year, starting_month, 1)

stop_year = 2015 #third quarter 2010
stop_month = 10
stop_date = date(stop_year, stop_month, 1)

# 1. Create a column with dates objects:
df['Date'] = 0
for i in range(0,len(df['Year'])):
	df.ix[i,'Date'] = date(df.ix[i,'Year'], df.ix[i,'Month'], 1)

#keep the selected time period: 
df = df.loc[df['Date'] <= stop_date]
df = df.loc[df['Date'] >= starting_date]

#Create
dateList = [] 
for index, row in df.iterrows():
	print(index)
	dateList.append(date(df.ix[index,'Year'], df.ix[index,'Month'], 1))

# Usual weekly earnings - in current dollars, first decile, both sexes: LEU0252911200
First_decile = df['LEU0252911200'].values

# Usual weekly earnings, first quartile, Employed full time, Wage and salary workers LEU0252911300
First_quartile = df['LEU0252911300'].values

#Median usual weekly earnings (second quartile), Employed full time, Wage and salary workers   LEU0252881500
Median = df['LEU0252881500'].values

# Usual weekly earnings, third quartile, Employed full time, Wage and salary workers  LEU0252911400
Third_quartile = df['LEU0252911400'].values

# Usual weekly earnings, ninth decile, Employed full time, Wage and salary workers    LEU0252911500
Ninth_decile = df['LEU0252911500'].values

##################
# Plot the series:
##################
plt.plot(dateList, First_decile, color = 'b')
plt.plot(dateList, First_quartile, 'k')
plt.plot(dateList, Median, color = 'r')
plt.plot(dateList, Third_quartile, color = 'g')
plt.plot(dateList, Ninth_decile, color = 'navy')

plt.title('Wages')
plt.legend(['First decile', 'First quartile', 'Median', 'Third quartile', 'Ninth decile'], loc='best', fancybox = True, framealpha=0.5)
plt.savefig(path_figure + 'Wages_2000_2015')
plt.show()

######################
# Plot ratio of series
######################
plt.plot(dateList, Ninth_decile/First_decile, color = 'b')
plt.plot(dateList, Median/First_decile, 'k')
plt.plot(dateList, Ninth_decile/Median, color = 'r')
plt.legend(['P90/P10','P50/P10','P90/P50'] , loc='best', fancybox = True, framealpha=0.2)
plt.title('Wages')
plt.savefig(path_figure + 'Wage_ratios_2000_2015')
plt.show()


#############################
# Plot the log of the series:
#############################
log_First_decile = np.log(First_decile)
log_First_quartile = np.log(First_quartile)
log_Median = np.log(Median)
log_Third_quartile = np.log(Third_quartile)
log_Ninth_decile = np.log(Ninth_decile)

plt.plot(dateList, log_First_decile, color = 'b')
plt.plot(dateList, log_First_quartile , 'k')
plt.plot(dateList, log_Median, color = 'r')
plt.plot(dateList, log_Third_quartile, color = 'g')
plt.plot(dateList, log_Ninth_decile , color = 'navy')
plt.title('Log Wages')
plt.savefig('Log_Wages_2000_2015')
plt.show()

#####################
#Plot ratios of logs
plt.plot(dateList, log_Ninth_decile/log_First_decile, color = 'b')
plt.plot(dateList, log_Median/log_First_decile, 'k')
plt.plot(dateList, log_Ninth_decile/log_Median, color = 'r')
plt.legend(['P90/P10','P50/P10','P90/P50'] , loc='best', fancybox = True, framealpha=0.2)
plt.title('Log Wages Ratios')
plt.savefig(path_figure + 'Log_Wage_ratios_2000_2015')
plt.show()


#####################################
# Get rid of a linear trend on wages:
#####################################
z = np.polyfit(df['Year'].values, log_First_decile, 1)
p = np.poly1d(z)
detrended_log_First_decile = log_First_decile - p(df['Year'].values) + np.mean(p(df['Year'].values)) #remove the trend and add the mean

z = np.polyfit(df['Year'].values, log_First_quartile, 1)
p = np.poly1d(z)
detrended_log_First_quartile = log_First_quartile - p(df['Year'].values) + np.mean(p(df['Year'].values))

z = np.polyfit(df['Year'].values, log_Median, 1)
p = np.poly1d(z)
detrended_log_Median = log_Median - p(df['Year'].values) + np.mean(p(df['Year'].values))

z = np.polyfit(df['Year'].values, log_Third_quartile, 1)
p = np.poly1d(z)
detrended_log_Third_quartile = log_Third_quartile - p(df['Year'].values) + np.mean(p(df['Year'].values))
 
z = np.polyfit(df['Year'].values, log_Ninth_decile, 1)
p = np.poly1d(z)
detrended_log_Ninth_decile = log_Ninth_decile - p(df['Year'].values) + np.mean(p(df['Year'].values))

#############################
# Plot detrented wage ratios:
#############################
# P90/P10: 90th to 10th percentile ratio
P_90_to_P10 = detrended_log_Ninth_decile/detrended_log_First_decile

# P50/P10
P_50_to_P10 = detrended_log_Median/detrended_log_First_decile

# P90/P50
P_90_to_50 = detrended_log_Ninth_decile/detrended_log_Median

plt.plot(df['Year'].values, P_90_to_P10)
plt.plot(df['Year'].values, P_50_to_P10)
plt.plot(df['Year'].values, P_90_to_50)
plt.legend(['P90/P10','P50/P10','P90/P50'] , loc='best', fancybox = True, framealpha=0.2)
plt.savefig(path_figure + 'delinearized_log_wage_decile_ratios_2000_2015')
plt.show()

######################################
# Standard deviations of wage deciles;
######################################
print(np.std(detrended_log_First_decile))
print(np.std(detrended_log_First_quartile))
print(np.std(detrended_log_Median))
print(np.std(detrended_log_Third_quartile))
print(np.std(detrended_log_Ninth_decile))

########################
# II. Data 1967 to 2005:
########################

df2 = pd.read_csv("individ_pctiles_work_hrs_wage.csv")

#Plot the deciles with the trend:

legend_list = []
#list_color = color_list = plt.cm.Set3(np.linspace(0, 1,13))
a = 0
cmap = plt.cm.Accent
line_colors = cmap(np.linspace(0,1,9)) 

for i in  np.arange(10, 100, 10):
	variable = 'avg_wage_%s' %i
	legend_name = '%sth percentile' %i
	plt.plot(df2['true_year'].values, df2[variable], color = line_colors[a])
	legend_list.append(legend_name)
	a=a+1

plt.title('Dynamics of wage deciles')
plt.savefig(path_figure + 'trended_wage_deciles')
plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.show()

#######################################
# A. HP Filter the log of the deciles :
#######################################

smoothing_parameter = 100 #Paramter for yearly data

cycle_avg_wage = {}
trend_avg_wage = {}
detrended_avg_wage = {}

# Decompose the trend and the cycle:
for i in  np.arange(10, 100, 10):
	variable = 'avg_wage_%s' %i
	cycle_avg_wage[i], trend_avg_wage[i] = sm.tsa.filters.hpfilter(np.log(df2[variable]), smoothing_parameter)
	detrended_avg_wage[i] = np.exp(cycle_avg_wage[i] + np.mean(trend_avg_wage[i])) # add the mean of the trend and take the exponential 


#Plot the trend:
a = 0
for i in  np.arange(10, 100, 10):
	plt.plot(df2['true_year'].values, trend_avg_wage[i], color = line_colors[a])
	a=a+1

plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.title('Trend in wage deciles')
plt.savefig(path_figure + 'Trend_wage_deciles')
plt.show()

#Plot the cycle component
a = 0
for i in  np.arange(10, 100, 10):
	plt.plot(df2['true_year'].values, cycle_avg_wage[i], color = line_colors[a])
	a=a+1

plt.title('Cycle component in wage deciles')
plt.savefig(path_figure + 'Cycle_wage_deciles')
plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.show()

#Plot the untrended data:
a = 0
for i in  np.arange(10, 100, 10):
	plt.plot(df2['true_year'].values, detrended_avg_wage[i], color = line_colors[a])
	a=a+1

plt.title('Detrended wage deciles HP Filter smoothing parameter = 100')
plt.savefig(path_figure + 'Detrended_wage_deciles')
plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.show()

# Table for the volatility of the HP filtered deciles:
table = [] #create a table to store the volatility
a = 0
for i in  np.arange(10, 100, 10):
	variable = '%sth Decile' %i
	print(np.std(detrended_avg_wage[i]))
	table.append([variable, np.std(detrended_avg_wage[i])])
	a=a+1

# Output table of standard errors of detrended data in latek:
print(tabulate(table, headers=['Decile', 'Volatility'], tablefmt="latex"))

#save the table in a csv format:
with open(path_table + 'table_volatility_detrended_wages.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	[writer.writerow(r) for r in table]


#####################################################
# B. Remove a linear trend, as in the original paper:
#####################################################

linear_trend_avg_wage = {}
delinear_avg_wage = {}
deviation_from_linear_trend = {}

for i in  np.arange(10, 100, 10):
	variable = 'avg_wage_%s' %i
	log_wage_decile = np.log(df2[variable]) #take the log
	z = np.polyfit(df2['true_year'].values, log_wage_decile , 1) #calculate the least squares line
	p = np.poly1d(z)
	linear_trend_avg_wage[i] = p(df2['true_year'])
	delinear_avg_wage[i] = np.exp(log_wage_decile - linear_trend_avg_wage[i] + np.mean(linear_trend_avg_wage[i])) # add the mean of the trend and take the exponential 
	deviation_from_linear_trend[i] = np.divide(log_wage_decile - linear_trend_avg_wage[i],linear_trend_avg_wage[i])

# Plot the untrended data:
a = 0
for i in  np.arange(10, 100, 10):
	plt.plot(df2['true_year'].values, delinear_avg_wage[i], color = line_colors[a])
	a=a+1

#plt.title('Detrended wage deciles, removing a linear trend')
plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.savefig(path_figure + 'delinearized_wage_deciles')
plt.show()

#Volatility of the delinearized deciles:
table = [] #create a table to store the volatility
a = 0
for i in  np.arange(10, 100, 10):
	variable = '%sth Decile' %i
	print(np.std(delinear_avg_wage[i]))
	table.append([variable, np.std(delinear_avg_wage[i])])
	a=a+1

# Output table of standard errors of delinearized data in latek:
print(tabulate(table, headers=['Decile', 'Volatility'], tablefmt="latex"))

#save the table in a csv format:
with open(path_table + 'table_volatility_delinearized_wages.csv', 'w') as csvfile:
	writer = csv.writer(csvfile)
	[writer.writerow(r) for r in table]

#######################################
# Plot deviations from the linear trend
#######################################
a = 0
for i in  np.arange(10, 100, 10):
	plt.plot(df2['true_year'].values, deviation_from_linear_trend[i], color = line_colors[a])
	a=a+1
#plt.title('Deviations of wage deciles from linear trend')
plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
plt.savefig(path_figure + 'Deviations_wages_deciles_from_linear_trend')
plt.show()

#####################
# Plot earning ratios
#####################

# P90/P10: 90th to 10th percentile ratio
P_90_to_P10 = delinear_avg_wage[90]/delinear_avg_wage[10] #indexing starts at 0

# P50/P10
P_50_to_P10 = delinear_avg_wage[50]/delinear_avg_wage[10]

# P90/P50
P_90_to_50 = delinear_avg_wage[90]/delinear_avg_wage[50]

plt.plot(df2['true_year'].values, P_90_to_P10, color = line_colors[1])
plt.plot(df2['true_year'].values, P_50_to_P10, color = line_colors[2])
plt.plot(df2['true_year'].values, P_90_to_50, color = line_colors[7])
plt.legend(['P90/P10','P50/P10','P90/P50'] , loc='best', fancybox = True, framealpha=0.2)
plt.savefig(path_figure + 'delinearized_wage_decile_ratios')
plt.show()

#######################################################
# III. Plot earning by educational attainment 2000-2015
#######################################################

df_wage = pd.read_csv("LEU_wage_education.csv")

W = df_wage.loc[df_wage['Series ID'] == 'LEU0252887700','Value'] #Median wage for every type of educational attainment
W1 = df_wage.loc[df_wage['Series ID'] == 'LEU0252916700','Value']#Median wage for Less than a high school diploma
W2 = df_wage.loc[df_wage['Series ID'] == 'LEU0252917300','Value']#Median wage for High school graduates, no college
W3 = df_wage.loc[df_wage['Series ID'] == 'LEU0254929400','Value']#Median wage for  Some college or associate degree
W4 = df_wage.loc[df_wage['Series ID'] == 'LEU0252918500','Value'] #Median wage for Bachelor's degree or higher

###############
# Scatter point
##############
plt.scatter(W, W1, color = 'b', alpha=0.5)
plt.scatter(W, W2, color = 'k', alpha=0.5)
plt.scatter(W, W3, color = 'r', alpha=0.5)
plt.scatter(W, W4, color = 'g', alpha=0.5)

#plt.scatter(Unemployment_rate_selected_years, degree_line, color = 'grey', alpha=0.2)
plt.legend(['Less than a High School Diploma', 'With a High School Diploma', 'Some College or Associate Degree','Bachelors Degree and Higher'], loc='upper left', fancybox = True, framealpha=0.5)
plt.xlabel('Median usual weekly earnings - in current dollars')
plt.ylabel('Median usual weekly earnings by educational attainment')
plt.savefig(path_figure + 'Overall_vs_group_edu_median_weekly_earning')
plt.show()

########
# Lines
#######

dateListWage = []
for i in range(0,len(W)):
    dateListWage.append(date(df_wage.loc[df_wage['Series ID'] == 'LEU0252887700','Year'].values[i],df_wage.loc[df_wage['Series ID'] == 'LEU0252887700','Month'].values[i], 1))


plt.plot(dateListWage, W1, color = 'b')
plt.plot(dateListWage, W2, '--k')
plt.plot(dateListWage, W3, color = 'r')
plt.plot(dateListWage, W4, color = 'g')

#fill in between:
plt.fill_between(dateListWage, 0, W1, color='b', alpha=0.2)
plt.fill_between(dateListWage, W1, W2, color='k', alpha=0.2)
plt.fill_between(dateListWage, W2, W3, color='r', alpha=0.2)
plt.fill_between(dateListWage, W3, W4, color='g', alpha=0.2)

#plt.title('Unemployment educational attainment')
plt.ylabel('Median usual weekly earnings in current dollars')
plt.legend(['Less than a High School Diploma', 'With a High School Diploma', 'Some College or Associate Degree','Bachelors Degree and Higher'], loc='upper left', fancybox = True, framealpha=0.5)
plt.savefig(path_figure + 'Wages_by_education')
plt.show()

