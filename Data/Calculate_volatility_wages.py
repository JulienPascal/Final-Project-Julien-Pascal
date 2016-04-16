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

path = '/home/julien/Documents/COURS/5A/MASTER THESIS/Labor Market/Data/Data'
os.chdir(path) #locate in the correct directory

###################
# Data 2000 to 2015
###################

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

# Plot them:
	plt.plot(dateList, First_decile, color = 'b')
	plt.plot(dateList, First_quartile, 'k')
	plt.plot(dateList, Median, color = 'r')
	plt.plot(dateList, Third_quartile, color = 'g')
	plt.plot(dateList, Ninth_decile, color = 'navy')
	
	plt.title('Wages')
	plt.legend(['First decile', 'First quartile', 'Median', 'Third quartile', 'Ninth decile'], loc='best', fancybox = True, framealpha=0.5)
	#plt.savefig('Unemployed_by_category')
	plt.show()

# Plot them:
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
plt.show()

#Get rid of a linear trend on wages:
z = np.polyfit(df['Year'].values, log_First_decile, 1)
p = np.poly1d(z)

detrended_log_First_decile = log_First_decile - p(df['Year'].values)

z = np.polyfit(df['Year'].values, log_First_quartile, 1)
p = np.poly1d(z)

detrended_log_First_quartile = log_First_quartile - p(df['Year'].values)

z = np.polyfit(df['Year'].values, log_Median, 1)
p = np.poly1d(z)

detrended_log_Median = log_Median - p(df['Year'].values)

detrended_log_First_quartile = log_First_quartile - p(df['Year'].values)


z = np.polyfit(df['Year'].values, log_Third_quartile, 1)
p = np.poly1d(z)

detrended_log_Third_quartile = log_Third_quartile - p(df['Year'].values)

z = np.polyfit(df['Year'].values, log_Ninth_decile, 1)
p = np.poly1d(z)

detrended_log_Ninth_decile = log_Ninth_decile - p(df['Year'].values)

print(np.std(detrended_log_First_decile))
print(np.std(detrended_log_First_quartile))
print(np.std(detrended_log_Median))
print(np.std(detrended_log_Third_quartile))
print(np.std(detrended_log_Ninth_decile))

###################
#Data 1967 to 2005:
###################

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
	plt.savefig('trended_wage_deciles')
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
		plt.savefig('Trend_wage_deciles')
		plt.show()

		#Plot the cycle component
		a = 0
		for i in  np.arange(10, 100, 10):
			plt.plot(df2['true_year'].values, cycle_avg_wage[i], color = line_colors[a])
			a=a+1

		plt.title('Cycle component in wage deciles')
		plt.savefig('Cycle_wage_deciles')
		plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
		plt.show()
 
		#Plot the untrended data:
		a = 0
		for i in  np.arange(10, 100, 10):
			plt.plot(df2['true_year'].values, detrended_avg_wage[i], color = line_colors[a])
			a=a+1

		plt.title('Detrended wage deciles HP Filter smoothing parameter = 100')
		plt.savefig('Detrended_wage_deciles')
		plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
		plt.show()

		#Volatility of the HP filtered deciles:
		a = 0
		for i in  np.arange(10, 100, 10):
			print(np.std(detrended_avg_wage[i]))
			a=a+1

#####################################################
# B. Remove a linear trend, as in the original paper:
#####################################################

	z = np.polyfit(a, b, 1)
	p = np.poly1d(z)

	linear_trend_avg_wage = {}
	delinear_avg_wage = {}

			for i in  np.arange(10, 100, 10):
			variable = 'avg_wage_%s' %i
			log_wage_decile = np.log(df2[variable]) #take the log
			z = np.polyfit(df2['true_year'].values, log_wage_decile , 1) #calculate the least squares line
			p = np.poly1d(z)
			linear_trend_avg_wage[i] = p(df2['true_year'])
			delinear_avg_wage[i] = np.exp(log_wage_decile - linear_trend_avg_wage[i] + np.mean(linear_trend_avg_wage[i])) # add the mean of the trend and take the exponential 

		# Plot the untrended data:
		a = 0
		for i in  np.arange(10, 100, 10):
			plt.plot(df2['true_year'].values, delinear_avg_wage[i], color = line_colors[a])
			a=a+1

		plt.title('Detrended wage deciles, removing a linear trend')
		plt.savefig('delinearized_wage_deciles')
		plt.legend(legend_list, loc='best', fancybox = True, framealpha=0.2)
		plt.show()

		#Volatility of the delinearized deciles:
		table = [] #create a table to store the volatility
		a = 0
		for i in  np.arange(10, 100, 10):
			variable = '%sth Decile' %i
			print(np.std(delinear_avg_wage[i]))
			table.append([variable, np.std(delinear_avg_wage[i])])
			a=a+1

print( tabulate(table, headers=['Decile', 'Volatility'], tablefmt="latex"))
