# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:18:49 2020

@author: Adrien SY

Capstone Project

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 
import seaborn as sns
#%% IMoprt, before adding all_data for more recent years

# plt.close('all')

# # go to folder with all all_data
# os.chdir("C:/Users/elira/Desktop/all_dataIncubator_Challenge/Capstone/all_dataIncubator/all_data")

# #import all_dataset for 2017
# all_data = pd.read_csv('chicago-energy-benchmarking-1.csv')
# all_data2=pd.read_csv('chicago-energy-benchmarking-covered-buildings-1.csv')

#%%
os.chdir("C:/Users/elira/Desktop/DataIncubator_Challenge/Capstone/DataIncubator/Data")

listFile = os.listdir()

all_data = pd.DataFrame()
for f in listFile:
    df = pd.read_csv(f)
    all_data = all_data.append(df,ignore_index=True)

all_data.describe()

#%% Look at age of building vs. Energy star score

fig,ax = plt.subplots()

plt.scatter(all_data['Year Built'],all_data['ENERGY STAR Score'],marker='.',c='b')
plt.xlabel('Year Built')
plt.ylabel('Energy Star score')

#%% Look at time , group by year

GroupYear=all_data.groupby(['Year Built']).count()


fig,ax = plt.subplots()


ax.bar(GroupYear.index,GroupYear.ID)
ax.set_xlabel('Year Built',fontsize=15)
ax.set_ylabel('Number of building',color='b',fontsize=15)
ax.grid()

# average ENERGY STAR Score by year 
ax2=ax.twinx()
GroupYear=all_data.groupby(['Year Built']).mean()
ax2.plot(GroupYear.index,GroupYear['ENERGY STAR Score'],'ro-')
ax2.set_ylabel('Average ENERGY STAR Score',color='r',fontsize=15)

#%% Look at size of building vs. Energy star score

fig,ax = plt.subplots()

plt.scatter(all_data['Gross Floor Area - Buildings (sq ft)'],all_data['ENERGY STAR Score'],marker='.',c='g')
plt.xlabel('Gross Floor Area - Buildings (sq ft)')
plt.ylabel('Energy Star score')


#%% GHG emission for a given year


YearlyData=all_data.groupby(['Data Year']).mean()

fig,ax = plt.subplots()
plt.scatter(YearlyData.index,YearlyData['Total GHG Emissions (Metric Tons CO2e)'],marker='o',c='g',s=100)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Total GHG Emissions (Metric Tons CO2e)',fontsize=15)


#%%
from sklearn.linear_model import LinearRegression

#Derived from https://www.weather.gov/lot/Annual_Temperature_Rankings_Chicago
Year=YearlyData.index
Temperature=[47.5,50.1,52.3,52.6,50.9]
Temperature=np.asanyarray(Temperature)
fig,ax = plt.subplots()
GHG=np.asfarray(YearlyData['GHG Intensity (kg CO2e/sq ft)'],float)
X, Y = Temperature.reshape(-1,1), GHG.reshape(-1,1)
plt.scatter(Temperature,GHG)
plt.plot( X, LinearRegression().fit(X, Y).predict(X),color="red")

ax.set_xlabel('Temperature in °F',fontsize=15)
ax.set_ylabel('GHG Intensity (kg CO2e/sq ft)',fontsize=15)
plt.title('Average GHG Intensity for years 2014-2016 and \n annual temperature values from The Chicago Building\n Energy Use Benchmarking Ordinance and NOAA',fontsize=20)

# reggression score
reg=LinearRegression().fit(X, Y)
print(reg.score(X, Y))




#%% Distribution of energy use


Energy=YearlyData[['Electricity Use (kBtu)',
 'Natural Gas Use (kBtu)',
 'District Steam Use (kBtu)',
 'District Chilled Water Use (kBtu)',
 'All Other Fuel Use (kBtu)']]


width = 0.35 
fig, ax = plt.subplots()

ax.bar(Energy.index, Energy['Electricity Use (kBtu)'], width, label='Electricity Use (kBtu)')
ax.bar(Energy.index, Energy['Natural Gas Use (kBtu)'], width, bottom=Energy['Electricity Use (kBtu)'],
       label='Natural Gas Use (kBtu)')
ax.bar(Energy.index, Energy['District Steam Use (kBtu)'], width, bottom=Energy['Natural Gas Use (kBtu)']+Energy['Electricity Use (kBtu)'],
       label='District Steam Use (kBtu)')
ax.bar(Energy.index, Energy['District Chilled Water Use (kBtu)'], width, bottom=Energy['Natural Gas Use (kBtu)']+Energy['Electricity Use (kBtu)']+Energy['District Steam Use (kBtu)'],
        label='District Chilled Water Use (kBtu)')
ax.bar(Energy.index, Energy['All Other Fuel Use (kBtu)'], width, bottom=Energy['Natural Gas Use (kBtu)']+Energy['Electricity Use (kBtu)']+Energy['District Steam Use (kBtu)']+Energy['District Chilled Water Use (kBtu)'],
        label='All Other Fuel Use (kBtu)')

ax.set_ylabel('Energy use in kBtu',fontsize=15)
ax.set_xlabel('Year',fontsize=15)
ax.legend(loc = 5)
ax.grid()
plt.show()

# average ENERGY STAR Score by year 
ax2=ax.twinx()
GroupYear=all_data.groupby(['Year Built']).mean()
ax2.plot(GroupYear.index[-len(Energy.index):],GroupYear['ENERGY STAR Score'][Energy.index[0]:Energy.index[-1]],'ro-')
ax2.set_ylabel('Average ENERGY STAR Score',color='r',fontsize=15)
ax2.tick_params(axis='y', colors='red')
plt.title('Average energy Use and ENERGY STAR Score for years 2014-2016 \n from The Chicago Building Energy Use Benchmarking Ordinance',fontsize=20)

#%% Look at size of building vs. total energy use and total GHG emission

all_data['Total energy use (kBtu)']=all_data[['Electricity Use (kBtu)',
 'Natural Gas Use (kBtu)',
 'District Steam Use (kBtu)',
 'District Chilled Water Use (kBtu)',
 'All Other Fuel Use (kBtu)']].sum(axis=1)



plt.figure()

plt.scatter(all_data['Gross Floor Area - Buildings (sq ft)'],all_data['Total GHG Emissions (Metric Tons CO2e)'],marker='.',c='b')
plt.xlabel('Gross Floor Area - Buildings (sq ft)')
plt.ylabel('Total GHG Emissions (Metric Tons CO2e)')

plt.figure()
plt.scatter(all_data['Gross Floor Area - Buildings (sq ft)'],all_data['Total energy use (kBtu)'],marker='.',c='g')
plt.xlabel('Gross Floor Area - Buildings (sq ft)')
plt.ylabel(['Total energy use (kBtu)'])



#%% Look at total energy use vs average temeprature in Chicago for the year; building level data

#Year=YearlyData.index
#Temperature=[47.5,50.1,52.3,52.6,50.9]

all_data['Avg Temp Chicago in degF']=np.nan


all_data.loc[all_data['Data Year']==2014, "Avg Temp Chicago in degF"] = Temperature[0]
all_data.loc[all_data['Data Year']==2015, "Avg Temp Chicago in degF"] = Temperature[1]
all_data.loc[all_data['Data Year']==2016, "Avg Temp Chicago in degF"] = Temperature[2]
all_data.loc[all_data['Data Year']==2017, "Avg Temp Chicago in degF"] = Temperature[3]
all_data.loc[all_data['Data Year']==2018, "Avg Temp Chicago in degF"] = Temperature[4]

fig, ax = plt.subplots()
sns.set(style="darkgrid")
g = sns.scatterplot(x='Gross Floor Area - Buildings (sq ft)', y='Total GHG Emissions (Metric Tons CO2e)', hue='Data Year',
                   data=all_data,legend="full")


plt.xlabel('Gross Floor Area - Buildings (sq ft)',fontsize=12)
plt.ylabel('Total GHG Emissions (Metric Tons CO2e)',fontsize=12)

#%%
fig, ax = plt.subplots()
sns.set(style="darkgrid")
g = sns.regplot(x='Gross Floor Area - Buildings (sq ft)', y='Total GHG Emissions (Metric Tons CO2e)',
                   data=all_data)


plt.xlabel('Gross Floor Area - Buildings (sq ft)',fontsize=12)
plt.ylabel('Total GHG Emissions (Metric Tons CO2e)',fontsize=12)

#%% test lm plot + hue = year

sns.set(style="darkgrid")
g = sns.lmplot(x='Gross Floor Area - Buildings (sq ft)', y='Total GHG Emissions (Metric Tons CO2e)',hue='Data Year',
                   data=all_data)


plt.xlabel('Gross Floor Area - Buildings (sq ft)',fontsize=12)
plt.ylabel('Total GHG Emissions (Metric Tons CO2e)',fontsize=12)

#%% continue GHG vs temperature, use scatter plot now

plt.figure()
plt.scatter(all_data["Avg Temp Chicago in degF"],all_data["Total energy use (kBtu)"])

# not really relvant


#%% 



# NEXT :    Look at energy use vs building type or energy use VS zip code, some some classification OR data conversion 

