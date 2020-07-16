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
from scipy import stats
import seaborn as sns

#%% IMoprt, before adding all_data for more recent years

# plt.close('all')

# # go to folder with all all_data
# os.chdir("C:/Users/elira/Desktop/all_dataIncubator_Challenge/Capstone/all_dataIncubator/all_data")

# #import all_dataset for 2017
# all_data = pd.read_csv('chicago-energy-benchmarking-1.csv')
# all_data2=pd.read_csv('chicago-energy-benchmarking-covered-buildings-1.csv')

#%%
os.chdir("C:/Users/elira/Desktop/DataIncubator/Capstone/DataIncubator/Data")

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



#%% Test linear regression OLS 

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats

all_data1=all_data.dropna(subset=['Gross Floor Area - Buildings (sq ft)','Year Built','# of Buildings','Source EUI (kBtu/sq ft)'])


#%%

# Apply one hot encoder to type of building
typeBuilding = pd.get_dummies(all_data1['Primary Property Type'])
# typeBuilding = pd.get_dummies(all_data["NewCategory"])

all_data1 = pd.merge(all_data1,typeBuilding,left_index=True, right_index=True)
list_building_type = list(typeBuilding)

# check outlier on the "Source" command
q_low = all_data1['Source EUI (kBtu/sq ft)'].quantile(0.01)
q_hi  = all_data1['Source EUI (kBtu/sq ft)'].quantile(0.99)

all_data_filtered1 = all_data1[(all_data1['Source EUI (kBtu/sq ft)'] < q_hi) & (all_data1['Source EUI (kBtu/sq ft)'] > q_low)]

# apply log transformation
all_data_filtered1['log Source EUI (kBtu/sq ft)'] = all_data_filtered1['Source EUI (kBtu/sq ft)'].apply(lambda x: np.log(x))
all_data_filtered1['log GFA'] = all_data_filtered1['Gross Floor Area - Buildings (sq ft)'].apply(lambda x: np.log(x))
all_data_filtered1['ZipCodeBIS'] = all_data1['ZIP Code'].astype(str).str[0:5]
all_data_filtered1['ZipCodeBIS'] = pd.to_numeric(all_data_filtered1['ZipCodeBIS'])

# filter outlier for Source EUI (kBtu/sq ft) , then apply a log transformation
all_data_X=all_data_filtered1[['log GFA','Year Built','# of Buildings','ZipCodeBIS'] + list_building_type]

all_data_Y=all_data_filtered1['log Source EUI (kBtu/sq ft)']



#%%

X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, \
                                                    test_size=0.33, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, test_size=0.33, random_state=42)

#%%
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Y_pred))

# The MAE
print('MAE: %.2f'
      % mean_absolute_error(y_test, Y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, Y_pred))


y_test.describe()
stats.describe(Y_pred)


# x = np.linspace(0, 10)
# plt.scatter(y_test,Y_pred)
# plt.axis('equal')
# plt.plot(x, x, linestyle='-')  # solid



#%%             CAS 2 ---- > moins de type building category


all_data2=all_data.dropna(subset=['Gross Floor Area - Buildings (sq ft)','Year Built','# of Buildings','Source EUI (kBtu/sq ft)'])


listOtherType= ['Adult Education',
 'Ambulatory Surgical Center',
 'Automobile Dealership',
 'Bank Branch',
 'College/University',
 'Convention Center',
 'Courthouse',
 'Distribution Center',
 'Enclosed Mall',
 'Financial Office',
 'Fitness Center/Health Club/Gym',
 'Hospital (General Medical & Surgical)',
 'Hotel',
 'Ice/Curling Rink',
 'Indoor Arena',
 'Laboratory',
 'Library',
 'Lifestyle Center',
 'Medical Office',
 'Mixed Use Property',
 'Movie Theater',
 'Museum',
 'Other',
 'Other - Education',
 'Other - Entertainment/Public Assembly',
 'Other - Lodging/Residential',
 'Other - Mall',
 'Other - Public Services',
 'Other - Recreation',
 'Other - Services',
 'Other - Specialty Hospital',
 'Outpatient Rehabilitation/Physical Therapy',
 'Performing Arts',
 'Pre-school/Daycare',
 'Prison/Incarceration',
 'Repair Services (Vehicle, Shoe, Locksmith, etc.)',
 'Residence Hall/Dormitory',
 'Residential Care Facility',
 'Retail Store',
 'Senior Care Community',
 'Social/Meeting Hall',
 'Stadium (Open)',
 'Strip Mall',
 'Supermarket/Grocery Store',
 'Urgent Care/Clinic/Other Outpatient',
 'Wholesale Club/Supercenter',
 'Worship Facility']


# too many paramaters of building type, reduce to 4 based the number of occurence : Mutlifamilly housing, k1 school, office and others 
all_data2["NewCategory"]= all_data2['Primary Property Type'].replace(listOtherType,'Others') 


#%%

# Apply one hot encoder to type of building
typeBuilding = pd.get_dummies(all_data2["NewCategory"])

all_data2 = pd.merge(all_data2,typeBuilding,left_index=True, right_index=True)

# check outlier on the "Source" command
q_low = all_data2['Source EUI (kBtu/sq ft)'].quantile(0.01)
q_hi  = all_data2['Source EUI (kBtu/sq ft)'].quantile(0.99)

all_data_filtered2 = all_data2[(all_data2['Source EUI (kBtu/sq ft)'] < q_hi) & (all_data2['Source EUI (kBtu/sq ft)'] > q_low)]

# apply log transformation
all_data_filtered2['log Source EUI (kBtu/sq ft)'] = all_data_filtered2['Source EUI (kBtu/sq ft)'].apply(lambda x: np.log(x))
all_data_filtered2['log GFA'] = all_data_filtered2['Gross Floor Area - Buildings (sq ft)'].apply(lambda x: np.log(x))
all_data_filtered2['ZipCodeBIS'] = all_data2['ZIP Code'].astype(str).str[0:5]
all_data_filtered2['ZipCodeBIS'] = pd.to_numeric(all_data_filtered2['ZipCodeBIS'])

# filter outlier for Source EUI (kBtu/sq ft) , then apply a log transformation
all_data_X=all_data_filtered2[['log GFA','Year Built','# of Buildings','ZipCodeBIS','K-12 School_x','Multifamily Housing_x','Office_x','Others_x']] 

all_data_Y=all_data_filtered2['log Source EUI (kBtu/sq ft)']



#%%

X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, \
                                                    test_size=0.33, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(all_data_X, all_data_Y, test_size=0.33, random_state=42)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Y_pred))

# The MAE
print('MAE: %.2f'
      % mean_absolute_error(y_test, Y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, Y_pred))


y_test.describe()
stats.describe(Y_pred)

#%% works better with all the categories  --- Try other regressor algortihn 

import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomrftClassifier

classifiers = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    linear_model.LinearRegression()]

for item in classifiers:
    print('-----------')
    print(item)
    clf = item
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    
        # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, Y_pred))
    
    # The MAE
    print('MAE: %.2f'
          % mean_absolute_error(y_test, Y_pred))
    
    # The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, Y_pred))
    

#%%   Test SVM with different kernel 


from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear',cache_size=7000)


svrs = [svr_rbf, svr_lin]


Cs = np.logspace(-2,2,5)
gammas = np.logspace(-2,2,5)
epsil = np.logspace(-2,2,5)
param_grid = {'C': Cs, 'gamma' : gammas,'epsilon':epsil}

for item in svrs:
    print('-----------')
    print(item)
    clf = item


    gs_est = GridSearchCV(clf, param_grid, cv=3, n_jobs=2, verbose=1,scoring='neg_mean_squared_error')
    gs_est.fit(X_train, y_train)
    print(gs_est.best_params_)
    
    opt_est = SVR(kernel='rbf', C=gs_est.best_params_['C'], gamma=gs_est.best_params_['gamma'],epsilon=gs_est.best_params_['epsilon'])
    opt_est.fit(X_train, y_train)
    Y_pred = opt_est.predict(X_test)
    
        # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, Y_pred))
    
    # The MAE
    print('MAE: %.2f'
          % mean_absolute_error(y_test, Y_pred))
    
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
      % r2_score(y_test, Y_pred))
    

#%%   Test with linear regression


svr_lin = SVR(kernel='linear',cache_size=7000)


svr_lin.fit(X_train, y_train)


Y_pred = svr_lin.predict(X_test)

    # The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Y_pred))

# The MAE
print('MAE: %.2f'
      % mean_absolute_error(y_test, Y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
  % r2_score(y_test, Y_pred))


#%%    Random forst test 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# --> tuning of RF

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(random_state = 42)# Train the model on training data


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
print(rf_random.best_params_)

#%%  then predcit and asses performance


rf = RandomForestRegressor(n_estimators=1000, min_samples_split=2,max_features='sqrt',max_depth=110, bootstrap=True,random_state = 42)# Train the model on training data
rf.fit(X_train, y_train)


# Use the forest's predict method on the test data
Y_pred = rf_random.predict(X_test)# Calculate the absolute errors
errors = abs(Y_pred - y_test)# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


    # The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, Y_pred))

# The MAE
print('MAE: %.2f'
      % mean_absolute_error(y_test, Y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
  % r2_score(y_test, Y_pred))

#%% Model is tuned, RF works the best and we have found optimal paramaters. Now go back to original predicted values 
# include exponentitation of result



#  all_data_Y=all_data_filtered1['log Source EUI (kBtu/sq ft)']
#  all_data_X=all_data_filtered1[['log GFA','Year Built','# of Buildings','ZipCodeBIS'] + list_building_type]

# predict on all the dataset :
log_model_result = rf_random.predict(all_data_X)# Calculate the absolute errors

# apply exp transf to result to get Source EUI (kBtu/sq ft)
model_result = np.exp(log_model_result)

all_data_filtered1['model_result']=model_result
                  

# compare it to all_data_filtered1['Source EUI (kBtu/sq ft)']

A = np.linspace(0,800,100)

#%% plotting model vs measured, using plt

plt.scatter(all_data_filtered1['Source EUI (kBtu/sq ft)'],model_result,label='$\mathregular{R^2}$: 0.76 \nAccuracy: 97.05 %')
plt.plot(A, A, '-k',label='Perfect fit');
plt.xlabel('Measured source EUI (kBtu/sq ft)', fontsize=18)
plt.ylabel('Modelled source EUI (kBtu/sq ft)', fontsize=18)
plt.title('Modelling of Source Energy Use Intensity (EUI) using Random Forest', fontsize=20)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(linestyle='dotted',color='gray')
plt.legend()
plt.show()


#%% vis with bokeh

from bokeh.plotting import figure, output_file, show

output_file('model_grah.html')

p = figure()

p.circle(all_data_filtered1['Source EUI (kBtu/sq ft)'],model_result,legend_label='R²: 0.76 | Accuracy: 97.05% | n=9,790')
p.line(A, A, color='black', legend_label='Perfect fit')
p.xaxis.axis_label = 'Measured source EUI (kBtu/sq ft)'
p.yaxis.axis_label ='Modelled source EUI (kBtu/sq ft)'
p.title.text = 'Modelling of Source Energy Use Intensity (EUI) using Random Forest'
p.xgrid.grid_line_color = 'gray'	
show(p)


#%%
# plt.figure()
# sns.regplot(x='Source EUI (kBtu/sq ft)',y='model_result',data=all_data_filtered1,scatter_kws={"color": "black"}, line_kws={"color": "red"},label=' Linear regression model fit')
# plt.xlabel('Measured source EUI (kBtu/sq ft)')
# plt.ylabel('Modelled source EUI (kBtu/sq ft)')
# plt.title('Modelling of Source Energy Use Intensity (EUI) using Random Forest \n$\mathregular{R^2}$: 0.76 | Accuracy: 97.05 %')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()