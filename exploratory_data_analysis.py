import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from matplotlib.pyplot import figure

sns.set(color_codes=True)


df = pd.read_csv("../../../data_joining/merged_sample_1_data.csv")
# Checking the data type
print(df.dtypes)
df = df.astype(float)
print(df.dtypes)


##---------Droping Duplicates---------#
## Total number of rows and columns
#print(df.shape)
#duplicate_rows_df = df[df.duplicated()]
#print("number of duplicate rows: ", duplicate_rows_df.shape)
## Used to count the number of rows before removing the data
#print(df.count())
## Dropping the duplicates 
#df = df.drop_duplicates()
## Counting the number of rows after removing duplicates.
#print(df.count())


##---------Dropping the missing or null values.---------#
## Finding the null values.
#print(df.isnull().sum())
## Dropping the missing values.
#df = df.dropna() 
## After dropping the values
#print(df.isnull().sum()) 


#---------Detecting Outliers---------#
fig,ax = plt.subplots(ncols=3,nrows=4,figsize=(35,40))
ax[3,0].set_xscale('log')

for i in [0,1,2,3]:
    for j in [0,1,2]:
        ax[i,j].set_xlabel(xlabel ='',fontsize = 20)
        ax[i,j].set_ylabel(ylabel ='',fontsize = 20)
        for tick in ax[i,j].get_xticklabels():
            tick.set_fontsize(20)
            tick.set_rotation(40)
sns.boxplot(x=df['R[Rsun]'].values,ax=ax[0,0])
ax[0,0].set_xlabel(xlabel ='R[Rsun]',fontsize = 20)

sns.boxplot(x=df['L[Rsun]'],ax=ax[0,1])
sns.boxplot(x=df['lon[Carr]'],ax=ax[0,2])
sns.boxplot(x=df['lat[Carr]'],ax=ax[1,0])
sns.boxplot(x=df['B[G]'],ax=ax[1,1])
sns.boxplot(x=df['A/A0'],ax=ax[1,2])
sns.boxplot(x=df['alpha[deg]'],ax=ax[2,0])
sns.boxplot(x=df['V/Cs'],ax=ax[2,1])
sns.boxplot(x=df['propag_dt[d]'],ax=ax[2,2])
sns.boxplot(x=df['n[cm^-3]'],ax=ax[3,0])
sns.boxplot(x=df['v[km/s]'],ax=ax[3,1])
sns.boxplot(x=df['T [MK]'],ax=ax[3,2])
fig.savefig("boxplot_univariate_outliers.pdf")
plt.cla()
plt.clf()
plt.close()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


#---------Getting general statistics---------#
print(df.describe())





#---------Plot different features against one another (scatter), against frequency (histogram)---------#

#---------Histograms---------#
fig,ax = plt.subplots(ncols=3,nrows=4,figsize=(35,40))
ax[3,0].set_xscale('log')
for i in [0,1,2,3]:
    for j in [0,1,2]:
        ax[i,j].set_xlabel(xlabel ='',fontsize = 25)
        ax[i,j].set_ylabel(ylabel ='',fontsize = 25)
        for tick in ax[i,j].get_xticklabels():
            tick.set_fontsize(25)
            tick.set_rotation(40)
sns.histplot(x=df['R[Rsun]'],bins=100,ax=ax[0,0])
ax[0,0].set_xlabel(xlabel ='R[Rsun]',fontsize = 20)
sns.histplot(x=df['L[Rsun]'],bins=100,ax=ax[0,1])
sns.histplot(x=df['lon[Carr]'],bins=100,ax=ax[0,2])
sns.histplot(x=df['lat[Carr]'],bins=100,ax=ax[1,0])
sns.histplot(x=df['B[G]'],bins=100,ax=ax[1,1])
sns.histplot(x=df['A/A0'],bins=100,ax=ax[1,2])
sns.histplot(x=df['alpha[deg]'],bins=100,ax=ax[2,0])
sns.histplot(x=df['V/Cs'],bins=100,ax=ax[2,1])
sns.histplot(x=df['propag_dt[d]'],bins=100,ax=ax[2,2])
sns.histplot(x=df['n[cm^-3]'],bins=100,ax=ax[3,0])
sns.histplot(x=df['v[km/s]'],bins=100,ax=ax[3,1])
sns.histplot(x=df['T [MK]'],bins=100,ax=ax[3,2])
fig.savefig("histograms.png", dpi=300)
plt.cla()
plt.clf()
plt.close()

#---------Heatmaps---------#
# Finding the relations between the variables.
#Correlation heatmap
df_corr = df.corr()
fig = figure(figsize=(35,35))
svm = sns.heatmap(df_corr,cmap='rocket', linecolor='white', linewidths=1.2)
svm.tick_params(labelsize=30,labelrotation=30)
fig = svm.get_figure()    
fig.savefig('heatmap3.png',)
plt.cla()
plt.clf()
plt.close()


#pairplots

#svm2 = sns.pairplot(data = df)

pp = sns.pairplot(data=df,
                  y_vars=['n[cm^-3]','v[km/s]','T [MK]'],
                  x_vars=['R[Rsun]','L[Rsun]','lon[Carr]','lat[Carr]','B[G]','A/A0','alpha[deg]','V/Cs','propag_dt[d]']) 
for ax in pp.axes.flatten():
    ax.tick_params(labelsize=10)    
   
pp.savefig('pairplot.png', dpi=500)
plt.cla()
plt.clf()
plt.close()






