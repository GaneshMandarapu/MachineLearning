# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 19:04:10 2016

@author: Ganesh
"""

#%%
from numpy import *

import numpy as np
import random
from scipy import stats
import pandas as pd
import os

#%%

#%%
path = "S:\Spring 2017\Big Data Anlytics\Project\OneDrive_2017-02-25"
os.chdir(path)

k_Data = genfromtxt("mean_temp_press.csv", delimiter=",")


#k_Data = np.matrix(k_Data)

#%%



#%%
k_Data.head()

#%%


#%%

k_Data = pd.DataFrame(k_Data)

x = (k_Data)

centers = k_Data.ix[np.random.choice(k_Data.index, 4)]

#centers = k_Data.iloc[17:19,0:]
#%%

#%%

def euclidnew(x, centers):
    distanceMatrix = pd.DataFrame(np.nan * np.ones(shape=(x.shape[0],centers.shape[0])))
    for j in range(0, centers.shape[0]):
        for i in range(0, x.shape[0]):
            distanceMatrix.iloc[i][j] = sqrt((centers.iloc[j][0] - x.iloc[i][0])**2 + (centers.iloc[j][1] - x.iloc[i][1])**2)
    #print (distanceMatrix)
    return distanceMatrix
 
 
#%%
 
#%%

euclidnew(x, centers)

#%%
            
#%%
def kmeans(x, centers, euclidnew, niter):
    clusterHistory = [[1]] * 10
    centerHistory =  [[1]] * 10
    
    for i in range (1, niter):
        distsToCenters = euclidnew(x, centers)
        clusters = distsToCenters.apply(lambda x: x.argmin(), axis=1)
        x.loc[:,2] = clusters
        centers = x.groupby(x.loc[:,2]).apply(lambda x: np.average(x.loc[:,0:1], axis=0))
        clusterHistory[i] = clusters
        centerHistory[i] = centers
    return(np.asmatrix(clusterHistory), centerHistory)
    

#%%



#%%


a = kmeans(x, centers, euclidnew, 8)




#%%


#Plotting clusters
#%%





#%%














#%%
[x[1] for x in a]
#%%


##Testing purpose

#%%
distsToCenters = euclidnew(x, centers)
        

clusters = distsToCenters.apply(lambda x: x.argmin(), axis=1)

x.loc[:,2] = clusters


centers = x.groupby(x.loc[:,2]).apply(lambda x: np.average(x.loc[:,0:1], axis=0))




index = pd.Index(['01/01/2012','01/01/2012','01/01/2012','01/02/2012','01/02/2012'], name='Date')

df = pd.DataFrame({'ID':[100,101,102,201,202],'wt':[.5,.75,1,.5,1],'value':[60,80,100,100,80]},index=index)

df.groupby(df.index).apply(lambda x: np.average(x.wt, weights=x.value))


df.groupby(df.index).apply(lambda x: np.average(x.wt))





sqrt( (centers.iloc[j][j] - x.iloc[i][j])**2 + (centers.iloc[j][j+1] - x.iloc[i][j+1])**2)


distanceMatrix[0][0] = sqrt((centers.iloc[0][0] - x.iloc[0][0])**2 + (centers.iloc[0][1] - x.iloc[0][1])**2)

distanceMatrix.iloc[0][1] = sqrt((centers.iloc[1][0] - x.iloc[0][0])**2 + (centers.iloc[1][1] - x.iloc[0][1])**2)


sqrt((centers.iloc[1][0] - x.iloc[0][0])**2 + (centers.iloc[1][1] - x.iloc[0][1])**2)

#%%