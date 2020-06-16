import pandas as pd
#import matplotlib.pyplot as plt
#import math
import numpy as np
#from scipy.stats import poisson
#from scipy.stats import binom
from scipy.stats import  multivariate_normal
#import sys
#from numpy import genfromtxt

PixelNum=784; #total numnber of pixels
PixValMax=255;  #max value of the pixel
# 0,1,2 ... 9 - Labels of identification
Labels = [i for i in range(0,10)];

#store training data in numpy array
train_data_pd = pd.read_csv('.\\mod_data\\deskewdata_train.csv', delimiter=',');
# calculate mean of all the pixel values for each digit
print ('calculating mean ...')
mean_np = np.zeros([len(Labels),PixelNum])
for ilab in Labels:
    data_np = train_data_pd[train_data_pd.label==ilab].iloc[:,1:].to_numpy()
    mean_np[ilab] = data_np.mean(axis=0)
# calculate covariance matrix
print ('calculating covariance ...')
cov_np = train_data_pd.iloc[:,1:].cov().to_numpy()

# return multivariate normal posterior prob 
def CalPosteriorProbab ( lab, pixvals):
    return multivariate_normal.logpdf (pixvals, mean_np[lab], cov_np,
                                       allow_singular=True);

verifydata = pd.read_csv('.\\mod_data\\deskewdata_test.csv', delimiter=',');
print (verifydata.head());

obsval=[]                            
passcount=0.0
startval = 0
numsamples=10000
failLabs = []
failIds = []

for count in range(startval, startval+numsamples):
    obsval=[]    
    for ilab in Labels:
        obsval.append( CalPosteriorProbab ( ilab, verifydata.iloc[count,1:] ) )

    if obsval.index( max(obsval)) == verifydata.iloc[count,0]:
        passcount = passcount+1;
    else:
        failLabs.append(verifydata.iloc[count,0]);
        failIds.append(startval+count)

    print ('Count:',count, ' Actual value: ',verifydata.iloc[count,0], ' Max probable value:',
           obsval.index( max(obsval) ), ' Perc. of success:', passcount*100/(count+1));

        

passperc =  passcount/(numsamples);
print ("Total count:",numsamples," Pass count:",passcount," Perc. of success:",passperc)
       
        
        

    

