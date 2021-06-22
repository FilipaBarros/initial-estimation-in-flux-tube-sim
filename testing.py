import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from keras.optimizers import SGD
import os
from pickle import load

path = '../../MULTI_VP_profiles/profiles_wso_CR1992/profile_wso_CR1992_line_0001.csv'

cols=['R[Rsun]','L[Rsun]','lon[Carr]','lat[Carr]','B[G]','A/A0','alpha[deg]','V/Cs','propag_dt[d]']
data = pd.read_csv(path,usecols=[0,1,2,3,4,5,6,7,8])
data.columns = cols
print(data)

testX = data.to_numpy()



# load the scalers
scaler_inputs = load(open('../training/scaler_inputs.pkl', 'rb'))
scaler_N = load(open('../training/scaler_N.pkl', 'rb'))
scaler_V = load(open('../training/scaler_V.pkl', 'rb'))
scaler_T = load(open('../training/scaler_T.pkl', 'rb'))
print("Loaded scalers from disk")



testX = scaler_inputs.transform(testX)


# It can be used to reconstruct the model identically.
model_N = keras.models.load_model("../fitting/my_model_N")
model_V = keras.models.load_model("../fitting/my_model_V")
model_T = keras.models.load_model("../fitting/my_model_T")
print("Loaded models from disk")



#predictions
predictionsN = model_N.predict(testX)
predictionsV = model_V.predict(testX)
predictionsT = model_T.predict(testX)

##de normalization: 
testX_dn = scaler_inputs.inverse_transform(testX).tolist()
predictionsN_dn = scaler_N.inverse_transform(predictionsN)
predictionsV_dn = scaler_V.inverse_transform(predictionsV)
predictionsT_dn = scaler_T.inverse_transform(predictionsT)


#print("Predictions")

#for i in random.sample(range(1, 110), 20):
#	print('%s => %s' % (testX_dn[i], predictionsN_dn[i],))

print(data)
data['n[cm^-3]'] = predictionsN_dn
data['v[km/s]'] = predictionsV_dn
data['T [MK]'] = predictionsT_dn
print(data)

#TODO: gerar csvs

data.to_csv("predicted_profile_wso_CR1992_line_0001.csv")