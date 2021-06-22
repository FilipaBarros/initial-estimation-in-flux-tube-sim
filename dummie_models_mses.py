#discover max values in each output 
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import QuantileTransformer

data = pd.read_csv("../../data_joining/merged_sample_1_data.csv")


data = data.to_numpy()


#inputs vs outputs
N = data[:,9:10]
V = data[:,10:11]
T = data[:,11:12]


scaler_N  = QuantileTransformer()
scaler_V  = QuantileTransformer()
scaler_T  = QuantileTransformer()

scaler_N.fit(N)
scaler_V.fit(V)
scaler_T.fit(T)

N = scaler_N.transform(N)
V = scaler_V.transform(V)
T = scaler_T.transform(T)


max_N = np.max(N)
min_N = np.min(N)
mean_N = np.mean(N)

max_V = np.max(V)
min_V = np.min(V)
mean_V = np.mean(V)

max_T = np.max(T)
min_T = np.min(T)
mean_T = np.mean(T)

print(min_T)
print(max_T)
#----------Dummie model - medians----------#
predictions_N = (np.full((12940, 1), mean_N))
predictions_V = (np.full((12940, 1), mean_V))
predictions_T = (np.full((12940, 1), mean_T))
predictions_N = scaler_N.transform(predictions_N)
predictions_V = scaler_V.transform(predictions_V)
predictions_T = scaler_T.transform(predictions_T)


mse_N = mean_squared_error(N,predictions_N)
mse_V = mean_squared_error(V,predictions_V)
mse_T = mean_squared_error(T,predictions_T)

print("median model mses: N->", mse_N, " V->" , mse_V, " T->", mse_T)

#----------random model----------#
predictions_N2 = np.random.uniform(min_N,max_N,12940)
predictions_V2 = np.random.uniform(min_V,max_V,12940)
predictions_T2 = np.random.uniform(min_T,max_T,12940)

#predictions_N = scaler_N.transform(predictions_N)
#predictions_V = scaler_V.transform(predictions_V)
#predictions_T = scaler_T.transform(predictions_T)



mse_N = mean_squared_error(N,predictions_N2)
mse_V = mean_squared_error(V,predictions_V2)
mse_T = mean_squared_error(T,predictions_T2)

print("random model mses: N->", mse_N, " V->" , mse_V, " T->", mse_T)
