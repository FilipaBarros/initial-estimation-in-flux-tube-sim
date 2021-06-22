import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from pickle import dump
from keras.layers import Dropout
from keras import regularizers
from sklearn.preprocessing import QuantileTransformer

#data = pd.read_csv("data_joining/merged_sample_1_data.csv")
data = pd.read_csv("../../data_joining/merged_sample_1_data.csv")
print(data)
print(data.dtypes)
#split data into testing and training 


msk = np.random.rand(len(data)) < 0.7
train = data[msk]
test_and_val = data[~msk]

msk_val = np.random.rand(len(test_and_val)) < 0.5

test = test_and_val[msk_val]
val = test_and_val[~msk_val]


training = train.to_numpy()
testing = test.to_numpy()
validation = val.to_numpy()

#inputs vs outputs
trainX = training[:,0:9] 
trainN = training[:,9:10]
trainV = training[:,10:11]
trainT = training[:,11:12]

testX = testing[:,0:9]
testN = testing[:,9:10]
testV = testing[:,10:11]
testT = testing[:,11:12]


valX = validation[:,0:9]
valN  = validation[:,9:10]
valV = validation[:,10:11]
valT = validation[:,11:12]

#scalling the data 
scaler_inputs = QuantileTransformer()
#scaler_N = QuantileTransformer()
#scaler_V = QuantileTransformer()
#scaler_T = QuantileTransformer()

scaler_inputs.fit(trainX)
#scaler_N.fit(trainN)
#scaler_V.fit(trainV)
#scaler_T.fit(trainT)

trainX = scaler_inputs.transform(trainX)
#trainN = scaler_N.transform(trainN)
#trainV = scaler_V.transform(trainV)
#trainT = scaler_T.transform(trainT)

testX = scaler_inputs.transform(testX)
#testN = scaler_N.transform(testN)
#testV = scaler_V.transform(testV)
#testT = scaler_T.transform(testT)


valX = scaler_inputs.transform(valX)
#valN = scaler_N.transform(valN)
#valV = scaler_V.transform(valV)
#valT = scaler_T.transform(valT)



#---------- define, compile and fit the keras models ----------
#---------- N ----------
model_N = Sequential()
model_N.add(Dense(44, input_dim=9, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_N.add(Dropout(0.2))
model_N.add(Dense(44, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_N.add(Dropout(0.2))
model_N.add(Dense(1, activation='relu')) #defines output layer with 3 nodes

model_N.compile(loss='mean_squared_error', optimizer='adam') #loss_weights = loss_weights 
history_N = model_N.fit(trainX, trainN, validation_data=(valX, valN), epochs=500)

#---------- V ----------
model_V = Sequential()
model_V.add(Dense(64, input_dim=9, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_V.add(Dropout(0.2))
model_N.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_N.add(Dropout(0.2))
model_V.add(Dense(1, activation='relu')) #defines output layer with 3 nodes

model_V.compile(loss='mean_squared_error', optimizer='adam') #loss_weights = loss_weights 
history_V = model_N.fit(trainX, trainV, validation_data=(valX, valV), epochs=500)


#---------- T ----------
model_T = Sequential()
model_T.add(Dense(64, input_dim=9, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_T.add(Dropout(0.2))
model_T.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model_T.add(Dropout(0.2))
model_T.add(Dense(1, activation='relu')) #defines output layer with 3 nodes

model_T.compile(loss='mean_squared_error', optimizer='adam') #loss_weights = loss_weights 
history_T = model_T.fit(trainX, trainT, validation_data=(valX, valT), epochs=500)







#---------- evaluate the models-----------
#---------- N ----------
train_mse = model_N.evaluate(trainX, trainN, verbose=0)
validation_mse = model_N.evaluate(valX,valN, verbose=0)
test_mse = model_N.evaluate(testX, testN, verbose=0)
print("Model N evaluation")
print('Train: ', train_mse, ' Validation:' , validation_mse, ' Test: ' , test_mse)
# plot loss during training
plt.title('N model - Loss / Mean Squared Error')
plt.plot(history_N.history['loss'][:100], label='train')
plt.plot(history_N.history['val_loss'][:100], label='val')
plt.legend()
plt.show()
plt.savefig("model_N_mse_history.png")

plt.cla()
plt.clf()
plt.close()

predictionsN = model_N.predict(testX)

##de normalization: 
testX_dn = scaler_inputs.inverse_transform(testX).tolist()
#testN_dn = scaler_N.inverse_transform(testN.tolist())
#predictionsN_dn = scaler_N.inverse_transform(predictionsN)

error_N_predictions = abs((predictionsN - testN))
plt.title('N Absolute Error')
x = range(1, len(error_N_predictions)+1)
plt.scatter(x,error_N_predictions)
plt.legend()
plt.show()
plt.savefig("model_N_predictions_mse.png")

plt.cla()
plt.clf()
plt.close()


#---------- V ----------
train_mse = model_V.evaluate(trainX, trainV, verbose=0)
validation_mse = model_V.evaluate(valX,valV, verbose=0)
test_mse = model_V.evaluate(testX, testV, verbose=0)
print("Model V evaluation")
print('Train: ', train_mse, ' Validation:' , validation_mse, ' Test: ' , test_mse)
# plot loss during training
plt.title('V  model -Loss / Mean Squared Error')
plt.plot(history_V.history['loss'][:100], label='train')
plt.plot(history_V.history['val_loss'][:100], label='val')
plt.legend()
plt.show()
plt.savefig("model_V_mse_history.png")

plt.cla()
plt.clf()
plt.close()

predictionsV = model_V.predict(testX)

##de normalization: 
#testV_dn = scaler_V.inverse_transform(testV.tolist())
#predictionsV_dn = scaler_V.inverse_transform(predictionsV)

error_V_predictions = abs((predictionsV - testV))
plt.title('V Absolute Error')
x = range(1, len(error_V_predictions)+1)
plt.scatter(x,error_V_predictions)
plt.legend()
plt.show()
plt.savefig("model_V_predictions_mse.png")

plt.cla()
plt.clf()
plt.close()

#---------- T ----------
train_mse = model_T.evaluate(trainX, trainT, verbose=0)
validation_mse = model_T.evaluate(valX,valT, verbose=0)
test_mse = model_T.evaluate(testX, testT, verbose=0)
print("Model T evaluation")
print('Train: ', train_mse, ' Validation:' , validation_mse, ' Test: ' , test_mse)
# plot loss during training
plt.title('T  model -Loss / Mean Squared Error')
plt.plot(history_T.history['loss'][:100], label='train')
plt.plot(history_T.history['val_loss'][:100], label='val')
plt.legend()
plt.show()
plt.savefig("model_T_mse_history.png")

plt.cla()
plt.clf()
plt.close()

predictionsT = model_T.predict(testX)

##de normalization: 
#testT_dn = scaler_T.inverse_transform(testT.tolist())
#predictionsT_dn = scaler_T.inverse_transform(predictionsT)

error_T_predictions = abs((predictionsT - testT))
plt.title('T Absolute Error')
x = range(1, len(error_T_predictions)+1)
plt.scatter(x,error_T_predictions)
plt.legend()
plt.show()
plt.savefig("model_T_predictions_mse.png")

plt.cla()
plt.clf()
plt.close()



# save the models
model_N.save("my_model_N")
model_V.save("my_model_V")
model_T.save("my_model_T")
# save the scaler
dump(scaler_inputs, open('scaler_inputs.pkl', 'wb'))
#dump(scaler_N, open('scaler_N.pkl', 'wb'))
#dump(scaler_V, open('scaler_V.pkl', 'wb'))
#dump(scaler_T, open('scaler_T.pkl', 'wb'))
print("Saved models and scalers to disk")



for i in random.sample(range(1, 110), 20):
	print('%s => %s' % (testX_dn[i], predictionsN[i]))