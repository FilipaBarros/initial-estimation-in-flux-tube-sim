import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from pickle import load
from sklearn.preprocessing import QuantileTransformer


data = pd.read_csv("../../../data_joining/merged_sample_1_data.csv")


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


# scale the data
scaler_inputs = load(open('../training/scaler_inputs.pkl', 'rb'))
scaler_N = load(open('../training/scaler_N.pkl', 'rb'))
scaler_V = load(open('../training/scaler_V.pkl', 'rb'))
scaler_T = load(open('../training/scaler_T.pkl', 'rb'))

testX = scaler_inputs.transform(testX)
trainX =scaler_inputs.transform(trainX)
valX = scaler_inputs.transform(valX)

trainN = scaler_N.transform(trainN)
testN = scaler_N.transform(testN)
valN = scaler_N.transform(valN)

trainV = scaler_V.transform(trainV)
testV = scaler_V.transform(testV)
valV = scaler_V.transform(valV)

trainT = scaler_T.transform(trainT)
testT = scaler_T.transform(testT)
valT = scaler_T.transform(valT)



# It can be used to reconstruct the model identically.
model_N = keras.models.load_model("../../../tunner/N/hyperband_search_model")
model_V = keras.models.load_model("../../../tunner/V/random_search_model")
model_T = keras.models.load_model("../../../tunner/T/random_search_model")
print("Loaded model from disk")

#fitting: 
history_N = model_N.fit(trainX, trainN, validation_data=(valX, valN), epochs=250)
history_V = model_V.fit(trainX, trainV, validation_data=(valX, valV), epochs=250)
history_T = model_T.fit(trainX, trainT, validation_data=(valX, valT), epochs=250)


#predictions
predictionsN = model_N.predict(testX)
predictionsV = model_V.predict(testX)
predictionsT = model_T.predict(testX)

##de normalization: 
testX_dn = scaler_inputs.inverse_transform(testX).tolist()
predictionsN_dn = scaler_N.inverse_transform(predictionsN)
predictionsV_dn = scaler_V.inverse_transform(predictionsV)
predictionsT_dn = scaler_T.inverse_transform(predictionsT)
testN_dn = scaler_N.inverse_transform(testN)
testV_dn = scaler_V.inverse_transform(testV)
testT_dn = scaler_T.inverse_transform(testT)

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


error_N_predictions = abs((predictionsN_dn - testN_dn))
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


error_V_predictions = abs((predictionsV_dn - testV_dn))
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


error_T_predictions = abs((predictionsT_dn - testT_dn))
plt.title('T Absolute Error')
x = range(1, len(error_T_predictions)+1)
plt.scatter(x,error_T_predictions)
plt.legend()
plt.show()
plt.savefig("model_T_predictions_mse.png")

plt.cla()
plt.clf()
plt.close()




#de-normalization:


print("Predictions")

for i in random.sample(range(1, 110), 20):
	print('%s => %s, %s => %s, %s => %s' % (testN_dn[i], predictionsN_dn[i], testV_dn[i],predictionsV_dn[i], testT_dn[i],predictionsT_dn[i]))


# save the models
model_N.save("my_model_N")
model_V.save("my_model_V")
model_T.save("my_model_T")