import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import QuantileTransformer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from keras import regularizers
from keras.layers import Dropout
import matplotlib.pyplot as plt

no_epochs = 100
optimizer = Adam()
verbosity = 1
num_folds = 10

#data = pd.read_csv("data_joining/merged_sample_1_data.csv")
data = pd.read_csv("../../../data_joining/merged_sample_1_data.csv")


data = data[data['B[G]'] >= -200]
data = data[data['A/A0'] >= 0]
data = data[data['T [MK]'] <= 6]
print(data)



msk = np.random.rand(len(data)) < 0.7
train = data[msk]
test = data[~msk]


training = train.to_numpy()
testing = test.to_numpy()

#inputs vs outputs
input_train = training[:,0:9] 
target_train = training[:,11:12]
input_test = testing[:,0:9]
target_test = testing[:,11:12]

#scalling the data 
scaler_inputs = QuantileTransformer()
scaler_outputs = QuantileTransformer()

scaler_inputs.fit(input_train)
scaler_outputs.fit(target_train)

input_train = scaler_inputs.transform(input_train)
target_train = scaler_outputs.transform(target_train)
input_test = scaler_inputs.transform(input_test)
target_test = scaler_outputs.transform(target_test)


#K-Folds cross-validator
#Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
#Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
# Define per-fold score containers
mse_per_fold = []
#loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(5, input_dim=9, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu')) #defines output layer with 1 nodes
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                epochs=no_epochs,
                verbose=verbosity)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores}')

    mse_per_fold.append(scores)

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(mse_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - mse: {mse_per_fold[i]}%')

plt.title('Loss / Mean Squared Error')
x = range(1, len(mse_per_fold)+1)
plt.scatter(x,mse_per_fold, label='mse_per_fold')
plt.legend()
plt.show()
plt.savefig("model_mse_1l_10nd_10folds.png")

plt.cla()
plt.clf()
plt.close()

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> mse: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')
print('------------------------------------------------------------------------')