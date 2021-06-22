import pandas as pd
import numpy as np
import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel
from kerastuner import RandomSearch, Hyperband, BayesianOptimization
from tensorflow.keras import models, layers
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from numpy.random import seed
from keras.utils.vis_utils import plot_model

class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape    

    def build(self, hp):
        model = Sequential()
        model.add(
            layers.Dense(
                units=hp.Int('units', 8, 64, 4, default=8),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=input_shape
            )
        )   
    
        model.add(
            layers.Dense(
                units=hp.Int('units', 16, 64, 4, default=16),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu')
            )
        )
    
        model.add(
            layers.Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.005,
                    step=0.01)
            )
        )
    
        model.add(layers.Dense(1))
    
        model.compile(
            optimizer='adam',loss='mse',metrics=['mse']
        )
    
        return model
    
    

#print(tf.__version__)
#print(kt.__version__)


data = pd.read_csv("../../data_joining/merged_sample_1_data.csv")
#print(data)
#print(data.dtypes)
#split data into testing and training 


msk = np.random.rand(len(data)) < 0.7
train = data[msk]
test = data[~msk]



training = train.to_numpy()
testing = test.to_numpy()

#inputs vs outputs
x_train = training[:,0:9] 
y_train = training[:,9:10]
x_test = testing[:,0:9]
y_test = testing[:,9:10]

#----------No search model----------
# set random seed
seed(42)
tf.random.set_seed(42)

# preprocessing - normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# model building
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1))

# compile model using adam
model.compile(optimizer='adam',loss='mse',metrics=['mse'])
# model training
history = model.fit(x_train_scaled, y_train, validation_split=0.2, epochs=1000)
# model evaluation
loss,mse = model.evaluate(x_test_scaled, y_test)

print("No Search")
print(model.summary())
print(loss,mse)






#----------Random Search----------
#hypermodel training

input_shape = (x_train.shape[1],)
hypermodel = RegressionHyperModel(input_shape)

tuner_rs = RandomSearch(
            hypermodel,
            objective='mse',
            seed=42,
            max_trials=10,
            executions_per_trial=2)

tuner_rs.search(x_train_scaled, y_train, epochs=500, validation_split=0.2, verbose=1)

best_model_r = tuner_rs.get_best_models(num_models=1)[0]
loss, mse = best_model_r.evaluate(x_test_scaled, y_test)
print("Random Search")
print(best_model_r.summary())
print(loss,mse)
best_model_r.save("random_search_model")



#----------Hyperband Search----------
tuner_hb = Hyperband(
            hypermodel,
            max_epochs=500,
            objective='mse',
            seed=42,
            executions_per_trial=2,
            overwrite=True
        )

tuner_hb.search(x_train_scaled, y_train, epochs=500, validation_split=0.2, verbose=1)

best_model_hb = tuner_hb.get_best_models(num_models=1)[0]
loss, mse = best_model_hb.evaluate(x_test_scaled, y_test)
print("Hyperband Search")
print(best_model_hb.summary())
print(loss,mse)
best_model_hb.save("hyperband_search_model")

