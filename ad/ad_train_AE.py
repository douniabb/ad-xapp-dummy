# ==================================================================================
#  Copyright (c) 2020 HCL Technologies Limited.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==================================================================================

import joblib
from ad_model.processing import PREPROCESS
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import IsolationForest
from database_dummy import DATABASE, DUMMY
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras.models import load_model, model_from_json

class AutoEncoder(keras.Model):
    """
    Parameters
    ----------
    output_units: int
        Number of ouput units
        
    code_size: int - Default: 8
        Number of units in bottleneck - needs to be smaller than number of features
    """
    def __init__(self, output_units, code_size=4):
        super(AutoEncoder,self).__init__()
        self.encoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(code_size, activation='relu')
        ])
        
        self.decoder = keras.Sequential([
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(output_units, activation='sigmoid')
        ])
      
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
    

class modelling(object):
    r""" The modelling class takes input as dataframe or array and train Isolation Forest model

    Paramteres
    .........
    data: DataFrame or array
        input dataset
    cols: list
        list of parameters in input dataset

    Attributes
    ----------
    actual:array
        actual label for test data
    X: DataFrame or array
        transformed values of input data
    """
    def __init__(self, data):
        self.data = data   # valid.csv without anomaly
        self.cols = data.columns   #cols : ['prb_usage', 'throughput', 'rsrq', 'rssinr']

    def read_test(self, db):
        """ Read test dataset for model validation"""

        db.read_data('valid') #same as train, with Anomaly column
        test = db.data
        self.actual = test['Anomaly'] 
        X = test[self.cols]
        sc = joblib.load('scale')
        self.X = sc.transform(X)

    def AE(self):
        """ Train isolation forest

         Parameters
          ----------
          output_units: int
            Number of output units (number of features)

          code_size: int
            Number of units in bottleneck - less than number of features
        """
        # Input: Use only normal data for training
        train_index = self.actual[self.actual==0].index
        train_data = self.data.loc[train_index]
        
        #Normalize data for value [0,1] --> min max scale the input data
        min_max_scaler =  sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
        x_train_scaled =  min_max_scaler.fit_transform(train_data.copy())
        x_test_scaled =  min_max_scaler.transform(self.X.copy())
        
        joblib.dump(min_max_scaler, 'AEscale')
        
        model = AutoEncoder(output_units=x_train_scaled.shape[1])
        model.compile(loss='mse', metrics=['accuracy'], optimizer='adam')
        
        history = model.fit(
            x_train_scaled,
            x_train_scaled,
            epochs=50,
            batch_size=512,
            validation_data=(x_test_scaled, x_test_scaled)
        )
        
        
        # Detect Anomalies on Test data
        # Anomalies are data points where reconstruction loss is higher
        # Reconstruction loss = MSE(y_test, y_pred)
        threshold = find_threshold(model, x_train_scaled)
        #print(f"Threshold for anomaly scores: {threshold}")
        joblib.dump(threshold, 'thr')
        
        joblib.dump(self.cols, 'params')
        
        model.save('model')



        
def find_threshold(model, x_train_scaled):   
    reconstructions = model.predict(x_train_scaled)
    # provides losses of individual instances - MSLE, MAE, MSE?
    reconstruction_errors = keras.losses.msle(reconstructions, x_train_scaled) #train_loss
    # Threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
    return threshold 


def get_predictions(model, data, threshold):
    predictions = model.predict(data)
    # provides losses of individual instances
    errors = keras.losses.msle(predictions, data)
    # 1 = anomaly, 0 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 1.0 if x == True else 0.0)
    return preds  


def train(thread=True):
    """
     Main function to perform training on input data
    """
    if thread:
        db = DUMMY()
    else:
        db = DATABASE('UEData')
    db.read_data('train')
    ps = PREPROCESS(db.data)
    ps.process()
    df = ps.data

    mod = modelling(df)
    mod.read_test(db)
    
    mod.AE() 

    print('Training Ends : ')



