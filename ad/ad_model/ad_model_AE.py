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
import keras
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from ad_train_AE import AutoEncoder


class modelling(object):
    r""" Filter dataframe based on paramters that were used to train model
    use transormer to transform the data
    load model and predict the label(normal/anomalous)

    Parameters:
    data:DataFrame
    """

    def __init__(self, data):

        with open('params', 'rb') as f:
            cols = joblib.load(f)
        if all(cols.isin(data.columns)):
            self.data = data[cols]
        else:
            print(data.columns, cols)

        self.transformation()

    def transformation(self):
        """ load transformer to transform data """

        sc = joblib.load('scale')
        self.data = sc.transform(self.data)
        
        #for AE - minmaxscaler
        aesc = joblib.load('AEscale')
        self.data = aesc.transform(self.data)
        
        

    def predict(self, name):
        """ Load the saved model and return predicted result.
        Parameters
        .........
        name:str
            name of model

        Return
        ......
        pred:int
            predict label on a given sample

        """      
        new_model = load_model('model')
        #new_model.summary()
        
        
        #self.data = tf.convert_to_tensor(self.data)
        pred = new_model(self.data)  #predict
        
        # provides losses of individual instances
        errors = keras.losses.msle(pred, self.data)
        #print(errors)
        # 1 = anomaly, 0 = normal
        threshold = joblib.load('thr')
        anomaly_mask = pd.Series(errors) > threshold
        pred = anomaly_mask.map(lambda x: 1 if x == True else 0)
        pred = pred.values.tolist()
        return pred



class CAUSE(object):
    r""""Rule basd method to find degradation type of anomalous sample

    Attributes:
    normal:DataFrame
        Dataframe that contains only normal sample
    """

    def __init__(self, db):
        db.read_data('train')
        self.normal = db.data[['rsrp', 'rsrq', 'rssinr', 'throughput', 'prb_usage', 'ue-id']]

    def cause(self, sample):
        """ Filter normal data for a particular ue-id to compare with a given sample
            Compare with normal data to find and return degradaton type
        """
        normal = self.normal[self.normal['ue-id'] == sample.iloc[0]['ue-id']].drop('ue-id', axis=1)
        param = self.find(sample, normal.max())
        return param

    def find(self, sample, Range):
        """ store if a particular parameter is below threshold and return """

        deg = []
        if sample.iloc[0]['throughput'] < Range['throughput']*0.5:
            deg.append('Throughput')
        if sample.iloc[0]['rsrp'] <= Range['rsrp']-20:
            deg.append('RSRP')
        if sample.iloc[0]['rsrq'] <= Range['rsrq']-20:
            deg.append('RSRQ')
        if sample.iloc[0]['rssinr'] <= Range['rssinr']-25:
            deg.append('RSSINR')
        if sample.iloc[0]['prb_usage'] <= Range['prb_usage']*0.5:
            deg.append('prb_usage')
        if len(deg) == 0:
            deg = False
        else:
            deg = ' '.join(deg)
        return deg


def ad_predict(df):
    """
        Call Predict method to predict whether a given sample is normal or anomalous
    """

    db = modelling(df)
    #db_df = db.predict('model')  # Calls predict module and store the result into db_df
    db_df = db.predict('model')
    del db
    return db_df
