import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import gc
import json
import time
import sklearn
from sklearn import *
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin

DATA_DIR = 'washed'
FILE_NAME = 'train_wfd.csv'
INPUT_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

COLUMN_NAMES = ['distance', 'pickup_longitude',
                'dropoff_longitude', 'pickup_hour', 'pickup_year']

CHUNKSIZE = 500000

JFK_RANGE = (-73.8250, -73.7746, 40.6397, 40.7121)

def get_jfk_mask(d):
    return (d.pickup_longitude >= JFK_RANGE[0]) & (d.pickup_longitude <= JFK_RANGE[1]) & \
           (d.pickup_latitude >= JFK_RANGE[2]) & (d.pickup_latitude <= JFK_RANGE[3]) | \
           (d.dropoff_longitude >= JFK_RANGE[0]) & (d.dropoff_longitude <= JFK_RANGE[1]) & \
           (d.dropoff_latitude >= JFK_RANGE[2]) & (d.dropoff_latitude <= JFK_RANGE[3])

df = pd.read_csv(INPUT_FILE_PATH, nrows=CHUNKSIZE)
df = df.assign(distance=((df.pickup_longitude-df.dropoff_longitude).pow(2) + (df.pickup_latitude-df.dropoff_latitude).pow(2)).pow(0.5))
df = df.assign(pickup_hour=(df.pickup_daytime/3600).round())

testdf = pd.read_csv('washed\\test_fd.csv')
testdf = testdf.assign(distance=((testdf.pickup_longitude-testdf.dropoff_longitude).pow(2) + (testdf.pickup_latitude-testdf.dropoff_latitude).pow(2)).pow(0.5))
testdf = testdf.assign(pickup_hour=(testdf.pickup_daytime/3600).round())

# jfkmask = get_jfk_mask(testdf)

# print (np.sum(jfkmask))

# exit(1)

X = df[COLUMN_NAMES]
y = df.fare_amount

train_in, test_in, train_f, test_f = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=1)

def write_predicted(d, modelname, datasize):
    tstring = str(time.time())
    d.to_csv('predicted\\{}_predicted.csv'.format(tstring), index=False)
    with open('predicted\\{}_feature.txt'.format(tstring), 'w') as f:
        f.write(', '.join(COLUMN_NAMES)+
                '\n{} data points used.'.format(datasize)+
                '\n{} model'.format(modelname))
        f.close()


xg_r = xgb.XGBRegressor(max_depth=3, n_estimators=500, learning_rate=0.01)\
    .fit(train_in, train_f)
xg_mse = mean_squared_error(test_f, xg_r.predict(test_in))
print("MSE: {}".format(xg_mse))
xg_pY = xg_r.predict(testdf[COLUMN_NAMES])
xg_predicted = testdf[['key']].copy()
xg_predicted = xg_predicted.assign(fare_amount=pd.Series(xg_pY))
write_predicted(xg_predicted, 'xgboost regression', CHUNKSIZE)