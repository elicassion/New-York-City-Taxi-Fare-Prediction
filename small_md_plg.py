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

DATA_DIR = 'washed'
FILE_NAME = 'train_wfd.csv'
INPUT_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

COLUMN_NAMES = ['pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
                'pickup_year', 'pickup_month', 'pickup_day', 'pickup_daytime']

CHUNKSIZE = 2000000

df = pd.read_csv(INPUT_FILE_PATH, nrows=CHUNKSIZE)

testdf = pd.read_csv('washed\\test_fd.csv')


X = df[COLUMN_NAMES]
y = df['fare_amount']

train_in, test_in, train_f, test_f = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=1)

lr = sklearn.linear_model.LinearRegression()
lr.fit(train_in, train_f)
# pY = lr.predict(test_in)
s = lr.score(test_in, test_f)
print (s)
pY = lr.predict(testdf[COLUMN_NAMES])

predicted = testdf[['key']].copy()
predicted = predicted.assign(fare_amount=pd.Series(pY))
predicted.to_csv('predicted\\{}_predicted.csv'.format(time.time()), index=False)
