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
import lightgbm as lgbm

DATA_DIR = 'washed'
FILE_NAME = 'train_w.csv'
INPUT_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

COLUMN_NAMES = ['distance',
                #'pickup_hour',
                'pickup_year',
                'pickup_longitude',
                'dropoff_longitude',
                'pickup_latitude',
                'dropoff_latitude',
                'move_longitude', 
                'move_latitude',
                'pickup_distance_center', 
                #'pickup_degree_center',
                'dropoff_distance_center',
                #'dropoff_degree_center',
                'jfk_trip',
                'lga_trip',
                'ewr_trip',
                #'reverse_distance',
                #'log_distance',
                'base_type',
                ]

EMPIRE_STATE_BUILDING = (-73.985428, 40.748817)

CHUNKSIZE = 1000000

JFK_RANGE = (-73.822381, -73.752368, 40.641365, 40.664807)  #40.641312,-73.778137
LGA_RANGE = (-73.885075, -73.854447, 40.766593, 40.776928) #40.776928,-73.873962
EWR_RANGE = (-74.201931, -74.148052, 40.669234, 40.709064) #40.703869,-74.176071

def get_airport_mask(d, airport):
    r = JFK_RANGE
    if airport == 'JFK':
        r = JFK_RANGE
    elif airport == 'LGA':
        r = LGA_RANGE
    elif airport == 'EWR':
        r = EWR_RANGE
    return ((d.pickup_longitude >= r[0]) & (d.pickup_longitude <= r[1]) & \
           (d.pickup_latitude >= r[2]) & (d.pickup_latitude <= r[3])) | \
           ((d.dropoff_longitude >= r[0]) & (d.dropoff_longitude <= r[1]) & \
           (d.dropoff_latitude >= r[2]) & (d.dropoff_latitude <= r[3]))

def get_night_hour_mask(d):
    return (d.pickup_hour >= 20) & (d.pickup_hour < 6)

def get_rush_hour_mask(d):
    return (d.pickup_hour >= 16) & (d.pickup_hour < 20) & (d.pickup_weekday >= 0) & (d.pickup_weekday <= 4)

def get_sphere_dist(d):
    R = 6371
    p_lon, p_lat, d_lon, d_lat = map(np.radians,[d.pickup_longitude, d.pickup_latitude, d.dropoff_longitude, d.dropoff_latitude])
    mlat = d_lat - p_lat
    mlon = d_lon - p_lon
    
    #Compute haversine distance
    a = np.sin(mlat/2.0)**2 + np.cos(p_lat) * np.cos(d_lat) * np.sin(mlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def gen_feature(d):
    # d = d.assign(distance=((d.pickup_longitude - d.dropoff_longitude).pow(2) + \
    #     (d.pickup_latitude - d.dropoff_latitude).pow(2)).pow(0.5))
    d = d.assign(distance=get_sphere_dist(d))
    d = d.assign(pickup_hour=np.floor(d.pickup_daytime/3600))
    d = d.assign(move_longitude=(d.dropoff_longitude - d.pickup_longitude).abs())
    d = d.assign(move_latitude=(d.dropoff_latitude - d.pickup_latitude).abs())
    d = d.assign(pickup_distance_center=((d.pickup_longitude - EMPIRE_STATE_BUILDING[0]).pow(2) + \
        (d.pickup_latitude - EMPIRE_STATE_BUILDING[1]).pow(2)).pow(0.5))
    #d = d.assign(pickup_degree_center=np.arcsin((d.pickup_latitude - EMPIRE_STATE_BUILDING[1])/(d.pickup_longitude - EMPIRE_STATE_BUILDING[0])))
    d = d.assign(dropoff_distance_center=((d.dropoff_longitude - EMPIRE_STATE_BUILDING[0]).pow(2) + \
        (d.dropoff_latitude - EMPIRE_STATE_BUILDING[1]).pow(2)).pow(0.5))
    #d = d.assign(dropoff_degree_center=np.arcsin((d.dropoff_latitude - EMPIRE_STATE_BUILDING[1])/(d.dropoff_longitude - EMPIRE_STATE_BUILDING[0])))
    d['jfk_trip'] = 0
    d.jfk_trip.loc[get_airport_mask(d, 'JFK')] = 1
    d['ewr_trip'] = 0
    d.ewr_trip.loc[get_airport_mask(d, 'EWR')] = 1
    d['lga_trip'] = 0
    d.lga_trip.loc[get_airport_mask(d, 'LGA')] = 1
    #d = d.assign(reverse_distance=(1/d.distance))
    # d = d.assign(log_distance=np.log(d.distance*110))
    d['base_type'] = 0
    d.base_type.loc[get_rush_hour_mask(d)] = 2
    d.base_type.loc[get_night_hour_mask(d)] = 1
    return d

df = pd.read_csv(INPUT_FILE_PATH, nrows=CHUNKSIZE)
df = gen_feature(df)
# print (df.describe())

testdf = pd.read_csv('washed\\test_fd.csv')
testdf = gen_feature(testdf)

# jfkmask = get_jfk_mask(testdf)

# print (np.sum(jfkmask))

# exit(1)

X = df[COLUMN_NAMES]
y = df.fare_amount

train_in, test_in, train_f, test_f = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=1)

def write_predicted(d, modelname, datasize, param='', rmse='na'):
    tstring = str(time.time())
    d.to_csv('predicted\\{}_predicted.csv'.format(tstring), index=False)
    with open('predicted\\{}_feature.txt'.format(tstring), 'w') as f:
        f.write(', '.join(COLUMN_NAMES)+
                '\n{} data points used.'.format(datasize)+
                '\n{} model'.format(modelname)+
                '\n{} xgboost param'.format(str(param))+
                '\n{} rmse'.format(rmse))
        f.close()
# exit()

# xgboost

xg_train = xgb.DMatrix(train_in, train_f)
xg_test = xgb.DMatrix(test_in, test_f)
xg_params = {
    'booster': 'gbtree',
    'boosting_type': 'gbdt',
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'eta': 0.05,
    'max_depth': 12,
    'seed': 2016,
    'silent': 1,
    'eval_metric': 'rmse'
}
watchlist = [(xg_test, 'test')]

xg_r = xgb.train(xg_params, xg_train, 200,
                 watchlist, early_stopping_rounds=50)

xg_rmse = np.sqrt(mean_squared_error(test_f, xg_r.predict(xgb.DMatrix(test_in))))
print ("MSE: {}".format(xg_rmse))
xg_pY = xg_r.predict(xgb.DMatrix(testdf[COLUMN_NAMES]))
xg_predicted = testdf[['key']].copy()
xg_predicted = xg_predicted.assign(fare_amount=pd.Series(xg_pY))
write_predicted(xg_predicted, 'xgboost regression', CHUNKSIZE, param=xg_params, rmse=xg_rmse)


#lightGBM

# lgbm_params = {
#         'boosting_type':'gbdt',
#         'objective': 'regression',
#         'nthread': 4,
#         'num_leaves': 31,
#         'learning_rate': 0.05,
#         'max_depth': -1,
#         'subsample': 0.8,
#         'bagging_fraction' : 1,
#         'max_bin' : 5000 ,
#         'bagging_freq': 20,
#         'colsample_bytree': 0.6,
#         'metric': 'rmse',
#         'min_split_gain': 0.5,
#         'min_child_weight': 1,
#         'min_child_samples': 10,
#         'scale_pos_weight':1,
#         'zero_as_missing': True,
#         'seed':0,
#         'num_rounds':50000
#     }
# catagorical_features = ['pickup_year',
#                 'jfk_trip',
#                 'lga_trip',
#                 'ewr_trip',
#                 'base_type',]
# lgbm_train = lgbm.Dataset(train_in, train_f, silent=False, categorical_feature=catagorical_features)
# lgbm_test = lgbm.Dataset(test_in, test_f, silent=False, categorical_feature=catagorical_features)
# lgbm_r = lgbm.train(lgbm_params, 
#                     train_set=lgbm_train, 
#                     valid_sets=lgbm_test,
#                     num_boost_round=10000,
#                     early_stopping_rounds=500,
#                     verbose_eval=500)
# lgbm_rmse = np.sqrt(mean_squared_error(test_f, lgbm_r.predict(test_in, num_iteration=lgbm_r.best_iteration)))
# print ("MSE: {}".format(lgbm_rmse))
# lgbm_pY = lgbm_r.predict(testdf[COLUMN_NAMES], num_iteration = lgbm_r.best_iteration)
# lgbm_predicted = testdf[['key']].copy()
# lgbm_predicted = lgbm_predicted.assign(fare_amount=pd.Series(lgbm_pY))
# write_predicted(lgbm_predicted, 'lightGBM regression', CHUNKSIZE, param=lgbm_params, rmse=lgbm_rmse)
