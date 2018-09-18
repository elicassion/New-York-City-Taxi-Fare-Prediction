import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import gc
import json
import time

DATA_DIR = 'data'
FILE_NAME = 'train.csv'
INPUT_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

CHUNKSIZE = 2000000
MAX_ROW_NUM = 50000000
# MAX_ROW_NUM = 3000000

NYC_RANGE = (-74.3400, -73.6700, 40.4500, 40.9500)
NYC_RANGE_EXTENDED = (-74.4667, -73.0083, 40.4203, 41.3077)
NYC_RANGE_FILENAME = 'gnyc.png'
nyc_map = plt.imread(os.path.join(DATA_DIR, NYC_RANGE_FILENAME))

NYC_EXTENDED_FILENAME = 'gnyce.png'
nyc_map_extended = plt.imread(os.path.join(DATA_DIR, NYC_EXTENDED_FILENAME))

WASHED_DATA_DIR = 'washed'
WASHED_FILE_NAME = 'train_w.csv'
OUTPUT_FILE_PATH = os.path.join(WASHED_DATA_DIR, WASHED_FILE_NAME)
if not os.path.exists(WASHED_DATA_DIR):
    os.makedirs(WASHED_DATA_DIR)


def drop_null(df):
    return df.dropna(how='any', axis='rows')

def drop_fare(df):
    return (df.fare_amount>0) & (df.fare_amount<=300)

def drop_out_of_box(df, box):
    return (df.pickup_latitude >= box[2]) & (df.pickup_latitude <= box[3]) & \
           (df.pickup_longitude >= box[0]) & (df.pickup_longitude <= box[1]) & \
           (df.dropoff_latitude >= box[2]) & (df.dropoff_latitude <= box[3]) & \
           (df.dropoff_longitude >= box[0]) & (df.dropoff_longitude <= box[1])

def lon_lat_to_pixel(lon, lat, cmap, box):
    s = cmap.shape
    return ((s[0] - abs((lat - box[2]) / (box[3] - box[2])) * s[0]).astype('int'),
            (abs((lon - box[0]) / (box[1] - box[0])) * s[1]).astype('int') )

def drop_in_water(df, cmap, box, mask):
    pickup_idx = lon_lat_to_pixel(df.pickup_longitude, df.pickup_latitude,
                                  cmap, box)
    dropoff_idx = lon_lat_to_pixel(df.dropoff_longitude, df.dropoff_latitude,
                                   cmap, box)
    merge_idx = mask[pickup_idx] & mask[dropoff_idx]
    # print ("Points in water:", np.sum(~merge_idx))
    return merge_idx
    

# def date_formatting(df):
    # new_years, new_months, new_days, new_daytime = zip(*[(int(d[0:4]), int(d[5:7]), int(d[8:10]), int(d[11:13])*3600+int(d[14:16])*60+int(d[17:19])) for d in df.pickup_datetime])
#     df = df.assign(pickup_year=new_years, pickup_month=new_months, pickup_day=new_days, pickup_daytime=new_daytime)
#     return df.drop(columns=['pickup_datetime'])

def date_formatting(df):
    # print ([d.weekday() for d in df.pickup_datetime])
    new_years, new_months, new_days, new_daytime, new_weekday = zip(*[(d.year, d.month, d.day, d.hour, d.weekday()) for d in df.pickup_datetime])
    df = df.assign(pickup_year=new_years, pickup_month=new_months, pickup_day=new_days, pickup_daytime=new_daytime, pickup_weekday=new_weekday)
    return df.drop(columns=['pickup_datetime', 'key'])

def drop_passenger_count(df):
    return (df.passenger_count >= 1) & (df.passenger_count <= 6)

def process(df, cmap, box, mask):
    ori_length = len(df)
    df = drop_null(df)
    after_null_len = len(df)
    df = df[drop_fare(df)]
    after_fare_len = len(df)
    df = df[drop_out_of_box(df, box)]
    after_box_len = len(df)
    df = df[drop_in_water(df, cmap, box, mask)]
    after_water_len = len(df)
    df = df[drop_passenger_count(df)]
    processed_length = len(df)
    df = date_formatting(df)


    if os.path.exists(OUTPUT_FILE_PATH):
        df.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, index=False)
    else:
        f = codecs.open(OUTPUT_FILE_PATH, 'w')
        f.close()
        df.to_csv(OUTPUT_FILE_PATH, mode='a', index=False)
    return ori_length, after_null_len, after_fare_len, after_box_len, after_water_len, processed_length

def get_water_mask():
    m = plt.imread(os.path.join(DATA_DIR, NYC_EXTENDED_FILENAME))
    return ~((m[:, :, 0] + 1e-2 >= 1) &
           (m[:, :, 1] + 1e-2 >= 1) &
           (m[:, :, 2] + 1e-2 >= 1))


#try:
    #pd.read_csv(INPUT_FILE_PATH, nrows=2)
#except:
    #exit(1)

# try:
start = time.time()
ori_sum = 0
processed_sum = 0
an_sum = 0
af_sum = 0
ab_sum = 0
aw_sum = 0
ap_sum = 0
range_map = nyc_map_extended
range_box = NYC_RANGE_EXTENDED
water_mask = get_water_mask()
if os.path.exists(OUTPUT_FILE_PATH):
    os.remove(OUTPUT_FILE_PATH)
for trunk in pd.read_csv(INPUT_FILE_PATH, chunksize = CHUNKSIZE, parse_dates=['pickup_datetime'], infer_datetime_format=True):
    # print (trunk.pickup_datetime)
    ori, an, af, ab, aw, processed = process(trunk, range_map, range_box, water_mask)
    ori_sum += ori
    an_sum += ori-an
    af_sum += an-af
    ab_sum += af-ab
    aw_sum += ab-aw
    ap_sum += aw-processed
    processed_sum = processed_sum + processed
    print ("[{}/{}] processed in Total, [{}/{}] for this trunk\n "
           "Found Null:[{}/{}] Abnormal Fare:[{}/{}] Out of Range:[{}/{}] In Water:[{}/{}] Wrong Passenger Count: [{}/{}]\n"
           "-------------------------------------------------------\n"
           .format(processed_sum, ori_sum, processed, ori,
                   ori-an, an_sum, an-af, af_sum,
                   af-ab, ab_sum, ab-aw, aw_sum, aw-processed, ap_sum))
print ("Time Used: {}s.".format(time.time() - start))
# except (ValueError, IOError, KeyError)as e:
#     print (e)
#     exit(1)
