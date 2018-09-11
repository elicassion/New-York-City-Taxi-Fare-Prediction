import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import codecs
import gc
import json
import time

DATA_DIR = 'washed'
FILE_NAME = 'train_w.csv'
INPUT_FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

WASHED_DATA_DIR = 'washed'
WASHED_FILE_NAME = 'train_wfd.csv'
OUTPUT_FILE_PATH = os.path.join(WASHED_DATA_DIR, WASHED_FILE_NAME)
if not os.path.exists(WASHED_DATA_DIR):
    os.makedirs(WASHED_DATA_DIR)

CHUNKSIZE = 2000000



def split_datetime(df):
    new_years, new_months, new_days, new_daytime = zip(*[(int(d[0:4]), int(d[5:7]), int(d[8:10]), int(d[11:13])*3600+int(d[14:16])*60+int(d[17:19])) for d in df.pickup_datetime])
    df = df.assign(pickup_year=new_years, pickup_month=new_months, pickup_day=new_days, pickup_daytime=new_daytime)
    return df.drop(columns=['pickup_datetime', 'key'])


try:
    start = time.time()
    ori_sum = 0
    processed_sum = 0
    if os.path.exists(OUTPUT_FILE_PATH):
        os.remove(OUTPUT_FILE_PATH)
    for trunk in pd.read_csv(INPUT_FILE_PATH, chunksize = CHUNKSIZE):
        trunk = split_datetime(trunk)
        if os.path.exists(OUTPUT_FILE_PATH):
            trunk.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, index=False)
        else:
            f = codecs.open(OUTPUT_FILE_PATH, 'w')
            f.close()
            trunk.to_csv(OUTPUT_FILE_PATH, mode='a', index=False)
    print ("Time Used: {}s.".format(time.time() - start))
except (ValueError, IOError, KeyError)as e:
    exit(1)