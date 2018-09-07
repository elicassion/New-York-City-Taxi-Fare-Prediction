import pyecharts
import pandas as pd
import numpy as np

import os
import codecs

DATA_DIR = 'data'
FILE_NAME = 'train.csv'
INPUT_FILE = os.path.join(DATA_DIR, FILE_NAME)

CHUNKSIZE = 10000
# MAX_ROW_NUM = 50000000
MAX_ROW_NUM = 10000

for cur_row_counter in range(0, MAX_ROW_NUM, CHUNKSIZE):
    df = pd.read_csv('data\\train.csv',
                     head = True,
                     chunksize = 10000,
                     )



