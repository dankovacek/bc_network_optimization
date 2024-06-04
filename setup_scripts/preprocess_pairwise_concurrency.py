import os

import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools
from time import time

import utility_functions as uf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS_DATA_PATH = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
HS_TS_PATH = os.path.join(HS_DATA_PATH, 'hysets_series')

all_station_ids = [f.split('.')[0] for f in os.listdir(HS_TS_PATH) if f.endswith('.csv')]


print(f'Checking {len(all_station_ids)} station records')

# inputs = [(pair[0], pair[1], pct_complete) for pair in pairs]
pool = mp.Pool(2)
results = pool.map(uf.check_completeness, all_station_ids)


df = pd.concat(results, join='outer', axis=1)
df.to_csv(f'processed_data/stn_observation_counts_by_year.csv')