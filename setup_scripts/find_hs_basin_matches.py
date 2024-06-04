import psycopg2
import psycopg2.extras as extras
import os
import warnings
import re
from time import time
import pandas as pd
import random
from shapely.validation import make_valid

import numpy as np
import multiprocessing as mp

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'input_data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
HS_DATA_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'

records_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'stn_observation_counts_by_year.csv'), index_col=0)
hs_properties_path = os.path.join(HS_DATA_DIR, 'HYSETS_watershed_properties.txt')
hs_df = pd.read_csv(hs_properties_path, sep=';')

t0 = time()
hs_basins_path = os.path.join(HS_DATA_DIR, 'HYSETS_watershed_boundaries/HYSETS_watershed_boundaries_20200730.shp')
# installing pyogrio speeds up geopandas read_file by 10x with use_arrow=True
hs_polygons_gdf = gpd.read_file(hs_basins_path, engine='pyogrio', use_arrow=True)
print(hs_polygons_gdf.head())
t1 = time()
print(f'Loaded {len(hs_polygons_gdf)} polygons in {t1 - t0:.1f} seconds')


conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}
schema_name = 'basins_schema'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WSC_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/WSC_data/'

def get_updated_WSC_polygon(stn_id):
    prefix = stn_id[:2]
    fpath = os.path.join(WSC_DIR, f'{prefix}/{stn_id}')
    
    polygon_file = f'{stn_id}_DrainageBasin_BassinDeDrainage.shp'
    ppt_file = f'{stn_id}_PourPoint_PointExutoire.shp'
    polygon_path = os.path.join(fpath, polygon_file)
    ppt_path = os.path.join(fpath, ppt_file)
    polygon_gdf = gpd.read_file(polygon_path)
    ppt_gdf = gpd.read_file(ppt_path)
    print(polygon_gdf.crs.to_epsg(), ppt_gdf.crs.to_epsg())
    print(polygon_gdf)
    print('')
    print(asdfads)
    return ppt_gdf, polygon_gdf

def main():
    for i, row in hs_df.iterrows():
        ab_flag = row['Flag_Artificial_Boundaries']
        source = row['Source']
        stn_id = row['Official_ID']
        if source == 'HYDAT':
            updated_polygon = get_updated_WSC_polygon(stn_id)
        print(asdfsadf)
        if ab_flag == 1:
            continue
        print(row)
        print(asdfasdf)
    pass


with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()

cur.close()
conn.close()