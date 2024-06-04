import os
from time import time
import pandas as pd
import numpy as np
import utility_functions as uf
import itertools
from utility_functions import Station
import multiprocessing as mp
import random
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
# print(hs_polygons_gdf.head())
# t1 = time()
# print(f'Loaded {len(hs_polygons_gdf)} polygons in {t1 - t0:.1f} seconds')
# print(asdf)

attr_cols = ['Drainage_Area_km2','Elevation_m',
       'Slope_deg', 'Gravelius', 'Perimeter', 
       'Aspect_deg', 'Land_Use_Forest_frac',
       'Land_Use_Grass_frac', 'Land_Use_Wetland_frac', 'Land_Use_Water_frac',
       'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac', 'Land_Use_Crops_frac',
       'Land_Use_Snow_Ice_frac', 
       'Permeability_logk_m2', 'Porosity_frac']
attr_df = hs_df[attr_cols].copy()
attr_df.index = hs_df['Official_ID'].values

# normalize the attribute columns to the range 0, 1
attr_df = (attr_df - attr_df.min()) / (attr_df.max() - attr_df.min())

stn_ids = records_df.columns

id_pairs = itertools.combinations(stn_ids, 2)
# id_pairs = itertools.permutations(stn_ids, 2)


def check_concurrency(pair, threshold):
    records = records_df[list(pair)].copy().dropna(how='any')
    # for each row, if one column value is >= 365, the row should be dropped if the other column value is < 90% of 365
    # if both values in the row are < 365, the row should be dropped if either value is < 95% of 365
    # otherwise keep the row
    records['neither_complete'] = records.apply(lambda row: (row < 365 * threshold).any(), axis=1)
    records = records[~records['neither_complete']]
    return records

def simulate_q(df, proxy, target):
    # print(f'proxy: {proxy.id}, target: {target.id}')
    # simulate a quantization of the proxy and target da
    target.log_sim_label = f'{target.id}_log10_sim'
    df[target.log_sim_label] = np.log10(df[proxy.Official_ID] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2))
    proxy.log_sim_label = f'{proxy.id}_log10_sim'
    df[proxy.log_sim_label] = np.log10(df[target.Official_ID] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2))
    return df

def process_compression(inputs):
    proxy, target, bitrate, completeness_threshold, n_years = inputs    
    
    result = {'proxy': proxy, 'target': target, 'bitrate': bitrate, 'completeness_threshold': completeness_threshold}
    
    concurrent_years = check_concurrency((proxy, target), completeness_threshold)

    result['num_concurrent_years'] = len(concurrent_years) 
    if len(concurrent_years) < 10:
        return None

    df = uf.retrieve_concurrent_data((proxy, target), concurrent_years.index.values)
    if df.empty:
        return result

    station_info = {
        'proxy': hs_df[hs_df['Official_ID'] == proxy].copy().to_dict(orient='records')[0],
        'target': hs_df[hs_df['Official_ID'] == target].copy().to_dict(orient='records')[0]
        }
    
    # compute distance
    result['distance'] = uf.compute_distance(station_info['proxy'], station_info['target'])

    # use the first n_years to quantize
    good_years = sorted(list(set(df['year'].values)))
    years_to_include = good_years[:n_years]
    basis = df[df['year'].isin(years_to_include)].copy()
    
    proxy = Station(station_info['proxy'])
    target = Station(station_info['target'])

    basis = uf.transform_and_jitter(basis, proxy, bitrate)
    basis = uf.transform_and_jitter(basis, target, bitrate)

    # basis = simulate_q(basis, proxy, target)

    df, uniform_edges, equiprob_edges, prop_edges = uf.digitize_series(basis, target, bitrate)
    result['uniform_edges'] = list(uniform_edges)
    result['equiprobable_edges'] = list(equiprob_edges)
    result['proportional_edges'] = list(prop_edges)

    # CHECK IF POLYGONS OVERLAP!  USE TP AS A PCT OF AREA
    # FEATURE 1: TARGET AREA OVERLAP WITH 

    for m in ['uniform', 'proportional', 'equiprobable']:
        try:
            uare = uf.compute_UARE(df, proxy, m, target, bitrate)
            result.update(uare)
        except Exception as e:
            print(e)
            print(m, proxy.id, target.id)

    # df = uf.compute_forecast_score(df, basis, proxy, target, bitrate)
    # find the L1 and L2 attribute distances
    result['L1_attr_dist'] = np.sum(np.abs(attr_df.loc[proxy.id] - attr_df.loc[target.id])) #+ result['distance']
    result['L2_attr_dist'] = np.sqrt(np.sum(np.square(attr_df.loc[proxy.id] - attr_df.loc[target.id]))) #+ result['distance']
    
    return result


t0 = time()

test_sample_size = 10000
# generate a smaller random sample of pairs
random.seed(42)
sample_pairs = random.sample(list(id_pairs), test_sample_size)
ta = time()
print(f'Generated {test_sample_size} random pairs in {ta - t0:.1f} seconds')
completeness_threshold = 0.9

for n_years in [1, 2, 3, 4, 5, 10]:
    for bitrate in [4, 5, 6, 7, 8]:
        print(f'processing {bitrate} bit signals ({n_years} years)')
        inputs = [(p, t, bitrate, completeness_threshold, n_years) for p, t in sample_pairs]

        pl = mp.Pool()
        results = pl.map(process_compression, inputs)
        results = [r for r in results if r is not None]
        pl.close()
        t1 = time()

        print(f'Processed {test_sample_size} in {t1 - t0:.1f} seconds')
        date = '20240126'
        results_fname = f'compression_test_results_{bitrate}bits_{n_years}years_{date}.csv'
        results_df = pd.DataFrame(results)

        results_df.to_csv(os.path.join(PROCESSED_DATA_DIR, results_fname), index=False)
