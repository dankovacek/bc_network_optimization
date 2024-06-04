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
from sqlalchemy import create_engine

import psycopg2

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
    'dbname': 'basins',
}
schema_name = 'basins_schema'
table_name = 'hysets_basins'
conn_str = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"

engine = create_engine(conn_str)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DATA_DIR = os.path.join(BASE_DIR, 'input_data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
HS_DATA_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'

records_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'stn_observation_counts_by_year.csv'), index_col=0)
hs_properties_path = os.path.join(HS_DATA_DIR, 'HYSETS_watershed_properties.txt')
hs_df = pd.read_csv(hs_properties_path, sep=';')

attr_cols = ['Drainage_Area_km2','Elevation_m',
       'Slope_deg', 'Gravelius', 'Perimeter', 
       'Aspect_deg', 'Land_Use_Forest_frac',
       'Land_Use_Grass_frac', 'Land_Use_Wetland_frac', 'Land_Use_Water_frac',
       'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac', 'Land_Use_Crops_frac',
       'Land_Use_Snow_Ice_frac', 
       'Permeability_logk_m2', 'Porosity_frac',]
climate_cols = ['tmax', 'tmin', 'prcp', 'srad', 'swe', 'vp', 
       'high_prcp_freq', 'low_prcp_freq', 'high_prcp_duration', 'low_prcp_duration',]

attr_df = hs_df[attr_cols].copy()
attr_df.index = hs_df['Official_ID'].values


def simulate_log_q(df, proxy, target):
    # print(f'proxy: {proxy.id}, target: {target.id}')
    # simulate a quantization of the proxy and target da
    # check if any proxy.id values are zero
    if df[proxy.id].min() == 0:
        raise ValueError(f'Zero values in {proxy.id} for {proxy.id}, {target.id}')
    df[target.log_sim_label] = np.log10(df[proxy.id] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2))
    # proxy.log_sim_label = f'{proxy.id}_log10_sim'
    # df[proxy.log_sim_label] = np.log10(df[target.id] * (proxy.Drainage_Area_km2 / target.Drainage_Area_km2))
    return df


def process_compression(inputs):
    proxy, target, bitrate, completeness_threshold, n_years = inputs
    bitrate = int(bitrate)
    completeness_threshold = float(completeness_threshold)
    n_years = int(n_years)
    proxy, target = str(proxy), str(target)
    
    station_info = {
        'proxy': hs_df[hs_df['Official_ID'] == proxy].copy().to_dict(orient='records')[0],
        'target': hs_df[hs_df['Official_ID'] == target].copy().to_dict(orient='records')[0]
    }
    
    # compute spatial distance
    centroid_distance = uf.compute_distance(station_info['proxy'], station_info['target'])
    if centroid_distance > 1000: return None

    # print(f'Stations {proxy} and {target} have {len(concurrent_years)} concurrent years')
    df = uf.retrieve_concurrent_data(proxy, target)

    if df.empty:
        return None
    
    complete_years, n_obs = uf.check_completeness(df, 0.9)
    # print(f'    {proxy}, {target} have {len(complete_years)} complete years and {n_obs} observations')
    
    # filter out as little as possible
    if len(complete_years) < n_years:
        # print(f'Not enough concurrent data for stations {proxy} and {target} ({len(concurrent_years)})')
        return None

    result = {'proxy': str(proxy), 'target': str(target), 'bitrate': bitrate, 'completeness_threshold': completeness_threshold}    
    result['centroid_distance'] = round(centroid_distance, 2)
    result['num_complete_years'] = int(len(complete_years))
    
    # don't need to add the attributes, these can be retrieved later.
    # HOWEVER, sacrificing a bit of disk space now saves computation
    for l in ['proxy', 'target']:
        for c in attr_cols + ['Centroid_Lat_deg_N', 'Centroid_Lon_deg_E']:
            result[f'{l}_{c}'] = station_info[l][c]
    
    # for stn in pair:
    proxy = Station(station_info['proxy'], bitrate)
    target = Station(station_info['target'], bitrate)

    # df = uf.transform_and_jitter(df, proxy)
    # df = uf.transform_and_jitter(df, target)

    # df = simulate_log_q(df, proxy, target)
    df[target.sim_label] = df[proxy.id] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2)

    df.reset_index(inplace=True, drop=True)

    # compute coefficient of determination
    result['cod'] = uf.compute_cod(df, proxy, target)

    # compute Nash-Sutcliffe efficiency
    result['nse'] = uf.compute_nse(df, proxy, target)

    # compute the Kling-Gupta efficiency
    result['kge'] = uf.compute_kge(df, proxy, target)

    # compute the bin edges based on equal width in log space
    # df, uniform_edges = uf.compute_bin_edges(df, target, bitrate)
    quantile_edges = uf.compute_quantile_bins(df, target, bitrate)

    # The quantization scheme based on observations (posterior P) 
    # is used to digitize both observed and similated time series
    # df = uf.quantize_series(df, uniform_edges, 'uniform', target, bitrate)
    df = uf.quantize_series(df, quantile_edges, 'quantile', target, bitrate)

    # computes the observed P and simulation Q distribution probabilities 
    # as dicts by bin number, probability key-value pairs
    # test a wide range of uniform priors via pseudo counts
    quant_label_obs = target.digit_label_obs
    quant_label_sim = target.digit_label_sim
    pseudo_counts = [1e-3, 1e-2, 1e-1, 0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    p_obs, p_sim = uf.compute_probabilities(
        df, quant_label_obs, quant_label_sim, bitrate, complete_years, pseudo_counts)
    
    dkl_df = uf.process_dkl(p_obs, p_sim, bitrate)
    # print(dkl_df)
    sum_dkl = dkl_df.sum()
    # print(f'Sum DKL: {sum_dkl.values}')
    # print(asfdd)
    if any(sum_dkl.values) < 0:        
        print(f'negative dkl')
        print(sum_dkl.values)
        print(f'{proxy.id}, {target.id}')
    elif any(sum_dkl) == 0:
        print(f'    DKL=0: {proxy.id}, {target.id} dkl = 0, n years = {len(complete_years)}')
    
    # update the result dict with the DKL sum values as a dict
    result.update(sum_dkl.to_dict())
    
    # compute the total variation distance (TVD)
    tvd = np.abs(p_sim['q_sim'].values - p_obs.values) / 2
    # compute total variation distance
    result['tvd'] = np.sum(tvd)
    
    return result


def process_climate_data(df, climate_cols=climate_cols):

    target_ids = df['target'].values
    proxy_ids = df['proxy'].values
    id_pairs = list(zip(proxy_ids, target_ids))
    diff_columns = ", ".join([f"a.{col} as target_{col}, b.{col} AS proxy_{col}" for col in climate_cols])

    batch_size = 100
    n_batches = max(len(id_pairs) // batch_size, 1)
    batches = np.array_split(id_pairs, n_batches)
    # [id_pairs[i:i + batch_size] for i in range(0, len(id_pairs), batch_size)]
    all_dfs = []
    for batch in batches:
        id_conditions = ", ".join([f"('{id_pair[0]}', '{id_pair[1]}')" for id_pair in batch])

        query = f"""
            SELECT 
                a.official_id as proxy, b.official_id as target,
                a.basin_geometry,{diff_columns}
            FROM basins_schema.hysets_basins a
            JOIN basins_schema.hysets_basins b ON (a.official_id, b.official_id) IN ({id_conditions});
            """
                
        # conn = psycopg2.connect(**conn_params)
        gdf = gpd.read_postgis(query, engine, geom_col='basin_geometry')
        ccols = []
        for col in climate_cols:
            ccols.append(f'proxy_{col}')
            ccols.append(f'target_{col}')
        all_dfs.append(gdf[['proxy', 'target'] + ccols])
        
    output = pd.concat(all_dfs) 
    
    return output
    
    
def check_processed_results(out_fpath):
    dtype_spec = {'proxy': str, 'target': str}
    if os.path.exists(out_fpath):
        results_df = pd.read_csv(out_fpath, dtype=dtype_spec)
        print(f'    Loaded {len(results_df)} existing results')
        return results_df
    else:
        print('    No existing results found')
        return pd.DataFrame()
    

def filter_processed_pairs(results_df, id_pairs):
    # Convert list of pairs to DataFrame for easy merging
    id_pairs_df = pd.DataFrame(id_pairs, columns=['proxy', 'target'])
    
    # Perform an outer merge and keep only those rows that are NaN in results_df index
    # This indicates that these rows were not found in results_df
    merged_df = id_pairs_df.merge(results_df, on=['proxy', 'target'], how='left', indicator=True)
    filtered_df = merged_df[merged_df['_merge'] == 'left_only']
    
    # Convert the filtered DataFrame back to a list of tuples
    remaining_pairs = list(zip(filtered_df['proxy'], filtered_df['target']))
    
    return remaining_pairs


def main(existing_results, bitrate, completeness_threshold, n_years, filtered_pairs):
    
    inputs = [(p, t, bitrate, completeness_threshold, n_years) for p, t in list(filtered_pairs)]
    
    if len(inputs) == 0:
        print('    No new pairs to process')
        return None

    n_batches = max(len(inputs) // batch_size, 1)
    batch_inputs = np.array_split(inputs, n_batches)
    print(f'    Processing {len(inputs)} pairs in {n_batches} batches at {bitrate} bits')
    for batch in batch_inputs:
        if len(batch) == 0:
            break
        with mp.Pool() as pool:
            results = pool.map(process_compression, batch)
            results = [r for r in results if r is not None]
        # results = []
        # for batch in batch_inputs[0]:
        #     result = process_compression(batch)
        #     if result is not None:
        #         results.append(result)
        if len(results) == 0:
            print('    No new results in current batch to save')
            continue
        
        t1 = time()
        print(f'    Processed {batch_size} ({len(results)} good results) in {t1 - t0:.1f} seconds')
        
        new_results_df = pd.DataFrame(results)
        new_results_df['target'] = new_results_df['target'].astype(str)
        new_results_df['proxy'] = new_results_df['proxy'].astype(str)

        if len(new_results_df) == 0:
            print('    No new results in current batch to save')
            continue
        else:
            print(f'    Saving {len(new_results_df)} new results to file.')
            climate_data = process_climate_data(new_results_df)
            combined_df = pd.merge(new_results_df, climate_data, on=['proxy', 'target'])
            existing_results = pd.concat([existing_results, combined_df], axis=0)
            existing_results['target'] = existing_results['target'].astype(str)
            existing_results['proxy'] = existing_results['proxy'].astype(str)
            existing_results.to_csv(out_fpath, index=False)
    return len(existing_results)

t0 = time()
revision_date = '20240409'
test_sample_size = 2000000
batch_size = 50000
# batch_size = 100
# generate a smaller random sample of pairs
random.seed(42)

# stn_ids = records_df.columns
id_query = "SELECT official_id,basin_geometry FROM basins_schema.hysets_basins;"
stn_ids = gpd.read_postgis(id_query, engine, geom_col='basin_geometry')['official_id'].values

id_pairs = list(itertools.combinations(stn_ids, 2))
# id_pairs = itertools.permutations(stn_ids, 2)
# sample_pairs = random.sample(list(id_pairs), test_sample_size)
ta = time()
print(f'Generated random input pairs in {ta - t0:.1f} seconds')
completeness_threshold = 0.9
n_years = 1 #[2, 3, 4, 5, 10]
bitrates = [4, 6, 8]

for b in [8]:#bitrates:
    print(f'Processing pairs at {b} bits')
    results_fname = f'DKL_results_{b}bits_{revision_date}.csv'
    out_fpath = os.path.join(PROCESSED_DATA_DIR, 'compression_test_results', results_fname)
    existing_results = check_processed_results(out_fpath)
    print(f'    {len(existing_results)} existing results loaded.')
    if existing_results.empty:
        id_pairs_filtered = id_pairs
        # id_pairs_filtered = list(sample_pairs)
    else:
        id_pairs_filtered = filter_processed_pairs(existing_results, id_pairs)

    n_results = main(existing_results, b, completeness_threshold, n_years, id_pairs_filtered)
    print(f'    Processed {n_results} pairs ({b} bits) in {time() - t0:.1f} seconds')
