import os
from time import time
import pandas as pd
import numpy as np
import utility_functions as uf
import itertools
from utility_functions import Station
import multiprocessing as mp
import random
from scipy.optimize import minimize, Bounds

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
       'Permeability_logk_m2', 'Porosity_frac']
attr_df = hs_df[attr_cols].copy()
attr_df.index = hs_df['Official_ID'].values

# normalize the attribute columns to the range 0, 1
# attr_df = (attr_df - attr_df.min()) / (attr_df.max() - attr_df.min())

stn_ids = records_df.columns

id_pairs = itertools.combinations(stn_ids, 2)
# id_pairs = itertools.permutations(stn_ids, 2)


def compute_nse(observed, simulated):
    mean_observed = np.mean(observed)
    nse = 1 - np.sum((simulated - observed) ** 2) / np.sum((observed - mean_observed) ** 2)
    return nse

def compute_kge(observed, simulated):
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / np.std(observed)
    beta = np.mean(simulated) / np.mean(observed)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge



def check_concurrency(pair, threshold):
    records = records_df[list(pair)].copy().dropna(how='any')
    # for each row, if one column value is >= 365, the row should be dropped if the other column value is < 90% of 365
    # if both values in the row are < 365, the row should be dropped if either value is < 95% of 365
    # otherwise keep the row
    records['neither_complete'] = records.apply(lambda row: (row < 365 * threshold).any(), axis=1)
    records = records[~records['neither_complete']]
    return records

def optimize_exponent(observed_series, area_ratio):
    # Define the objective function to minimize
    # print('area ratio: ', area_ratio)
    def sse_objective(params, observed_series):
        exponent, constant_multiplier = params
        simulated_series = constant_multiplier * observed_series * np.power(area_ratio, exponent)
        residual = np.sum((simulated_series - observed_series) ** 2)
        return residual
        
    if area_ratio > 1:
        initial_guess = [1.0, 1.1]
    else:
        initial_guess = [1.0, 0.8]
    
    guess_bounds = Bounds([0.5, 0.1], [2, 10])

    # Perform the optimization
    result = minimize(sse_objective, x0=initial_guess, 
                      args=(observed_series, ), bounds=guess_bounds,
                      method='L-BFGS-B')

    if result.success:
        optimized_exponent, optimized_multiplier = result.x
        return optimized_exponent, optimized_multiplier
    else:
        # raise ValueError("Optimization failed: " + result.message)
        return np.nan, np.nan


def simulate_q_AR(df, proxy, target):
    # print(f'proxy: {proxy.id}, target: {target.id}')
    # simulate a quantization of the proxy and target da
    target.log_sim_label_AR = f'{target.id}_log10_sim_AR'
    proxy.log_sim_label_AR = f'{proxy.id}_log10_sim_AR'
    
    df[target.log_sim_label_AR] = df[proxy.log_obs_label] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2)
    df[proxy.log_sim_label_AR] = df[target.log_obs_label] * (proxy.Drainage_Area_km2 / target.Drainage_Area_km2)
    
    # compute the sse between the simulated and observed series
    target_sse = np.sum((df[target.log_obs_label] - df[target.log_sim_label_AR])**2)

    nse_ar = compute_nse(df[target.log_obs_label], df[target.log_sim_label_AR])
    kge_ar = compute_kge(df[target.log_obs_label], df[target.log_sim_label_AR])

    return df, target_sse, nse_ar, kge_ar


def simulate_q_AR_exponent(df, proxy, target, result):
    # print(f'proxy: {proxy.id}, target: {target.id}')
    # simulate a quantization of the proxy and target da
    target.log_sim_label_AR_exp = f'{target.id}_log10_sim_AR_exp'
    proxy.log_sim_label_AR_exp = f'{proxy.id}_log10_sim_AR_exp'
    target_exponent, target_multiplier = optimize_exponent(df[target.log_obs_label].values, target.Drainage_Area_km2 / proxy.Drainage_Area_km2)
    proxy_exponent, proxy_multiplier = optimize_exponent(df[proxy.log_obs_label].values, proxy.Drainage_Area_km2 / target.Drainage_Area_km2)

    result['proxy_exponent'] = proxy_exponent
    result['target_exponent'] = target_exponent
    result['proxy_multiplier'] = proxy_multiplier
    result['target_multiplier'] = target_multiplier

    # add the areas to the result
    result['proxy_area'] = proxy.Drainage_Area_km2
    result['target_area'] = target.Drainage_Area_km2

    # compute the sse between the simulated and observed series
    if np.isnan(target_exponent) or np.isnan(proxy_exponent):
        target_sse, proxy_sse = np.nan, np.nan
        target_r2, proxy_r2 = np.nan, np.nan
        nse_AR_exp, kge_AR_exp = np.nan, np.nan
    else:
        df[target.log_sim_label_AR_exp] = proxy_multiplier * df[proxy.log_obs_label] * (target.Drainage_Area_km2 / proxy.Drainage_Area_km2)**target_exponent
        df[proxy.log_sim_label_AR_exp] = target_multiplier * df[target.log_obs_label] * (proxy.Drainage_Area_km2 / target.Drainage_Area_km2)**proxy_exponent
        target_sse = np.sum((1000*df[target.log_obs_label] - df[target.log_sim_label_AR_exp])**2)
        proxy_sse = np.sum((1000*df[proxy.log_obs_label] - df[proxy.log_sim_label_AR_exp])**2)

        # compute the r2 between the simulated and observed series
        target_r2, _ = uf.compute_cod(df[target.log_obs_label], df[target.log_sim_label_AR_exp])
        proxy_r2, _ = uf.compute_cod(df[proxy.log_obs_label], df[proxy.log_sim_label_AR_exp])
        # print(f'proxy: {proxy_multiplier:.2f}*Ar^{proxy_exponent:.2f} (sse={proxy_sse:.2f}, r^2={proxy_r2:.2f}), target: {target_multiplier:.2f}*Ar^{target_exponent:.2f} (target sse: {target_sse:.2f}, r^2={target_r2:.2f})') 
        nse_AR_exp = compute_nse(df[target.log_obs_label], df[target.log_sim_label_AR_exp])
        kge_AR_exp = compute_kge(df[target.log_obs_label], df[target.log_sim_label_AR_exp])

    result['target_sse'] = target_sse
    result['proxy_sse'] = proxy_sse
    result['target_r2'] = target_r2
    result['proxy_r2'] = proxy_r2
    result['nse_AR_exp'] = nse_AR_exp
    result['kge_AR_exp'] = kge_AR_exp
    
    return df, result

def exponent_optimization(inputs):
    proxy, target, completeness_threshold = inputs    
    
    result = {'proxy': proxy, 'target': target, 'completeness_threshold': completeness_threshold}
    
    concurrent_years = check_concurrency((proxy, target), completeness_threshold)

    result['num_concurrent_years'] = len(concurrent_years) 
    if len(concurrent_years) < 10:
        # print(f'Not enough concurrent data for stations {proxy} and {target} ({len(concurrent_years)})')
        return None

    # print(f'Stations {proxy} and {target} have {len(concurrent_years)} concurrent years')
    df = uf.retrieve_concurrent_data((proxy, target), concurrent_years.index.values)
    # print(len(df))
    if df.empty:
        # print('   No data remaining after filtering')
        return result

    station_info = {
        'proxy': hs_df[hs_df['Official_ID'] == proxy].copy().to_dict(orient='records')[0],
        'target': hs_df[hs_df['Official_ID'] == target].copy().to_dict(orient='records')[0]
        }
    
    # use the first year to quantize
    basis = df[df['year'] == df['year'].min()].copy()
    
    # for stn in pair:
    proxy = Station(station_info['proxy'])
    target = Station(station_info['target'])

    basis = uf.transform_and_jitter(basis, proxy)
    basis = uf.transform_and_jitter(basis, target)

    basis, AR_sse, nse_AR, kge_AR = simulate_q_AR(basis, proxy, target)
    result['AR_sse'] = AR_sse
    result['nse_AR'] = nse_AR
    result['kge_AR'] = kge_AR

    basis, result = simulate_q_AR_exponent(basis, proxy, target, result)

    # add each attr_df difference to the result
    for col in attr_df.columns:
        result[f'{col}_diff'] = attr_df.loc[proxy.id, col] - attr_df.loc[target.id, col]

    # find the L1 and L2 attribute distances
    result['L1_attr_dist'] = np.sum(np.abs(attr_df.loc[proxy.id] - attr_df.loc[target.id])) #+ result['distance']
    result['L2_attr_dist'] = np.sqrt(np.sum(np.square(attr_df.loc[proxy.id] - attr_df.loc[target.id]))) #+ result['distance']

    # compute spatial distance between basin centroids
    result['distance'] = uf.compute_distance(station_info['proxy'], station_info['target'])
    
    return result


t0 = time()

test_sample_size = 100000
# generate a smaller random sample of pairs
random.seed(42)
sample_pairs = random.sample(list(id_pairs), test_sample_size)
ta = time()
print(f'Generated {test_sample_size} random pairs in {ta - t0:.1f} seconds')
completeness_threshold = 0.9


inputs = [(p, t, completeness_threshold) for p, t in sample_pairs]

pl = mp.Pool()
results = pl.map(exponent_optimization, inputs)
results = [r for r in results if r is not None]
pl.close()
# for inp in inputs[:100]:
#     res = exponent_optimization(inp)


t1 = time()

print(f'Processed {test_sample_size} in {t1 - t0:.1f} seconds')
results_fname = f'compression_test_results_exponent_optimization.csv'
results_df = pd.DataFrame(results)

results_df.to_csv(os.path.join(PROCESSED_DATA_DIR, results_fname), index=False)
