import os
from time import time
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
from shapely.geometry import Point
import geopandas as gpd
import scipy.stats as stats
import warnings
from joblib import Parallel, delayed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS_DATA_PATH = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
HS_TS_PATH = os.path.join(HS_DATA_PATH, 'hysets_series')

warnings.filterwarnings('error')

class Station:
    def __init__(self, station_info, bitrate) -> None:
        # the input station_info is a dict, 
        # unpack the dict into attributes
        for k, v in station_info.items():
            setattr(self, k, v)

        self.id = self.Official_ID
        self.sim_label = f'{self.id}_sim'
        self.obs_label = f'{self.id}'
        self.log_sim_label = f'{self.id}_log10_sim'
        self.log_obs_label = f'{self.id}_log10_obs'
        self.digit_label_obs = f'qq_{self.id}_obs_{bitrate}b_quantile_digitized'
        self.digit_label_sim = f'qq_{self.id}_sim_{bitrate}b_quantile_digitized'


def bootstrap_quantiles(data, n_bootstrap=100):
    """
    Estimate the distribution of specified percentiles using bootstrap resampling
    in a vectorized manner.

    Parameters:
    - data: array-like, the dataset from which to draw bootstrap samples.
    - percentiles: array-like, the target percentiles to estimate (values between 0 and 1).
    - n_bootstrap: int, the number of bootstrap samples to generate.

    Returns:
    - quantile_estimates: 2D array where each row represents a percentile and
      each column represents a bootstrap sample's quantile estimate.

      If we want 8 bit quantization, only set 254 bins so that we can leave
      one at either end for out-of-range values when comparing the simulated 
      values to compute the KL divergence.
    """
    np.random.seed(42)  # For reproducibility
    n = len(data)

    mid_range = np.arange(0.01, 0.99, 0.0125/3)
    lows = np.arange(0.001, 0.01, 0.001)
    highs = np.arange(0.991, 0.999, 0.001)
    percentiles = list(lows) + list(mid_range) + list(highs)

    # Generate bootstrap indices in a vectorized way
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = data[bootstrap_indices]

    # Calculate the quantiles for each bootstrap sample in a vectorized way
    quantile_estimates = np.quantile(bootstrap_samples, percentiles, axis=1)

    # Calculate the requested percentiles across the second axis (bootstrap samples)
    # and transpose the result for alignment with quantiles as rows
    confidence_levels = [2.5, 50, 97.5]
    intervals = np.percentile(quantile_estimates, confidence_levels, axis=1).T
    # Create a DataFrame from the results, setting columns names as confidence levels
    intervals_df = pd.DataFrame(intervals, columns=[f"{cl}%" for cl in confidence_levels])
    return intervals_df


def get_timeseries_data(stn_id):
    
    fpath = os.path.join(HS_TS_PATH, f'{stn_id}.csv')
    df = pd.read_csv(fpath, parse_dates=['time'], 
                     engine='pyarrow')
    # df.dropna(subset=['discharge'], inplace=True, axis=0)
    stn_id = str(stn_id)
    min_flow_artificial = 1e-4
    df[f'{stn_id}_low_flow_flag'] = df['discharge'] < min_flow_artificial
    
    # get the smallest nonzero value
    # min_nonzero = df['discharge'][df['discharge'] > 0].min()

    # some stations report 0 flow and flow < 0.0001 m3/s which
    # is not measurable.  We'll set these values to 0.0001 m3/s
    df['discharge'].clip(lower=min_flow_artificial, inplace=True)
    df['year'] = df['time'].dt.year
    # rename the discharge column to the station id    
    df.rename(columns={'discharge': stn_id}, inplace=True)
    return df


def check_completeness(df, threshold=0.9):
    df.dropna(inplace=True, how='any')
    annual_counts = df.groupby('year').count()
    complete_years = annual_counts[annual_counts >= threshold*365]
    complete_years = complete_years.dropna().index.values
    return complete_years, len(df)


def retrieve_concurrent_data(proxy, target):
    df1 = get_timeseries_data(proxy)
    df2 = get_timeseries_data(target)
    
    df = pd.merge(df1, df2, on='time', how='inner')
    # count number of nan rows
    n_nan_rows = df.isnull().sum(axis=1).sum()
    
    if n_nan_rows > 0:
        print(f'Warning: {n_nan_rows} nan values found in concurrent data.')
        raise Exception('Concurrent data contains nan values.')
    df.drop(columns=['year_x', 'year_y'], inplace=True)    
    df['year'] = df['time'].dt.year
    return df


def find_max_precision(data):
    """
    Find the maximum precision in an array of float values.

    Parameters:
    data (np.array): Array of float values.
    Returns:
    float: The smallest non-zero difference between any two values in the array.
    """
    data = np.array(data)
    data = data[~np.isnan(data) & ~np.isinf(data)]
    data = np.sort(data)
    diffs = np.diff(data)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0
    min_diff = np.min(diffs)
    return abs(min_diff)


def transform_and_jitter(df, station):
    # add a random amount of noise below the measurement precision
    depth = 1e-9
    noise = np.random.uniform(-depth, depth, size=len(df))
    df[station.id] += noise

    if df[station.id].min() < 1e-5:
        print(df[station.id].min())
        msg = f'Noise addition creates values < 1e-5 (ID: {station.id})'
        print(msg)
        # raise Exception(msg) 
    label = f'{station.id}_log10_obs'
    # station.log_obs_label = label
    df[label] = np.log10(df[station.id])
    return df


def uniform_bins(data, station, bitrate, method, n_bins, epsilon = 1e-10):
    
    obs_label, sim_label = station.log_obs_label, station.log_sim_label
    quant_label = f'qq_{station.id}_obs_{bitrate}b_{method}_digitized'
    station.uniform_quant_label_obs = quant_label
    flag_cols = [e for e in data.columns if e.endswith('zero_flow_flag')]
    zero_flow_flag = data[flag_cols].any(axis=1).any()

    nonzero_data = data[~data[flag_cols].any(axis=1)].copy()
    min_val = nonzero_data[[obs_label, sim_label]].min().min()
    max_val = nonzero_data[[obs_label, sim_label]].max().max()

    # if there's a zero flow flag, set the left-most edge 
    # to -4 (corresponding to 1e-4 L/s/km^2) unless the
    # minimum observed value is less than -4,
    # in which case use the minimum observed value
    if zero_flow_flag & (min_val > -4):            
        bin_edges = np.linspace(
            min_val,
            max_val,
            n_bins
        )        
        bin_edges = [-4] + list(bin_edges)
    else:
        bin_edges = np.linspace(
            min_val,
            max_val,
            n_bins + 1
        ).tolist()

    return data, bin_edges


def compute_probabilities(df, quant_label_obs, quant_label_sim, bitrate, years, priors):
    # count the occurrences of each quantized value
    obs_count = df.groupby(quant_label_obs).count()
    sim_count = df.groupby(quant_label_sim).count()

    # reindex to 1, 2, 3, ... 2**bitrate 
    # note that the values in all columns are the same after groupby    
    obs_count = obs_count[['time']].reindex(range(1, 2**bitrate+1)).fillna(0)
    sim_count = sim_count[['time']].reindex(range(1, 2**bitrate+1)).fillna(0)

    n_obs, n_sim = np.nansum(obs_count.values), np.nansum(sim_count.values)
    # doesn't matter what label you use, all columns are the same    
    p_obs = obs_count['time'] / n_obs
    p_sim = sim_count['time'] / n_sim
    # here we add one pseudo-count to each bin to represent the uniform prior
    # doesn't matter what label you use, all columns are the same
    L = len(years)
    q_sim = pd.DataFrame()
    q_sim[f'q_sim'] = p_sim
    
    # compute the posterior probabilities based on 
    # a wide range of priors to test sensitivity
    for pseudo_counts in priors:
        adjusted_counts = sim_count + pseudo_counts
        q_sim[f'q_post_{pseudo_counts}R'] = (adjusted_counts) / (np.nansum(adjusted_counts))
    
    return p_obs, q_sim


def equiprobable_bins(df, station, bitrate, method, n_bins, epsilon=1e-10, jittered=False):
    
    label = station.log_obs_label
    quant_label = f'qq_{station.id}_obs_{bitrate}b_{method}_digitized'
    station.equiprob_quant_label_obs = quant_label    
    observed_series = df[label].values

    # Calculate the percentile values that correspond to the bin edges
    sorted_vals = sorted(observed_series)
    groups = np.array_split(sorted_vals, n_bins)
    # bin_edges = [min(groups[0])]
    # set the left edge to a value smaller than the minimum value
    # used at import to prevent log0 or the minimum value if
    # for some insane reason it's less than 1e-4
    min_val = min(groups[0])
    bin_edges = [min_val - 1]

    for grp in range(len(groups)):
        try:
            bin_edges += [min(groups[grp])]
        except Exception as e:
            print('empty sequence')
            print(len(sorted_vals), n_bins)
            n_per_grp = [len(e) for e in groups]
            print(n_per_grp)
    # add the far right edge
    bin_edges.append(max(groups[-1]) + np.log10(1.0001))
    # Ensure bin widths are not too small
    bin_widths = np.subtract(bin_edges[1:], bin_edges[:-1])
    if np.any(bin_widths == 0.):  # or some small threshold of your choice
        raise Exception('Jittering failed to create unique bins.')
        
    hist, bin_edges = np.histogram(observed_series, bins=bin_edges, density=True)
    return df, bin_edges


def linear_pct_bin_widths(a, bin_width_pct, n_bins):
    return a * ((1+bin_width_pct)/(1-bin_width_pct))**n_bins


def new_proportion_finder(a, b, n_bins, bitrate):
    def objective_function(bin_width_pct):
        b_new = linear_pct_bin_widths(a, bin_width_pct, n_bins)
        return b_new - b

    # Use a root-finding algorithm to find the bin width as a pct of midpoint
    tolerance_range = [0.05, 0.6]
    if bitrate > 5:
        tolerance_range = [0.01, 0.3]
    elif bitrate > 6:
        tolerance_range = [0.001, 0.1]
    result = root_scalar(objective_function, bracket=tolerance_range, 
                         method='brentq', rtol=1e-2, maxiter=10000)
    if result.converged:
        final_tolerance = result.root
        # final_difference = objective_function(final_tolerance)
        # print(f'Final Tolerance: {final_tolerance}, Final Difference: {final_difference}')
        return final_tolerance
    else:
        print('Optimization failed to converge.')
        print(a, b, n_bins)
        raise ValueError("Could not find a suitable tolerance within the acceptance tolerance.")


def set_edges_linear_pct(a, b, tolerance, n_bins):
    
    if a - 1e-6 <= 0:
        print(f'linear pct left edge <= 0 ({a:.3e})')
        raise Exception('Left edge of bin is negative or zero.')
    # adjust the left edge by a very small number to ensure
    # the smallest observed values falls in the first bin
    edges = [a - 1e-6]
    midpoints = [a / (1 - tolerance)]
    for _ in range(n_bins):
        # add the right edge
        next_edge = (1 + tolerance) * midpoints[-1]
        edges.append(next_edge)
        # find the next midpoint
        new_mp = ((1+tolerance)/(1-tolerance)) * midpoints[-1]
        midpoints.append(new_mp)
    
    return edges
    

def proportional_width_constant(df, station, bitrate, method, n_bins, tolerance=0.1, epsilon=1e-6):
    # use the first year to develop a quantization
    label = station.log_obs_label

    quant_label = f'qq_{station.id}_test_{bitrate}b_{method}_digitized'
    # station.proportional_quant_label_obs = quant_label
    
    # n_bins = np.power(2, bitrate)
    a, b = df[station.id].min(), df[station.id].max()

    if a < 1e-5:
        print(a)
        raise Exception(f'Minimum observed value is less than 1e-4: {a:.3e}')
    
    # if the left edge is negative, shift the interval to the positive 
    try:
        new_tolerance = new_proportion_finder(a, b, n_bins, bitrate)
    except Exception as ex:
        print(a, b)
        print('')
        print('Exception in solving for constant proportion (new_proportion_finder())')
        print(ex)
        print('')
        return df, None

    if new_tolerance is None:
        print('')
        print('No tolerance found')
        print('')
        print('')
        return df, None
    else:
        bin_edges = set_edges_linear_pct(a, b, new_tolerance, n_bins)
        # print(f'last bin edge: {bin_edges[-1]:.4f} vs. max observed: {b:.4f}')
        
    # adjust the bin edges by the offset
    bin_edges = [np.log10(e) for e in bin_edges]
    # quantize the signal to check if the range is covered
    df[quant_label] = np.digitize(df[label], bin_edges) 
    if bin_edges[-1] <= df[label].max():
        bin_undershoot = df[label].max() - bin_edges[-1]
        if (bin_undershoot >= 0) & (bin_undershoot < 0.1):
            bin_edges[-1] += bin_undershoot + epsilon
            # print(f'  {len(bin_edges)} bins Adjusting upper bin right edge by {bin_undershoot + epsilon:.2e} cms) to encompass full range and re-digitizing.')
        else:
            print(bin_edges)
            print(df[label].max(), bin_edges[-1])
            print(bin_undershoot)
            raise Exception('Bin adjustment failed, larger than 0.1')
        
    return df, bin_edges


def quantize_series(df, edge_set, model, location, bitrate):
    # digitize the posterior (observed) series based on the posterior edges
    for s in ['obs', 'sim']:
        # quant_label = f'qq_{location.id}_{s}_{bitrate}b_{model}_digitized'
        # quant_label = f'qq_{location.id}_{s}_{bitrate}b_{model}_digitized'
        quant_label = location.digit_label_sim
        val_label = location.sim_label
        if s == 'obs':
            quant_label = location.digit_label_obs
            val_label = location.obs_label
        try:
            # df[quant_label] = np.digitize(df[f'{location.id}_log10_{s}'], edge_set)
            df[quant_label] = np.digitize(df[val_label], edge_set)
        except Exception as ex:
            print('Exception occurred digitizing series')
            print(model, bitrate, 'bits')
            print(edge_set)
            print(ex)
            raise Exception(ex)
    return df


def compute_quantile_bins(df, location, bitrate):
    # computes bin edges for 8 bit quantization 
    # based on the observed series
    saved_quantiles_path = os.path.join(BASE_DIR, 'processed_data', 'bootstrapped_quantiles')
    if not os.path.exists(saved_quantiles_path):
        # print('Creating directory:', saved_quantiles_path)
        os.makedirs(saved_quantiles_path)
    # print(os.listdir(saved_quantiles_path))
    quantiles_fname = f'{location.id}_{bitrate}b_quantiles.csv'
    fpath = os.path.join(saved_quantiles_path, quantiles_fname)
    if os.path.exists(fpath):
        # print('loading previously computed quantiles')
        quantile_df = pd.read_csv(os.path.join(saved_quantiles_path, f'{location.id}_{bitrate}b_quantiles.csv'))
    else:
        observed_values = df[location.id].values
        quantile_df = bootstrap_quantiles(observed_values, n_bootstrap=1000)
        quantile_df.to_csv(fpath, index=False)
        print(f'Saved quantiles to {fpath}')
    
    medians = quantile_df['50%'].values.tolist()
    return [1e-4] + medians + [10 * medians[-1]]


def compute_bin_edges(df, location, bitrate):
    # get the range of the specified log 
    # transformed series create the bins    
    n_bins = np.power(2, bitrate) 

    min_obs = df[location.id].min()
    if min_obs <= 1e-5:
        print(df[location.id].min())
        raise Exception('Compute_bin_edges: Minimum observed value is less than 1e-5')

    # digitize the observed series (posterior) and reserve 
    # the two outer quantization levels for out-of-range value
    df, uniform_edges = uniform_bins(df, location, bitrate, 'uniform', n_bins)

    return df, uniform_edges#, equiprob_edges#, prop_edges


def compute_distance(stn1, stn2):
    p1 = Point(stn1['Centroid_Lon_deg_E'], stn1['Centroid_Lat_deg_N'])
    p2 = Point(stn2['Centroid_Lon_deg_E'], stn2['Centroid_Lat_deg_N'])
    gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs='EPSG:4326')
    gdf = gdf.to_crs('EPSG:3005')
    distance = gdf.distance(gdf.shift()) / 1000
    return distance.values[1]


def compute_dirichlet_prior(pmf, alpha_0, N=1):
    """
    Compute the Dirichlet prior for a given discrete PMF.

    Parameters:
    pmf (list): The probability mass function, a list of probabilities for each event.
    alpha_0 (float): The concentration parameter (pseudocount) to be added for smoothing.
    N (int, optional): Total count of observations. Default is 1, assuming uniform distribution of counts.

    Returns:
    list: The computed Dirichlet prior parameters.
    """
    return [p * N + alpha_0 for p in pmf]


def compute_hellinger_distance(df, bitrate, proxy, target):
    """
    Proxy is the posterior (P), target is the prior (Q)
    """
    pmf = pd.DataFrame()
    pmf['bin'] = list(range(1, 2**bitrate+1))
    for m in ['uniform', 'equiprobable', 'proportional']:
        for l in [proxy, target]:
            quantized_proxy_label = f'qq_{l.id}_obs_{bitrate}b_{m}_digitized'
            hist = df.groupby(quantized_proxy_label).count()
            hist = hist[[proxy.id]].reindex(range(1, 2**bitrate+1))
            # print(hist)
            hist.fillna(0, inplace=True)
            pmf[f'P_{l.id}_{m}'] = (hist.values / sum(hist.values)).round(3)
        
        posterior_pmf = pmf[f'P_{proxy.id}_{m}'].values
        prior_pmf = pmf[f'P_{target.id}_{m}'].values
        hd = np.sqrt(np.sum((np.sqrt(posterior_pmf) - np.sqrt(prior_pmf)) ** 2)) / np.sqrt(2)
            
        print(f'{m} Hellinger Distance: {hd:.2f}')
    print(asdfasdf)


def compute_histogram(df, bitrate, location, which_series, model):
    
    quantized_label = f'qq_{location.id}_{which_series}_{bitrate}b_{model}_digitized'
    
    hist = df.groupby(quantized_label).count()
    # print(hist[[location.id]])
    hist = hist[[location.id]].divide(hist[location.id].sum())
    if (which_series == 'obs') & (len(hist) > (2**bitrate - 2)):
        print(f'min obs: {df[location.id].min():.2f}')
        print(min(hist.index), max(hist.index))
        print('')        
        raise Exception(f'{which_series} {model} histogram found observations outside range.')

    hist = hist.reindex(range(1, 2**bitrate+1))
    hist.fillna(0, inplace=True)
    return hist.values


def compute_mean_divergence(row, label, M_label):
    if (row[M_label] == 0) | (row[label] == 0):
        # print('     M and/or P|Q are zero.')
        return 0.
    else:
        try:
            return row[label] * np.log(row[label] / row[M_label])            
        except Exception as ex:
            print('     something broke ', row[label], row[M_label])
            raise Exception(ex)


def compute_jensen_shannon_divergence(df, bitrate, location, m):

    pmf = pd.DataFrame()
    pmf['bin'] = list(range(1, 2**bitrate+1))
    
    # compute histogram of the observed series
    for s in ['obs', 'sim']:
        hist = compute_histogram(df, bitrate, location, s, m)
        pmf[f'P_{location.id}_{m}_{s}'] = hist
    
    P_label = f'P_{location.id}_{m}_obs'
    Q_label = f'P_{location.id}_{m}_sim'
    M_label = f'P_{location.id}_{m}_mean'

    # posterior_pmf = pmf[f'P_{location.id}_{m}_obs'].values
    # prior_pmf = pmf[f'P_{location.id}_{m}_sim'].values

    pmf[M_label] = pmf[[P_label, Q_label]].mean(axis=1)           

    pmf[f'P_{location.id}_{m}_Q_to_M'] = pmf.apply(lambda row: compute_mean_divergence(row, Q_label, M_label), axis=1)
    pmf[f'P_{location.id}_{m}_P_to_M'] = pmf.apply(lambda row: compute_mean_divergence(row, P_label, M_label), axis=1)

    prior_to_mean = pmf[f'P_{location.id}_{m}_Q_to_M'].sum() / 2
    posterior_to_mean = pmf[f'P_{location.id}_{m}_P_to_M'].sum() / 2
    jsd = prior_to_mean + posterior_to_mean
    
    # print(f'{m} Jensen-Shannon Divergence: {jsd:.2f}')
    # print('')
    return round(jsd,3), [round(e,3) for e in pmf[M_label].values]


def compute_obs_hist(df, bitrate, proxy, target):
    pmf = pd.DataFrame()
    pmf['bin'] = list(range(1, 2**bitrate+1))
    for m in ['uniform', 'equiprobable', 'proportional']:
        for l in [proxy, target]:
            quantized_proxy_label = f'qq_{l.id}_obs_{bitrate}b_{m}_digitized'
            hist = df.groupby(quantized_proxy_label).count()
            # reindex to range(1, 2**bitrate+1)
            hist = hist[[proxy.id]].reindex(range(1, 2**bitrate+1))
            # print(hist)
            hist.fillna(0, inplace=True)
            pmf[f'P_{l.id}_{m}'] = (hist.values / sum(hist.values)).round(3)
    print(pmf)
    # pmf.to_csv(f'{proxy.id}_{target.id}_bitrate.csv')
    # check if any values are zero
    if np.any(pmf.values == 0):
        print('')


        # add a small amount of noise to the pmf
        # in the form of the dirichlet prior
        # in essence, you would expect to wait 
        # 2N + 1 observations to see the unobserved state?

        # how could the unobserved state
        # be defined or bound?

        # extreme value -- we don't know the full range
        # how do we create a new bound to extend the range?
        # is it some bound of information?
        # i.e a conservative way of saying something outside the 
        # state space. 
        # something outside the state space should occur every
        # 2**state_space_size + 1 observations,  If it were
        # 1 year of observations, we would expect something 

        # Minimize the DKL between the observed and approximated distribution
        # We could use the KL divergence to find the best fit distribution
        # But fat tails will be a problem because the KL divergence
        # diverges as the probability we assign to the unexplored state goes to zero

        # in this sense, the DKL is sensitive to outliers / long tails, etc.
        # we then need to associate the unexplored state with a probability
        # that we care about.  i.e. if a dam spillway is designed to pass 
        # 100 cubic meters of water per second and in 10 years we've seen a max of 50,
        # we don't say we'd expect to see 51 in another ten years, we 
        # say how long until we meet/exceed 100 (or the catastrophic magnitude).
        # given what we have observed?  There are then at least two unobserved states,
        # one that is (50, 100), and one that is [100, infinity].
        # i.e., an event greater than currently observed but not catastrophic,
        # and an unobserved event that is catastrophic.
        #
        # we set the alpha to represent some value that reflects
        # a) how accurate was the measurement of 50 or 90, and
        # b) when will the bridge need to be torn down or replaced?
        # how long our planning horizon (design life) of the structure.  
        # We can then be conservative or optimistic.

        # could we say that the unmonitored state is defined as:
        #      

        raise Exception('Zero values in PMF')
    

def compute_dkl(p, q):
    dkl = []
    for i in range(len(p)):
        if q[i] == 0:
            dkl.append(np.nan)
        elif p[i] + q[i] == 0:
            dkl.append(np.nan)
        elif p[i] == 0:
            dkl.append(0)
        else:
            dkl.append(p[i] * np.log2(p[i] / q[i]))
    return dkl


def process_dkl(p_obs, p_sim, bitrate):
    df = pd.DataFrame()
    df['bin'] = range(1,2**bitrate+1)
    df.set_index('bin', inplace=True)
    for c in p_sim.columns:
        q = p_sim[c].values
        p = p_obs.values
        label = 'dkl_' + '_'.join(c.split('_')[1:])
        df[label] = compute_dkl(p, q)
    return df


def compute_cod(df, obs, sim):
    # compute the coefficient of determination 
    # between the obs and sim series using the scipy.stats.linregress function
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[obs.id], df[sim.id])
    except Exception as ex:
        print('busted')
        print(obs.id, sim.id)
        print(df)
    return r_value**2

def compute_nse(df, obs, sim):
    # compute the Nash-Sutcliffe Efficiency
    # between the obs and sim series
    obs_mean = df[obs.id].mean()
    return 1 - (df[sim.id] - df[obs.id]).pow(2).sum() / (df[obs.id] - obs_mean).pow(2).sum()

def compute_kge(df, obs, sim):
    # compute the Kling-Gupta Efficiency
    # between the obs and sim series
    obs_mean = df[obs.id].mean()
    sim_mean = df[sim.id].mean()
    obs_std = df[obs.id].std()
    sim_std = df[sim.id].std()
    r = np.corrcoef(df[obs.id], df[sim.id])[0,1]
    beta = sim_mean / obs_mean
    gamma = (sim_std / sim_mean) / (obs_std / obs_mean)
    return 1 - np.sqrt((r-1)**2 + (beta - 1)**2 + (gamma - 1)**2)


def compute_forecast_score(df, basis, proxy, target, bitrate):
    for c in basis.columns:
        print(c)
        print(basis[[c]].head())
    print(sadfasdf)


def compute_UARE(df, proxy, method, target, bitrate):
    df['target_vol_error'] = df[f'{target.Official_ID}'] - np.power(10, df[f'{target.id}_log10_sim'])
    df['target_ur_error'] = 1000 * df['target_vol_error'] / target.Drainage_Area_km2
    
    if method == 'uniform':
        label = target.uniform_quant_label_obs
    elif method == 'equiprobable':
        label = target.equiprob_quant_label_obs
    elif method == 'proportional':
        label = target.proportional_quant_label_obs
    else:
        raise ValueError(f'Invalid method: {method}')
    
    good_cols = [e for e in df.columns if e.endswith('error')] + [label]
    grouped_ur_error = df[good_cols].groupby(label).sum()
    
    # fill in missing indices
    complete_index = list(range(1, 2**bitrate+1))
    grouped_ur_error = grouped_ur_error.reindex(complete_index)
    grouped_ur_error.fillna(0, inplace=True)

    output_label = f'UARE_pct_{method}_{bitrate}b'
    grouped_ur_error[output_label] = grouped_ur_error['target_ur_error'] / grouped_ur_error['target_ur_error'].abs().sum()
    mean_label = f'UARE_mean_{method}_{bitrate}b'
    mean_ur_error = df[good_cols].groupby(label).mean()
    mean_ur_error = mean_ur_error.reindex(complete_index)
    # mean_ur_error.fillna(0, inplace=True)

    result = {
        output_label: list(grouped_ur_error[output_label].round(2).values), 
        mean_label: list(mean_ur_error['target_ur_error'].round(0).values)
        }

    return result