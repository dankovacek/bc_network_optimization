import os
from time import time

import utility_functions as ut

import numpy as np
import pandas as pd

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def bootstrap_quantiles(data, percentiles, n_bootstrap=1000):
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
    """
    np.random.seed(42)  # For reproducibility
    n = len(data)

    # Generate bootstrap indices in a vectorized way
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = data[bootstrap_indices]

    # Calculate the quantiles for each bootstrap sample in a vectorized way
    quantile_estimates = np.quantile(bootstrap_samples, percentiles, axis=1)

    return quantile_estimates


def calculate_ecdf(values, bin_edges):
    """
    Calculate the Empirical Cumulative Distribution Function (ECDF) for each bin.

    Parameters:
    - values: Array of values.
    - bin_edges: Array of bin edges.

    Returns:
    - A dict where keys are bin numbers (integers), and values are ECDFs for each bin.
    """
    # Digitize the values to find out which bins they fall into
    bins = np.digitize(values, bin_edges, right=True)
    # Initialize the ECDFs dictionary
    ecdfs = {}
    # Iterate through bins to calculate ECDF for each
    for bin_number in np.unique(bins):
        # Select values in the current bin
        bin_values = values[bins == bin_number]
        # Sort values in the bin
        sorted_values = np.sort(bin_values)
        # Calculate ECDF: proportion of values less than or equal to each value
        ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        # Assign ECDF to the corresponding bin number in the dict
        ecdfs[bin_number] = (sorted_values, ecdf)
    return ecdfs


def compare_predictions(prediction1, prediction2, bin_edges, ecdfs):
    """
    Compare two predictions to determine the probability that prediction 1 is greater than prediction 2
    using their corresponding ECDFs.

    Parameters:
    - prediction1: First prediction value.
    - prediction2: Second prediction value.
    - bin_edges: Array of bin edges.
    - ecdfs: Dict of ECDFs for each bin.

    Returns:
    - Probability that prediction 1 is greater than prediction 2.
    """
    # Find bins for each prediction
    bin1 = np.digitize([prediction1], bin_edges, right=True)[0]
    bin2 = np.digitize([prediction2], bin_edges, right=True)[0]
    
    # Retrieve the ECDFs for the corresponding bins
    values1, ecdf1 = ecdfs[bin1]
    values2, ecdf2 = ecdfs[bin2]
    
    # Probability that prediction 1 is greater than prediction 2:
    # We integrate over the range where prediction 1's ECDF is greater than prediction 2's ECDF.
    # Since ECDFs are step functions based on empirical data, we approximate this by
    # counting how many times values in bin1 are greater than values in bin2,
    # weighted by their ECDF values.
    probability = 0
    for v1, e1 in zip(values1, ecdf1):
        # For each value in bin1, find the proportion of values in bin2 it is greater than.
        # This is approximated by the ECDF value of the largest value in bin2 that is less than v1.
        less_than_v1 = values2 < v1
        if np.any(less_than_v1):
            max_ecdf2_less_than_v1 = np.max(ecdf2[less_than_v1])
            probability += max_ecdf2_less_than_v1 / len(values1)
    
    return probability

hs_data_path = os.path.join(BASE_DIR, 'input_data', 'HYSETS_watershed_properties.txt')
hs_df = pd.read_csv(hs_data_path, sep=';')


# Load the predictive model
# load the residual model
# pre-compute the probability that a prediction (bin) 
#      is less than a prediction in any bin to the right.

# 1. load the set of ungauged basins
# 2. for each ungauged location, compute "baseline" DKL (best of existing network)
# 3. for each ungauged location, compute expected DKL to all other ungauged locations
#      - compare 

