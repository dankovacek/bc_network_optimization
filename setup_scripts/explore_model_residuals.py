import os
import pandas as pd
import numpy as np

# import geopandas as gpd

from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, column, row
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.palettes import Category20, Set2
import numpy as np
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import HYSETS data
# hysets_stn_fpath = 'input_data/BC_study_region_HYSETS_stns.geojson'
# stn_gdf = gpd.read_file(hysets_stn_fpath, dtype={'Official_ID': str})
# stn_gdf.set_index('Official_ID', inplace=True)
# foopath = os.path.join(BASE_DIR, 'input_data', 'HYSETS_watershed_properties.txt')
# stn_df = pd.read_csv(foopath, sep=';', index_col='Official_ID')
# stn_gdf = gpd.GeoDataFrame(stn_df, geometry=gpd.points_from_xy(stn_df['Centroid_Lon_deg_E'], stn_df['Centroid_Lat_deg_N']))

# stn_gdf.crs = 'EPSG:4326'
# stn_gdf = stn_gdf.to_crs('EPSG:3005')

stn_path = os.path.join(BASE_DIR, 'input_data', 'HYSETS_stns.geojson')
stn_gdf = gpd.read_file(stn_path)
stn_gdf.set_index('Official_ID', inplace=True)
# stn_gdf.to_file(foopath, driver='GeoJSON')

def load_validation_results(sim_identifier, max_depth_id, lr_id):
    validation_results_folder = os.path.join(BASE_DIR, 'processed_data', 'xval_results')
    sim_files = sorted([e for e in os.listdir(validation_results_folder) if sim_identifier in e])
    sim_files = [e for e in sim_files if max_depth_id in e]
    sim_files = [e for e in sim_files if round(float(e.split('_')[16]), 3) == lr_id]

    sim_df = pd.DataFrame()
    sim_df['file'] = sim_files
    sim_df["sim_no"] = sim_df["file"].apply(lambda x: int(x.split("_")[0]))
    sim_df["bits"] = sim_df["file"].apply(lambda x: int(x.split("_")[2]))
    sim_df["max_depth"] = sim_df["file"].apply(lambda x: int(x.split("_")[10]))
    sim_df['prior'] = sim_df['file'].apply(lambda x: float(x.split('_')[4]))
    sim_df["n_estimators"] = sim_df["file"].apply(lambda x: int(x.split("_")[13]))
    sim_df["learning_rate"] = sim_df["file"].apply(lambda x: float(x.split("_")[16]))
    sim_df["colsample_bytree"] = sim_df["file"].apply(lambda x: float(x.split("_")[19]))
    # sim_df["lambda"] = sim_df["file"].apply(lambda x: float(x.split("_")[19]))
    # sim_df["alpha"] = sim_df["file"].apply(lambda x: float(x.split("_")[21]))
    sim_df["mse"] = sim_df["file"].apply(lambda x: float(x.split("_")[21]))
    sim_df["n_train"] = sim_df["file"].apply(lambda x: int(x.split("_")[22]))
    sim_df["n_test"] = sim_df["file"].apply(lambda x: int(x.split("_")[24]))
    sim_df.sort_values('sim_no', inplace=True)
    return sim_df


def compute_percentiles(df, column_name, bin_size):
    """
    Compute the 5th, 50th, and 95th percentiles for every bin_size chunk of ordered rows 
    in a DataFrame based on the specified column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    column_name (str): The name of the column to compute percentiles for.
    bin_size (int): The number of rows in each chunk.

    Returns:
    pandas.DataFrame: A DataFrame containing the percentiles for each chunk.
    """
    # Initialize a list to store the percentile results
    percentiles_list = []
    final_bin = False
    for start_idx in range(0, len(df), bin_size):
        
        end_idx = start_idx + bin_size
        
        if len(df) - end_idx < bin_size:
            # Adjust the end index of the previous chunk to include the last bit
            end_idx = len(df)
            final_bin = True
        chunk = df.iloc[start_idx:end_idx]
        
        # Calculate the percentiles for the current chunk
        percentiles = np.percentile(chunk[column_name], [5, 50, 95])
        mean_predicted = chunk['predicted'].mean()
        
        # Append the results to the list
        percentiles_list.append({
            'start_index': start_idx,
            'end_index': end_idx,
            'mean_predicted': mean_predicted,
            '5': percentiles[0],
            '50': percentiles[1],  # Median
            '95': percentiles[2],
        })
        if final_bin:
            break
    
    # Convert the list of dictionaries to a DataFrame and return
    return pd.DataFrame(percentiles_list)


def compute_edges(sim_df):
    # first, lets get an understanding of the distribution of predicted values
    predictions = []
    actuals = []
    eqp_edges = []
    eqw_edges = []
    n_bins = 20
    all_max, all_min = -1e6, 1e6
    for i, row in sim_df.iterrows():
        sim_id = row['sim_no']
        file = row['file']
        mae = row['mse']
        sim = pd.read_csv(os.path.join(BASE_DIR, 'processed_data', 'xval_results', file))

        preds = sim['predicted'].values
        predictions += list(preds)
        actuals += list(sim['actual'].values)
        if min(preds) < all_min:
            all_min = min(preds)
        if max(preds) > all_max:
            all_max = max(preds)
        # add a small amount of random noise to avoid ties
        preds += np.random.uniform(-1e-6, 1e-6, len(preds))
        sim_sorted = sorted(preds)
        hist, edges = np.histogram(sim_sorted, n_bins-1, density=True)
        eqw_edges.append(list([0] + edges.tolist()))
        
        # Calculate the percentiles to split the preds array
        percentiles = np.linspace(0, 100, n_bins + 1)
        
        # Compute the bin edges using the percentiles
        bin_edges = np.percentile(preds, percentiles)
        
        # Ensure uniqueness by handling potential duplicates in bin_edges
        unique_bin_edges = np.unique(bin_edges)
        # Handle the rare case where uniqueness filtering reduces the number of edges
        if len(unique_bin_edges) < len(bin_edges):
            raise ValueError("Reduced bin edges due to duplicates. Consider fewer bins or different data.")
        else:
            eqp_edges.append(unique_bin_edges)
    
    ed_df = pd.DataFrame()
    ed_df['ew'] = pd.DataFrame(eqw_edges).mean(0)
    ed_df['ep'] = pd.DataFrame(eqp_edges).mean(0)
    ed_df.loc[0, :] = all_min - 1e-6
    ed_df.loc[max(ed_df.index), :] = all_max + 1e-6
    return ed_df


def compute_pdfs(sim_df, ed_df, n_bins=20):
    ew_counts, ep_counts = [], []
    ep_probs, ew_probs = [], []
    for i, row in sim_df.iterrows():
        sim_id = row['sim_no']
        file = row['file']
        mae = row['mse']
        sim = pd.read_csv(os.path.join(BASE_DIR, 'processed_data', 'xval_results', file))
        preds = sim['predicted'].values
        # ew_bin_counts = np.zeros(n_bins)
        # ep_bin_counts = np.zeros(n_bins)
        ew_edges = ed_df['ew'].to_numpy()
        ep_edges = ed_df['ep'].to_numpy()
        
        ew_bins = np.digitize(preds, ew_edges) - 1
        ep_bins = np.digitize(preds, ep_edges) - 1
        # unique_bins, counts = np.unique(ew_bins, return_counts=True)
        
        ew_bin_counts = np.bincount(ew_bins, minlength=n_bins)
        ep_bin_counts = np.bincount(ep_bins, minlength=n_bins)
            
        # Calculate probabilities
        ew_prob = ew_bin_counts / ew_bin_counts.sum()
        ep_prob = ep_bin_counts / ep_bin_counts.sum()
        
        ew_probs.append(ew_prob.tolist())
        ep_probs.append(ep_prob.tolist())

    pdf = pd.DataFrame()
    pdf['ew'] = pd.DataFrame(ew_probs).mean(0)
    pdf['ep'] = pd.DataFrame(ep_probs).mean(0)
    return pdf


def aggregate_simulation_data(sim_identifier, max_depth_id, lr_id):
    data = {}
    bin_vals = {}
    label_dict = {}
    pred_vals, obs_vals = [], []
    proxies, targets = [], []
    for method in ['ep', 'ew']:
        method_results = []
        bin_vals[method] = {}
        label_dict[method] = {}
        sim_df = load_validation_results(sim_identifier, max_depth_id, lr_id)
        ed_df = compute_edges(sim_df)
        pdf = compute_pdfs(sim_df, ed_df)
        
        for i, row in sim_df.iterrows():
            sim_id = row['sim_no']
            file = row['file']
            mae = row['mse']
            sim = pd.read_csv(os.path.join(BASE_DIR, 'processed_data', 'xval_results', file))

            sim.drop('Unnamed: 0', axis=1, inplace=True)
            # preds = sim['predicted'].values
            # sort the dataframe by the predicted value
            sim.sort_values('predicted', inplace=True)
            sim.reset_index(inplace=True, drop=True)
            pred_vals += list(sim['predicted'].values)
            obs_vals += list(sim['actual'].values)
            proxies += list(sim['proxy'].values)
            targets += list(sim['target'].values)


    # sort the aggregated data by predicted value and compute the confidence interval for every 1000 pts
    adf = pd.DataFrame({'predicted': pred_vals, 'actual': obs_vals,
                        'proxy': proxies, 'target': targets})
    adf.sort_values('predicted', inplace=True)

    ew_widths = np.diff(ed_df['ew'])
    ew_cents = ed_df['ew'][:-1] + ew_widths / 2

    hs_df = pd.DataFrame()
    hs_df['ew_cents'] = ew_cents
    hs_df['ew_p'] = pdf['ew'].values
    hs_df['ew_w'] = ew_widths
    return adf, ed_df, pdf, hs_df


# get the files for one simulation
# sim_identifier = 'n_estimators_75'
# max_depth_id = 'max_depth_7'
# lr_id = 0.144
sim_identifier = 'n_estimators_30'
max_depth_id = 'max_depth_8'
lr_id = 0.107

adf, ed_df, pdf, hs_df = aggregate_simulation_data(sim_identifier, max_depth_id, lr_id)

ci_df = compute_percentiles(adf, 'actual', 5000)

# Initial empty data for the DataTable
table_data = {'proxy': [], 'proxy_counts': [], 'target': [], 'target_counts': [], 'distance_km': []}
table_source = ColumnDataSource(data=table_data)

# Define the DataTable columns
columns = [
    TableColumn(field="proxy", title="Proxy"),
    TableColumn(field="proxy_counts", title="Proxy Count"),
    TableColumn(field="target", title="Target"),
    TableColumn(field="target_counts", title="Target Count"),
    TableColumn(field="distance_km", title="Distance [km]"),
]

# Create the DataTable
data_table = DataTable(source=table_source, columns=columns, width=400, height=300)

# Layout
source = ColumnDataSource(hs_df)
h = figure(width=600, height=200, title="Distribution of Predicted Values", output_backend='webgl',
           x_axis_label=r'Predicted $$D_{KL} (\hat Y)$$', y_axis_label='Probability')

# Add vertical bars to the plot
h.vbar(x='ew_cents', top='ew_p', width='ew_w', source=source, fill_alpha=0.6,
       legend_label="Equal Width", fill_color='navy', line_color='white')

h.legend.click_policy='hide'

TOOLS="pan,box_select,wheel_zoom,box_zoom,reset,save"

source = ColumnDataSource(adf)
afig = figure(width=600, height=400, title='Aggregated MC Simulations', output_backend='webgl', tools=TOOLS)
afig.circle('predicted', 'actual', color='dodgerblue', size=0.5, alpha=0.25, source=source)
afig.line(ci_df['mean_predicted'], ci_df['50'], line_width=3, line_color='black')
afig.line(ci_df['mean_predicted'], ci_df['5'], line_width=3, line_color='black', line_dash='dotted')
afig.line(ci_df['mean_predicted'], ci_df['95'], line_width=3, line_color='black', line_dash='dotted')
afig.line([0, max(adf['predicted'])], [0, max(adf['predicted'])], color='red', width=3, line_dash='dashed')
afig.xaxis.axis_label = r'Predicted $$D_{KL} (\hat Y)$$'
afig.yaxis.axis_label = r'Actual $$D_{KL}$$'

layout = row(column(afig, h), data_table)

def update(attr, old, new):
    
    if new:
        selected = source.selected.indices
        # get the official ID values associated with the selection set
        selected_data = adf.loc[selected].copy()
        
        # compute the distances between each proxy and target based on the 
        # point geometries in stn_gdf
        distances = selected_data.apply(lambda x: stn_gdf.loc[x['proxy'], 'geometry'].distance(stn_gdf.loc[x['target'], 'geometry']), axis=1) / 1000
        proxies = selected_data['proxy']
        targets = selected_data['target']
        
        proxy_unique, proxy_counts = np.unique(proxies, return_counts=True)
        target_unique, target_counts = np.unique(targets, return_counts=True)

        for i, row in selected_data.iterrows():
            if row['proxy'] in proxy_unique:
                idx = np.where(proxy_unique == row['proxy'])[0][0]
                selected_data.loc[i, 'proxy_counts'] = proxy_counts[idx]
            else:
                selected_data.loc[i, 'proxy_counts'] = 0
            
            if row['target'] in target_unique:
                idx = np.where(target_unique == row['target'])[0][0]
                selected_data.loc[i, 'target_counts'] = target_counts[idx]
            else:
                selected_data.loc[i, 'target_counts'] = 0

        tc = selected_data['target_counts'].tolist()
        pc = selected_data['proxy_counts'].tolist()        
        
        # Here, for demonstration, we'll just use the selected indices as stn_id
        # and some arbitrary count. You'll need to adjust this logic based on your actual data.
        selected_data = {
            'proxy': proxies.tolist(), 'proxy_counts': pc, 
            'target': targets.tolist(), 'target_counts': tc, 
            'distance_km': distances.round(1).tolist()
            }
        
        table_source.data = selected_data
    else:
        # Reset the data if no selection
        table_source.data = {'proxy': [], 'proxy_counts': [], 'target': [], 'target_counts': [], 'distance_km': []}



# Add a custom JS callback to be executed whenever the selection changes
source.selected.on_change('indices', update)

curdoc().add_root(layout)


