
import psycopg2
import psycopg2.extras as extras
import os
from time import time
import pandas as pd

import numpy as np
from shapely import wkb
import multiprocessing as mp

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

import warnings

warnings.filterwarnings('ignore')


# The KL divergence of P from Q is the expected excess surprise
# from using Q as a model when the actual distribution is P
revision_date = '20230901'

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}

schema_name = 'basins_schema'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
HYSETS_DIR = os.path.join(BASE_DIR, 'input_data/HYSETS_data')


def circular_diff(a, b, n=""):
    # print(f'{n} circ diff: ', a, b)
    return np.minimum(np.abs(a - b), 360 - np.abs(a - b))

    
def log_diff(a, b, n=""):
    # set zero values to 1E-6 to avoid divide by zero errors
    a = np.where(a == 0, 1E-9, a)
    b = np.where(b == 0, 1E-9, b)
    return np.abs(np.log(a) - np.log(b))


def abs_diff(a, b, n=""):
    return np.abs(a - b)


def spatial_dist(a, b, n=""):
    return a.distance(b) / 1E6


def get_cml_locations(region_polygon):
    # given a polygon query the database for all cmls within that polygon
    # return a geodataframe of the cmls and their properties
    
    # convert the polygon to wkt
    region_polygon = region_polygon.to_crs(3005)
    polygon_wkt = region_polygon.geometry.iloc[0].wkt

    query = f"""
    SELECT * 
    FROM basins_schema.basin_attributes
    WHERE ST_Within(pour_pt, ST_GeomFromText(%s, 3005))
    LIMIT 10;
    """
    
    with psycopg2.connect(**conn_params) as conn:
        # cur = conn.cursor() 
        # cur.execute(query, (polygon_wkt,))
        
        # get the results
        # results = cur.fetchall()
        cml_df = gpd.read_postgis(query, conn, geom_col='pour_pt', params=(polygon_wkt,))
        for c in cml_df.columns:
            print(cml_df[[c]].head())
            print('')
        print(cml_df)
    
    # cur.close()
    # conn.close()
    return cml_df


def get_hysets_locations(region_polygon, search_buffer=1000):
    # convert the polygon to wkt
    region_polygon = region_polygon.to_crs(3005)
    polygon_wkt = region_polygon.geometry.iloc[0].wkt
    
    query = f"""
    SELECT * 
    FROM basins_schema.hysets_basins
    WHERE ST_Within(centroid, ST_GeomFromText(%s, 3005))
    LIMIT 10;
    """
    with psycopg2.connect(**conn_params) as conn:
        # cur = conn.cursor() 
        # cur.execute(query, (polygon_wkt,))
        
        # get the results
        # results = cur.fetchall()
        hs_df = gpd.read_postgis(query, conn, geom_col='centroid', params=(polygon_wkt,))
    return hs_df


def create_normalized_views(schema_name, table_name, cols):
    norm_string_array = [f'({col} - (SELECT MIN({col}) FROM {schema_name}.{table_name})) / (SELECT MAX({col}) - MIN({col}) FROM {schema_name}.{table_name}) AS {col}_normalized' for col in cols]
    normalized_strings = ',\n'.join(norm_string_array)
    query = f"""
    CREATE VIEW normalized_{table_name} AS
    SELECT
        id,
        {normalized_strings}
    FROM {schema_name}.{table_name};
    """
    print(query)
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            print('Query executed successfully')   


def get_extreme_val(schema_name, table_name, col, which_extreme='MIN'):
    # get the min and max values for each column in the table
    # col_string = ','.join([f"{which_extreme}({c})" for c in cols])
    # get the extreme values for each column in the table
    query = f"SELECT {which_extreme}({col}) AS max_{col} FROM {schema_name}.{table_name} WHERE {col} = {col}"
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            result = cur.fetchone()[0]
            return result


def get_minmax_dict(schema_name, static_attributes):
    minmax_dict = {}
    for table in ['basin_attributes', 'hysets_basins']:
        minmax_dict[table] = {}
        for c in static_attributes:
            max_vals = get_extreme_val(schema_name, table, c, 'MAX')
            min_vals = get_extreme_val(schema_name, table, c, 'MIN')
            minmax_dict[table][c] = {'min': min_vals, 'max': max_vals}
    return minmax_dict

    
def get_table_cols(schema_name, table_name):
    query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{schema_name}'
    AND table_name   = '{table_name}'
    ORDER BY table_name, column_name;
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query) 
            results = cur.fetchall()
            return [e[0] for e in results]
    
    
def check_if_columns_exist(schema_name, table_name, columns):
    col_exists = []
    for c in columns:
        query = f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = '{schema_name}'
                AND table_name   = '{table_name}'
                AND column_name = '{c}'
            ) AS column_exists;
            """
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                result = cur.fetchone()[0]
                col_exists.append(result)
    return col_exists
       

def get_attributes(schema_name, table_name, static_attributes):
    col_string = ','.join(static_attributes + ['centroid']) 
    query= f'SELECT {col_string} FROM {schema_name}.{table_name};'
    gdf = gpd.read_postgis(query, psycopg2.connect(**conn_params), geom_col='centroid')
    return gdf
    
    
def determine_nearest_proxy(inputs):
    cml, obs, functions = inputs
    # attributes = list(functions.keys())
    # dist_cols = [f'diff_{k}' for k in functions.keys()]
    # calculate the difference between the observed and the row values on each attribute
    # and return the index of the smallest sum of differences
    for _, cml_row in cml.iterrows():
        # get the minimum value and the index of the minimum value
        db_id = cml_row['id']
        distances = pd.DataFrame()
        for k, _ in functions.items():
            if k in functions.keys():
                mapping_function = functions[k]
            else:
                mapping_function = abs_diff
            try:
                distances[f'diff_{k}'] = mapping_function(cml_row[k], obs[k])
            except Exception as ex:
                print(k, ex)
                raise Exception; 'Calumnias!!!'
        
        distances['L1_norm'] = distances.abs().sum(axis=1)
        
        min_idx = distances['L1_norm'].idxmin()
        val = distances['L1_norm'].iloc[min_idx]
        cml.loc[cml['id'] == db_id, 'baseline_station_idx'] = min_idx
        cml.loc[cml['id'] == db_id, 'baseline_station_dist'] = val
    
    return cml
    

def find_unprocessed_distance_idxs(schema_name, table_name, cols):
    
    # join the null query checks on all columns with an OR statement
    col_string = ' OR '.join([f'{col} is NULL' for col in cols])
    query = f"""
    SELECT id
    FROM {schema_name}.{table_name}
    -- where any of the columns are null
    WHERE {col_string};
    """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return [e[0] for e in results]


def run_query(query, tuples=None):
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                if tuples is None:
                    cur.execute(query)
                else:
                    extras.execute_values(cur, query, tuples, page_size=1000)                
                # conn.commit()
                print('Query executed successfully')
    except Exception as e:
        print(e)
        print('Query failed.')
    finally:
        conn.close()


def add_distance_col(schema_name, table_name, column):
    if column.endswith('dist'):
        dt = 'FLOAT'
    elif column.endswith('idx'):
        dt = 'INTEGER'
    else:
        raise Exception; 'Column name must end with "dist" or "idx" or define another data type.'
    query = f"""
    ALTER TABLE {schema_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {column} {dt};
    """
    run_query(query)
         
         
def update_database_with_distances(new_results, schema_name, table_name):
    data = new_results[['id', 'baseline_station_idx', 'baseline_station_dist']].copy()
    n_pts = len(data)
    print(f'  Updating database with new {n_pts} baseline distances...')
    # filter out nan rows for any column
    data.dropna(inplace=True, axis=0, how='any', subset=['id', 'baseline_station_idx', 'baseline_station_dist'])
    n_good_pts = len(data)
    print(f'  Dropped {n_pts - n_good_pts} na values...')
    tuples = list(data.itertuples(index=False, name=None))
    
    # Step 1: Create a temporary table
    try: 
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS temp_update_table
                    (id BIGINT primary key, baseline_station_idx BIGINT, baseline_station_dist FLOAT);
                """)
                # Step 2: Insert your data into the temporary table
                extras.execute_values(
                    cur,
                    """
                    INSERT INTO temp_update_table (id, baseline_station_idx, baseline_station_dist) VALUES %s;
                    """,
                    tuples
                )
                conn.commit()
                # Step 3: Update the main table
                cur.execute(f"""
                    UPDATE {schema_name}.{table_name} main
                    SET 
                        baseline_station_idx = temp.baseline_station_idx,
                        baseline_station_dist = temp.baseline_station_dist
                    FROM temp_update_table temp
                    WHERE main.id = temp.id;
                """)
                conn.commit()
    except Exception as ex:
        print(ex)
    finally:
        conn.close()
    
    
def check_if_table_exists(schema_name, table_name):
    query = f"""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = '{schema_name}'
            AND table_name   = '{table_name}'
        ) AS table_exists;
        """
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()[0]
            return result


def generate_batches(idxs, df, hs_df, diff_funcs, batch_size=500):
    """generator to avoid creating copies of the dataframe in 
    memory outside of each process."""
    n_batches = int(np.ceil(len(idxs) / batch_size))
    batch_idxs = np.array_split(idxs, n_batches)
    for b in batch_idxs:
        yield (df.loc[df['id'].isin(b), :].copy(), hs_df.copy(), diff_funcs)

       
def calculate_cml_distance(cml_id, target_loc, df, diff_funcs):
    
    diff_cols = []
    for k, _ in diff_funcs.items():
        if k == 'centroid':
            target_data = target_loc['centroid']
        else:
            target_data = [target_loc[k].values[0]] * len(df)
        df[f'diff_{k}'] = diff_funcs[k](target_data, df[k])
        diff_cols.append(f'diff_{k}')
    # calculate the sum of all rows starting with 'diff_k'
    df['L1_norm'] = df[diff_cols].abs().sum(axis=1)
    df = df[df['L1_norm'] > 0]
    df['distance_change'] = df['baseline_station_dist'] - df['L1_norm']
    # filter out any rows where the distance change is negative, 
    # menaing the new distance is greater than the baseline distance
    improved_locs = df[df['distance_change'] > 0].copy() 
    
    tot_reduction = improved_locs['distance_change'].sum()
    n_improved_locs = len(improved_locs)
    mean_improved_dist = improved_locs['distance_change'].mean()
    return cml_id, tot_reduction, n_improved_locs, mean_improved_dist


def generate_inputs(df, diff_funcs, batch_ids, max_distance):
    """generator to avoid creating copies of the dataframe in 
    memory outside of each process."""
    for i in batch_ids:
        target = df.loc[df['id'] == i, :].copy()
        # filter for only the cmls within some distance of the target
        # based on distance between basin centroids
        close_cmls = df.loc[df.geometry.distance(target.geometry.iloc[0]) / 1E3 < max_distance, :].copy()
        yield (i, target, close_cmls, diff_funcs)


def compute_distance(row, df, max_distance):
    assert type(row) == pd.Series
    target = df.loc[df.index == row.name].copy()
    close_cmls = df[df.geometry.distance(target.geometry.iloc[0]) / 1E3 < max_distance].copy()
    return calculate_cml_distance(row.name, target, close_cmls, diff_funcs)


diff_funcs = {
    # 'centroid_lat_deg_n': abs_diff, 'centroid_lon_deg_e': abs_diff, 
    'centroid': spatial_dist,
    'drainage_area_km2': log_diff,
    'aspect_deg': circular_diff,
    # all others are abs_diff
}

cml_cols = get_table_cols(schema_name, 'basin_attributes')
hysets_cols = get_table_cols(schema_name, 'hysets_basins')

static_attributes = [c for c in cml_cols if (c in hysets_cols) & (c != 'centroid')]

# minmax_dict = get_minmax_dict(schema_name, static_attributes)

# cml_df = get_attributes(schema_name, 'basin_attributes', ['id'] + static_attributes)
cml_df = gpd.read_postgis(f'SELECT * FROM basins_schema.basin_attributes', psycopg2.connect(**conn_params), geom_col='centroid')

# assert that the 'id' column contains unique values
assert len(cml_df['id'].unique()) == len(cml_df)
cml_df.set_index('id', inplace=True)

hs_df = get_attributes(schema_name, 'hysets_basins', ['watershed_id'] + static_attributes)
assert cml_df.crs == hs_df.crs


RESULTS_DIR = os.path.join(BASE_DIR, 'results')
# results_fpath = os.path.join(RESULTS_DIR, 'nearest_stations.geojson')
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


def get_pair_distances(df, max_distance, idx_to_process):
    """
    Given a geodataframe of basin attributes, return a dataframe of the cartesian product of
    distances between each pair of basin centroids, also return the absolute
    difference between each pair of basins for each attribute.
    """

    land_use_cols = [c for c in df.columns if c.startswith('land_use')]
    soil_cols = ['logk_ice_x100', 'porosity_x100']

    
    abs_diff_query_string = ','.join([f' CAST(ABS(a.{c} - b.{c}) AS SMALLINT) as diff_{c}' for c in land_use_cols + soil_cols])
    abs_diff_col_string = ','.join([f' diff_{c}' for c in land_use_cols + soil_cols])

    all_diff_cols = land_use_cols + soil_cols + ['log_drainage_area_km2', 'slope_deg']
    all_diff_cols_str = ' + '.join([f' diff_{c}' for c in all_diff_cols])

    unprocessed_id_str = ' ,'.join([i for i in idx_to_process])
    # query the database for the cartesian product of centroid
    # pairs that are within some distance of each other
    query = f"""
    WITH CartesianProduct AS (
        SELECT
            a.id AS id_a,
            b.id AS id_b,
            b.baseline_station_dist as baseline_distance,
            CAST((ST_Distance(a.centroid, b.centroid) / 1000) AS SMALLINT) AS distance_km,
            LN(a.drainage_area_km2) - LN(b.drainage_area_km2) AS diff_log_drainage_area_km2,
            CAST(LEAST(ABS(a.slope_deg - b.slope_deg), 360 - ABS(a.slope_deg - b.slope_deg)) AS SMALLINT) AS diff_Slope_deg,
            {abs_diff_query_string}
        FROM {schema_name}.basin_attributes a, {schema_name}.basin_attributes b
        WHERE a.id in ({unprocessed_id_str})
        AND a.id < b.id 
    ),
    L1Norms AS (
        SELECT
            id_a,
            id_b,
            distance_km,
            baseline_distance - ({all_diff_cols_str}) AS potential_difference
        FROM
            CartesianProduct
        WHERE 
            distance_km BETWEEN 0 and {int(max_distance)}
        AND (baseline_distance - ({all_diff_cols_str})) > 1 
    )
    SELECT 
        id_a, 
        COUNT(id_b) as count_idb, 
        SUM(potential_difference) AS potential_diff_sum, 
        AVG(potential_difference) AS potential_diff_mean
    FROM 
        L1Norms
    GROUP BY
        id_a;
    """
    print(query)
    t0 = time()
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            # cols = ['id_a', 'id_b', 'distance', 'diff_log_drainage_area_km2', 'diff_slope_deg']
            # cols += [f'diff_{c}' for c in land_use_cols + soil_cols]
            cols = ['id', 'count_improved', 'potential_difference_sum', 'potential_difference_mean']
            df = pd.DataFrame(results, columns=cols)
            t1 = time()
            print(f'Query time: {t1-t0:.2f}s for {len(df)} rows.')
            return df



def main():

    baseline_cols = ['baseline_station_idx', 'baseline_station_dist']
    potential_distance_cols = ['count_improved', 'potential_difference_sum', 'potential_difference_mean']
    dist_cols_exist = check_if_columns_exist(schema_name, 'basin_attributes', baseline_cols + potential_distance_cols)

    for v, c in zip(dist_cols_exist, baseline_cols + potential_distance_cols):
        if not v:
            print(f'   ...adding {c} columns to BCUB dataset.')
            add_distance_col(schema_name, 'basin_attributes', c)

    print('starting up....')
    baseline_cols = [c for c in cml_df.columns if 'baseline' in c]
    unprocessed_distance_idxs = find_unprocessed_distance_idxs(schema_name, 'basin_attributes', baseline_cols)

    if len(unprocessed_distance_idxs) > 0:    
        print(f'Found unprocessed baseline distances.  Processing {len(unprocessed_distance_idxs)} locations...')
        # these ids should match the "id" column in basin_attributes table
        unprocessed_ids = cml_df.loc[cml_df['id'].isin(sorted(unprocessed_distance_idxs)), 'id'].copy().values
        with mp.Pool(8) as pool:
            # results = pool.map(determine_nearest_proxy, batches)
            results = pool.map(determine_nearest_proxy, generate_batches(unprocessed_ids, cml_df, hs_df, diff_funcs))
            # print(f'Batch time: {t_batch-t0:.2f}s ({ut:.3f}s per row, N={len(b)} batch size)')

        cml_df = gpd.GeoDataFrame(pd.concat(results, axis=0), crs=cml_df.crs, geometry='centroid')

        update_database_with_distances(cml_df, 'basins_schema', 'basin_attributes')

    
    unprocessed_potential_idxs = find_unprocessed_distance_idxs(schema_name, 'basin_attributes', potential_distance_cols)
    if len(unprocessed_potential_idxs) > 0:
        print('Adding expected CML distances to BCUB dataset.')
        t0 = time()

        max_distance = 100 #km -- search radius for a CML's influence on neighbours
        distance_df = get_pair_distances(cml_df, max_distance, unprocessed_potential_idxs)

        # update the database with the new distance columns
        # update_database_with_distances(distance_df, 'basins_schema', 'basin_attributes')
        
        t1 = time()
        ut = (t1-t0) / len(cml_df)
        print(f'Total time: {t1-t0:.2f}s ({ut:.2f}s/row, N={len(cml_df)})')
    
    # set the distance_df index to the id row
    distance_df.set_index('id', inplace=True)
    # concatenate the distance df with the cml_df along the index
    output_df = cml_df[[c for c in cml_df.columns if c not in ['basin']]].copy()
    output_df.set_geometry('pour_pt')
    results_df = pd.concat([output_df, distance_df], join='inner', axis=1)
    

    result_fpath = os.path.join(RESULTS_DIR, f'best_sites_{revision_date}.geojson')
    results_df.to_file(result_fpath)
    print(results_df.head())

    
if __name__ == '__main__':
    # client = Client()
    main()