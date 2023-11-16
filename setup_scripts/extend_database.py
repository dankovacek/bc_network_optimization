
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

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}
schema_name = 'basins_schema'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#########################
# update these file paths
#########################
DATA_DIR = os.path.join(BASE_DIR, 'input_data/')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data/')

dtype_dict = {
    'double': 'FLOAT',
    'int64': 'INT',
    'float64': 'FLOAT',
    'string': 'VARCHAR(255)',
    'object': 'VARCHAR(255)',
    'bool': 'SMALLINT',
}

# PostgreSQL data type mapping
postgres_types = {
    'int64': 'INTEGER',
    'float64': 'DOUBLE PRECISION',
    'bool': 'BOOLEAN',
    'datetime64[ns]': 'TIMESTAMP',
    'object': 'TEXT',
    # Add more mappings as needed
}

# def basic_table_change(query):
#     cur.execute(query)
#     conn.commit()
    

# def basic_query(query):
#     cur.execute(query)
#     result = cur.fetchall()

#     if len(result) > 0:
#         return [e[0] for e in result]
#     else:
#         return None


# def add_table_columns(schema_name, new_cols):
#     table_name = f'basin_attributes'
#     # print(f'    adding new cols {new_cols}.')
#     # add the new columns to the table
    
#     for col in new_cols:
#         if 'land_use' in col:
#             dtype = 'INT'
#         elif 'FLAG' in col:
#             dtype = 'INT'
#         elif ('prcp' in col) & ~('duration' in col):
#             dtype = 'INT'
#         elif 'sample_size' in col:
#             dtype = 'INT'
#         else:
#             dtype = 'FLOAT'

#         cur.execute(f'ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS {col} {dtype};')
#     conn.commit()


# def alter_column_names(schema_name, old_cols, new_cols):

#     table_name = f'basin_attributes'
#     for i in range(len(old_cols)):
#         print(f'    renaming {old_cols[i]} to {new_cols[i]}')
#         cur.execute(f'ALTER TABLE {schema_name}.{table_name} RENAME COLUMN {old_cols[i]} TO {new_cols[i]};')

#     conn.commit()
    



def get_unprocessed_attribute_rows(column, region=None):

    id_query = f"""
    SELECT id, region_code FROM basins_schema.basin_attributes 
    WHERE ({column} IS NULL OR {column} != {column}) """
    # for c in columns[1:]:
    #     id_query += f"OR ({c} IS NULL OR {c} != {c}) "
    
    if region is not None:
        id_query += f"AND (region_code = '{region}') "
    
    id_query += f"ORDER by id ASC;"
    cur.execute(id_query)
    results = cur.fetchall()
    df = pd.DataFrame(results, columns=['id', 'region_code'])
    # groups = df.groupby('region_code').count()
    # print(groups)
    return df['id'].values




def update_database(new_data, schema_name, table_name, column_suffix=""):
    
    # update the database with the new land use data
    # ids = tuple([int(e) for e in new_data['id'].values])
    data_tuples = list(new_data.itertuples(index=False, name=None))
    
    cols = new_data.columns.tolist()
    set_string = ', '.join([f"{e}{column_suffix} = data.v{j}" for e,j in zip(cols[1:], range(1,len(cols[1:])+1))])
    v_string = ', '.join([f"v{e}" for e in range(1,len(cols[1:])+1)])
    
    query = f"""
    UPDATE {schema_name}.{table_name} AS basin_tab
        SET {set_string}
        FROM (VALUES %s) AS data(id, {v_string})
    WHERE basin_tab.id = data.id;
    """
    t0 = time()
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                extras.execute_values(cur, query, data_tuples)
                # commit the changes
                conn.commit()
    t1 = time()
    ut = len(data_tuples) / (t1-t0)
    # print(f'    {t1-t0:.1f}s to update {len(data_tuples)} polygons ({ut:.1f}/second)')


def check_all_null_or_nan(values):
    """Check if all values in a list are null or nan."""
    return all(x is None or (isinstance(x, float) and np.isnan(x)) for x in values)


def nearest_pixel_query(bid, basin_geom_table, raster_table):
    q = f"""
    SELECT
        b.id,
        b.area,
        ST_Distance(b.centroid, ST_Centroid(vals.geom)) AS dist_to_nearest_pixel,
        vals.val AS nearest_pixel_value
    FROM
        {basin_geom_table} b,
        LATERAL (
            SELECT (ST_PixelAsPoints(r.rast)).*
            FROM {raster_table} r
            ORDER BY b.centroid <-> (ST_PixelAsPoints(r.rast)).geom LIMIT 1
        ) AS vals
    WHERE 
        b.id = {bid};
    """
    # create a new connection for each process
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            # Fetch the results
            results = cur.fetchall()
            return results[0]


def create_hysets_table(df, table_name):
    """Create a table and populate with data from the HYSETS dataset.
    The 'geometry' column represents the basin centroid.
    
    ########
    # This is an awful function and yes, I am ashamed.
    ########

    Args:
        df (pandas dataframe): dataframe containing the HYSETS data
    """
    # rename the Watershed_ID column to id
    print('Creating HYSETS table...')    
    # convert centroid geometry to 3005 
    df = df.to_crs(3005)
        
    # get columns and dtypes
    hysets_cols = ['Name', 'Official_ID', 'Watershed_ID',
                   'Centroid_Lat_deg_N', 'Centroid_Lon_deg_E', 'Drainage_Area_km2', 
                   'Drainage_Area_GSIM_km2', 'Flag_GSIM_boundaries', 
                   'Flag_Artificial_Boundaries', 'Elevation_m', 'Slope_deg', 
                   'Gravelius', 'Perimeter', 'Flag_Shape_Extraction', 'Aspect_deg', 
                   'Flag_Terrain_Extraction', 'Land_Use_Forest_frac', 'Land_Use_Grass_frac', 'Land_Use_Wetland_frac', 
                   'Land_Use_Water_frac', 'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac', 'Land_Use_Crops_frac', 
                   'Land_Use_Snow_Ice_frac', 'Flag_Land_Use_Extraction', 
                   'Permeability_logk_m2', 'Porosity_frac', 'Flag_Subsoil_Extraction',
                    'YEAR_FROM', 'YEAR_TO', 'RECORD_LENGTH', 'AGENCY', 'STATUS', 
                   ]
    # convert flag columns to boolean
    df = df[hysets_cols + ['geometry']]
    flag_cols = [e for e in sorted(df.columns) if e.lower().startswith('flag')]
    df[flag_cols] = df[flag_cols].astype(bool)
    df['Watershed_ID'] = df['Watershed_ID'].astype(int)
    
    # add 2010 suffix to land use columns to match bcub dataset
    land_use_cols = [e for e in hysets_cols if e.startswith('Land_Use')]
    
    # print(asdfsd)
    soil_cols = ['Permeability_logk_m2', 'Porosity_frac']    
    
    # drop rows with null values
    # in the future, we should probably fill these values by 
    # deriving the basin polygon and extracting the values
    # this could be part of a validation procedure
    df = df[~df[soil_cols + land_use_cols].isna().any(axis=1)]

    # remap soil columns to match the bcub dataset
    df.rename(columns={'Permeability_logk_m2': 'logk_ice_x100', 
                       'Porosity_frac': 'porosity_x100',}, inplace=True)
    
    
    
    # remap land use columns to match the bcub dataset
    df.rename(columns={e: f'{e}_2010' for e in land_use_cols}, inplace=True)
    
    land_use_cols = [e for e in df.columns if e.startswith('Land_Use')]
    df[land_use_cols] = 100 * df[land_use_cols]
    
    soil_cols = ['logk_ice_x100', 'porosity_x100']
    # multiply the soil values by 100 to match the format of the GLHYMPS data 
    # in the BCUB
    df[soil_cols] = 100 * df[soil_cols].round(1)
    

    # convert the centroid geometry to WKB
    df['centroid'] = df['geometry'].to_wkb(hex=True)
    
    cols = [e for e in df.columns if e not in ['geometry', 'centroid']]
    
    df = df[['centroid'] + cols]

    # get column dtypes 
    dtypes = [postgres_types[str(df[c].dtype)] for c in cols]
    
    # create the table query
    q = f'''CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
        centroid geometry(POINT, 3005),
        '''
    for c, d in zip(cols, dtypes):
        q += f'{c} {d},'
    q = q[:-1] + ');'
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute(q)                
                # convert the centroid geometry to WKB
                cur.execute(f"UPDATE {schema_name}.{table_name} SET centroid = ST_PointFromWKB(decode(centroid, 'hex'));")    

      
                tuples = list(df[['centroid'] + cols].itertuples(index=False, name=None))
                cols_str = ', '.join(['centroid'] + cols)
                query = f"""
                INSERT INTO {schema_name}.{table_name} ({cols_str})
                VALUES %s;
                """
                extras.execute_values(cur, query, tuples)
    
    return df
    

def check_if_table_exists(table_name):
    """
    """
    query = f"""
    SELECT EXISTS (SELECT 1 FROM information_schema.tables 
    WHERE table_name = '{table_name}') AS table_exists;
    """
    cur.execute(query)
    return cur.fetchone()[0]


def check_if_column_exists(schema_name, table_name, column_name):
    """Check if a column exists in a table.

    Args:
        table_name (str): name of the table to check

    Returns:
        bool: true or false if the index exists
    """
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM   information_schema.columns
        WHERE
            table_schema = '{schema_name}'
            AND table_name = '{table_name}'
            AND column_name = '{column_name}'
    ) as column_exists;
    """
    cur.execute(query)
    return cur.fetchone()[0]


def create_spatial_index(schema_name, table_name, geom, geom_idx_name):
    print(f'Creating spatial index for {schema_name}.{table_name}') 
    query = f'CREATE INDEX {geom_idx_name} ON {schema_name}.{table_name} USING GIST ({geom});'
    cur.execute(query)
    conn.commit()


def check_spatial_index(table_name, idx_name, schema_name):
    """Check if a spatial index exists for the given table.  If not, create it.

    Args:
        table_name (str): name of the table to check

    Returns:
        bool: true or false if the index exists
    """
    query = f"""
    SELECT EXISTS (
        SELECT 1
        FROM   pg_indexes
        WHERE
            schemaname = '{schema_name}'
            AND tablename = '{table_name}'
            AND indexname = '{idx_name}'
    ) as index_exists;
    """ 
    cur.execute(query)
    return cur.fetchone()[0]

    

def main():

    ta = time()
    # add HYSETS station data to the database
    hysets_table_name = 'hysets_basins'
    hysets_fpath = os.path.join(DATA_DIR, 'HYSETS_data/HYSETS_watershed_properties_BCUB_with_status.geojson')
    hysets_df = gpd.read_file(hysets_fpath)
    
    hysets_table_exists = check_if_table_exists(hysets_table_name)
    if not hysets_table_exists:
        # remove punctuation marks from column data
        hysets_df['Name'] = hysets_df['NAME'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        hysets_df = create_hysets_table(hysets_df, hysets_table_name)
    
    hysets_spatial_index = 'hysets_centroid_idx'
    # check if the index exists
    hysets_index_exists = check_spatial_index(hysets_table_name, hysets_spatial_index, schema_name)
    if not hysets_index_exists:
        # add a spatial index to the basin centroid points
        create_spatial_index(schema_name, hysets_table_name, 'centroid', hysets_spatial_index)

    tb = time()
    print(f'Created HYSETS data table and adding attributes in {tb - ta:.2f} seconds.')
    

with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main()

cur.close()
conn.close()
