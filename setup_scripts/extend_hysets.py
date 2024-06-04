
import os
import json
import rioxarray as rxr
import numpy as np
import pandas as pd

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import psycopg2

from urllib.request import urlopen

from shapely.geometry import Point, Polygon

# import extras
from psycopg2 import extras


import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSC_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/WSC_data'

hs_data_path = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
hs_attrs_path = 'HYSETS_watershed_properties_BCUB_with_status.geojson'

# api url for nwis sites
usgs_api_url = 'https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/'

schema_name = 'basins_schema'

daymet_path = '/home/danbot2/code_5820/large_sample_hydrology/bcub/processed_data/DAYMET/'
daymet_params = ['prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp',
                 'high_prcp_freq', 'high_prcp_duration', 'low_prcp_freq', 'low_prcp_duration',
                 ]

conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
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

    df.columns = [e.lower() for e in df.columns]
        
    # get columns and dtypes
    hysets_cols = [e.lower() for e in ['Watershed_ID', 'Name', 'Official_ID', 
                   'Centroid_Lat_deg_N', 'Centroid_Lon_deg_E', 'Drainage_Area_km2', 
                   'Drainage_Area_GSIM_km2', 'Flag_GSIM_boundaries', 
                   'Flag_Artificial_Boundaries', 'Elevation_m', 'Slope_deg', 
                   'Gravelius', 'Perimeter', 'Flag_Shape_Extraction', 'Aspect_deg', 
                   'Flag_Terrain_Extraction', 'Land_Use_Forest_frac', 'Land_Use_Grass_frac', 'Land_Use_Wetland_frac', 
                   'Land_Use_Water_frac', 'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac', 'Land_Use_Crops_frac', 
                   'Land_Use_Snow_Ice_frac', 'Flag_Land_Use_Extraction', 
                   'Permeability_logk_m2', 'Porosity_frac', 'Flag_Subsoil_Extraction', 
                   'YEAR_FROM', 'YEAR_TO', 'RECORD_LENGTH', 'AGENCY', 'STATUS', 
                   ]]
    # convert flag columns to boolean
    df = df[hysets_cols + ['geometry']]
    flag_cols = [e for e in sorted(df.columns) if e.lower().startswith('flag')]
    df[flag_cols] = df[flag_cols].astype(bool)
    df['watershed_id'] = df['watershed_id'].astype(int)
    
    # add 2010 suffix to land use columns to match bcub dataset
    land_use_cols = [e for e in hysets_cols if e.startswith('land_use')]
    
    # print(asdfsd)
    soil_cols = ['permeability_logk_m2', 'porosity_frac']
    
    # don't drop rows with missing values for now
    # These will be infilled and flagged
    # there are roughly 400 missing geometries
    # that could be estimated from official sources
    # or at worst from the station location
    # df = df[~df[soil_cols + land_use_cols].isna().any(axis=1)]

    # remap soil columns to match the bcub dataset
    df.rename(columns={'permeability_logk_m2': 'logk_ice_x100',
                       'porosity_frac': 'porosity_x100',}, inplace=True)
    
    # remap land use columns to match the bcub dataset
    df.rename(columns={e: f'{e}_2010' for e in land_use_cols}, inplace=True)
    
    land_use_cols = [e for e in df.columns if e.startswith('land_use')]
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
    
    cur.execute(q)

    # set a uniqueness constraint on the official_id column
    q = f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'unique_official_id'
        ) THEN
            ALTER TABLE basins_schema.hysets_basins
            ADD CONSTRAINT unique_official_id UNIQUE (official_id);
        END IF;
    END
    $$;
    """
    cur.execute(q)
    
    # convert the centroid geometry to WKB
    cur.execute(f"UPDATE {schema_name}.{table_name} SET centroid = ST_PointFromWKB(decode(centroid, 'hex'));")    
    
        
    print('   ...hysets table created.')
    tuples = list(df[['centroid'] + cols].itertuples(index=False, name=None))
    cols_str = ', '.join(['centroid'] + cols)
    query = f"""
    INSERT INTO {schema_name}.{table_name} ({cols_str})
    VALUES %s ON CONFLICT (official_id) DO NOTHING;
    """
    extras.execute_values(cur, query, tuples)

    print('   ...hysets table populated.')
    
    return df


# load the study region bounds (to filter the HYSETS basins within the study region)
study_region_bounds_fpath = '/home/danbot2/code_5820/large_sample_hydrology/bc_network_optimization/input_data/BC_study_region_polygon_4326.geojson'
study_region_gdf = gpd.read_file(study_region_bounds_fpath)

# load the existing hysets data
hs_attr_df = gpd.read_file(
    os.path.join(BASE_DIR, f'HYSETS_data/{hs_attrs_path}'), 
    dtype_dict={'Official_ID': str}
)

n_basins = len(hs_attr_df)
print(f'There are {n_basins} stations in the HYSETS dataset within the study region.')

# retrieve the basin polygons and add them to the database
bcub_table_name = 'basin_boundaries'
hs_bounds_fpath = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/HYSETS_watershed_boundaries/HYSETS_watershed_boundaries_20200730.shp'
hs_gdf = gpd.read_file(hs_bounds_fpath)
# check type of officialid
hs_gdf['OfficialID'] = hs_gdf['OfficialID'].astype(str)
hs_gdf = hs_gdf.set_crs(4326)

# filter the HYSETS basins within the study region
hs_gdf = hs_gdf[hs_gdf['OfficialID'].isin(hs_attr_df['Official_ID'])]

assert hs_gdf.crs == hs_attr_df.crs, 'Data must be in the same CRS.'

# project to 3005 to match the daymet data
hs_gdf = hs_gdf.to_crs(3005)
hs_attr_df = hs_attr_df.to_crs(3005)


def usgs_basin_polygon_query(url):
    response = urlopen(url)
    json_data = response.read().decode('utf-8', 'replace')
    d = json.loads(json_data)
    df = gpd.GeoDataFrame.from_features(d['features'], crs='EPSG:4326')
    return df.to_crs(3005)


def get_nearest_bcub_basins(stn, stn_gdf, conn):
    # find all BCUB pour points within 250 m of the station coordinates
    pour_pt_wkt = stn_gdf[['geometry']].to_wkt().values[0][0]
    
    ptq = f"""
        SELECT id,drainage_area_km2,pour_pt,basin
        FROM basins_schema.basin_attributes
        WHERE ST_DWithin(pour_pt, ST_GeomFromText('{pour_pt_wkt}', 3005), 1000)
        ORDER BY drainage_area_km2 DESC;
        """   
    pt_df = gpd.read_postgis(ptq, conn, geom_col='pour_pt')

    # save the geometry
    pt_path = os.path.join(BASE_DIR, f'processed_data/basin_matches/bcub_ppt_{stn}.geojson')
    pt_df.to_file(pt_path, driver='GeoJSON')
    if pt_df.empty:
        print(f'No BCUB pour points found within 1 km of stn.')
        return pd.DataFrame()

    # get the corresponding BCUB polygon
    id_str = ','.join([str(e) for e in pt_df['id'].tolist()])
    bq = f"""
        SELECT id,drainage_area_km2,basin
        FROM basins_schema.basin_attributes
        WHERE id IN ({id_str});
        """ 
    bdf = gpd.read_postgis(bq, conn, geom_col='basin')
    return bdf


def get_basin_from_bcub(stn, stn_gdf):
    basin_matches = get_nearest_bcub_basins(stn, stn_gdf, conn)
    if basin_matches.empty:
        print(f' No BCUB basin matches found for {stn} within search criteria.')
        return pd.DataFrame()
    hysets_area = hs_attr_df[hs_attr_df['Official_ID'] == stn]['Drainage_Area_km2'].values[0]

    basin_matches['distance'] = basin_matches.distance(stn_gdf.geometry.values[0])
    basin_matches['area_pct_diff'] = 100 * np.abs(basin_matches['drainage_area_km2'] - hysets_area) / hysets_area
    basin_matches = basin_matches.sort_values(['area_pct_diff','distance'], ascending=[True, False])
    close_matches = basin_matches[basin_matches['area_pct_diff'] < 5].copy()
    if close_matches.empty:
        print(f'No BCUB basin matches found for {stn}.')
        match_fpath = os.path.join(BASE_DIR, f'processed_data/basin_matches/bcub_basin_matches_{stn}.geojson')
        basin_matches.to_file(match_fpath, driver='GeoJSON')
        stn_fpath = os.path.join(BASE_DIR, f'processed_data/basin_matches/USGS_stn_loc_{stn}.geojson')
        stn_gdf.to_file(stn_fpath, driver='GeoJSON')
        print(f'  Hysets area = {hysets_area:.2f} km2')
        print(basin_matches)
        return pd.DataFrame()
    
    close_matches.reset_index(inplace=True, drop=True)
    match = close_matches.loc[[0]]
    return match


def retrieve_usgs_stn_data(stn):
    # query the NWIS with the station number to get the station coordinates    
    try:
        query_url = usgs_api_url + f'USGS-{stn}'
        usgs_data = pd.read_json(query_url)
        usgs_stn_loc = usgs_data['features'][0]['geometry']['coordinates']
        stn_pt = Point(*usgs_stn_loc)
        stn_gdf = gpd.GeoDataFrame(geometry=[stn_pt], crs=4326)
        stn_gdf = stn_gdf.to_crs(3005)
        stn_fpath = os.path.join(BASE_DIR, f'processed_data/USGS_stn_loc_{stn}.geojson')
        stn_gdf.to_file(stn_fpath, driver='GeoJSON')
        return stn_gdf
    except Exception as ex:
        msg = f'USGS station query failed for {stn}. {ex}'
        print(msg)
        return pd.DataFrame()



def retrieve_usgs_basin_data(stn):
    """Retrieve the USGS basin polygon and station location from the NLDI API. 
    If there is no basin for the station, use the NLDI to retrieve upstream 
    and downstream boundaries.  
    Pick the one closest in (HYSETS published) area to the station location."""    
    
    # query the basin polygon from USGS
    basin_query = usgs_api_url + f'USGS-{stn}/basin?simplified=false&splitCatchment=false'    
    try:
        usgs_basin_df = usgs_basin_polygon_query(basin_query)
        # dissolve the basin polygons
        usgs_basin_df = usgs_basin_df.dissolve()
        usgs_basin_df = usgs_basin_df.to_crs(3005)
        # check if geometry is multipolygon
        if usgs_basin_df.geometry.type.values[0] == 'MultiPolygon':
            print(f'   ...MultiPolygon detected, attemping to make geometry valid.')
            usgs_basin_df = usgs_basin_df.explode()
            usgs_basin_df['area'] = usgs_basin_df.geometry.area / 1E6
            usgs_basin_df['area_pct'] = usgs_basin_df['area'] / usgs_basin_df['area'].sum()
            usgs_basin_df = usgs_basin_df[usgs_basin_df['area_pct'] > 0.95]
            if len(usgs_basin_df) > 1:
                raise Exception('USGS basin polygon query returned multiple polygons.')

        return usgs_basin_df
    except Exception as ex:
        print(f'USGS basin polygon query failed for {stn}.  {ex}')
        return pd.DataFrame()



def retrieve_wsc_basin(stn):
    """Retrieve the WSC basin polygon and station 
    location from the July 2022 updated polygon set.
    """
    basin_df, ppt_df, stn_df = gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    region_prefix = stn[:2]

    wsc_basin_fpath = os.path.join(WSC_DIR, f'{region_prefix}/{stn}/{stn}_DrainageBasin_BassinDeDrainage.shp')
    wsc_ppt_fpath = os.path.join(WSC_DIR, f'{region_prefix}/{stn}/{stn}_PourPoint_PointExutoire.shp')
    wsc_stn_fpath = os.path.join(WSC_DIR, f'{region_prefix}/{stn}/{stn}_Station.shp')
    
    basin_exists = os.path.exists(wsc_basin_fpath)
    ppt_exists = os.path.exists(wsc_ppt_fpath)
    stn_pt_exists = os.path.exists(wsc_stn_fpath)
    if not (basin_exists & ppt_exists & stn_pt_exists):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if os.path.exists(wsc_basin_fpath):
        basin_df = gpd.read_file(wsc_basin_fpath)
        basin_df = basin_df.to_crs(3005)    
        # check if the geometry is multipolygon
        if basin_df.geometry.type.values[0] == 'MultiPolygon':
            basin_df = basin_df.explode()
            basin_df['area'] = basin_df.geometry.area / 1E6
            basin_df['area_pct'] = basin_df['area'] / basin_df['area'].sum()
            basin_df = basin_df[basin_df['area_pct'] > 0.95]
            if len(basin_df) > 1:
                raise Exception('WSC basin polygon query returned multiple polygons.')
    
    if os.path.exists(wsc_ppt_fpath):
        ppt_df = gpd.read_file(wsc_ppt_fpath)
        ppt_df = ppt_df.to_crs(3005)
    
    if os.path.exists(wsc_stn_fpath):
        stn_df = gpd.read_file(wsc_stn_fpath)
        stn_df = stn_df.to_crs(3005)
    
    return basin_df, ppt_df, stn_df



def add_basin_geom_column(df, table_name, schema_name='basins_schema'):
    q = f"""
        DO $$
        BEGIN
            -- Check if the column does not exist
            IF NOT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND column_name = 'basin_geometry'
            ) THEN
                -- If the column does not exist, create it
                ALTER TABLE {schema_name}.{table_name}
                ADD COLUMN basin_geometry geometry(POLYGON, 3005);                      
            END IF;
        END
        $$;
        """
    print('')
    print('Adding basin polygon geometry column to the hysets_basins table...')
    # print(q)
    cur.execute(q)    
    
    cur.execute(f"UPDATE {schema_name}.{table_name} SET basin_geometry = ST_PolygonFromWKB(decode(basin_geometry, 'hex'));")    
    
    # create a flag column for updated official basins if the column doesn't exist
    q = f"""
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_schema = '{schema_name}'
            AND table_name   = '{table_name}'
            AND column_name  = 'updated_official_basin'
        ) THEN
            ALTER TABLE {schema_name}.{table_name} ADD COLUMN updated_official_basin BOOLEAN;
        END IF;
    END
    $$;
    """
    cur.execute(q)

    # commit the changes
    conn.commit()


def add_basin_geometries(df, table_name):
    # add basin geometries from the HYSETS basin boundaries shapefile 
    # to the database column 'basin_geometry'
    print('Adding basin geometries to the HYSETS table...')
    # hs_df columns:)
    # 'features', 'Name', 'OfficialID', 'FlagPAVICS', 'Source', 'Area', 'geometry'
    for i, row in df.iterrows():
        stn = row['official_id']
        source = row['agency']
        ab_flag = row['flag_artificial_boundaries']

        if (stn in hs_gdf['OfficialID'].values) & (not ab_flag):
            basin = hs_gdf[hs_gdf['OfficialID'] == stn].copy()
            # convert the geometry to WKB
            wkb = basin.geometry.to_wkb(hex=True)
            bcub_basin = False
            updated_basin = False
        else: # retrieve the basin geometry from the official source USGS or WSC            
            if source == 'USGS':
                stn_gdf = retrieve_usgs_stn_data(stn)
                basin_df = retrieve_usgs_basin_data(stn)
            elif source == 'WSC':
                basin_df, ppt_gdf, stn_gdf = retrieve_wsc_basin(stn)
            else:
                raise Exception(f'Source {source} not recognized.')
            
            bcub_basin = False
            updated_basin = False
            if basin_df.empty:
                print(f'    {stn} basin geometry not found in {source} (ab_flag={ab_flag}).')
                if source == 'USGS':
                    basin_df = get_basin_from_bcub(stn, stn_gdf)
                    if not basin_df.empty:
                        updated_basin = True    
                else:
                    if ppt_gdf.empty:
                        print(f'No pour point found for {stn} in BCUB meeting match criteria.')
                        basin_df = hs_gdf[hs_gdf['OfficialID'] == stn].copy()
                    else:
                        basin_df = get_basin_from_bcub(stn, ppt_gdf)
                        if not basin_df.empty:
                            bcub_basin = True
                            updated_basin = True
            
            if basin_df.empty:
                print(f'No basin found for {stn} in BCUB meeting match criteria, using fallback HYSETS polygon.')
                basin_df = hs_gdf[hs_gdf['OfficialID'] == stn].copy()
                bcub_basin = False
                updated_basin = False


            wkb = basin_df.geometry.to_wkb(hex=True)
            print(f'Updating {stn} basin geometry from {source}... (ab_flag={ab_flag})')

        # update the database
        q = f"UPDATE {schema_name}.{table_name} SET basin_geometry = decode('{wkb.values[0]}', 'hex') WHERE official_id = '{stn}';"
        cur.execute(q)
        
        # update the updated_official_basin boolean column
        q = f"""UPDATE {schema_name}.{table_name} SET updated_official_basin = {updated_basin};"""
        cur.execute(q)

        # update the source to be BCUB instead of USGS or WSC
        if bcub_basin:
            q = f"""UPDATE {schema_name}.{table_name} SET agency = 'BCUB' WHERE official_id = '{stn}';"""
        cur.execute(q)
        if i % 200 == 0:
            print(f'    {i} of {len(df)} basins processed.')
    conn.commit()


def get_basin_from_hysets_database(stn):
    print(f'querying station {stn} from hysets')
    query = f"""SELECT * FROM basins_schema.hysets_basins WHERE official_id = '{stn}';"""
    basin_df = gpd.read_postgis(query, conn, geom_col='basin_geometry')
    print(basin_df)
    print(asdfasd)


def process_daymet_data(hs_gdf, hs_attr_df, daymet_path, daymet_params):
    for p in daymet_params:
        # load the corresponding raster
        raster_fpath = os.path.join(daymet_path, f'{p}_mosaic_3005.tiff')
        # open the raster as an xarray dataset
        ds = rxr.open_rasterio(raster_fpath)
        print(f'Processing {p}...')
        print(ds)
        print(asdfasdf)
        # inputs = [(i, ds, hs_gdf[hs_gdf['OfficialID'] == row['Official_ID']].copy(), 'EPSG:3005') for i, row in hs_attr_df.iterrows()]
        n = 0
        for i, row in hs_attr_df.iterrows():
            n += 1
            stn = row['Official_ID']
            # query basin from HYSETS basins
            basin = get_basin_from_hysets_database(stn)
            ab_flag = row['Flag_Artificial_Boundaries']
            if (len(basin) == 0):
                print(f'    {stn} basin geometry not included in HYSETS basin boundaries.')
                continue
            clipped = ds.rio.clip(basin.geometry.values, 'EPSG:3005')
            hs_attr_df.loc[i, p] = np.nanmean(clipped.values)
            if n % 200 == 0:
                print(f'    {i} of {len(hs_attr_df)} basins processed.')
    return hs_attr_df


def main(hs_attr_df):

    # create the hysets table
    # df = create_hysets_table(hs_attr_df, 'hysets_basins')
    # add_basin_geom_column(df, 'hysets_basins') 

    # add basin geometries
    # add_basin_geometries(df, 'hysets_basins')
    # now derive the daymet data for all basin polygons
    hs_attr_df = process_daymet_data(hs_gdf, hs_attr_df, daymet_path, daymet_params)

    # output path
    out_path = hs_attrs_path.replace('.geojson', '_daymet.geojson')

    # save the data
    hs_attr_df.to_file(os.path.join(BASE_DIR, f'HYSETS_data/{out_path}'), driver='GeoJSON')


with psycopg2.connect(**conn_params) as conn:
    cur = conn.cursor() 
    main(hs_attr_df)

cur.close()
conn.close()

