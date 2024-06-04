import os
import time
import itertools
import pandas as pd
import psycopg2
from shapely.geometry import Point, LineString, Polygon
from urllib.request import urlopen
import json

import warnings
warnings.filterwarnings('ignore')

os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import numpy as np

import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYSETS_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
basin_attrs_fname = 'HYSETS_watershed_properties_BCUB_with_status.geojson'
HYSETS_ATTRS_FPATH = os.path.join(BASE_DIR, f'HYSETS_data/{basin_attrs_fname}')

WSC_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/WSC_data'

# open the hysets attributes table
hysets_attrs_fpath = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/HYSETS_watershed_properties.txt'
hs_df = pd.read_csv(hysets_attrs_fpath, delimiter=';', dtype={'Official_ID': str})

# hs_basins_fpath = os.path.join(HYSETS_DIR, f'HYSETS_watershed_boundaries/HYSETS_watershed_boundaries_20200730.shp')
# hs_basins = gpd.read_file(hs_basins_fpath)
# hs_basins = hs_basins.set_crs(4326)

# PostgreSQL connection params
conn_params = {
    'dbname': 'basins',
    'user': 'postgres',
    'password': 'pgpass',
    'host': 'localhost',
    'port': '5432',
}

def read_with_pyarrow(p):
    data = pd.read_csv(os.path.join(HYSETS_DIR, f'hysets_series/{p}.csv'), 
                      parse_dates=['time'],
                      engine='pyarrow')
    data['date'] = pd.to_datetime(data['time'])
    data.drop('time', axis=1, inplace=True)
    data.set_index('date', inplace=True)
    # rename 'discharge' column to the basin id
    data.rename(columns={'discharge': p}, inplace=True)
    return data

           
def usgs_query(url):
    response = urlopen(url)
    json_data = response.read().decode('utf-8', 'replace')
    d = json.loads(json_data)
    df = gpd.GeoDataFrame.from_features(d['features'], crs='EPSG:4326')
    # for c in df.columns:
    #     print(f'{c}: {df[c].values[0]}')
    return df.to_crs(3005)


def retrieve_usgs_stn_data(stn):
    # query the NWIS with the station number to get the station coordinates
    query_url = f'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={stn}'
    try:
        usgs_df = pd.read_csv(query_url, header=29, delimiter='\t')    
        usgs_df = usgs_df[~usgs_df['dec_lat_va'].str.endswith('s')]
        stn_pt = Point(usgs_df['dec_long_va'].values[0], usgs_df['dec_lat_va'].values[0])
        stn_gdf = gpd.GeoDataFrame(geometry=[stn_pt], crs=4326)
        stn_gdf = stn_gdf.to_crs(3005)
        stn_fpath = os.path.join(BASE_DIR, f'processed_data/USGS_stn_loc_{stn}.geojson')
        stn_gdf.to_file(stn_fpath, driver='GeoJSON')
        
    except Exception as ex:
        print('NWIS station query failed')
        print(ex)
        print(asfasdf)
    # query the basin from USGS
    # note this will only work if the missing basin is in the US
    basin_query = f'https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{stn}/basin?simplified=false&splitCatchment=false'
    
    try:
        usgs_basin_df = usgs_query(basin_query)
        usgs_basin_df = usgs_basin_df.to_crs(3005)
        usgs_area = usgs_basin_df.geometry.area.values[0]              
    except Exception as ex:
        print('USGS basin query failed')
        print(ex)
        usgs_basin_df = gpd.GeoDataFrame(geometry=[], crs=3005)
        usgs_area = np.nan
        return stn_gdf, usgs_basin_df, usgs_area
        
    return stn_gdf, usgs_basin_df, usgs_area


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
    pt_path = os.path.join(BASE_DIR, f'processed_data/bcub_ppt_{stn}.geojson')
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


def retrieve_wsc_basin(stn):
    region_prefix = stn[:2]
    wsc_basin_fpath = os.path.join(WSC_DIR, f'{region_prefix}/{stn}/{stn}_DrainageBasin_BassinDeDrainage.shp')
    if not os.path.exists(wsc_basin_fpath):
        return gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(geometry=[]), np.nan
    wsc_basin_df = gpd.read_file(wsc_basin_fpath)
    wsc_basin_df = wsc_basin_df.to_crs(3005)
    wsc_area = wsc_basin_df.geometry.area.values[0]
    wsc_stn_fpath = os.path.join(WSC_DIR, f'{region_prefix}/{stn}/{stn}_PourPoint_PointExutoire.shp')
    if not os.path.exists(wsc_stn_fpath):
        return gpd.GeoDataFrame(geometry=[]), gpd.GeoDataFrame(geometry=[]), np.nan
    stn_df = gpd.read_file(wsc_stn_fpath)
    stn_df = stn_df.to_crs(3005)
    return stn_df, wsc_basin_df, wsc_area


def compute_missing_geometry(stn):

    # query the hysets attributes table for the basin geometry    
    with psycopg2.connect(**conn_params) as conn:

        # first query if the database contains a basin geometry for both ids
        q = f"""SELECT official_id,basin_geometry FROM basins_schema.hysets_basins WHERE "official_id" = '{stn}';"""
        poly_df = gpd.read_postgis(q, conn, geom_col='basin_geometry')
        
        if poly_df.empty:
            
            # find the station info (centroid, drainage area, etc.)
            hs_stn_data = hs_df[hs_df['Official_ID'] == stn].copy()
            hs_area = hs_stn_data['Drainage_Area_km2'].values[0]

            stn_source = hs_stn_data['Source'].values[0]
            print(f'Checking geometry for {stn_source} {stn} (HYSETS area: {hs_area:.2f} km2))')
            # retrieve quality flags
            shape_flag, gsim_flag, ab_flag,  no_official_basin_flag = False, False, False, False
            if hs_stn_data['Flag_Artificial_Boundaries'].values[0] == 1:
                print('artificial boundaries')
                ab_flag = True
            elif hs_stn_data['Flag_GSIM_boundaries'].values[0] == 1:
                gsim_area = hs_stn_data['Drainage_Area_GSIM_km2'].values[0]
                print('GSIM boundaries: ', gsim_area)
                gsim_flag = True
            elif hs_stn_data['Flag_Shape_Extraction'].values[0] == 1:
                print('shape extraction')
                shape_flag = True
            
            if stn_source == 'USGS':
                stn_gdf, official_basin_df, official_area = retrieve_usgs_stn_data(stn)
            else:
                region_prefix = stn[:2]
                stn_gdf, official_basin_df, official_area = retrieve_wsc_basin(stn)
                if stn_gdf.empty:
                    print(f'No WSC data found for {stn}.')
                    return False

            
            official_basin_df['source'] = stn_source
            official_basin = official_basin_df[['source', 'geometry']].copy()

            bdf = get_nearest_bcub_basins(stn, stn_gdf, conn)
            
            if bdf.empty:
                print(f'No BCUB basins found within 1 km of {stn}.')
                return False
            
            # merge the polygons
            bcub_basins = bdf[['id','basin']].copy()
            # bcub_basin = overlap_df[['id','basin']].copy()
            bcub_basins.rename(columns={'basin': 'geometry'}, inplace=True)
            bcub_basins.set_geometry('geometry', inplace=True)
            bcub_basins['source'] = 'BCUB'
            bdf_path = os.path.join(BASE_DIR, f'processed_data/bcub_basin_{stn}.geojson')
            bdf.to_file(bdf_path, driver='GeoJSON')            

            # test if the overlapping region is >= 95%
            # update the hysets_basins table with the new geometry
            if len(official_basin) > 1:
                raise Exception(f'{stn_source} basin query returned too many results.')
            elif len(official_basin) == 0:
                no_official_basin_flag = True
                official_polygon = None
                print(f'{stn_source} basin query returned no results.')
            else:
                official_polygon = official_basin.geometry.values[0]
                # create a boolean column to indicate if any part of the bcub_basin
                # polygons overlap with the usgs basin                
                bcub_basins['overlaps'] = bcub_basins.geometry.intersects(official_polygon)
                bcub_basins['area'] = bcub_basins.geometry.area
                bcub_basins['TP'] = bcub_basins.geometry.intersection(official_polygon).area / official_area * 100
                bcub_basins['FP'] = bcub_basins.difference(official_polygon).area / official_area * 100
                # get the part of the usgs basin that does not overlap
                bcub_basins['FN'] = official_polygon.difference(bcub_basins.geometry).area / official_area * 100
                bcub_basins['area_pct_diff'] = np.abs(bcub_basins['area'] - official_area) / official_area * 100

                basin_match = bcub_basins[(bcub_basins['TP'] >= 95) & (bcub_basins['FP'] <= 5) & (bcub_basins['FN'] <= 5)]

                bcub_basins['area'] =  bcub_basins['area'] /1E6
                basin_match['area'] = basin_match['area'] / 1E6
                print(f'{stn} Area: {official_area/1E6:.2f} km2, HYSETS Area: {hs_area:.2f} km2')
                pcols = ['id', 'TP', 'FP', 'FN', 'area', 'area_pct_diff', 'overlaps']
                print(bcub_basins[pcols].sort_values(['TP', 'FP', 'FN'], ascending=[False, True, True]))
                pcols1 = ['id', 'TP', 'FP', 'FN', 'area', 'area_pct_diff']
                if len(basin_match) > 1:
                    print('Too many basins matched.')
                    print(basin_match[pcols1])
                    basin_match = basin_match.sort_values(['TP', 'FP', 'FN', 'area_pct_diff'], ascending=[False, True, True, True])
                    basin_match.reset_index(inplace=True, drop=True)
                    basin_match = basin_match.loc[[0]]
                    
                if basin_match.empty:
                    # print(stn)
                    # print(bcub_basin[['id', 'pct_pve_overlap', 'pct_nve_overlap']])
                    print(f'No basins matched for {stn}.')
                    out_path = os.path.join(BASE_DIR, f'processed_data/basin_comparison_{stn}_nomatch.geojson')
                    merged = pd.concat([bcub_basins, official_basin[['source', 'geometry']]], axis=0)                    
                else:
                    print('final match:')
                    print(basin_match[['id', 'TP', 'FP', 'FN', 'area', 'area_pct_diff']])
                    # merge the usgs and bcub polygons and output to file for inspection
                    merged = pd.concat([basin_match, official_basin[['source', 'geometry']]], axis=0)
                    out_path = os.path.join(BASE_DIR, f'processed_data/basin_comparison_{stn}.geojson')

                    # update hysets_basins table with missing geometry from official source
                    # ST_PolygonFromWKB(decode(basin_geometry, 'hex'))
                    new_polygon_wkt = official_basin.geometry.to_wkt().values[0]
                    q = f"""
                    INSERT INTO basins_schema.hysets_basins (official_id, drainage_area_km2, basin_geometry)
                    VALUES ('{stn}', {official_area/1E6:.2f}, ST_GeomFromText('{new_polygon_wkt}'))
                    ON CONFLICT (official_id)
                    DO UPDATE SET
                        basin_geometry = EXCLUDED.basin_geometry;
                    """
                    with conn.cursor() as cur:
                        print(f'Updating HYSETS_basins database geometry for {stn}')
                        # if official_area / 1E6 < 25:
                        #     print(q)
                        cur.execute(q)
                        conn.commit()
                    
                comparison_df = gpd.GeoDataFrame(merged, crs=3005)
                comparison_df.to_file(out_path, driver='GeoJSON')
                
                # also need to save some kind of flag to indicate 
                # that the geometry was updated from (source)
                print('')
                return True
        

def centroid_distance_query(p):
    """ Returns the distance between the centroids of two basins in km.
    """
    q = f"""
    SELECT ST_Distance(
        (SELECT centroid FROM basins_schema.hysets_basins WHERE official_id = '{p[0]}'),
        (SELECT centroid FROM basins_schema.hysets_basins WHERE official_id = '{p[1]}')
    ) AS centroid_distance;
    """
    
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            centroid_distance = cur.fetchone()[0]
            return centroid_distance / 1000

            # if centroid_distance is None:
            #     compute_missing_geometry(p)

            
    
# 1.  Find all pairs of basins satisfying the following conditions:
#     a. the basins have minimum 20 years concurrent data
#     b. the basins are within 500 km of each other 

# For a) start by plotting # of stations in the network 
# as a function of the minimum number of years of concurrent data.
# For a) think of how to test...

# get the stations within the study region
basin_df = gpd.read_file(HYSETS_ATTRS_FPATH, dtype_dict={'Official_ID': str})

# create list of pairs of "Offical_ID"s
ids = basin_df['Official_ID'].tolist()

pairs = list(itertools.combinations(ids, 2))

print(f'There are {len(pairs)} pairs of basins in the study region.')


def filter_incomplete_months(df, p, completeness_threshold=0.9):
    a, b = p
    
    df.dropna(subset=p, inplace=True)

    # filter the dataframe for months at least 90% complete
    df['year'] = df.index.year
    df['month'] = df.index.month    
    df['daysinmonth'] = df.index.daysinmonth
    df['YearMonth'] = df.index.map(lambda x: (x.year, x.month))
    df['MonthlyCount'] = df.copy().groupby('YearMonth')['YearMonth'].transform('count')
    
    # filter out months less than 90% complete
    df = df[df['MonthlyCount'] >= completeness_threshold * df['daysinmonth']]
    
    complete_months = df['YearMonth'].unique()
    
    drop_cols = ['year', 'month', 'daysinmonth', 'YearMonth', 'MonthlyCount']

    df = df[[c for c in df.columns if c not in drop_cols]]
    return complete_months, df


# can we quantify the marginal information gain from changing 
# the minimum concurrent record requirement?


def filter_incomplete_years(df):
    # filter the dataframe for months at least 90% complete
    pass


# create an empty matrix of nxn of the length of the number of ids
record_matrix = np.full((len(ids), len(ids)), np.nan)

t0 = time.time()

results = []

for input in ids:
    found_geom = compute_missing_geometry(input)
    # results.append(find_concurrent_record(input))
    # if len(results) > 40:
    #     t1 = time.time()
    #     print(f'Elapsed time: {t1-t0:.1f} seconds ({len(results)/(t1-t0):.2f}/s).')
    #     print(asdfsad)

# with mp.Pool() as pool:
#     results = pool.map(find_concurrent_record, inputs)


# fill the matrix with the number of concurrent months
for r in results:
    i, j, n_complete_months = r
    record_matrix[i, j] = n_complete_months

t1 = time.time()
ut = len(results) / (t1-t0)
print(f'Elapsed time: {t1-t0:.1f} seconds ({ut:.2f}/s).')

# save the matrix
out_fpath = os.path.join(BASE_DIR, 'processed_data/concurrent_record_matrix.npy')
np.save(out_fpath, record_matrix)
# print(record_matrix)


