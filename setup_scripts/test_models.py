import os
import pandas as pd
import numpy as np

import geopandas as gpd


# import HYSETS data
hs_fpath = 'data/HYSETS_watershed_properties_BCUB_with_status.geojson'
hs_gdf = gpd.read_file(hs_fpath)

# import streamflow data
