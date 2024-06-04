''' Present a scatter plot with linked histograms on both axes.

Use the ``bokeh serve`` command to run the example by executing:

    bokeh serve selection_histogram.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/selection_histogram

in your browser.

'''
import os
import numpy as np
import pandas as pd
import geopandas as gpd

from bokeh import events
from bokeh.layouts import gridplot
from bokeh.models import BoxSelectTool, LassoSelectTool, TapTool, HoverTool, Tabs, TabPanel, DataTable
from bokeh.models import CustomJS, Slider, Dropdown, Switch, RadioButtonGroup, AutocompleteInput
from bokeh.models import ColumnDataSource, TableColumn, DataTable

from bokeh.layouts import column, row
from bokeh.plotting import curdoc, figure, ColumnDataSource
from bokeh.palettes import Vibrant3

from shapely.geometry import Point, Polygon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'processed_data')

HS_DATA_PATH = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
TS_PATH = os.path.join(HS_DATA_PATH, 'hysets_series')

hs_properties = pd.read_csv(os.path.join(HS_DATA_PATH, 'HYSETS_watershed_properties.txt'),
                            sep=';', index_col='Official_ID')

# bounds_path = os.path.join(HS_DATA_PATH, 'HYSETS_watershed_boundaries/HYSETS_watershed_boundaries_20200730.shp')
# hs_bounds = gpd.read_file(bounds_path)
    
def get_stn_properties(stn):
    stn = hs_properties.loc[stn].to_dict()
    point = Point(stn['Centroid_Lon_deg_E'], stn['Centroid_Lat_deg_N'])
    stn['geometry'] = point
    gdf = gpd.GeoDataFrame([stn], geometry=[point], crs='EPSG:4326')
    gdf = gdf.to_crs('EPSG:3857')
    stn['x'] = gdf['geometry'].x.values[0]
    stn['y'] = gdf['geometry'].y.values[0]
    return stn

def retrieve_ts_data(proxy, target):
    proxy_df = pd.read_csv(os.path.join(TS_PATH, f'{proxy}.csv'), 
                           index_col=['time'], parse_dates=['time'])
    proxy_df.rename(columns={'discharge': 'proxy'}, inplace=True)
    target_df = pd.read_csv(os.path.join(TS_PATH, f'{target}.csv'), 
                            index_col=['time'], parse_dates=['time'])
    target_df.rename(columns={'discharge': 'target'}, inplace=True)
    df = pd.concat([proxy_df, target_df], axis=1, join='inner')
    target_props = get_stn_properties(target)
    proxy_props = get_stn_properties(proxy)
    
    area_ratio = target_props['Drainage_Area_km2'] / proxy_props['Drainage_Area_km2']
    df['sim_target'] = df['proxy'] * area_ratio
    df.reset_index(inplace=True)
    return df


def retrieve_data(distance_measure='L1'):
    b = bitrate_slider.value
    # create three normal population samples with different parameters
    data_path = os.path.join(DATA_DIR, 'compression_test_results', f'DKL_test_results_{b}bits_20240212.csv')
    df = pd.read_csv(data_path)
    print(df.columns)
    df.dropna(inplace=True)
    df['dist_scaled'] = (df['centroid_distance'] - df['centroid_distance'].min()) / (df['centroid_distance'].max() - df['centroid_distance'].min())
    # df[f'{distance_measure}_attr_dist'] += df['dist_scaled']
    # print(df.head())
    # print(df.columns)
    # print(asdfsd)
    return df

bitrate_slider = Slider(start=4, end=8, value=4, step=1, title="Bitrate")
bin_model_options = [
    ('Equal Interval', 'uniform'),
    ('Equal Probability', 'equiprobable'), 
    ('Proportional Interval', 'proportional')
]
bin_model_dict = {k: {'model_tag': bin_model_options[k][1],
                      'model_label': bin_model_options[k][0],
                      } for k in range(3)}

model_selector = RadioButtonGroup(labels=[e[0] for e in bin_model_options], active=0)

data = retrieve_data()

proxy_stations = sorted(list(data['proxy'].astype(str).unique())) 
target_stations = sorted(list(data['target'].astype(str).unique()))
stn_pair = [proxy_stations[0], target_stations[1]]
target_input = AutocompleteInput(title='Pick a target station', 
                                 value=stn_pair[0], completions=proxy_stations)
proxy_input = AutocompleteInput(title='Pick a proxy station', 
                                value=stn_pair[1], completions=target_stations)

models = ['uniform', 'proportional', 'equiprobable']
cp = {k: v for k, v in zip(models, Vibrant3)}
color_dict = {k: v for k, v in zip(models, cp)}
num_symbol_cols = sorted([e for e in data.columns if e.endswith('num_symbols')])
binning_model_cols = sorted([f'{e}_compression_ratio' for e in models])

TOOLS="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

# create the scatter plot
p = figure(tools=TOOLS, width=600, height=450, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_location=None, y_axis_location=None,
           title="Linked Histograms", y_axis_type='log', output_backend='webgl')
p.background_fill_color = "#fafafa"
p.select(BoxSelectTool).continuous = False
p.select(LassoSelectTool).continuous = False

model_tag = bin_model_dict[model_selector.active]['model_tag']
x_label = bin_model_dict[model_selector.active]['model_label']
print(model_tag)
print(data.columns)
print(safasdfsd)

x = data[f'{model_tag}_num_symbols'].values
y = data[f'{model_tag}_compression_ratio'].values

hhist, hedges = np.histogram(x, bins=20, density=True)
hzeros = np.zeros(len(hedges)-1)
hmax = max(hhist)*1.1

ph = figure(toolbar_location=None, width=p.width, height=150, x_range=p.x_range,
            y_range=(-hmax, hmax), min_border=10, min_border_left=50, y_axis_location="right")

bitrate_line = ph.line([2**bitrate_slider.value, 2**bitrate_slider.value], [-max(hhist), max(hhist)], 
                color="black", line_dash='dashed', line_width=3,
                legend_label = 'encoding bit depth')

ph.legend.location = "top_left"
ph.legend.background_fill_alpha = 0.3

ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"
ph.xaxis.axis_label = 'Residual dictionary size [# symbols]'

# create the vertical histogram

sorted_y = np.sort(y)
y_bin_edges = np.quantile(sorted_y, np.linspace(0, 1, 21))
y_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
          1.7, 1.8, 2.0, 2.1, 2.5, 3., 5., max(y)*1.1 ]
vhist, vedges = np.histogram(y, bins=y_bins, density=True)
vzeros = np.zeros(len(vedges)-1)
vmax = max(vhist)*1.1

pv = figure(toolbar_location=None, width=150, height=p.height, 
            x_range=(-vmax, vmax), y_axis_type='log',
            y_range=p.y_range, min_border=10, y_axis_location="right")

pv.ygrid.grid_line_color = None
pv.xaxis.major_label_orientation = np.pi/4
pv.background_fill_color = "#fafafa"
pv.yaxis.axis_label = 'Compression [%]'

source = ColumnDataSource(data=dict(x=x, y=y, target=data['target'].values, proxy=data['proxy'].values))
r = p.scatter('x', 'y', size=3, color=cp[model_tag], source=source,
              alpha=0.6, legend_label='AR model sim.')

hover_tooltips = [('Proxy', '@proxy'), ('Target', '@target')]
hover = HoverTool(tooltips=hover_tooltips)
p.add_tools(hover)

equal_c_line = p.line([min(x), max(x)], [1, 1], color="red", 
       line_dash='dashed', line_width=3, legend_label='0 compression')

p.legend.location = "top_left"

LINE_ARGS = dict(color=cp[model_tag], line_color=None)
hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], 
                top=hhist, alpha=0.75, **LINE_ARGS)
hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], 
                top=hzeros, alpha=0.25, **LINE_ARGS)

no_compression_line = pv.line([-max(vhist), max(vhist)], [1, 1], color="red", line_dash='dashed', line_width=3)
LINE_ARGS = dict(color=cp[model_tag], line_color=None)
vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, alpha=0.75, **LINE_ARGS)
vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.25, **LINE_ARGS)


def update_colors(model_tag):
    
    c = cp[model_tag]

    hh1.glyph.fill_color = c
    hh2.glyph.fill_color = c
    vh1.glyph.fill_color = c
    vh2.glyph.fill_color = c

    hh1.glyph.line_color = c
    hh2.glyph.line_color = c
    vh1.glyph.line_color = c
    vh2.glyph.line_color = c
    
    r.glyph.fill_color = c
    r.glyph.line_color = c
    # label = bin_model_dict[model_selector.active]['model_label']
    # p.legend[0].items[0].label.value = label
    # p.legend.update()

def update(attr, old, new):
    inds = new
    mtag = bin_model_dict[model_selector.active]['model_tag']
    
    x = data[f'{mtag}_num_symbols'].values
    y = data[f'{mtag}_compression_ratio'].values
    
    if len(inds) == 0 or len(inds) == len(x):
        hhist1, hhist2 = hzeros, hzeros
        vhist1, vhist2 = vzeros, vzeros
    else:
        neg_inds = np.ones_like(x, dtype=bool)
        neg_inds[inds] = False
        hhist1, _ = np.histogram(x[inds], bins=hedges)
        vhist1, _ = np.histogram(y[inds], bins=vedges)
        hhist2, _ = np.histogram(x[neg_inds], bins=hedges)
        vhist2, _ = np.histogram(y[neg_inds], bins=vedges)

        vmax = max(vhist1.max(), vhist1.max())
        hmax = max(hhist1.max(), hhist2.max())

        # pick the first selection if there are multiple
        # and get the proxy and target values
        proxy = data['proxy'].values[inds[0]]
        target = data['target'].values[inds[0]]
        update_ts_layout(proxy, target)

    hh1.data_source.data["top"]   =  hhist1
    hh2.data_source.data["top"]   = -hhist2
    vh1.data_source.data["right"] =  vhist1
    vh2.data_source.data["right"] = -vhist2

    ph.y_range.start, ph.y_range.end = -1.1 * hmax, 1.1 * hmax
    pv.x_range.start, pv.x_range.end = -1.1 * vmax, 1.1 * vmax

    equal_c_line.data_source.data['x'] = [min(x), max(x)]
    no_compression_line.data_source.data['x'] = (-vmax, vmax)    

    bitrate_line.data_source.data['x'] = (2**bitrate_slider.value, 2**bitrate_slider.value)

def update_bitrate(attr, old, new):
    # create the vertical histogram
    new_data = retrieve_data()
    model = bin_model_dict[model_selector.active]['model_tag']

    x_label = f'{model}_num_symbols'
    y_label = f'{model}_compression_ratio'
    xx = new_data[x_label].values
    yy = new_data[y_label].values
    vhist, vedges = np.histogram(yy, bins=y_bins, density=True)
    vmax = max(vhist)*1.1

    # left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros

    # vh1.data_source.data["right"] = vhist
    
    vh1.data_source.data["left"] = vzeros
    vh2.data_source.data["left"] = vzeros

    vh1.data_source.data["right"] = vhist
    vh2.data_source.data["right"] = vzeros
    
    vh1.data_source.data["bottom"] = vedges[:-1]
    vh1.data_source.data["top"] = vedges[1:]
    vh2.data_source.data["bottom"] = vedges[:-1]
    vh2.data_source.data["top"] = vedges[1:]

    pv.x_range.start, pv.x_range.end = -vmax * 1.1, vmax * 1.1
    pv.y_range.start, pv.y_range.end = min(vedges), max(vedges)

    hhist, hedges = np.histogram(xx, bins=20, density=True)
    hzeros = np.zeros(len(hedges)-1)
    hmax = max(hhist)*1.1

    ph.y_range.start, ph.y_range.end = -hmax * 1.1, hmax * 1.1
    ph.x_range.start, ph.x_range.end = min(hedges), max(hedges)

    hh1.data_source.data["left"] = hedges[:-1]
    hh1.data_source.data["right"] = hedges[1:]
    hh2.data_source.data["left"] = hedges[:-1]
    hh2.data_source.data["right"] = hedges[1:]
    hh1.data_source.data["top"] = hhist    
    hh2.data_source.data["top"] = hzeros
    hh1.data_source.data["bottom"] = hzeros
    hh2.data_source.data["bottom"] = hzeros

    r.data_source.data["x"] = xx
    r.data_source.data["y"] = yy
    
    equal_c_line.data_source.data['x'] = [min(xx), max(xx)]
    no_compression_line.data_source.data['x'] = (-max(vhist), max(vhist))    
    bitrate_line.data_source.data['x'] = (2**bitrate_slider.value, 2**bitrate_slider.value)
    update_colors(model)
    
def update_ts_layout(attr, old, new):
    # update the autocomplete values
    proxy = proxy_input.value
    target = target_input.value

    ts_source.data = ts_source.from_df(retrieve_ts_data(proxy, target))
    ts_plot.title.text = f'{proxy} -> {target}'

    map_data = create_map_df(proxy, target)
    mplt.data_source.data = map_data


stn_pair = [proxy_stations[0], target_stations[1]]
stn_dict = {k: v for k, v in zip(['proxy', 'target'], stn_pair)}
ts_data = retrieve_ts_data(*stn_pair)
ts_source = ColumnDataSource(data=ts_data)
ts_plot = figure(width=1000, height=300, 
                 title="Time series", 
                 x_axis_type='datetime')

i = 0
da_dict = {}
for label, stn_id in stn_dict.items():
    da_dict[stn_id] = Point(hs_properties.loc[ stn_id, 'Centroid_Lon_deg_E'],
                            hs_properties.loc[stn_id, 'Centroid_Lat_deg_N'])
    ts_plot.line('time', label, legend_label=label, source=ts_source, 
                 color=Vibrant3[i], line_width=2)
    if label == 'target':
        ts_plot.line('time', 'sim_target', legend_label=f'sim. target', source=ts_source, 
                     color=Vibrant3[i], line_width=2, line_dash='dashed')
    i += 1


def create_map_df(proxy, target):
    proxy_props = get_stn_properties(proxy)
    target_props = get_stn_properties(target)
    
    # make the data dicts into a dataframe where the columns 
    # are the station names
    data = pd.DataFrame()
    for p in proxy_props.keys():
        if p != 'geometry':
            data[p] = [proxy_props[p], target_props[p]]
    
    proxy_loc = Point(proxy_props['x'], proxy_props['y'])
    target_loc = Point(target_props['x'], target_props['y'])
    distance = proxy_loc.distance(target_loc) / 1000
    data['distance_km'] = [distance, distance]
    return data

map_df = create_map_df(*stn_pair)

map_source = ColumnDataSource(map_df)
map_plot = figure(width=600, height=400, 
                  title="Location Map", 
                  x_axis_type="mercator", y_axis_type="mercator")
# map_plot.add_tile("CartoDB Positron", retina=True)
map_plot.add_tile("Esri World Imagery", retina=True)
map_plot.xgrid.grid_line_color = None
map_plot.ygrid.grid_line_color = None
mplt = map_plot.circle('x', 'y', source=map_source, size=10, color=Vibrant3[0], alpha=0.5)

flag_cols = [e for e in map_df.columns if e.startswith('Flag')]
exclude_cols = ['Watershed_ID', 'x', 'y', 'geometry',
                'Centroid_Lon_deg_E', 'Centroid_Lat_deg_N',
                 'Drainage_Area_GSIM_km2'] + flag_cols
lu_cols = [e for e in map_df.columns if e.startswith('Land_Use')]

table_data = map_df.copy()
table_data = table_data[[e for e in table_data.columns if e not in exclude_cols]]
table_source = ColumnDataSource(table_data)
table_cols = [
    TableColumn(field=e, title=e) for e in table_data.columns
]
data_table = DataTable(source=table_source, columns=table_cols, width=1000, height=100)

r.data_source.selected.on_change('indices', update)
bitrate_slider.on_change('value', update_bitrate)
model_selector.on_change('active', update_bitrate)
proxy_input.on_change('value', update_ts_layout)
target_input.on_change('value', update_ts_layout)

layout1 = gridplot([[model_selector, bitrate_slider], 
                   [p, pv], 
                   [ph, None]], 
                   merge_tools=False)

layout2 = gridplot([[ts_plot],
                    [data_table],
                    [row(map_plot, column(proxy_input, target_input))]],
                   merge_tools=False)

tab1 = TabPanel(child=layout1, title='Selection Histogram')
tab2 = TabPanel(child=layout2, title='Paired Signal')

layout = Tabs(tabs=[tab1, tab2])

curdoc().add_root(layout)
curdoc().title = "Selection Histogram"
