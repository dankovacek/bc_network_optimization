import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

tf.keras.mixed_precision.set_global_policy('mixed_float16')

#  The id encodes the model architecture variables:
#  prefix  bitrate  batch size  divergence metric  training length (frac)  dropout_rate  # layers  # nodes/layer
#    nn      b         bs           m                      s                     dr           nl           nn


# Experiment 1: Test the effect of the number of layers and the number of nodes per layer, dropout rate, and batch size
# Experiment 2: Test the effect of geographic sample bias in training data 
#             (e.g. sort by latitude and take the first 10% of the data, then the first 50% of the data, etc.)
# Experiment 3: Test the effect of 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class WeightHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weight_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Assuming model has a single layer or specifying the layer of interest
        weights = self.model.layers[0].get_weights()[0] # Adjust layer index as needed
        self.weight_history.append(weights)


class NNModel():
    def __init__(self, data, 
                 feature_columns, target_column, bitrate, 
                 model_arch_dict, dropout_rate, training_frac,
                 bin_model, divergence_metric, epochs=50):
        
        self.feature_columns = feature_columns
        self.target_column = target_column

        # self.target_column = f'{bin_model}_{divergence_metric}'
        self.bitrate = bitrate
        self.model_arch_dict = model_arch_dict
        self.dropout_rate = dropout_rate
        # self.training_size = int(training_frac * len(data))
        self.training_frac = training_frac
        
        self.input_df = data[feature_columns + [target_column]].copy().dropna(how='any')
        
        self.epochs = epochs

        self.define_callbacks()

        self.run_model()

    def run_model(self):
        model, X_train, y_train, X_test, y_test = self.prepare_model()
        
        model.fit(X_train, y_train, epochs=self.epochs, callbacks=self.callback_list, 
          validation_split=0.25, batch_size=self.model_arch_dict['batch_size'])
        
        self.test_loss = model.evaluate(X_test, y_test)[0]
        model_id = self.model_arch_dict['id']

        print(f'    Test Loss: {self.test_loss:.2f}  {model_id}')
        predictions = model.predict(X_test)
        results_df = pd.DataFrame({'actual': y_test, 'predicted': predictions.flatten()})
        
        result_fname = self.model_arch_dict['id']
        result_fpath = os.path.join('processed_data', 'nn_results')
        if not os.path.exists(result_fpath):
            os.makedirs(result_fpath)
        results_df.to_csv(os.path.join(result_fpath, f'{result_fname}.csv'))

    def define_callbacks(self):
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=1, mode='min', restore_best_weights=True)
        weight_history_callback = WeightHistory()
        self.callback_list = [weight_history_callback, early_stopping_callback]
            
    def prepare_model(self):
        # keep just the first rows corresponding to the training size
        # dataframe = self.input_df.loc[:self.training_size, :].copy()
        # Split the data into features and target
        X = self.input_df[self.feature_columns].values
        y = self.input_df[self.target_column].values
        
        # Standardize the features
        # scaler = StandardScaler()
        scaler = QuantileTransformer(output_distribution='normal')
        X_scaled = scaler.fit_transform(X)

        # Split the dataset into training and testing sets
        test_frac = 1 - self.training_frac
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_frac, random_state=42)

        model = keras.Sequential(self.model_arch_dict['layers'])

        # Compile the model
        optimizer = self.model_arch_dict['optimizer']
        loss = self.model_arch_dict['loss']
        metrics = self.model_arch_dict['metrics']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model, X_train, y_train, X_test, y_test


HS_DATA_DIR = '/home/danbot2/code_5820/large_sample_hydrology/common_data/HYSETS_data/'
hs_properties_path = os.path.join(HS_DATA_DIR, 'HYSETS_watershed_properties.txt')
hs_df = pd.read_csv(hs_properties_path, sep=';')
hs_df.set_index('Official_ID', inplace=True)

def add_coords(stn, col):
    return hs_df.loc[stn, col]   

def reformulate_jdf_model_result(model, df, b):
    last_col = 2**b - 1
    jdf = df[f'jsd_{model}'].str.split(', ', expand=True)
    # print(jdf[[0, last_col]].head())
    jdf[0] = jdf[0].apply(lambda r: r.split('[')[1])
    jdf[last_col] = jdf[last_col].apply(lambda r: r.split(']')[0])
    
    for c in jdf.columns:
        jdf[c] = jdf[c].astype(float)
    jdf.columns = [f'{c:03d}' for c in jdf.columns]
    
    low_flow_cols = list(jdf.columns)[:2]
    high_flow_cols = list(jdf.columns)[-2:]
    mid_range_cols = [e for e in list(jdf.columns) if e not in low_flow_cols + high_flow_cols]
    for quantile_focus in [f'low_flow_jsd_{model}', f'mid_range_jsd_{model}', f'high_flow_jsd_{model}']:
        if quantile_focus.startswith('low'):
            df[quantile_focus] = jdf.loc[:, low_flow_cols].sum(1)
        elif quantile_focus.startswith('mid'):
            df[quantile_focus] = jdf.loc[:, mid_range_cols].sum(1)
        else:
            df[quantile_focus] = jdf.loc[:, high_flow_cols].sum(1)
    return df, jdf


def reformulate_dkl_model_result(model, metric, df, b):
    last_col = 2**b - 1
    col = f'{model}_{metric}'
        
    ddf = df[f'{col}_disagg'].str.split(', ', expand=True)
    ddf[0] = ddf[0].apply(lambda r: r.split('[')[1])
    ddf[last_col] = ddf[last_col].apply(lambda r: r.split(']')[0])
    
    for c in ddf.columns:
        ddf[c] = ddf[c].astype(float)
    ddf.columns = [f'{c:03d}' for c in ddf.columns]
    
    low_flow_cols = list(ddf.columns)[:2]
    high_flow_cols = list(ddf.columns)[-2:]
    mid_range_cols = [e for e in list(ddf.columns) if e not in low_flow_cols + high_flow_cols]
    for quantile_focus in [f'low_flow_{metric}_{model}', f'mid_range_{metric}_{model}', f'high_flow_{metric}_{model}']:
        if quantile_focus.startswith('low'):
            df[quantile_focus] = ddf.loc[:, low_flow_cols].sum(1)
        elif quantile_focus.startswith('mid'):
            df[quantile_focus] = ddf.loc[:, mid_range_cols].sum(1)
        else:
            df[quantile_focus] = ddf.loc[:, high_flow_cols].sum(1)
    return df, ddf


def generate_layer_list(nl, nn, input_size, dr):
    l1 = layers.Dense(nn, activation='relu',
                      input_shape=(input_size,), 
                      kernel_initializer='he_uniform')
    layer_list = [l1]
    layer_no = 1
    for i in range(nl):
        ln = layers.Dense(nn, activation='relu', 
                          kernel_initializer='he_uniform')
        if dr > 0:
            layer_list.append(layers.Dropout(dr))
        layer_list.append(ln)
    print(f'    {len(layer_list)-1} hidden layers, {nn} nodes/layer.')
    # the last (output later) should have a single node
    layer_list.append(layers.Dense(1))
    return layer_list


def load_data(b):
    # fname = f'compression_test_results_{b}bits_20240212.csv'
    fname = f'DKL_test_results_{b}bits_20240212.csv'
    fpath = os.path.join(BASE_DIR, 'processed_data', 'compression_test_results', fname)
    return pd.read_csv(fpath, low_memory=False)

bitrates = [4, 6, 8]
sample_sizes = [0.01, 0.1, 0.5, 0.7, 0.95, 0.75]
number_of_layers = [1, 2, 5, 10, 20, 50]
neurons_per_layer = [1, 2, 8, 32, 64, 128, 256, 512, 1024]
batch_sizes = [32, 64, 128, 16]
dropout_rates = [0, 0.5]
losses = ['mean_absolute_error']#, 'mean_squared_error'], 
divergence_metrics = ['dkl'] # also 'tdv'
bin_model = 'uniform' # also equiprobable

attributes = [
    'Centroid_Lat_deg_N', 'Centroid_Lon_deg_E',
    'Drainage_Area_km2',
    'Elevation_m', 'Slope_deg',
    'Aspect_deg',
    'Gravelius', 'Perimeter',
    'Land_Use_Forest_frac', 'Land_Use_Grass_frac', 'Land_Use_Wetland_frac',
    'Land_Use_Snow_Ice_frac', 'Land_Use_Urban_frac', 'Land_Use_Shrubs_frac',
    'Land_Use_Crops_frac', 'Land_Use_Water_frac',
    'Permeability_logk_m2', 'Porosity_frac',
    'tmax', 'tmin',
    'prcp',
    'srad', 'swe', 'vp',
    'high_prcp_freq',
    'high_prcp_duration',
    'low_prcp_freq',
    'low_prcp_duration', 
]

features = []
for c in attributes:
    features.append(f'proxy_{c}')
    features.append(f'target_{c}')

# add the distance feature
features.append('centroid_distance')

results_path = os.path.join(BASE_DIR, 'processed_data', 'nn_results', 'NN_test_results.csv')

results_df = pd.DataFrame()
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)

n_experiments = len(results_df)
for b in bitrates:
    df = load_data(b)
    
    if len(divergence_metrics) > 1:
        raise Exception('Update data loading for multiple divergence metrics.')
    
    df, ddf = reformulate_dkl_model_result(bin_model, divergence_metrics[0], df, b)
    

    for m in divergence_metrics:
        target_column = f'{bin_model}_{m}'
        for l in losses:
            print('loss: ', l)
            loss_label = 'mse'
            if l == 'mean_absolute_error':
                loss_label = 'mae'
            for bs in batch_sizes:
                for s in sample_sizes:
                    input_size = int(s * len(df))
                    for dr in dropout_rates:
                        for nl in number_of_layers:
                            for nn in neurons_per_layer:
                                if (nn >= 256) & (nl >= 20):
                                    continue

                                # check if the model has already been run
                                model_id = f'test_nn_{b}_{bs}_{m}_{loss_label}_{s}_{dr}_{nl}_{nn}'
                                if model_id in results_df['model_id'].values:
                                    print(f'    {model_id} already run.')
                                    continue

                                layer_list = generate_layer_list(nl, nn, len(features), dr)
                                
                                model_arch_dict = {
                                    'id': model_id,
                                    'layers': layer_list,
                                    'optimizer': 'adam',
                                    'loss': l,
                                    'metrics': [l],
                                    'batch_size': bs,
                                    'training_frac': s,
                                }  
                        
                                nn_model = NNModel(df, features, target_column, b, model_arch_dict, 
                                                dr, s, bin_model, m, epochs=50)
                                
                                # update the results dataframe with all the parameters
                                results_df.loc[n_experiments, 'bitrate'] = b
                                results_df.loc[n_experiments, 'batch_size'] = bs
                                results_df.loc[n_experiments, 'sample_size'] = input_size
                                results_df.loc[n_experiments, 'dropout_rate'] = dr
                                results_df.loc[n_experiments, 'number_of_layers'] = nl
                                results_df.loc[n_experiments, 'neurons_per_layer'] = nn
                                results_df.loc[n_experiments, 'loss'] = loss_label
                                results_df.loc[n_experiments, 'divergence_metric'] = m
                                results_df.loc[n_experiments, 'training_frac'] = s
                                results_df.loc[n_experiments, 'model_id'] = model_arch_dict['id']
                                results_df.loc[n_experiments, 'test_loss'] = nn_model.test_loss

                                print(f' {b}bits {loss_label} ')
                                
                                # set the index value to n_experiments
                                if n_experiments % 10 == 0:                                  
                                    results_df.to_csv(results_path, index=False)

                                n_experiments += 1
                            






