import os
from time import time
import pandas as pd
import numpy as np

import optuna
import matplotlib.pyplot as plt

import xgboost as xgb
# from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler#, QuantileTransformer

#  The id encodes the model architecture variables:
#  prefix  bitrate  batch size  divergence metric  training length (frac)  dropout_rate  # layers  # nodes/layer
#    nn      b         bs           m                      s                     dr           nl           nn

# Run a series of experiments to test the following effects:
# Experiment 1: the number of layers and the number of nodes per layer, dropout rate, and batch size
# Experiment 2: geographic sample bias in training data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HS_DATA_DIR = os.path.join(BASE_DIR, "HYSETS_data")
hs_properties_path = os.path.join(HS_DATA_DIR, "HYSETS_watershed_properties.txt")
hs_df = pd.read_csv(hs_properties_path, sep=";")
hs_df.set_index("Official_ID", inplace=True)

revision_date = "20240409"

def add_coords(stn, col):
    return hs_df.loc[stn, col]


def load_attributes(df, attributes, fname, fpath):
    # Ensure all required attributes are in the DataFrame
    required_columns = [f'proxy_{a}' for a in attributes] + [f'target_{a}' for a in attributes]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if not missing_columns:
        return df  # All attributes already loaded
    
    # Add missing columns initialized with NaN
    for col in missing_columns:
        df[col] = pd.NA

    # Function to adjust IDs (add leading zero if missing)
    def adjust_id(id_val):
        # Check if ID exists in hs_df, if not, prepend zero and check again
        if id_val not in hs_df.index:
            adjusted_id = f"0{id_val}"
            if adjusted_id in hs_df.index:
                return adjusted_id
        return id_val

    # Apply the adjust_id function to both 'proxy' and 'target'
    df['proxy'] = df['proxy'].apply(adjust_id)
    df['target'] = df['target'].apply(adjust_id)

    # Fetch attributes for valid proxies and targets
    for a in attributes:
        if a in hs_df.columns:
            df[f'proxy_{a}'] = df['proxy'].map(hs_df[a])
            df[f'target_{a}'] = df['target'].map(hs_df[a])

        # Else, you might want to handle or log the missing attribute case

    df.to_csv(os.path.join(fname, fpath))      
    return df


def load_data(b):
    # fname = f'compression_test_results_{b}bits_20240212.csv'
    fname = f"DKL_results_{b}bits_{revision_date}.csv"
    fpath = os.path.join(BASE_DIR, "processed_data", "dkl_test_results", fname)
    return pd.read_csv(fpath, low_memory=False)


class ObjectiveFunction:
    def __init__(
        self, df, cv_param, bitrate, feature_columns, loss, 
        target_column="dkl_post_0R", n_simulations=10,
    ):
        self.input_df = df
        # self.cv_param = cv_param
        self.bitrate = bitrate
        self.loss = loss
        self.target_column = target_column
        self.feature_columns = feature_columns
        # self.filter_feature_cols()
        # self.create_feature_diff_cols()
        self.stations = pd.unique(df[["proxy", "target"]].values.ravel("K"))
        self.K = cv_param
        self.max_dist = 1000  # km
        self.n_simulations = n_simulations
        self.feature_scores = []
        print('    study initialized.')

    def filter_feature_cols(self):
        to_keep = [
            'proxy_prcp', 'target_prcp',
            'proxy_Slope_deg', 'target_Slope_deg', 
            'target_low_prcp_freq', 'proxy_low_prcp_freq',
            'target_low_prcp_duration', 'proxy_low_prcp_duration',
            'proxy_Land_Use_Snow_Ice_frac', 'target_Land_Use_Snow_Ice_frac',
            'proxy_Porosity_frac', 'target_Porosity_frac',
            'proxy_Elevation_m', 'target_Elevation_m',
            'proxy_Land_Use_Forest_frac', 'target_Land_Use_Forest_frac',
            'proxy_srad', 'target_srad',
            'target_Drainage_Area_km2', 'proxy_Drainage_Area_km2',
            'target_Centroid_Lon_deg_E', 'proxy_Centroid_Lon_deg_E',
            ]
        # to_keep = ['proxy_prcp', 'target_prcp']
        # self.feature_columns = to_keep
        self.feature_columns = [c for c in self.feature_columns if 'Centroid' not in c]
        
    def create_feature_diff_cols(self):
        features = list(set(['_'.join(e.split('_')[1:]) for e in self.feature_columns]))
        features = [f for f in features if f != 'distance']
                        
        for c in features:
            self.input_df[f"{c}_diff"] = self.input_df[f"proxy_{c}"] - self.input_df[f"target_{c}"]
     
    def leave_out_at_random(self):
        """
        Leave out K stations at random from the training data
        """
        excluded_stations = np.random.choice(self.stations, self.K, replace=False)
        training_df = self.input_df[
            ~self.input_df["proxy"].isin(excluded_stations) & ~self.input_df["target"].isin(excluded_stations)
        ].copy()
        test_df = self.input_df[
            self.input_df["proxy"].isin(excluded_stations) & self.input_df["target"].isin(excluded_stations)
        ].copy()
        return training_df, test_df
    
    def leave_out_by_geography(self):
        """
        Pick a point (station) at random and leave out all stations within a certain distance
        """
        central_station = np.random.choice(self.stations, 1)
        central_coords = hs_df.loc[central_station, ["Centroid_Lat_deg_N", "Centroid_Lon_deg_E"]].values
        # compute the distance from the central station

        # find the 

    def filter_training_data(self):
        train_df, test_df = self.leave_out_at_random()
        filtered_train_df = train_df[
            train_df["centroid_distance"] <= self.max_dist
        ].copy()
        filtered_test_df = test_df[test_df["centroid_distance"] <= self.max_dist].copy()
        return filtered_train_df, filtered_test_df

    def prepare_input_data(self):
        # keep just the first rows corresponding to the training size
        # dataframe = self.input_df.loc[:self.training_size, :].copy()
        # Split the data into features and target
        X_train_filtered, X_test_filtered = self.filter_training_data()

        X_train = X_train_filtered[self.feature_columns].values
        X_test = X_test_filtered[self.feature_columns].values
        
        Y_train = X_train_filtered[self.target_column].values
        Y_test = X_test_filtered[self.target_column].values

        # Standardize the features
        # self.scaler = QuantileTransformer(output_distribution='normal')
        # X_scaled = scaler.fit_transform(X)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        empty_data = any([len(X_train_scaled) == 0, len(Y_train) == 0, len(X_test_scaled) == 0, len(Y_test) == 0])
        if empty_data:
            raise Exception("Empty data.")

        # data = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        # label = pd.DataFrame(y_train, columns=[self.target_column])
        # dtrain = xgb.DMatrix(data, label=label)
        return X_train_scaled, Y_train, X_test_scaled, Y_test, X_test_filtered
    
    def save_feature_importance_plot(self):

        # Sort the feature importances
        feature_importances = self.model.get_booster().get_score(importance_type='weight')

        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

        # Unpack the labels and values
        labels, values = zip(*sorted_importances)
        
        # get the corresponding feature names
        self.feature_labels = [self.feature_columns[int(e.split('f')[1])] for e in labels]
        self.feaure_score_dict = {k: v for k, v in zip(self.feature_labels, values)}
        self.feature_scores.append(self.feaure_score_dict)
        fs_df = pd.DataFrame(self.feature_scores).T
        

        # return fs_df
        # Create the plot
        # fig, ax = plt.subplots()
        # ax.barh(self.feature_labels, values)
        # ax.set_xlabel('Importance')
        # ax.set_title('Feature Importance')

        # Save the plot as a PNG file
        # fig_dir = os.path.join(BASE_DIR, 'processed_data', 'feature_importance_figs')
        # plt.gcf().set_size_inches(8, 12)
        # plt.savefig(os.path.join(fig_dir,
        #             f'{self.model_id}.png'),
                    # bbox_inches='tight')

        # Close the plot to avoid displaying it
        # plt.close()
        return fs_df
        
    def __call__(self, trial):
        
        # Define hyperparameter space
        max_depth = trial.suggest_int("max_depth", 1, 8)
        n_estimators = trial.suggest_int("n_estimators", 1, 100)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.4)
        # reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1000.0)  # L1 reg
        # reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 2.0, log=True)  # L2 reg
        # subsample = trial.suggest_float("subsamples", 0.5, 0.8)
        colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        model_params = {
            "objective": f"reg:{self.loss}",
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "colsample_bytree": colsample,
            # "reg_alpha": reg_alpha
            # "lambda": 0.0,
            # "alpha": reg_alpha,
            "random_state": 42,
            "tree_method": "hist",
        }
        self.common_params = model_params
        prior = self.target_column.split("_")[-1].split('R')[0]
        # formulate a unique model id using the model parameters
        model_id = f"xgb_{self.bitrate}_b_{prior}_p"
        for k, v in self.common_params.items():
            if k in ["loss", "max_features", "tree_method", "random_state"]:
                continue
            if v == "reg:absoluteerror":
                v = "mae"
                self.loss_code = 'mae'
            elif v == "reg:squarederror":
                v = "mse"
                self.loss_code = 'mse'
            if isinstance(v, float):
                v = f"{v:1.3e}"
            model_id += f"_{k}_{v}"

        self.model_id = model_id
        mean_error = self.run_model()
        return mean_error

    def run_model(self):
        simulation_test_errors = []        
        running_mean_errors = []
        for n in range(self.n_simulations):
            t0 = time()
            X_train, Y_train, X_test, Y_test, X_test_original = self.prepare_input_data()
            t1 = time()
            # print(f"    Prepared input data in {t1 - t0:.3f} seconds")
            n_train, n_test = len(X_train), len(X_test)

            self.model = xgb.XGBRegressor(
                **self.common_params,
            )
            t2 = time()

            model_save_fpath = os.path.join(BASE_DIR, "processed_data", "xgb_models", f"{self.model_id}.json")
            # train the model
            self.model.fit(X_train, Y_train)
            t3 = time()
            self.model.save_model(model_save_fpath)
            # get the model fit mae
            # training_error = root_mean_squared_error(Y_train, self.model.predict(X_train))
            # evaluate the model
            predictions = self.model.predict(X_test)
            t4 = time()
            res_df = pd.DataFrame({"actual": Y_test, "predicted": predictions.flatten(),
                                   'proxy': X_test_original['proxy'].values, 'target': X_test_original['target'].values})

            # expected_mse = root_mean_squared_error(Y_test, predictions)
            expected_error = mean_absolute_error(Y_test, predictions)
            simulation_fname = f"{n}_{self.model_id}_{self.loss_code}_{expected_error:1.2f}_{n_train}_ntrain_{n_test}_ntest.csv"
            res_df.to_csv(os.path.join("processed_data", "xval_results", simulation_fname))
            simulation_test_errors.append(expected_error)

            fs_df = self.save_feature_importance_plot()
            t5 = time()
            if n % 25 == 0:
                running_mean_error = np.mean(simulation_test_errors)
                change_in_running_mean = running_mean_error
                if len(running_mean_errors) > 0:
                    change_in_running_mean = running_mean_errors[-1] - running_mean_error
                running_mean_errors.append(running_mean_error)
                
                print(f'    Completed simulation {n} in {t5 - t0:.2f} seconds')
                print(f'    ...trained model {n} in {t3 - t2:.3f} seconds')
                print(f'    ...computed metrics {n} in {t5 - t4:.3f} seconds')
                print(f'    ...running mean error: {running_mean_error:.2f}, change in mean error: {change_in_running_mean:.2f}')
                print('')
        fs_df.to_csv(f'processed_data/feature_importance/{self.model_id}_fscores.csv')
        mean_sim_err = np.mean(simulation_test_errors)
        std_sim_err = np.std(simulation_test_errors)
        print(f"    Mean Loss ({self.loss}): {mean_sim_err}, std loss: {std_sim_err}")

        feature_score_df = pd.DataFrame(self.feature_scores)
        feature_fpath = f'processed_data/feature_importance/{self.model_id}_fscores.csv'
        feature_score_df.to_csv(feature_fpath)
        return np.mean(simulation_test_errors)

attributes = [
    # "Centroid_Lat_deg_N",
    # "Centroid_Lon_deg_E",
    "Drainage_Area_km2",
    "Elevation_m",
    "Slope_deg",
    "Aspect_deg",
    # "Gravelius",
    # "Perimeter",
    "Land_Use_Forest_frac",
    "Land_Use_Grass_frac",
    "Land_Use_Wetland_frac",
    "Land_Use_Snow_Ice_frac",
    "Land_Use_Urban_frac",
    "Land_Use_Shrubs_frac",
    "Land_Use_Crops_frac",
    "Land_Use_Water_frac",
    "Permeability_logk_m2",
    "Porosity_frac",
    "tmax",
    "tmin",
    "prcp",
    "srad",
    "swe",
    "vp",
    "high_prcp_freq",
    "high_prcp_duration",
    "low_prcp_freq",
    "low_prcp_duration",
]

features = []
for c in attributes:
    features.append(f"proxy_{c}")
    features.append(f"target_{c}")

# add the distance feature
features.append("centroid_distance")

for b in [8]:
    loss = "squarederror"
    loss = 'absoluteerror'
    fname = f"DKL_results_{b}bits_{revision_date}.csv"
    fpath = os.path.join(BASE_DIR, "processed_data", "dkl_test_results", fname)
    df = pd.read_csv(fpath, low_memory=False, dtype={"proxy": str, "target": str},
                     engine='c')

    # add attribute differences
    for attr in attributes:
        feat = f"{attr}_diff"
        df[feat] = df[f"proxy_{attr}"] - df[f"target_{attr}"]
        features.append(feat) 

    t0 = time()
    df = load_attributes(df, attributes, fname, fpath)
    t1 = time()
    print(f'    Loaded attributes in {t1 - t0} seconds')
    cv_param = 40

    priors = [0, 1, 1e-1, 10, 1e-2, 100, 1e-3, 1000]
    for alpha in priors:
        print(f'   Testing prior: {alpha}')
        tc = f'dkl_post_{alpha}R'
        objective = ObjectiveFunction(
            df, cv_param, b, features, loss, target_column=tc,
            n_simulations=100,
        )
        study = optuna.create_study(
            direction="minimize",
            study_name=f"XGBoost Hist DKL ({b} bits) MSE",
            storage="sqlite:///study.db",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=15)

best = study.best_trial

print(f"  Value: {best.value}")
print("  Params: ")
for key, value in best.params.items():
    print(f"    {key}: {value}")
