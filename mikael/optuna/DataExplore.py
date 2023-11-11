# Data handling
from category_encoders import OrdinalEncoder
import pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from feature_engine.timeseries.forecasting import LagFeatures

# Data science
from sklearn.preprocessing import MinMaxScaler


class Data:
    target: pd.Series = None
    train: pd.DataFrame = None
    frame: pd.DataFrame = None
    frame_without_tuning_data: pd.DataFrame = None
    tune_data: pd.DataFrame = None
    test: pd.DataFrame = None
    location: str = None

    X_scaler: MinMaxScaler = MinMaxScaler()
    Y_scaler: MinMaxScaler = MinMaxScaler()
    frame_scaler: MinMaxScaler = MinMaxScaler()

    def __init__(self, location: str):
        self.location = location

        self.target = pd.read_parquet(f"../../{location}/train_targets.parquet")
        self.target.rename(columns={"time": "date_forecast"}, inplace=True)
        self.test = pd.read_parquet(f"../../{location}/X_test_estimated.parquet")
        self.test["observed_or_estimated"] = 1
        observed = pd.read_parquet(f"../../{location}/X_train_observed.parquet")
        observed["observed_or_estimated"] = 0
        estimated = pd.read_parquet(f"../../{location}/X_train_estimated.parquet")
        estimated["observed_or_estimated"] = 1
        self.train = pd.concat(
            [
                observed,
                estimated,
            ],
            ignore_index=True,
        ).reset_index(drop=True)
        
        self.frame = self.train.copy().groupby(pd.Grouper(key="date_forecast", freq="1H")).mean()
        self.frame = self.frame.copy().merge(self.target.copy(), how="inner", on="date_forecast")
        self.frame = self.remove_consecutive_measurments(self.frame.copy())
        
    def set_dtypes(self):
        categorical_colummns = [c for c in self.frame.columns if "idx" in c]
        self.frame[categorical_colummns] = self.frame[categorical_colummns].astype("category")
        self.test[categorical_colummns] = self.test[categorical_colummns].astype("category")
        self.frame["date_forecast"] = pd.to_datetime(self.frame["date_forecast"])
        self.test["date_forecast"] = pd.to_datetime(self.test["date_forecast"])
        self.frame["date_calc"] = pd.to_datetime(self.frame["date_calc"])
        self.test["date_calc"] = pd.to_datetime(self.test["date_calc"])

    # @mats
    def remove_consecutive_measurments(self, df: pd.DataFrame, consecutive_threshold=25, consecutive_threshold_for_zero=24):
        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

        df['consecutive_count'] = df.groupby(
            mask).transform('count')[column_to_check]

        mask = (df['consecutive_count'] > consecutive_threshold)
        mask_zero = (df['consecutive_count'] > consecutive_threshold_for_zero) & (
            df[column_to_check] == 0)
        df.drop(columns=["consecutive_count"], inplace=True)

        df = df.loc[~mask]
        df = df.loc[~mask_zero]
        return df.reset_index(drop=True)

    def fit_predictions_to_submission_length(self, predictions: np.ndarray):
        submission_skeleton = pd.read_csv("../../test.csv")
        submission_skeleton = submission_skeleton.loc[submission_skeleton["location"] == self.location]
        submission_skeleton.drop(columns=["id", "location", "prediction"], inplace=True)
        submission_skeleton.rename(columns={"time": "date_forecast"}, inplace=True)
        submission_skeleton["date_forecast"] = pd.to_datetime(submission_skeleton["date_forecast"])

        test = self.test.copy()
        test["predictions"] = predictions

        test_shortened = pd.merge(submission_skeleton, test, on="date_forecast", how="left")
        return test_shortened["predictions"]
    
    def reduce_test_to_submission_length(self):
        submission_skeleton = pd.read_csv("../../test.csv")
        submission_skeleton = submission_skeleton.loc[submission_skeleton["location"] == self.location]
        submission_skeleton.drop(columns=["id", "location", "prediction"], inplace=True)
        submission_skeleton.rename(columns={"time": "date_forecast"}, inplace=True)
        submission_skeleton["date_forecast"] = pd.to_datetime(submission_skeleton["date_forecast"])

        test = self.test.copy()
        test["date_forecast"] = pd.to_datetime(test["date_forecast"])

        self.test = pd.merge(submission_skeleton, test, on="date_forecast", how="left")
        
    # @mats
    def unzip_date_feature(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df[date_column] = pd.to_datetime(df[date_column])
        df["date_calc"] = pd.to_datetime(df["date_calc"])
        df["day_of_year"] = df[date_column].dt.day_of_year
        df['time_of_day'] = df[date_column].dt.hour + df['date_forecast'].dt.minute / 60
        df["hour"] = df[date_column].dt.hour
        df["month"] = df[date_column].dt.month
        # Utfør sinus- og cosinus-transformasjoner
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)

        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        df["minute_calc_diff"] = (df[date_column] - df["date_calc"]).dt.seconds / 60
        df["minute_calc_diff"].fillna(0, inplace=True)
        df.drop(columns=[date_column, "date_calc", "hour", "month", "day_of_year", "time_of_day"], inplace=True)
        return df
    
    def catboost(self):
        X = self.unzip_date_feature(self.frame.copy())
        X = X.loc[X["pv_measurement"].notna()]
        test = self.unzip_date_feature(self.test.copy())
        y = X["pv_measurement"]
        
        # drop features
        drop = ['wind_speed_u_10m:ms',
                'wind_speed_v_10m:ms',
                'wind_speed_w_1000hPa:ms']
        X.drop(columns=drop)
        test.drop(columns=drop)
        
        # drop where nans
        X.dropna(axis=1, inplace=True)
        test[X.columns.drop(["pv_measurement"])].dropna(axis=1, inplace=True)
        
        # Categorical data
        cat_features = [c for c in X.columns if ":idx" in c]
        enc = OrdinalEncoder()
        X[cat_features] = enc.fit_transform(test[cat_features])
        test[cat_features] = enc.transform(test[cat_features])
        
        no_cat_features = [c for c in X.columns if ":idx" not in c]
        # Lag features
        lag_cols = X[no_cat_features].select_dtypes(include=["number", "float", "int"]).columns.to_list()
        lag_cols.remove("pv_measurement")
        no_cat_features = [c for c in lag_cols if ":idx" not in c]
        lag_f = LagFeatures(variables=no_cat_features, periods=4)
        X_tr = lag_f.fit_transform(X[no_cat_features].select_dtypes(include=["number", "float", "int"]))
        test_tr = lag_f.fit_transform(test[no_cat_features].select_dtypes(include=["number", "float", "int"]))
        X[X_tr.columns] = X_tr
        test[test_tr.columns] = test_tr
        
        X = X.loc[X["pv_measurement"].notna()]
        X.drop(columns=["pv_measurement"], inplace=True)
        
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            kf = KFold(n_splits=10, shuffle=True, random_state=1)

            params = {
                "iterations": 1000,
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "subsample": trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            }
            # Initialize a list to store the mean absolute errors for each fold
            mae_list = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

                print(f"Training Fold {fold}...")
                model = CatBoostRegressor(**params, silent=True)
                model.fit(X_fold_train, y_fold_train)
                predictions = model.predict(X_fold_val)
                mae_fold = mean_absolute_error(y_fold_val, predictions)
                mae_list.append(mae_fold)
                print(f"Fold {fold} MAE: {mae_fold}")

            # Calculate the mean of the mean absolute errors for all folds
            mae = sum(mae_list) / len(mae_list)
            print(f"Mean MAE for all Folds: {mae}")
            return mae

        # Create an Optuna study
        study = optuna.create_study(direction="minimize")

        # Optimize the objective function
        print("Optimizing hyperparameters...")
        study.optimize(objective, n_trials=40)
        print("Optimization complete!")
        
        return CatBoostRegressor(**study.best_params).fit(X, y).predict(test)
