# Data handling
import pickle
import json
import pandas as pd

# Helper functions
from functions import get_days_sinse_beginning_of_year, get_seconds_of_day

# Types handling
from typing import Optional
import numpy as np

# Data science
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Feature engineering
from feature_engine.timeseries.forecasting import LagFeatures

# Machine learning tool
import xgboost as xgb

# Optimization / feature engineering tools
import optuna

# plotting
import matplotlib.pyplot as plt


class MasterDataframes:
    df_a: pd.DataFrame = None
    df_b: pd.DataFrame = None
    df_c: pd.DataFrame = None

    X_scaler: MinMaxScaler = None
    Y_scaler: MinMaxScaler = None

    # def __init__(self):
    #     self.df_a = self.prep_dataset_x(X_train_observed_a, Y_train_observed_a)
    #     self.df_b = self.prep_dataset_x(X_train_observed_b, Y_train_observed_b)
    #     self.df_c = self.prep_dataset_x(X_train_observed_c, Y_train_observed_c)

    def prep_dataset_x_y(self, location: str) -> pd.DataFrame:
        if location == "A":
            X_train_x = X_train_observed_a
            Y_train_x = Y_train_observed_a
        if location == "B":
            X_train_x = X_train_observed_b
            Y_train_x = Y_train_observed_b
        if location == "C":
            X_train_x = X_train_observed_c
            Y_train_x = Y_train_observed_c

        X_train_group = X_train_x.groupby(pd.Grouper(key="date_forecast", freq="1H")).mean().reset_index()
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)

        inner_merge = pd.merge(X_train_group, Y_train_x, on="time", how="inner")
        id_columns = [c for c in inner_merge.columns if ":idx" in c]

        filled = self._fill_df(inner_merge, id_columns)
        df_new_features = self._add_features_full_df(filled)
        cleaned_df = self._clean_df(df_new_features)

        X = cleaned_df[[c for c in cleaned_df.columns if "pv_measure" not in c]].astype("float")
        X = self._add_lag_features(X)

        Y = cleaned_df["pv_measurement"].fillna(0).astype("float")

        # X_scaled, Y_scaled = self._scale_and_create_scaler(X, Y)
        # return X_scaled, Y_scaled
        return X, Y

    def _scale_and_create_scaler(self, X: pd.DataFrame, Y: pd.Series):
        self.Y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(Y.values.reshape(1, -1))
        self.X_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)

        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y.values.reshape(1, -1))

        df_X = pd.DataFrame(X_scaled, columns=X.columns)
        s_Y = pd.Series(Y_scaled.ravel(), name=Y.name).reset_index()

        return df_X, s_Y

    def _add_features_full_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).year)
        df["month"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).month)
        df["seconds_in_day"] = df["time"].apply(lambda datestring: get_seconds_of_day(datestring))
        df["since_jan_1"] = df["time"].apply(lambda datestring: get_days_sinse_beginning_of_year(datestring))

        df["effective_energy"] = df["clear_sky_rad:W"] * (df["total_cloud_cover:p"] / 100)

        return df

    def _add_lag_features(self, X: pd.DataFrame) -> pd.DataFrame:
        no_nan_columns_df = X[[c for c in X.columns if len(X[X[c].isna()].index) == 0]]
        no_nan_columns_df = no_nan_columns_df.select_dtypes(include=["number", "float", "int"])
        no_nan_columns = no_nan_columns_df[~no_nan_columns_df.isna()].columns.to_list()

        lag_f = LagFeatures(variables=no_nan_columns, periods=1)

        X_tr = lag_f.fit_transform(no_nan_columns_df)
        X[X_tr.columns] = X_tr

        return X

    def _fill_df(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        inner_to_split = df.copy()
        non_id_columns = [c for c in inner_to_split.columns if ":idx" not in c]

        inner_to_split[categorical_columns].fillna(0, inplace=True)
        inner_to_split[non_id_columns].fillna(method="ffill", inplace=True)

        return inner_to_split

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["time"], inplace=True)
        df = df.astype("float")

        # non_categorical = [c for c in df.columns if ":idx" not in c]

        # lower_bound = df[non_categorical].quantile(0.05)
        # upper_bound = df[non_categorical].quantile(0.95)

        # df[non_categorical] = df[(df[non_categorical] >= lower_bound) & (df[non_categorical] <= upper_bound)]

        return df

    def _shorten_dataset_to_prediction_scale(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        test_df["time_merge"] = pd.to_datetime(test_df["time"])
        df["time_merge"] = pd.to_datetime(df["time"])
        test_mapped = test_df.loc[test_df["location"] == location]

        train_mapped = df.merge(
            test_mapped[["time", "time_merge"]], how="left", on="time_merge", suffixes=("_train", "_test")
        )

        df_c = train_mapped[train_mapped["time_test"].notna()]

        df_c = df_c.drop(columns=[c for c in df_c.columns if "tim" in c or "date" in c])
        df_c = df_c.astype("float")

        return df_c

    def prep_test(self, location) -> pd.DataFrame:
        if location == "A":
            df = X_test_estimated_a
        if location == "B":
            df = X_test_estimated_b
        if location == "C":
            df = X_test_estimated_c

        df.rename(columns={"date_forecast": "time"}, inplace=True)

        categorical = [c for c in df.columns if ":idx" in c]
        filled = self._fill_df(df, categorical)

        df_new_features = self._add_features_full_df(filled)
        df_with_lag = self._add_lag_features(df_new_features)
        df_shortened = self._shorten_dataset_to_prediction_scale(df_with_lag, location)

        print(f"Location {location}. length: {str(len(df_with_lag))}")

        return df_shortened


# Estimate
X_train_estimated_a = pd.read_parquet("../A/X_train_estimated.parquet")
X_train_estimated_b = pd.read_parquet("../B/X_train_estimated.parquet")
X_train_estimated_c = pd.read_parquet("../C/X_train_estimated.parquet")

# Test estimates
X_test_estimated_a = pd.read_parquet("../A/X_test_estimated.parquet")
X_test_estimated_b = pd.read_parquet("../B/X_test_estimated.parquet")
X_test_estimated_c = pd.read_parquet("../C/X_test_estimated.parquet")

# Full test estimate
X_test_A = pd.read_parquet("../A/X_test_A.parquet")
X_test_B = pd.read_parquet("../B/X_test_B.parquet")
X_test_C = pd.read_parquet("../C/X_test_C.parquet")

test_df = pd.read_csv("../test.csv")

# Observations
X_train_observed_a = pd.read_parquet("../A/X_train_observed.parquet")
X_train_observed_b = pd.read_parquet("../B/X_train_observed.parquet")
X_train_observed_c = pd.read_parquet("../C/X_train_observed.parquet")

# Targets
Y_train_observed_a = pd.read_parquet("../A/train_targets.parquet")
Y_train_observed_b = pd.read_parquet("../B/train_targets.parquet")
Y_train_observed_c = pd.read_parquet("../C/train_targets.parquet")


class MLModel:
    model: Optional[xgb.XGBRegressor] = None
    selection_model: Optional[xgb.XGBRegressor] = None
    selection: Optional[SelectFromModel] = None
    M_df: Optional[MasterDataframes] = None
    y_pred = None
    y_test = None
    X_train = None
    X_test = None
    params = None

    def __init__(
        self,
        model=None,
        selection_model=None,
        M_df=None,
        y_pred=None,
        y_test=None,
        X_train=None,
        X_test=None,
        params=None,
        selection=None,
    ):
        self.model = model
        self.selection_model = selection_model
        self.M_df = M_df
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.params = params
        self.selection = selection

    def plot_important_features(self, top: int):
        feature_important = self.model.get_booster().get_score(importance_type="weight")
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.nlargest(top, columns="score").plot(kind="barh", figsize=(20, 10))

    def plot_pred_vs_test(self):
        plt.scatter(self.y_pred, self.y_test)
        plt.title("Time Series Plot")
        plt.xlabel("Time (absolute)")
        plt.ylabel("Prediction")
        plt.xticks(rotation=45)

        plt.show()

    def predict_test_data(self, location: str):
        df = self.M_df.prep_test(location)

        if self.selection is not None:
            df_c = self.selection.transform(df)
            y_pred = self.selection_model.predict(df_c)
        else:
            y_pred = self.model.predict(df)

        # print("Shape of y_pred before inverse transform:", y_pred.shape)
        # print("Sample of y_pred before inverse transform:", y_pred[:5])

        # reverted_y_pred = self.M_df.Y_scaler.inverse_transform(y_pred[:, 1].reshape(-1, 1))

        # # Debugging statements
        # print("Shape of reverted_y_pred:", reverted_y_pred.shape)
        # print("Sample of reverted_y_pred:", reverted_y_pred[:5])

        # return reverted_y_pred
        return y_pred

    def _get_missing_indexes(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        df_c = df.copy()
        test_df["time_merge"] = pd.to_datetime(test_df["time"])
        df_c["time_merge"] = pd.to_datetime(df["time"])
        test_mapped = test_df.loc[test_df["location"] == location]
        test_mapped = df_c.merge(test_mapped[["time", "time_merge"]], how="left", on="time_merge")

        return test_mapped


class ModelTrainer:
    modelA = None
    modelB = None
    modelC = None

    M_df = MasterDataframes()

    def train_model(self, location: str, trials: int):
        X, Y = self.M_df.prep_dataset_x_y("C")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.20)

        objective_list_reg = ["reg:squarederror"]
        tree_method = ["approx", "hist"]
        metric_list = ["mae"]

        def objective(trial):
            param = {
                "objective": trial.suggest_categorical("objective", objective_list_reg),
                "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
                "tree_method": trial.suggest_categorical("tree_method", tree_method),
                "max_depth": trial.suggest_int("max_depth", 3, 10),  # Adjust the range
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),  # Increase the range
                "gamma": trial.suggest_float("gamma", 0.1, 1.0),  # Increase the lower bound
                "subsample": trial.suggest_discrete_uniform("subsample", 0.6, 1.0, 0.05),  # Reduce the range
                "colsample_bytree": trial.suggest_discrete_uniform(
                    "colsample_bytree", 0.6, 1.0, 0.05
                ),  # Reduce the range
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
                "random_state": trial.suggest_int("random_state", 1, 1000),
            }
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return mean_absolute_error(y_test, y_pred)

        study = optuna.create_study(direction="minimize", study_name="regression")
        study.optimize(objective, n_trials=trials, n_jobs=6)

        model = xgb.XGBRegressor(**study.best_params)
        model.fit(X_train, y_train)

        selection = SelectFromModel(model, threshold=0.004002, prefit=True)
        select_X_train = selection.transform(X_train)

        selection_model = xgb.XGBRegressor(**study.best_params)
        selection_model.fit(select_X_train, y_train)

        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)

        M_df_c = self.M_df
        switched_model = MLModel(
            model=model,
            selection_model=selection_model,
            M_df=M_df_c,
            y_pred=y_pred,
            y_test=y_test,
            X_test=X_test,
            X_train=X_train,
            params=study.best_params,
            selection=selection,
        )
        if location == "A":
            self.modelA = switched_model
        if location == "B":
            self.modelB = switched_model
        if location == "C":
            self.modelC = switched_model

        MAE = mean_absolute_error(y_test, y_pred)

        filename = f"models/{location}_xgb_MAE_{str(int(MAE))}.pkl"
        model_params = f"models/{location}_xgb_MAE_{str(int(MAE))}_best_params.json"
        with open(model_params, "w") as f:
            f.write(json.dumps(study.best_params, indent=4))
        f.close()
        pickle.dump(model, open(filename, "wb"))

        print("R2: ", r2_score(y_test, y_pred))
        print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
        print("graded! MAE: ", MAE)
        print("Best params: " + json.dumps(study.best_params, indent=4))

        return switched_model
