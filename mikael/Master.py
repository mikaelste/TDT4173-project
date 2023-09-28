# Data handling
import pickle
import json
from typing import Optional, Union
import pandas as pd

# Helper functions
from functions import get_days_sinse_beginning_of_year, get_seconds_of_day

# Types handling
import numpy as np
from fractions import Fraction

# Data science
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Machine learning tool
import xgboost as xgb

# Optimization / feature engineering tools
import optuna

# plotting
import matplotlib.pyplot as plt


class MLModel:
    model: Optional[xgb.XGBRegressor] = None
    y_pred = None
    y_test = None
    X_train = None
    X_test = None
    params = None

    def __init__(self, model=None, y_pred=None, y_test=None, X_train=None, X_test=None, params=None):
        self.model = model
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.params = params

    def plot_important_features(self, top: int):
        feature_important = self.model.get_booster().get_score(importance_type="weight")
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.nlargest(top, columns="score").plot(kind="barh", figsize=(20, 10))

    def plot_pred_vs_test(self):
        df_y = pd.DataFrame({"Y pred": self.y_pred, "Y test": self.y_test})

        plt.plot(np.arange(len(df_y)), df_y, linestyle="dotted")

        # Add title and axis labels
        plt.title("Time Series Plot")
        plt.xlabel("Time (absolute)")
        plt.ylabel("Prediction")
        plt.xticks(rotation=45)

        # Display the plot
        plt.show()

    def predict_test_data(self, location: str):
        MasterDF = MasterDataframes()
        df = MasterDF.prep_test(location)

        y_pred = self.model.predict(df)
        return y_pred


class ModelTrainer:
    modelA = None
    modelB = None
    modelC = None

    def train_model(self, X, Y, location: str, trials: int):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.20)

        def objective(trial):
            param = {
                "objective": "reg:linear",
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.01, 1.0),
                "subsample": trial.suggest_float("subsample", 0.01, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0),
                "random_state": trial.suggest_int("random_state", 1, 1000),
            }
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return mean_absolute_error(y_test, y_pred)

        study = optuna.create_study(direction="minimize", study_name="regression")
        study.optimize(objective, n_trials=trials)

        model = xgb.XGBRegressor(**study.best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        switched_model = MLModel(
            model=model, y_pred=y_pred, y_test=y_test, X_test=X_test, X_train=X_train, params=study.best_params
        )
        if location == "A":
            self.modelA = switched_model
        if location == "B":
            self.modelB = switched_model
        if location == "C":
            self.modelC = switched_model

        MAE = mean_absolute_error(y_test, y_pred)

        filename = f"models/{location}_xgb_MAE_{str(int(MAE))}_.pkl"
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


class MasterDataframes:
    df_a = None
    df_b = None
    df_c = None

    def __init__(self):
        self.df_a = self.prep_dataset_x(X_train_observed_a, Y_train_observed_a)
        self.df_b = self.prep_dataset_x(X_train_observed_b, Y_train_observed_b)
        self.df_c = self.prep_dataset_x(X_train_observed_c, Y_train_observed_c)

    def split_df_X_Y(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        X = df[[c for c in df.columns if c != "pv_measurement"]]
        Y = df["pv_measurement"].fillna(0)
        return X, Y

    def prep_dataset_x(self, X_train_x: pd.DataFrame, Y_train_x: Optional[pd.DataFrame]) -> pd.DataFrame:
        X_train_group = X_train_x.groupby(pd.Grouper(key="date_forecast", freq="1H")).mean().reset_index()
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)

        inner_merge = pd.merge(X_train_group, Y_train_x, on="time", how="inner")
        id_columns = [c for c in inner_merge.columns if ":idx" in c]

        df_encoded_categorical = self._target_encode_categorical(inner_merge, id_columns)

        df_new_features = self._add_features(df_encoded_categorical)

        return self._clean_df(df_new_features, id_columns)

    def _target_encode_categorical(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        target = ["pv_measurement"]
        target_encode_df = df[categorical_columns + target].reset_index().drop(columns="index", axis=1)
        target_name = target[0]
        target_df = pd.DataFrame()
        for embed_col in categorical_columns:
            val_map = target_encode_df.groupby(embed_col)[target].mean().to_dict()[target_name]
            target_df[embed_col] = target_encode_df[embed_col].map(val_map).values

        score_target_drop = df.drop(categorical_columns, axis=1).reset_index().drop(columns="index", axis=1)
        score_target = pd.concat([score_target_drop, target_df], axis=1)

        return score_target

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).year)
        df["month"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).month)
        df["seconds_in_day"] = df["time"].apply(lambda datestring: get_seconds_of_day(datestring))
        df["days_sinse_jan_1"] = df["time"].apply(lambda datestring: get_days_sinse_beginning_of_year(datestring))
        return df

    def _clean_df(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        inner_to_split = df.copy().drop(columns=[c for c in df.columns if "tim" in c or "date" in c])
        non_id_columns = [c for c in inner_to_split.columns if ":idx" not in c]

        inner_to_split[categorical_columns].fillna(0, inplace=True)
        inner_to_split[non_id_columns].fillna(inner_to_split.mean(), inplace=True)

        inner_to_split = inner_to_split.astype("float")

        return inner_to_split

    def prep_test(self, location) -> pd.DataFrame:
        if location == "A":
            df = X_test_estimated_a
        if location == "B":
            df = X_test_estimated_b
        if location == "C":
            df = X_test_estimated_c

        X_train_group = df.groupby(pd.Grouper(key="date_forecast", freq="1H")).mean().reset_index()
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)

        categorical = [c for c in X_train_group.columns if ":idx" in c]

        return self._clean_df(self._add_features(X_train_group), categorical)


# Estimate
X_train_estimated_a = pd.read_parquet("../A/X_train_estimated.parquet")
X_train_estimated_b = pd.read_parquet("../B/X_train_estimated.parquet")
X_train_estimated_c = pd.read_parquet("../C/X_train_estimated.parquet")

# Test estimates
X_test_estimated_a = pd.read_parquet("../A/X_test_estimated.parquet")
X_test_estimated_b = pd.read_parquet("../B/X_test_estimated.parquet")
X_test_estimated_c = pd.read_parquet("../C/X_test_estimated.parquet")

# Observations
X_train_observed_a = pd.read_parquet("../A/X_train_observed.parquet")
X_train_observed_b = pd.read_parquet("../B/X_train_observed.parquet")
X_train_observed_c = pd.read_parquet("../C/X_train_observed.parquet")

# Targets
Y_train_observed_a = pd.read_parquet("../A/train_targets.parquet")
Y_train_observed_b = pd.read_parquet("../B/train_targets.parquet")
Y_train_observed_c = pd.read_parquet("../C/train_targets.parquet")
