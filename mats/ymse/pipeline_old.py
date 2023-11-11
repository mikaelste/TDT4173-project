import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

from sklearn.metrics import mean_absolute_error

import os

current_dir = os.getcwd()
print("Current working directory:", current_dir)


PATH = "/Users/matsalexander/Desktop/Forest Gump/"
# Estimate
X_train_estimated_a: pd.DataFrame = pd.read_parquet(
    PATH + 'A/X_train_estimated.parquet')
X_train_estimated_b: pd.DataFrame = pd.read_parquet(
    PATH + "B/X_train_estimated.parquet")
X_train_estimated_c: pd.DataFrame = pd.read_parquet(
    PATH + "C/X_train_estimated.parquet")

# Test estimates
X_test_estimated_a: pd.DataFrame = pd.read_parquet(
    PATH + "A/X_test_estimated.parquet")
X_test_estimated_b: pd.DataFrame = pd.read_parquet(
    PATH + "B/X_test_estimated.parquet")
X_test_estimated_c: pd.DataFrame = pd.read_parquet(
    PATH + "C/X_test_estimated.parquet")

# Observations
X_train_observed_a: pd.DataFrame = pd.read_parquet(
    PATH + "A/X_train_observed.parquet")
X_train_observed_b: pd.DataFrame = pd.read_parquet(
    PATH + "B/X_train_observed.parquet")
X_train_observed_c: pd.DataFrame = pd.read_parquet(
    PATH + "C/X_train_observed.parquet")

# Targets
Y_train_observed_a: pd.DataFrame = pd.read_parquet(
    PATH + "A/train_targets.parquet")
Y_train_observed_b: pd.DataFrame = pd.read_parquet(
    PATH + "B/train_targets.parquet")
Y_train_observed_c: pd.DataFrame = pd.read_parquet(
    PATH + "C/train_targets.parquet")

test_df_example = pd.read_csv(PATH + "test.csv")

best_submission: pd.DataFrame = pd.read_csv(
    PATH + "mikael/submissions/fourth_submission.csv")

optins = {
    "randomize": False,
    "consecutive_threshold": 6,
    "normalize": False,
    "group_by_hour": True,
    "unzip_date_feature": True,
}

# make a options class with the options as attributes


class Options:
    randomize = False
    consecutive_threshold = 6
    normalize = False
    group_by_hour = True
    unzip_date_feature = True

    def __init__(self, randomize=False, consecutive_threshold=6, normalize=False, group_by_hour=True, unzip_date_feature=True) -> None:
        self.randomize = randomize
        self.consecutive_threshold = consecutive_threshold
        self.normalize = normalize
        self.group_by_hour = group_by_hour
        self.unzip_date_feature = unzip_date_feature


class Pipin:

    def __init__(self):
        pass

    def get_data(self, location: str, randomize=False, consecutive_threshold=6, normalize=False, group_by_hour=True, unzip_date_feature=True):
        x, y = self.get_training_data_by_location(location)
        return self.handle_data(x, Y_train_x=y, randomize=randomize, consecutive_threshold=consecutive_threshold, normalize=normalize, group_by_hour=group_by_hour, unzip_date_feature=unzip_date_feature)

    def get_test_data(self, location: str,  normalize=False) -> pd.DataFrame:
        if location == "A":
            df = X_test_estimated_a
        elif location == "B":
            df = X_test_estimated_b
        elif location == "C":
            df = X_test_estimated_c
        else:
            raise Exception("location must be A, B or C")
        return self.handle_data(df.copy(), normalize=normalize)

    def handle_data(self, X_train, Y_train_x=pd.DataFrame(), randomize=False, consecutive_threshold=3, normalize=False, group_by_hour=True, unzip_date_feature=True):

        # ––––––––––––––––––––– for test data and train data –––––––––––––––––––––
        X_train["date_forecast"] = pd.to_datetime(
            X_train["date_forecast"])
        # add a column with the date_calc of the forecast
        X_train = self.onehot_estimated(X_train)
        # adjust feature data
        if group_by_hour:
            X_train = self.grouped_by_hour(X_train)
        # rename the date_forecast column to time to merge with the target data
        if unzip_date_feature:
            X_train = self.unzip_date_feature(X_train)

        # distance from date_calc to date_forcast forecast in seconds
        X_train["date_calc"] = pd.to_datetime(X_train["date_calc"])
        X_train["calculated_ago"] = (
            X_train["date_forcast"] - X_train["date_calc"]).dt.total_seconds()
        X_train["calculated_ago"] = X_train["time_difference_seconds"].fillna(
            0)

        X_train.rename(columns={"date_forecast": "time"}, inplace=True)

        if normalize:
            X_train = self.normalize(X_train)

        # ––––––––––––––––––––– only for train data –––––––––––––––––––––
        if not Y_train_x.empty:
            Y_train_x = self.remove_consecutive_measurments(
                Y_train_x, consecutive_threshold=consecutive_threshold, consecutive_threshold_for_zero=30)

        # Merge the targets and features and remove bad targets
        if not Y_train_x.empty:
            merged = pd.merge(X_train, Y_train_x, on="time", how="inner")
            mask = merged["pv_measurement"].notna()
            merged = merged.loc[mask].reset_index(drop=True)
        else:
            merged = X_train

        if randomize:
            merged = merged.sample(frac=1).reset_index(drop=True)

        return merged

    def get_combined_datasets(self,  randomize=False, data_sets: set = {"A", "B", "C"}, consecutive_threshold=6, normalize=False, group_by_hour=True, offset_years=False, unzip_date_feature=False):
        # get for location A, B and C and concatinate them
        if not data_sets.issubset({"A", "B", "C"}):
            raise Exception("set must contain A, B or C")

        df_a = self.get_data(
            "A", consecutive_threshold=consecutive_threshold, normalize=normalize, group_by_hour=group_by_hour, unzip_date_feature=unzip_date_feature) if "A" in data_sets else pd.DataFrame()
        df_b = self.get_data(
            "B", consecutive_threshold=consecutive_threshold, normalize=normalize, group_by_hour=group_by_hour, unzip_date_feature=unzip_date_feature) if "B" in data_sets else pd.DataFrame()
        df_c = self.get_data(
            "C", consecutive_threshold=consecutive_threshold, normalize=normalize, group_by_hour=group_by_hour, unzip_date_feature=unzip_date_feature) if "C" in data_sets else pd.DataFrame()

        dataSets = [df_a, df_b, df_c]
        location = ["A", "B", "C"]
        dataSets = self.add_location_to_datasets(
            dataSets, location, offset_years=offset_years)

        df = pd.concat([df_a, df_b, df_c]).reset_index(drop=True)
        if randomize:
            df = df.sample(frac=1).reset_index(drop=True)
        # place the target column at the end
        df = df[[c for c in df if c not in ['pv_measurement']] +
                ['pv_measurement']]
        return df

    def get_combined_test_data(self,  data_sets: set = {"A", "B", "C"}):
        if not data_sets.issubset({"A", "B", "C"}):
            raise Exception("set must contain A, B or C")

        df_a = self.get_test_data(
            "A") if "A" in data_sets else pd.DataFrame()
        df_b = self.get_test_data(
            "B") if "B" in data_sets else pd.DataFrame()
        df_c = self.get_test_data(
            "C") if "C" in data_sets else pd.DataFrame()

        dataSets = [df_a, df_b, df_c]
        locations = ["A", "B", "C"]
        dataSets = self.add_location_to_datasets(dataSets, locations)

        df = pd.concat([df_a, df_b, df_c])

        df.reset_index().drop(
            columns=["index"])

        return df

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––– helper funciton ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def get_training_data_by_location(self, location):
        if location == "A":
            X_train_observed_x = X_train_observed_a
            X_train_estimated_x = X_train_estimated_a
            Y_train_x = Y_train_observed_a
        elif location == "B":
            X_train_observed_x = X_train_observed_b
            X_train_estimated_x = X_train_estimated_b
            Y_train_x = Y_train_observed_b
        elif location == "C":
            X_train_observed_x = X_train_observed_c
            X_train_estimated_x = X_train_estimated_c
            Y_train_x = Y_train_observed_c
        else:
            raise Exception("location must be A, B or C")
        train = pd.concat(
            [X_train_observed_x, X_train_estimated_x]).reset_index(drop=True)
        return train, Y_train_x

    def add_location_to_datasets(self, dfs: list, locations: list,  offset_years=False):
        if not set(locations).issubset({"A", "B", "C"}) or len(dfs) != len(locations):
            raise Exception("set must contain A, B or C")
        for location, dataset in zip(locations, dfs):
            if dataset.empty:
                continue
            dataset = self.onehot_location(dataset, location)
            if offset_years:
                offset = 5*(ord(location)-ord("A"))
                dataset["time"] = dataset["time"] + pd.DateOffset(years=offset)
        return dfs

    def get_categorical_features(self, df: pd.DataFrame, feature_selection=False):
        categorical_columns = [c for c in df.columns if ":idx" in c]
        if feature_selection:
            categorical_columns = list(
                set(categorical_columns) & set(important_features))
        return categorical_columns

    def get_irrelevant_features(self, df=None, feature_selection=False):
        irrelevant = ["date_calc", "consecutive_count"]
        if df is None:
            return irrelevant
        if feature_selection:
            irrelevant = list(
                set(irrelevant) & set(important_features))
        return list(set(df.columns) & set(irrelevant))

    def get_numeric_features(self, df: pd.DataFrame, feature_selection=False):
        categorical_features = self.get_categorical_features(df)
        ignore_features = self.get_irrelevant_features(
            df, feature_selection=feature_selection)

        numerical_features = list(set(
            df.columns) - set(categorical_features) - set(ignore_features) - set(['pv_measurement']))
        if feature_selection:
            numerical_features = list(
                set(numerical_features) & set(important_features))
        return numerical_features

    def grouped_by_hour(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df = df.groupby(pd.Grouper(key=date_column, freq="1H")
                        ).mean(numeric_only=True)
        all_nan_mask = df.isnull().all(axis=1)
        df = df[~all_nan_mask]
        return df.reset_index()
        # return df.groupby(pd.Grouper(key=date_column, freq="1H")
        #                   ).mean().reset_index()

    def grouped_by_hour_old(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        def custom_agg_categorical(x):
            if x.empty:
                return np.nan
            else:
                mode_result = x.mode()
                if not mode_result.empty:
                    return mode_result.iloc[0]
                else:
                    return np.nan

        categorical_columns = self.get_categorical_features(df)
        numeric_columns = list(
            set(df.columns) - set(categorical_columns) - {date_column})
        X_train_group = df.groupby(pd.Grouper(key=date_column, freq="1H")).agg(
            {**{col: 'mean' for col in numeric_columns}, **{col: custom_agg_categorical for col in categorical_columns}}).reset_index()
        return X_train_group

    def unzip_date_feature(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df[date_column] = pd.to_datetime(df[date_column])
        df["day_of_year"] = df["date_forecast"].dt.day_of_year
        df["hour"] = df["date_forecast"].dt.hour
        df["month"] = df["date_forecast"].dt.month
        return df

    def remove_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3):
        df = df[np.abs(df[column]-df[column].mean())
                <= (threshold*df[column].std())]
        return df

    def normalize(self, df: pd.DataFrame):
        numerical_features = self.get_numeric_features(df)
        df[numerical_features] = df[numerical_features].apply(
            pd.to_numeric, errors='coerce')
        scaler = MinMaxScaler()
        df[numerical_features] = scaler.fit_transform(
            df[numerical_features])
        return df

    def remove_consecutive_measurments(self, df: pd.DataFrame, consecutive_threshold=6, consecutive_threshold_for_zero=12):
        if consecutive_threshold < 2:
            return df

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

    def get_features_selection(self, df: pd.DataFrame):
        features = list(df.columns)
        selected_features = list(set(features) & set(important_features))
        return selected_features + ["pv_measurement"]

    def scale_targets_in_dataset(self, df: pd.DataFrame, location: str):
        locations_to_scale = list(
            set(["A", "B", "C"]) - set(location))
        df = df.copy()
        avg_targt_y = df.loc[df[location] == 1, "pv_measurement"].mean()
        for loca in locations_to_scale:
            # for all rows with location loca, scale the target with the average target
            avg_targt_x = df.loc[df[loca]
                                 == 1, "pv_measurement"].mean()
            scale = avg_targt_y/avg_targt_x
            df.loc[df[loca] == 1,
                   "pv_measurement"] = df.loc[df[loca] == 1, "pv_measurement"]*scale
        return df

    def onehot_estimated(self, df):
        df["estimated"] = 0  # Initialize both columns to 0
        df["observed"] = 0
        estimated_mask = df["date_calc"].notna()
        df.loc[estimated_mask, "estimated"] = 1
        df.loc[~estimated_mask, "observed"] = 1
        return df

    def onehot_location(self, df, location):
        if location == "A":
            df["A"], df["B"], df["C"] = 1, 0, 0
        elif location == "B":
            df["A"], df["B"], df["C"] = 0, 1, 0
        elif location == "C":
            df["A"], df["B"], df["C"] = 0, 0, 1
        return df

    def split_train_tune(self, df: pd.DataFrame):
        df = df.copy()
        df_estimated = df.loc[df["estimated"] == 1]
        df_observed = df.loc[df["estimated"] == 0]

        num_rows = len(df_estimated)
        middle_index = num_rows // 2

        df_estimated.sample(frac=1, random_state=42)
        train_estimated = df.iloc[:middle_index]
        tune = df.iloc[middle_index:]

        train = pd.concat([df_observed, train_estimated])
        return train, tune

    def compare_mae(self, df: pd.DataFrame):
        best_submission: pd.DataFrame = pd.read_csv(
            PATH+"mats/submissions/big_gluon_best.csv")
        best_submission = best_submission[["prediction"]]

        if best_submission.shape != df.shape:
            print("best_submission", best_submission.shape)
            print("df", df.shape)
            raise Exception("Dataframe shape must be the same")

        return mean_absolute_error(
            best_submission["prediction"], df["prediction"])

    def post_processing(self, df: pd.DataFrame, prediction_column: str = "prediction_label"):
        df = df[[prediction_column]].rename(
            columns={prediction_column: "prediction"}).reset_index(drop=True).rename_axis(index="id")

        df["prediction"] = df["prediction"].clip(lower=0)
        return df


# pipin = Pipin()
# x = pipin.get_combined_datasets(data_sets={"A"})

# get all date_calc.rows that are nan


# pipin.compare_mae(pd.DataFrame({"prediction": [1,2,3,4,5]}))

# print("df", big_data.head())
# pipin = Pipin()
# test = pipin.get_combined_test_data()
# pipin.get_data("B")

important_features = [
    'time',
    'direct_rad:W',
    'diffuse_rad:W',
    'sun_azimuth:d',
    'sun_elevation:d',
    'clear_sky_energy_1h:J',
    'clear_sky_rad:W',
    'total_cloud_cover:p',
    'effective_cloud_cover:p',
    'rain_water:kgm2',
    'precip_5min:mm',
    'wind_speed_10m:ms',
    'wind_speed_w_1000hPa:ms',
    'super_cooled_liquid_water:kgm2',
    'air_density_2m:kgm3',
    'pressure_100m:hPa',
    'pressure_50m:hPa',
    'sfc_pressure:hPa',
    'msl_pressure:hPa',
    'dew_point_2m:K',
    'is_day:idx',
    'is_in_shadow:idx',
    'elevation:m',

    "snow_melt_10min:mm",
    "snow_density:kgm3",
    "fresh_snow_6h:cm",
    "fresh_snow_1h:cm",
    "snow_water:kgm2",
    "fresh_snow_12h:cm",
    "fresh_snow_3h:cm",
    "fresh_snow_24h:cm",
    "snow_depth:cm",

    'A',
    'B',
    'C',
    "estimated",
    "observed",
]
