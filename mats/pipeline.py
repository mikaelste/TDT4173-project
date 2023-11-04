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


class Pipin:

    def __init__(self):
        pass

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
        for location, dataset in zip(location, dataSets):
            if dataset.empty:
                continue
            dataset["location:idx"] = location
            if offset_years:
                offset = 5*(ord(location)-ord("A"))
                dataset["time"] = dataset["time"] + pd.DateOffset(years=offset)

        df = pd.concat([df_a, df_b, df_c]).reset_index(drop=True)
        if randomize:
            df = df.sample(frac=1).reset_index(drop=True)
        return df

    def get_data(self, location: str, randomize=False, consecutive_threshold=6, normalize=False, group_by_hour=True, unzip_date_feature=True):
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

        X_train = pd.concat(
            [X_train_observed_x, X_train_estimated_x]).reset_index().drop(columns=["index"])
        X_train["date_forecast"] = pd.to_datetime(
            X_train["date_forecast"])

        # add a column with the date_calc of the forecast
        X_train['is_estimated:idx'] = X_train['date_calc'].notna().astype(
            int)
        # adjust feature data
        if group_by_hour:
            X_train = self.grouped_by_hour(X_train)
        # rename the date_forecast column to time to merge with the target data
        if unzip_date_feature:
            X_train = self.unzip_date_feature(X_train)
        X_train.rename(columns={"date_forecast": "time"}, inplace=True)

        # normalize the data
        if normalize:
            X_train = self.normalize(X_train)

        # clean the traget data
        Y_train_x = self.remove_consecutive_measurments(
            Y_train_x, consecutive_threshold=consecutive_threshold, consecutive_threshold_for_zero=consecutive_threshold*2)

        # Merge the targets and features and remove bad targets
        merged = pd.merge(X_train, Y_train_x, on="time", how="inner")
        mask = merged["pv_measurement"].notna()
        merged = merged.loc[mask].reset_index(drop=True)

        if randomize:
            merged = merged.sample(frac=1).reset_index(drop=True)

        return merged

    def scale_targets_in_dataset(self, df: pd.DataFrame, location: str):
        locations_to_scale = list(
            set(df["location:idx"].unique()) - set(location))
        df = df.copy()
        avg_targt_y = df.loc[df["location:idx"] ==
                             location, "pv_measurement"].mean()
        for loca in locations_to_scale:
            # for all rows with location loca, scale the target with the average target
            avg_targt_x = df.loc[df["location:idx"]
                                 == loca, "pv_measurement"].mean()
            scale = avg_targt_y/avg_targt_x

            df.loc[df["location:idx"] == loca,
                   "pv_measurement"] = df.loc[df["location:idx"] == loca, "pv_measurement"]*scale
        return df

    def split_train_tune(self, df: pd.DataFrame):
        df = df.copy()
        df_estimated = df.loc[df["is_estimated:idx"] == 1]
        df_observed = df.loc[df["is_estimated:idx"] == 0]

        num_rows = len(df_estimated)
        middle_index = num_rows // 2

        df_estimated.sample(frac=1, random_state=42)
        train_estimated = df.iloc[:middle_index]
        tune = df.iloc[middle_index:]

        train = pd.concat([df_observed, train_estimated])
        return train, tune

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
        location = ["A", "B", "C"]
        for location, dataset in zip(location, dataSets):
            if dataset.empty:
                continue
            dataset["location:idx"] = location

        df = pd.concat([df_a, df_b, df_c])

        df.reset_index().drop(
            columns=["index"])

        return df

    def get_test_data(self, location: str = None, normalize=False) -> pd.DataFrame:
        if location == "A":
            df = X_test_estimated_a
        elif location == "B":
            df = X_test_estimated_b
        elif location == "C":
            df = X_test_estimated_c
        else:
            raise Exception("location must be A, B or C")

        df = self.grouped_by_hour(df)
        df = self.unzip_date_feature(df)
        df.rename(columns={"date_forecast": "time"}, inplace=True)
        df['is_estimated:idx'] = df['date_calc'].notna().astype(
            int)

        df = self.filter_for_dates_in_test_example(df, location)
        df.reset_index().drop(columns=["index"])

        if normalize:
            df = self.normalize(df)

        return df

    def add_location_to_datasets(self, dfs: list, locations: list):
        if not locations.issubset({"A", "B", "C"}) or len(dfs) != len(locations):
            raise Exception("set must contain A, B or C")
        for location, dataset in zip(location, dfs):
            if dataset.empty:
                continue
            dataset["location:idx"] = location
        return dfs

    def filter_for_dates_in_test_example(self, df: pd.DataFrame, location: str = None):
        test_df = test_df_example
        if location == "A" or location == "B" or location == "C":
            # location is not marked at a idx in the test_df
            test_df = test_df.loc[test_df["location"] == location]

        test_df = test_df[["time"]]
        test_df["time"] = pd.to_datetime(test_df["time"])

        filter_on_time = df.merge(test_df, on="time", how="right")
        return filter_on_time

    def get_categorical_features(self, df: pd.DataFrame, feature_selection=False):
        categorical_columns = [c for c in df.columns if ":idx" in c]
        if feature_selection:
            categorical_columns = list(
                set(categorical_columns) & set(important_features))
        return categorical_columns

    def get_irrelevant_features(self, df=None, feature_selection=False):
        irrelevant = ["date_calc", "time", "consecutive_count"]
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
        X_train_group = df.groupby(pd.Grouper(key=date_column, freq="1H", )).agg(
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

    def compare_mae(self, df: pd.DataFrame):
        best_submission: pd.DataFrame = pd.read_csv(
            PATH+"mats/submissions/lightgbm.csv")
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
    'snow_depth:cm',
    'snow_melt_10min:mm',
    'fresh_snow_3h:cm',
    'fresh_snow_1h:cm',
    'snow_water:kgm2',
    'super_cooled_liquid_water:kgm2',
    'snow_density:kgm3',
    'snow_drift:idx',
    'air_density_2m:kgm3',
    'pressure_100m:hPa',
    'pressure_50m:hPa',
    'sfc_pressure:hPa',
    'msl_pressure:hPa',
    'dew_point_2m:K',
    'is_day:idx',
    'is_in_shadow:idx',
    'elevation:m',
    'location:idx'
]
