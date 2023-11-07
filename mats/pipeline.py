import pandas as pd
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


class Pipeline:

    def __init__(self):
        pass

    def get_combined_data(self, test_data=False):
        locations = ["A", "B", "C"]
        dfs = []
        for index, location in enumerate(locations):
            if test_data:
                dfs.append(self.get_test_data(location))
            else:
                dfs.append(self.get_data(location))

            dfs[index] = self.onehot_location(dfs[index], location)
        df = pd.concat(dfs).reset_index(drop=True)

        if test_data:
            return df
        return df[[c for c in df if c not in ['pv_measurement']] +  # pv measurement is the target and is at the end columns
                  ['pv_measurement']]

    def get_data(self, location: str) -> pd.DataFrame:
        train, targets = self.get_training_data_by_location(location)
        return self.handle_data(train, targets)

    def get_test_data(self, location: str) -> pd.DataFrame:
        test_data = self.get_test_data_by_location(location)
        return self.handle_data(test_data)

    def handle_data(self, df, targets=pd.DataFrame()):
        df["date_calc"] = pd.to_datetime(df["date_calc"])
        df["date_forecast"] = pd.to_datetime(df["date_forecast"])

        # df = self.add_time_since_calucation(df)
        df = self.onehot_estimated(df)
        df = self.unzip_date_feature(df)
        df = self.grouped_by_hour(df)

        df["time"] = df["date_forecast"]
        df.drop(["date_forecast"], axis=1, inplace=True)

        # denne kjører bare når vi prossessere train data (med targets som parameter)
        if not targets.empty:
            # targets = self.remove_consecutive_measurments(targets, 3)
            df = self.merge_train_target(df, targets)

        df.drop(["time"], axis=1, inplace=True)

        # denne var ikke med i 148 3 modeller med gluon.
        df = self.absolute_values(df)
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

    def get_test_data_by_location(self, location: str,  normalize=False) -> pd.DataFrame:
        if location == "A":
            df = X_test_estimated_a
        elif location == "B":
            df = X_test_estimated_b
        elif location == "C":
            df = X_test_estimated_c
        else:
            raise Exception("location must be A, B or C")
        return df.copy()

    def unzip_date_feature(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df[date_column] = pd.to_datetime(df[date_column])
        df["day_of_year"] = df["date_forecast"].dt.day_of_year
        df["hour"] = df["date_forecast"].dt.hour
        # df["month"] = df["date_forecast"].dt.month
        return df

    def add_time_since_calucation(self, df):  # denne er ikke så dum.
        df["date_calc"] = pd.to_datetime(df["date_calc"])
        df["calculated_ago"] = (
            df["date_forecast"] - df["date_calc"]).dt.total_seconds()
        df["calculated_ago"] = df["calculated_ago"].fillna(
            0) / 60/30
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

    def grouped_by_hour(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df = df.groupby(pd.Grouper(key=date_column, freq="1H")
                        ).mean(numeric_only=True)
        all_nan_mask = df.isnull().all(axis=1)
        df = df[~all_nan_mask]
        return df.reset_index()

    def merge_train_target(self, x, y):
        # henning får med alle pv measurments selv om han merger på inner time. Fordi resample fyller nan rows for alle timer som ikke er i datasettet.
        merged = pd.merge(x, y, on="time", how="right")
        mask = merged["pv_measurement"].notna()
        merged = merged.loc[mask].reset_index(drop=True)
        return merged

    def absolute_values(self, df: pd.DataFrame):
        df[df.columns] = df[df.columns].abs()
        df = df.replace(-0.0, 0.0)
        return df

    def remove_consecutive_measurments(self, df: pd.DataFrame, consecutive_threshold=3, consecutive_threshold_zero=12, return_removed_rows=False):
        if consecutive_threshold < 2:
            return df

        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift(1)).cumsum()
        df['consecutive_count'] = df.groupby(
            mask).transform('count')[column_to_check]

        mask_non_zero = ((df['consecutive_count'] >= consecutive_threshold)
                         & (df["pv_measurement"] > 0))
        mask_zero = ((df['consecutive_count'] >= consecutive_threshold_zero)
                     & (df["pv_measurement"] == 0))

        mask = mask_non_zero | mask_zero

        removed_rows = df.copy().loc[mask]
        df = df.loc[~mask]

        if return_removed_rows:
            return df, removed_rows
        return df.reset_index(drop=True)

    def lag_features_by_1_hour(df, columns_to_lag):
        lag_columns = [c for c in df.columns if "_1h:" in c]
        df[lag_columns] = df[lag_columns].shift(1)
        return df

    def remove_consecutive_measurments_new(self, df: pd.DataFrame, consecutive_threshold=3, consecutive_threshold_zero=12, return_removed_rows=False):
        if consecutive_threshold < 2:
            return df
        column_to_check = 'pv_measurement'

        mask = (df[column_to_check] != df[column_to_check].shift(1)).cumsum()
        df['consecutive_group'] = df.groupby(
            mask).transform('count')[column_to_check]

        df["is_first_in_consecutive_group"] = False
        df['is_first_in_consecutive_group'] = df['consecutive_group'] != df['consecutive_group'].shift(
            1)

        # masks to remove rows
        mask_non_zero = (df['consecutive_group'] >= consecutive_threshold) & (
            df["pv_measurement"] > 0) & (df["is_first_in_consecutive_group"] == False)

        mask_zero = (df['consecutive_group'] >= consecutive_threshold_zero) & (
            df["pv_measurement"] == 0)
        mask = mask_non_zero | mask_zero

        removed_rows = df.loc[mask]
        df = df.loc[~mask]

        df.drop(["consecutive_group", "is_first_in_consecutive_group"],
                axis=1, inplace=True)

        if return_removed_rows:
            return df, removed_rows
        return df.reset_index(drop=True)

    def compare_mae(self, df: pd.DataFrame):
        best_submission: pd.DataFrame = pd.read_csv(
            PATH+"mats/submissions/best_gluon_3.csv")
        best_submission = best_submission[["prediction"]]

        if best_submission.shape != df.shape:
            print("best_submission", best_submission.shape)
            print("df", df.shape)
            raise Exception("Dataframe shape must be the same")

        return mean_absolute_error(
            best_submission["prediction"], df["prediction"])

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

    def post_processing(self, df: pd.DataFrame, prediction_column: str = "prediction_label"):
        df = df[[prediction_column]].rename(
            columns={prediction_column: "prediction"}).reset_index(drop=True).rename_axis(index="id")

        df["prediction"] = df["prediction"].clip(lower=0)
        return df


# pipe = Pipeline()
# df00 = pipe.get_combined_data()
# df00


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
