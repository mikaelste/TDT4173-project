import pandas as pd
import numpy as np

import sys

from sklearn.metrics import mean_absolute_error
sys.path.append(sys.path[0][0:-5])
PATH = sys.path[-1]
# Estimate
X_train_estimated_a: pd.DataFrame = pd.read_parquet(
    PATH+'/A/X_train_estimated.parquet')
X_train_estimated_b: pd.DataFrame = pd.read_parquet(
    PATH+"/B/X_train_estimated.parquet")
X_train_estimated_c: pd.DataFrame = pd.read_parquet(
    PATH+"/C/X_train_estimated.parquet")

# Test estimates
X_test_estimated_a: pd.DataFrame = pd.read_parquet(
    PATH+"/A/X_test_estimated.parquet")
X_test_estimated_b: pd.DataFrame = pd.read_parquet(
    PATH+"/B/X_test_estimated.parquet")
X_test_estimated_c: pd.DataFrame = pd.read_parquet(
    PATH+"/C/X_test_estimated.parquet")

# Observations
X_train_observed_a: pd.DataFrame = pd.read_parquet(
    PATH+"/A/X_train_observed.parquet")
X_train_observed_b: pd.DataFrame = pd.read_parquet(
    PATH+"/B/X_train_observed.parquet")
X_train_observed_c: pd.DataFrame = pd.read_parquet(
    PATH+"/C/X_train_observed.parquet")

# Targets
Y_train_observed_a: pd.DataFrame = pd.read_parquet(
    PATH+"/A/train_targets.parquet")
Y_train_observed_b: pd.DataFrame = pd.read_parquet(
    PATH+"/B/train_targets.parquet")
Y_train_observed_c: pd.DataFrame = pd.read_parquet(
    PATH+"/C/train_targets.parquet")

test_df_example = pd.read_csv(PATH+"/test.csv")

best_submission: pd.DataFrame = pd.read_csv(
    PATH+"/mikael/submissions/fourth_submission.csv")


class Pipin:

    def __init__(self):
        pass

    def get_combined_datasets(self, timeSeries=False, randomize=True, data_sets: set = {"A", "B", "C"}):
        # get for location A, B and C and concatinate them
        if not data_sets.issubset({"A", "B", "C"}):
            raise Exception("set must contain A, B or C")

        df_a = self.get_data(
            "A", timeSeries) if "A" in data_sets else pd.DataFrame()
        df_b = self.get_data(
            "B", timeSeries) if "B" in data_sets else pd.DataFrame()
        df_c = self.get_data(
            "C", timeSeries) if "C" in data_sets else pd.DataFrame()
        
        dataSets = [df_a, df_b, df_c]
        location = ["A", "B", "C"]
        for location, dataset in zip(location, dataSets):
            if dataset.empty:
                continue
            dataset["location:idx"] = location
                
        if timeSeries:
            # hypotese: The year of the data is not important, and the data can be used from all locations to make a better model
            df_b.index = df_b.index + pd.DateOffset(years=5)
            df_c.index = df_c.index + pd.DateOffset(years=10)
            all_dfs = pd.concat([df_a, df_b, df_c])
            all_dfs["time"] = all_dfs.index
            groupe_for_ts = self.grouped_by_hour_ts(all_dfs)
            return groupe_for_ts
        else:
            df = pd.concat([df_a, df_b, df_c]).reset_index(drop=True)
            if randomize:
                df = df.sample(frac=1).reset_index(drop=True)
            return df
        
    def add_location_to_datasets(self, dfs: list, locations: list):
        if not locations.issubset({"A", "B", "C"}) or  len(dfs) != len(locations):
            raise Exception("set must contain A, B or C")
        for location, dataset in zip(location, dfs):
            if dataset.empty:
                continue
            dataset["location:idx"] = location
        return dfs
        
            

    def get_data(self, location: str, timeSeries: bool = False):
        if location == "A":
            Y_train_x = Y_train_observed_a
            X_train_observed_x = X_train_observed_a
            X_train_estimated_x = X_train_estimated_a
        if location == "B":
            X_train_observed_x = X_train_observed_b
            Y_train_x = Y_train_observed_b
            X_train_estimated_x = X_train_estimated_b
        if location == "C":
            X_train_observed_x = X_train_observed_c
            Y_train_x = Y_train_observed_c
            X_train_estimated_x = X_train_estimated_c

        # concatinate the estimated and observed data

        X_train_total = pd.concat(
            [X_train_observed_x, X_train_estimated_x]).reset_index().drop(columns=["index"])
        X_train_total["date_forecast"] = pd.to_datetime(
            X_train_total["date_forecast"])

        # adjust feature data

        X_train_group = self.grouped_by_hour(X_train_total)
        if not timeSeries:
            X_train_group = self.unzip_date_feature(X_train_group)
        # rename the date_forecast column to time to merge with the target data
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)

        # clean the traget data
        # Y_train_x = self.remove_consecutive_measurments(
        #     Y_train_x, consecutive_threshold=24)

        # Merge the targets and features and remove bad targets
        merged = pd.merge(X_train_group, Y_train_x, on="time", how="inner")
        mask = merged["pv_measurement"].notna()
        merged = merged.loc[mask].reset_index().drop(columns=["index"])

        # add the location to the data
        # merged["location:idx"] = location # do this when getting combined data

        if timeSeries:
            # make sure all houres are present. Even empty ones
            merged = self.grouped_by_hour_ts(merged)
        return merged

    def get_irrelevant_features(self):
        return ["date_calc", "time", "consecutive_count"] #"location:idx" # this is relevant when getting combined data and not single location. 

    def get_combined_test_data(self,   time_series: bool = False, data_sets: set = {"A", "B", "C"}):
        if not data_sets.issubset({"A", "B", "C"}):
            raise Exception("set must contain A, B or C")
        
        df_a = self.get_test_data(
            "A", time_series) if "A" in data_sets else pd.DataFrame()
        df_b = self.get_test_data(
            "B", time_series) if "B" in data_sets else pd.DataFrame()
        df_c = self.get_test_data(
            "C", time_series) if "C" in data_sets else pd.DataFrame()
        
        dataSets = [df_a, df_b, df_c]
        location = ["A", "B", "C"]
        for location, dataset in zip(location, dataSets):
            if dataset.empty:
                continue
            dataset["location:idx"] = location

        df = pd.concat([df_a, df_b, df_c])

        if not time_series:
            df.reset_index().drop(
                columns=["index"])

        return df

    def get_test_data(self, location: str = None, time_series=False) -> pd.DataFrame:
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
        # df = self._shorten_dataset_to_prediction_scale(df, location)
        df = self.scale_after_test_example_csv(df, location)
        df.reset_index().drop(columns=["index"])
        # add the location to the data
        # df["location:idx"] = location

        if time_series:
            df = self.grouped_by_hour_ts(df)
            # df["time"] = df.index
            # df = self.scale_after_test_example_csv(df, location)
            # df.index = df["time"]
            # df.drop(columns=["time"], inplace=True)

        return df

    def scale_after_test_example_csv(self, df: pd.DataFrame, location: str = None):
        test_df = test_df_example
        if location == "A" or location == "B" or location == "C":
            # location is not marked at a idx in the test_df
            test_df = test_df.loc[test_df["location"] == location]

        test_df = test_df[["time"]]
        test_df["time"] = pd.to_datetime(test_df["time"])

        filter_on_time = df.merge(test_df, on="time", how="right")
        return filter_on_time

    def get_categorical_features(self, df: pd.DataFrame):
        categorical_columns = [c for c in df.columns if ":idx" in c]
        return categorical_columns

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

    def grouped_by_hour_ts(self, df: pd.DataFrame):
        # print("df", df.columns)
        categorical_columns = self.get_categorical_features(df)
        numeric_columns = list(
            set(df.columns) - set(categorical_columns) - {"time"})

        df["index"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)
        df.set_index('index', inplace=True)

        def custom_agg_categorical(x):
            if x.empty:
                return np.nan
            else:
                mode_result = x.mode()
                if not mode_result.empty:
                    return mode_result.iloc[0]
                else:
                    return np.nan

        numeric_df = df[numeric_columns]
        numeric_df.ffill(inplace=True)
        numeric_df = numeric_df.resample("H").mean()

        categorical_df = df[categorical_columns]
        categorical_df = categorical_df.resample(
            "H").apply(custom_agg_categorical)

        merged_df = pd.merge(categorical_df, numeric_df,
                             left_index=True, right_index=True)
        return merged_df

    def unzip_date_feature(self, df: pd.DataFrame, date_column: str = "date_forecast"):
        df["date_forecast"] = pd.to_datetime(df["date_forecast"])
        df["day_of_year"] = df["date_forecast"].dt.day_of_year
        df["hour"] = df["date_forecast"].dt.hour
        df["month"] = df["date_forecast"].dt.month
        return df

    def remove_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3):
        df = df[np.abs(df[column]-df[column].mean())
                <= (threshold*df[column].std())]
        return df

    def remove_consecutive_measurments(self, df: pd.DataFrame, consecutive_threshold: int = 48):
        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

        df['consecutive_count'] = df.groupby(
            mask).transform('count')[column_to_check]
        mask = (df['consecutive_count'] > consecutive_threshold)
        df.drop(columns=["consecutive_count"], inplace=True)
        return df.loc[~mask]

    def find_consecutive_measurements(self, df: pd.DataFrame):
        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift()).cumsum()

        df['consecutive_count'] = df.groupby(
            mask).transform('count')[column_to_check]
        start_date = '2020-07-12 15:00:00'
        end_date = '2020-08-26'
        too_hig = 48
        # mask = (df['time'] > start_date) & (df['time'] <= end_date)
        mask = (df['consecutive_count'] > too_hig) & (
            df["pv_measurement"].notna()) & (df["pv_measurement"] > 0)
        df_sub = df.loc[mask]
        df_sub['time'] = pd.to_datetime(df_sub['time'])
        print("sub", df_sub)

    def compare_mae(self, df: pd.DataFrame):
        best_submission: pd.DataFrame = pd.read_csv(
            PATH+"/mats/submissions/lightgbm.csv")
        best_submission = best_submission[["prediction"]]

        if best_submission.shape != df.shape:
            print("best_submission", best_submission.shape)
            print("df", df.shape)
            raise Exception("Dataframe shape must be the same")

        # calculate mae of the predicitons
        mae = mean_absolute_error(
            best_submission["prediction"], df["prediction"])
        return mae


# pipin = Pipin()
# x = pipin.get_combined_test_data(time_series=True)
# print(x)


# data0 = pipin.get_data("A", timeSeries=True)

# pipin.compare_mae(pd.DataFrame({"prediction": [1,2,3,4,5]}))

# print("df", big_data.head())
# pipin = Pipin()
# test = pipin.get_combined_test_data()
# pipin.get_data("B")
