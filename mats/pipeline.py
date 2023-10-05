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

    def get_combined_datasets(self):
        # get for location A, B and C and concatinate them
        df_a = self.get_data("A")
        df_b = self.get_data("B")
        df_c = self.get_data("C")

        # randmize the order of the data
        df = pd.concat([df_a, df_b, df_c]).sample(
            frac=1).reset_index().drop(columns=["index"])
        return df

    def get_data(self, location: str):
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
        X_train_group = self.unzip_date_feature(X_train_group)
        # rename the date_forecast column to time to merge with the target data
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)

        # clean the traget data
        Y_train_x = self.remove_consecutive_measurments(
            Y_train_x, consecutive_threshold=24)

        # Merge the targets and features and remove bad targets
        merged = pd.merge(X_train_group, Y_train_x, on="time", how="inner")
        mask = merged["pv_measurement"].notna()
        merged = merged.loc[mask].reset_index().drop(columns=["index"])

        # add the location to the data
        merged["location"] = location
        return merged

    def get_irrelevant_features(self):
        return ["date_calc", "time", "consecutive_count", "location"]

    def get_combined_test_data(self):
        df_a = self.get_test_data("A")
        df_b = self.get_test_data("B")
        df_c = self.get_test_data("C")
        df = pd.concat([df_a, df_b, df_c]).reset_index().drop(
            columns=["index"])
        return df

    def get_test_data(self, location: str = None) -> pd.DataFrame:
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
        df["location"] = location
        return df

    def scale_after_test_example_csv(self, df: pd.DataFrame, location: str = None):
        test_df = test_df_example
        if location == "A" or location == "B" or location == "C":
            test_df = test_df.loc[test_df["location"] == location]

        test_df = test_df[["time"]]
        test_df["time"] = pd.to_datetime(test_df["time"])

        filter_on_time = df.merge(test_df, on="time", how="right")
        return filter_on_time

    def get_categorical_features(self, df: pd.DataFrame):
        categorical_columns = [c for c in df.columns if ":idx" in c]
        return categorical_columns

    def grouped_by_hour(self, df: pd.DataFrame):
        categorical_columns = self.get_categorical_features(df)
        numeric_columns = list(
            set(df.columns) - set(categorical_columns) - {"date_forecast"})
        X_train_group = df.groupby(pd.Grouper(key="date_forecast", freq="1H", )).agg(
            {**{col: 'mean' for col in numeric_columns}, **{col: lambda x: x.mode().iloc[0] if not x.empty else np.nan for col in categorical_columns}}).reset_index()
        return X_train_group

    def unzip_date_feature(self, df: pd.DataFrame):
        df["date_forecast"] = pd.to_datetime(df["date_forecast"])
        df["day_of_year"] = df["date_forecast"].dt.day_of_year
        df["hour"] = df["date_forecast"].dt.hour
        df["month"] = df["date_forecast"].dt.month
        df["year"] = df["date_forecast"].dt.year
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
        mae = mean_absolute_error(best_submission["prediction"], df["prediction"])
        return mae
        


# pipin = Pipin()
# pipin.compare_mae(pd.DataFrame({"prediction": [1,2,3,4,5]}))
# big_data = pipin.get_combined_datasets()
# print("df", big_data.head())
# pipin = Pipin()
# test = pipin.get_combined_test_data()
# pipin.get_data("B")
