# Data handling
import pandas as pd

# Helper functions
from functions import get_days_sinse_beginning_of_year, get_hours_of_day, remove_consecutive_measurments

# Types handling
import numpy as np

# Data science
from sklearn.preprocessing import MinMaxScaler

# Feature engineering
from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures
from feature_engine.timeseries.forecasting import LagFeatures

non_equal_value_columns = [
    "dew_or_rime:idx",
    "dew_point_2m:K",
    "elevation:m",
    "fresh_snow_12h:cm",
    "fresh_snow_1h:cm",
    "fresh_snow_24h:cm",
    "fresh_snow_3h:cm",
    "fresh_snow_6h:cm",
    "is_day:idx",  # A and B
    "is_in_shadow:idx",  # A and B
    "precip_5min:mm",  # C and B
    "precip_type_5min:idx",  # A and B
    "prob_rime:p",
    "snow_depth:cm",
    "snow_melt_10min:mm",
    "wind_speed_w_1000hPa:ms",  # A and B
]


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

    def prep_dataset(self, location: str, merge_dfs: bool = False):
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

        if merge_dfs:
            X_train_total_a = pd.concat([X_train_observed_a, X_train_estimated_a]).reset_index()
            X_train_total_b = pd.concat([X_train_observed_b, X_train_estimated_b]).reset_index()
            X_train_total_c = pd.concat([X_train_observed_c, X_train_estimated_c]).reset_index()

            if location != "A":
                X_train_total_a = X_train_total_a.drop(columns=non_equal_value_columns)
            if location != "B":
                X_train_total_b = X_train_total_b.drop(columns=non_equal_value_columns)
            if location != "C":
                X_train_total_c = X_train_total_c.drop(columns=non_equal_value_columns)
            X_train_total = pd.merge(
                pd.merge(
                    X_train_total_a,
                    X_train_total_b,
                    on="date_forecast",
                    how="inner",
                    suffixes=("_a", "_b"),
                ),
                X_train_total_c,
                on="date_forecast",
                how="inner",
                suffixes=("", "_c"),
            )
        else:
            X_train_total = pd.concat([X_train_observed_x, X_train_estimated_x]).reset_index()
        return X_train_total, Y_train_x

    def prep_dataset_x_y(self, location: str, drop_features=False, merge_dfs=False) -> pd.DataFrame:
        X_train_total, Y_train_total = self.prep_dataset(location, merge_dfs)

        # X_train_group = X_train_total.groupby(pd.Grouper(key="date_forecast", freq="1H")).mean().reset_index()
        X_train_group = X_train_total.groupby(pd.Grouper(key="date_forecast", freq="1H")).first().reset_index()
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)
        X_train_group.drop(columns=["date_calc"], inplace=True)
        X_train_group.drop(columns=[c for c in X_train_group.columns if "index" in c], inplace=True)

        inner_merge = pd.merge(X_train_group, Y_train_total, on="time", how="inner")
        id_columns = [c for c in inner_merge.columns if ":idx" in c]

        filled = self._fill_df(inner_merge, id_columns)
        df_new_features = self._add_features_full_df(filled)
        cleaned_df = self._clean_df(df_new_features)
        drop_consecutive_pv_zeros = remove_consecutive_measurments(cleaned_df)

        X = drop_consecutive_pv_zeros.drop(columns=["pv_measurement"]).astype("float")
        # X = self._add_lag_features(X, drop_features=drop_features)

        Y = drop_consecutive_pv_zeros["pv_measurement"].fillna(0).reset_index(drop=True)

        # X_scaled, Y_scaled = self._scale_and_create_scaler(X, Y)
        # return X_scaled, Y_scaled
        return X, Y

    def _scale_and_create_scaler(self, X: pd.DataFrame, Y: pd.Series):
        self.Y_scaler = MinMaxScaler(feature_range=(0, 1)).fit(Y.values.reshape(-1, 1))
        self.X_scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)

        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y.values(-1, 1))

        df_X = pd.DataFrame(X_scaled, columns=X.columns)
        s_Y = pd.Series(Y_scaled.ravel(), name=Y.name).reset_index(drop=True)

        return df_X, s_Y

    def _add_features_full_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).year)
        df["month"] = df["time"].apply(lambda datestring: np.datetime64(datestring).astype(pd.Timestamp).month)
        df["hours"] = df["time"].apply(lambda datestring: get_hours_of_day(datestring))
        df["since_jan_1"] = df["time"].apply(lambda datestring: get_days_sinse_beginning_of_year(datestring))

        df["effective_energy"] = df["clear_sky_rad:W"] * (df["total_cloud_cover:p"] / 100)

        return df

    def _add_lag_features(self, X: pd.DataFrame, drop_features: bool = False) -> pd.DataFrame:
        no_nan_columns_df = X[[c for c in X.columns if len(X[X[c].isna()].index) == 0]]
        no_nan_columns_df = no_nan_columns_df.select_dtypes(include=["number", "float", "int"])
        no_nan_columns = no_nan_columns_df[~no_nan_columns_df.isna()].columns.to_list()

        lag_f = LagFeatures(variables=no_nan_columns, periods=1)

        X_tr = lag_f.fit_transform(no_nan_columns_df)
        X[X_tr.columns] = X_tr

        if drop_features:
            tr = DropCorrelatedFeatures(variables=None, method="pearson", threshold=0.8)
            Xdc = tr.fit_transform(X)

            transformer = DropConstantFeatures(tol=0.7, missing_values="ignore")
            X = transformer.fit_transform(Xdc)

        return X

    def _fill_df(self, df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
        inner_to_split = df.copy()
        # non_id_columns = [c for c in inner_to_split.columns if ":idx" not in c]

        inner_to_split[categorical_columns].fillna(0, inplace=True)
        # inner_to_split[non_id_columns].fillna(method="bfill", inplace=True)

        return inner_to_split

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=[c for c in df.columns if "date" in c or "tim" in c], inplace=True)
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
        ).reset_index(drop=True)

        df_c = train_mapped[train_mapped["time_test"].notna()]

        df_c = df_c.drop(columns=[c for c in df_c.columns if "tim" in c or "date" in c])
        df_c = df_c.astype("float")

        return df_c

    def prep_test(self, location: str, merge_dfs=False) -> pd.DataFrame:
        if merge_dfs:
            X_test_a = X_test_estimated_a
            X_test_b = X_test_estimated_b
            X_test_c = X_test_estimated_c
            if location != "A":
                X_test_a = X_test_a.drop(columns=non_equal_value_columns)
            if location != "B":
                X_test_b = X_test_b.drop(columns=non_equal_value_columns)
            if location != "C":
                X_test_c = X_test_c.drop(columns=non_equal_value_columns)
            df = pd.merge(
                pd.merge(
                    X_test_a,
                    X_test_b,
                    on="date_forecast",
                    how="inner",
                    suffixes=("_a", "_b"),
                ),
                X_test_c,
                on="date_forecast",
                how="inner",
                suffixes=("", "_c"),
            )
        else:
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
        # df_with_lag = self._add_lag_features(df_new_features)
        df_shortened = self._shorten_dataset_to_prediction_scale(df_new_features, location)

        print(f"Location {location}. length: {str(len(df_new_features))}")

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
