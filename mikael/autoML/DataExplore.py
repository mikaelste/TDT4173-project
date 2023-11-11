# Data handling
from category_encoders import OrdinalEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import mean_absolute_error
import autogluon.core as ag
from feature_engine.timeseries.forecasting import LagFeatures


import numpy as np

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
        # self.test["observed_or_estimated"] = 1
        observed = pd.read_parquet(f"../../{location}/X_train_observed.parquet")
        # observed["observed_or_estimated"] = 0
        estimated = pd.read_parquet(f"../../{location}/X_train_estimated.parquet")
        # estimated["observed_or_estimated"] = 1
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
        
        # self.set_dtypes()
        
    def extract_tuning_data(self, last=False, tuning_start_date='2021-05-01', tuning_end_date='2021-08-31'):
        # Extract tuning data from self.frame
        if last:
            tuning_start_idx = int(len(self.frame) * 0.8)
            tuning_condition = self.frame.index >= tuning_start_idx
        else:
            tuning_condition = (self.frame['date_forecast'] >= tuning_start_date) & (self.frame['date_forecast'] <= tuning_end_date)

        self.tune_data = self.frame.loc[tuning_condition]
        self.frame_without_tuning_data = self.frame.loc[~tuning_condition]


    def set_dtypes(self):
        categorical_colummns = [c for c in self.frame.columns if "idx" in c]
        self.frame[categorical_colummns] = self.frame[categorical_colummns].astype("category")
        self.test[categorical_colummns] = self.test[categorical_colummns].astype("category")
        self.frame["date_forecast"] = pd.to_datetime(self.frame["date_forecast"])
        self.test["date_forecast"] = pd.to_datetime(self.test["date_forecast"])
        self.frame["date_calc"] = pd.to_datetime(self.frame["date_calc"])
        self.test["date_calc"] = pd.to_datetime(self.test["date_calc"])

    def transform_frame(self):
        self.frame = self.frame_scaler.fit_transform(self.frame.copy())

    def fit_transform_test(self):
        return self.X_scaler.transform(self.test)

    def fit_scalers(self):
        self.Y_scaler.fit(self.target.to_numpy().reshape(-1, 1))
        self.X_scaler.fit(self.train)

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

    def set_neg_to_zero(self):
        self.frame["pv_measurement"] = self.frame["pv_measurement"].apply(lambda x: max(0, x))

    def drop_index(self):
        self.train = self.train.drop(columns=["index"])
        self.frame = self.frame.drop(columns=["index"])
        # self.frame_without_tuning_data = self.frame_without_tuning_data.drop(columns=["index"])
        # self.tune_data = self.tune_data.drop(columns=["index"])

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
        df.drop(columns=["date_calc", "hour", "month", "day_of_year", "time_of_day"], inplace=True)
        return df
    
    def predict_location_GLUON(self, num_bag_folds=10, num_bag_sets=2, num_stack_levels=1):
        X = self.unzip_date_feature(self.frame.copy())
        X = X.loc[X["pv_measurement"].notna()]
        test = self.unzip_date_feature(self.test.copy())
        
        # drop features
        drop = ['wind_speed_u_10m:ms',
                'wind_speed_v_10m:ms',
                'wind_speed_w_1000hPa:ms'
                ]
        X.drop(columns=drop, inplace=True)
        test.drop(columns=drop, inplace=True)
        
        # drop where nans
        X.dropna(axis=1, thresh=100, inplace=True)
        X.fillna(X.mean(), inplace=True)
        test = test[X.columns.copy().drop("pv_measurement")]
        
        # Lag features
        no_cat_features_1h = [c for c in X.columns if "_1h:" in c]
        lag_cols = X[no_cat_features_1h].select_dtypes(include=["number", "float", "int"]).columns.to_list()
        lag_f = LagFeatures(variables=lag_cols, periods=1)
        X_tr = lag_f.fit_transform(X[lag_cols].select_dtypes(include=["number", "float", "int"]))
        test_tr = lag_f.fit_transform(test[lag_cols].select_dtypes(include=["number", "float", "int"]))
        X[X_tr.columns] = X_tr
        test[test_tr.columns] = test_tr
        
        
        X = X.loc[X["pv_measurement"].notna()]
        self.test = test
        
        train = TabularDataset(X)

        time_limit = 60 * 60
        # path = f"autogluon_models_{self.location}_f{num_bag_folds}_s{num_bag_sets}_s{num_stack_levels}"
        path = f"autogluon_models_{self.location}_{5}"
        
        print(f"""Starting....
              X-lenght: {len(X)}, from original lenght: {len(self.train)}
              X-features: {len(X.columns)}, from original length: {len(self.train.columns)}
              """)

        predictor = TabularPredictor(
            problem_type="regression", 
            eval_metric=mean_absolute_error, 
            label="pv_measurement", 
            path=path,
            ).fit(
            train,
            presets="experimental_zeroshot_hpo_hybrid",
            time_limit=time_limit
        )
        return predictor     

if __name__ == "__main__":
    num_folds = [5, 7]
    num_sets = [2, 4, 7, 30]
    num_stacks = [0, 1, 2]
    
    round = 0
    for num_fold in num_folds:
        for num_set in num_sets:
            for num_stack in num_stacks:
                print(f""".... predicting\n\n\n
                      num folds: {num_fold}
                      num set: {num_set}
                      num stack: {num_stack}
                      
                      Round: {round + 1} of {len(num_stacks) * len(num_folds) * len(num_sets)}
                      
                      """)
                prediction = []
                for location in ["A", "B", "C"]:
                    data = Data(location=location)
                    predictor = data.predict_location_GLUON(num_bag_folds=num_fold, num_bag_sets=num_set, num_stack_levels=num_stack)
                    test = TabularDataset(data.test)
                    y_pred = predictor.predict(test, as_pandas=False)
                    y_pred = data.fit_predictions_to_submission_length(y_pred).to_numpy()
                    prediction += list(y_pred)
                df = pd.DataFrame({"prediction": prediction}).rename_axis(index="id")
                df.to_csv(f"submissions/auto/auto_f{num_fold}_s{num_set}_s{num_stack}_submission.csv")