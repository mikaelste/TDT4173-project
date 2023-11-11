# Data handling
import datetime
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.metrics import mean_absolute_error
import autogluon.core as ag

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

        self.target = pd.read_parquet(f"../../{location}/train_targets.parquet")["pv_measurement"]
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
        ).reset_index()
        
        # self.train = self.train.groupby(pd.Grouper(key="date_forecast", freq="1H")).first().reset_index()
        # self.test = self.test.groupby(pd.Grouper(key="date_forecast", freq="1H")).first().reset_index()

        self.frame = self.train.copy()
        self.frame["pv_measurement"] = self.target
        # self.frame = self.frame.loc[self.frame["pv_measurement"].notna()]
        
        # self.drop_consequtives()
        # self.frame = self.frame.loc[self.frame["pv_measurement"].notna()]
        # self.tune_data = self.tune_data.loc[self.tune_data["pv_measurement"].notna()]
        # self.frame_without_tuning_data = self.frame_without_tuning_data.loc[self.frame_without_tuning_data["pv_measurement"].notna()]
        
        self.set_neg_to_zero()
        self.set_dtypes()
        self.drop_index()
        self.exstract_tuning_data()
        # self.reduce_test_to_submission_length()
        # self.fit_scalers()
        # self.transform_frame()
        
    def exstract_tuning_data(self):
        # exstract tuning data from self.frame
        tuning_start_date = '2021-05-01'
        tuning_end_date = '2021-08-31'
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

    def drop_consequtives(self, consecutive_threshold=int(24)):
        column_to_check = "pv_measurement"
        df = self.frame
        mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

        df["consecutive_count"] = df.groupby(mask).transform("count")[column_to_check]
        mask = df["consecutive_count"] > consecutive_threshold
        df.drop(columns=["consecutive_count"], inplace=True)
        self.frame = df.loc[~mask]

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

    def predict_location_H2O(self):
        frame = h2o.H2OFrame(self.frame)

        y = "C47"
        x = frame.columns
        x.remove(y)

        aml = H2OAutoML(
            max_models=20,
            project_name="regression_" + str(self.location),
        )

        aml.train(x=x, y=y, training_frame=frame)
        return aml

    def predict_location_MLJAR(self):
        columns = [f"C_{i}" for i in range(len(self.frame[0]))]
        frame = pd.DataFrame(self.frame, columns=columns)
        frame.fillna(value=np.nan, inplace=True)

        y = frame["C_46"]
        X = frame.drop(columns=["C_46"])
        automl = AutoML(
            ml_task="regression",
            mode="Compete",
            algorithms=[
                "Random Forest",
                "Extra Trees",
                "LightGBM",
                "Xgboost",
                "CatBoost",
            ],
            # eval_metric="rmse",
            # optuna_time_budget=3600,
            # optuna_init_params={},
            # algorithms=["LightGBM", "Xgboost", "Extra Trees"],
            # total_time_limit=8 * 3600,
        )
        automl.fit(X=X, y=y)

        test = self.test.rename(
            columns={self.test.columns[i]: f"C_{i}" for i in range(len(self.frame[0]) - 1)}
        )  # not possible to use frame[0] when not scaling! Rename is thus unecessary
        predictions = automl.predict(test)
        return predictions
    
    def predict_location_GLUON(self, num_bag_folds=7, num_bag_sets=2, num_stack_levels=1):
        frame = self.frame_without_tuning_data.copy().drop(columns=[c for c in self.frame.columns if "index" in c])
        tune = self.tune_data.copy().drop(columns=[c for c in self.frame.columns if "index" in c])
        self.test = self.test.drop(columns=[c for c in self.test.columns if "index" in c])
        
        # frame = frame.loc[frame["pv_measurement"].notna()]
        # tune = tune.loc[tune["pv_measurement"].notna()]
        
        train = TabularDataset(frame)   
        tune = TabularDataset(tune)

        time_limit = 3 * 60 * 60
        # path = f"autogluon_models_{self.location}_f{num_bag_folds}_s{num_bag_sets}_s{num_stack_levels}"
        path = f"autogluon_models_{self.location}"

        predictor = TabularPredictor(
            problem_type="regression", 
            eval_metric=mean_absolute_error, 
            label="pv_measurement", 
            path=path,
            ).fit(
            train,
            tuning_data=tune,
            presets="best_quality",
            time_limit=time_limit,
            hyperparameters={
                'NN_TORCH': {},
                'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, 'GBMLarge'],
                'CAT': {},
                'XGB': {},
                'FASTAI': {},
                'RF': [{'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}}],
                'XT': [{'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}}],
            },
            use_bag_holdout=True
            # num_bag_folds=num_bag_folds, 
            # num_bag_sets=num_bag_sets, 
            # num_stack_levels=num_stack_levels
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