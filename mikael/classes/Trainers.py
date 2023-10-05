# Data handling
import pickle
import json
from classes.DataFrameHandler import MasterDataframes

# Data science
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Feature engineering
from sklearn.ensemble import HistGradientBoostingRegressor


# Machine learning tool
import xgboost as xgb

# Optimization / feature engineering tools
import optuna

from classes.Models import MonoModel


class MonoModelTrainer:
    modelA = None
    modelB = None
    modelC = None

    M_df = MasterDataframes()

    def train_model(self, trials: int, location: str, drop_features=True, merge_dfs=False):
        X, Y = self.M_df.prep_dataset_x_y(location, merge_dfs=merge_dfs)

        non_id_columns = [c for c in X.columns if ":idx" not in c]

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X[non_id_columns] = scaler_x.fit_transform(X[non_id_columns])
        Y = scaler_y.fit_transform(Y.to_numpy().reshape(-1, 1))

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

        # selection = RFECV(estimator=model, verbose=1, step=1, cv=KFold(5), scoring=make_scorer(mean_absolute_error))
        # selection.fit(X_train, y_train)

        # select_X_train = selection.transform(X_train)
        # select_X_test = selection.transform(X_test)

        # selection_model = xgb.XGBRegressor(**study.best_params)
        # selection_model.fit(select_X_train, y_train)

        y_pred = selection_model.predict(select_X_test)

        M_df_c = self.M_df
        switched_model = MonoModel(
            model=model,
            selection_model=selection_model,
            M_df=M_df_c,
            y_pred=y_pred,
            y_test=y_test,
            X_test=X_test,
            X_train=X_train,
            params=study.best_params,
            scaler_x=scaler_x,
            scaler_y=scaler_y,
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


def HGBRegressorObjectiveModel(self, X, Y):
    non_id_columns = [c for c in X.columns if ":idx" not in c]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X[non_id_columns] = scaler_x.fit_transform(X[non_id_columns])
    Y = scaler_y.fit_transform(Y.to_numpy().reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.20)

    def objective(trial):
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 40)
        learning_rate = trial.suggest_float("learning_rate", 0, 0.5)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 40)
        max_bins = trial.suggest_int("max_bins", 50, 255)

        model = HistGradientBoostingRegressor(
            # task_type='GPU',
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_bins=max_bins,
            max_leaf_nodes=max_leaf_nodes,
            random_state=973,
        )
        model.fit(X_train, y_train)

        # Make predictions and calculate RMSE
        y_pred = model.predict(X_test)
        return mean_absolute_error(y_test, y_pred)

    return X_train, X_test, y_train, y_test, scaler_x, scaler_y, objective
