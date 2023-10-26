# Data handling
import pickle
import json
from classes.DataFrameHandler import MasterDataframes

# Data science
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Optimization / feature engineering tools
from classes.ModelObjectives import XGBRCVObjective  # , HGBRObjective, RFRObjective, XGBRObjective

from classes.Models import MonoModel


class MonoModelTrainer:
    modelA = None
    modelB = None
    modelC = None

    M_df = MasterDataframes()

    def train_model(self, trials: int, location: str, drop_features=True, merge_dfs=False):
        X, Y = self.M_df.prep_dataset_x_y(location, merge_dfs=merge_dfs)
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X = scaler_x.fit_transform(X)
        Y = scaler_y.fit_transform(Y.to_numpy().reshape(-1, 1)).flatten()

        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=10, test_size=0.20)

        model, selection_model, study, selection, select_X_test = XGBRCVObjective(
            X_train, X_test, y_train, y_test, trials
        )

        if selection_model:
            y_pred = selection_model.predict(select_X_test)
        else:
            y_pred = model.predict(X_test)

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
