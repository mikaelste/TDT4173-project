# Data handling
import pandas as pd
from classes.DataFrameHandler import MasterDataframes

# Types handling
from typing import Optional

# Data science
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel  # , RFECV

# Feature engineering
# from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures
# from feature_engine.timeseries.forecasting import LagFeatures
# from sklearn.ensemble import HistGradientBoostingRegressor


# Machine learning tool
from xgboost import XGBRegressor

# plotting
import matplotlib.pyplot as plt


class MonoModel:
    model: Optional[XGBRegressor] = None
    selection_model: Optional[XGBRegressor] = None
    selection: Optional[SelectFromModel] = None
    M_df: Optional[MasterDataframes] = None
    y_pred = None
    y_test = None
    X_train = None
    X_test = None
    scaler_x: MinMaxScaler = None
    scaler_y: MinMaxScaler = None
    params = None

    def __init__(
        self,
        model=None,
        selection_model=None,
        M_df=None,
        y_pred=None,
        y_test=None,
        X_train=None,
        X_test=None,
        scaler_x=None,
        scaler_y=None,
        params=None,
        selection=None,
    ):
        self.model = model
        self.selection_model = selection_model
        self.M_df = M_df
        self.y_pred = y_pred
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.params = params
        self.selection = selection

    def plot_important_features(self, top: int):
        feature_important = self.model.get_booster().get_score(importance_type="weight")
        keys = list(feature_important.keys())
        values = list(feature_important.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        data.nlargest(top, columns="score").plot(kind="barh", figsize=(20, 10))

    def plot_pred_vs_test(self):
        plt.scatter(self.y_pred, self.y_test)
        plt.title("Time Series Plot")
        plt.xlabel("Time (absolute)")
        plt.ylabel("Prediction")
        plt.xticks(rotation=45)

        plt.show()

    def predict_test_data(self, location: str, merge_df=False):
        df = self.M_df.prep_test(location, merge_df)

        non_id_columns = [c for c in df.columns if ":idx" not in c]
        df[non_id_columns] = self.scaler_x.transform(df[non_id_columns])

        if self.selection is not None:
            df_c = self.selection.transform(df)
            y_pred = self.selection_model.predict(df_c)
        else:
            y_pred = self.model.predict(df)

        y_pred_reverted = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        # return y_pred
        return y_pred_reverted
