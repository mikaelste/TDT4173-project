import pandas as pd
import numpy as np

import sys
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


class Pipe:
    
    def get_data(self, location: str, debug: bool = False):
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

        X_train_total = pd.concat(
            [X_train_observed_x, X_train_estimated_x]).reset_index().drop(columns=["index"])
        
        X_train_total["date_forecast"] = pd.to_datetime(
            X_train_total["date_forecast"])
        
        
        categorical_columns = [c for c in X_train_total.columns if ":idx" in c]
        numeric_columns = list(
            set(X_train_total.columns) - set(categorical_columns)- {"date_forecast"})
        X_train_group = X_train_total.groupby(pd.Grouper(key="date_forecast", freq="1H", )).agg(
            {**{col: 'mean' for col in numeric_columns}, **{col: lambda x: x.mode().iloc[0] if not x.empty else np.nan for col in categorical_columns}}).reset_index()
        
        Y_train_x = self.remove_consecutive_measurments(Y_train_x, consecutive_threshold=24)
        
        
        X_train_group.rename(columns={"date_forecast": "time"}, inplace=True)
        merged = pd.merge(X_train_group, Y_train_x, on="time", how="inner")
        merged = self.remove_consecutive_measurments(merged)
        
        
        # merged = merged[merged["pv_measurement"].notna()] # all rows were populated
        return merged
        
    def remove_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3):
        df = df[np.abs(df[column]-df[column].mean()) <= (threshold*df[column].std())]
        return df
    
    def find_consecutive_measurements(self, df: pd.DataFrame):
        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift()).cumsum()

        df['consecutive_count'] = df.groupby(mask).transform('count')[column_to_check]
        start_date = '2020-07-12 15:00:00'
        end_date = '2020-08-26'
        too_hig = 48
        # mask = (df['time'] > start_date) & (df['time'] <= end_date)
        mask = (df['consecutive_count'] > too_hig ) & (df["pv_measurement"].notna()) & (df["pv_measurement"] > 0 ) 
        df_sub = df.loc[mask]
        df_sub['time'] = pd.to_datetime(df_sub['time'])
        print("sub",df_sub)
    
    def remove_consecutive_measurments(self, df: pd.DataFrame, consecutive_threshold: int = 48):
        column_to_check = 'pv_measurement'
        mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

        df['consecutive_count'] = df.groupby(mask).transform('count')[column_to_check]
        mask = (df['consecutive_count'] > consecutive_threshold )

        return df.loc[~mask]
        
        


pipe = Pipe()
pipe.get_data("B")
