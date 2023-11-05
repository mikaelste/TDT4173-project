import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor


class Master:
    @staticmethod
    def timestamp(df,name_time_col):
        # Create a sample DataFrame
        df_stamped = df.copy()

# Convert the name_time_col to datetime format
        df_stamped[name_time_col] = pd.to_datetime(df_stamped[name_time_col])

# Calculate 'days_since_jan' and 'hour' and create new columns for them
        df_stamped['days_since_jan'] = df_stamped[name_time_col].dt.dayofyear
        df_stamped['hour'] = df_stamped[name_time_col].dt.hour

# Drop the original datetime column
        df_stamped.drop(name_time_col, axis=1, inplace=True)

# Display the DataFrame
        return df_stamped
    
    @staticmethod
    def averaged(df,name_time_col):
        df_averaged = df.copy()
        df_averaged = pd.DataFrame(df_averaged)
        df_averaged[name_time_col] = pd.to_datetime(df_averaged[name_time_col])
        df_averaged.set_index(name_time_col, inplace=True)
        df_resampled = df_averaged.resample('1H').mean()
        df_resampled.reset_index(inplace=True)

        return df_resampled
    

    def averaged_redundant(df, name_time_col, column_to_sum):
        df_averaged = df.copy()
        df_averaged[name_time_col] = pd.to_datetime(df_averaged[name_time_col])
        df_averaged.set_index(name_time_col, inplace=True)

        # Resample and take the mean for the entire DataFrame
        df_resampled = df_averaged.resample('1H').mean()

        # Resample the specific column you want to sum over 15-minute intervals
        sum_resampled = df_averaged[column_to_sum].resample('15T').sum()

        # Adjust the 1-hour resampled DataFrame with the summed values for every 15 minutes
        df_resampled[column_to_sum] = sum_resampled.resample('1H').sum()
        
        df_resampled.reset_index(inplace=True)

        return df_resampled
    

    def averaged_test(df, name_time_col):
        df_averaged = df.copy()
        df_averaged[name_time_col] = pd.to_datetime(df_averaged[name_time_col])
    
    # Extract date and hour from the timestamp
        df_averaged['date'] = df_averaged[name_time_col].dt.date
        df_averaged['hour'] = df_averaged[name_time_col].dt.hour
    
    # Group by date and hour to get the mean
        df_resampled = df_averaged.groupby(['date', 'hour']).mean().reset_index()
    
    # Reconstruct the timestamp from the date and hour
        df_resampled[name_time_col] = pd.to_datetime(df_resampled['date'].astype(str) + ' ' + df_resampled['hour'].astype(str) + ':00:00')
        df_resampled.drop(columns=['date', 'hour'], inplace=True)

        return df_resampled
    



    @staticmethod
    def target_join(df1,df2,df_target):
        df_merge = pd.concat([df1.copy(),df2.copy()],ignore_index=True)
        df_merge = df_merge[df_merge['date_forecast'].isin(df_target.copy()['time'])]
        return df_merge
    
    @staticmethod
    def randomForest(df_features,df_targets):

        
        X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_targets, test_size=0.2, random_state=42)

        # If you meant to keep only the first column of Y_train and Y_test
        Y_train = Y_train.iloc[:, 0]
        Y_test = Y_test.iloc[:, 0]

        # Handle NaN values in feature matrices
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        # Handle NaN values in target variables with caution, or maybe consider dropping NaNs if applicable.
        # If you decide to fill NaN in targets:
        Y_train.fillna(0, inplace=True)
        Y_test.fillna(0, inplace=True)


        model = RandomForestRegressor(n_estimators=100, random_state=42)  # Create a Random Forest Regressor model
        model.fit(X_train, Y_train)  # Train the model with the training data

        predictions = model.predict(X_test)  # Make predictions on the test data
        mae = mean_absolute_error(Y_test, predictions)  # Calculate the Mean Absolute Error
        #print(f"Mean Absolute Error: {mae}")
        return mae
    
   

    def inner_join_old(feature_df, target_df):
    # Inner join on 'date_forecast' and 'time' columns

        target_df = target_df.dropna(subset=['pv_measurement'])


        merged_df = feature_df.merge(target_df, left_on='date_forecast', right_on='time', how='inner')

    # Extract columns from merged_df for each DataFrame
        cols_feature = feature_df.columns
        cols_target = target_df.columns

        aligned_feature_df = merged_df[cols_feature]
        aligned_target_df = merged_df[cols_target]

    # Drop all rows from the original DataFrames
        feature_df.drop(feature_df.index, inplace=True)
        target_df.drop(target_df.index, inplace=True)

        # Append the aligned rows using pandas.concat
        feature_df = pd.concat([feature_df, aligned_feature_df], ignore_index=True)
        target_df = pd.concat([target_df, aligned_target_df], ignore_index=True)

        return feature_df, target_df
    

    def inner_join(feature_df, target_df):
    # Assuming 'pv_measurement' is the target variable column in target_df.
    # Perform an inner join on 'date_forecast' from feature_df and 'time' from target_df.
    # This will only include rows where the keys match in both DataFrames.

        target_df = target_df.dropna(subset=['pv_measurement'])

        merged_df = feature_df.merge(target_df[['time', 'pv_measurement']], 
                                    left_on='date_forecast', 
                                    right_on='time', 
                                    how='inner')
        
        # If you do not want to keep the 'time' column from target_df after the join,
        # you can drop it from the resulting DataFrame.
        merged_df.drop('time', axis=1, inplace=True)

        return merged_df
    

    def onehot_estimated(df,estimated_bool):
        
        if estimated_bool:
            df["estimated"]=1 
            df["observed"]=0

        else:
            df["estimated"]=0
            df["observed"]=1

        return df
        
    
    def onehot_location(df,location):
        
        #This is garbage, write cleaner

        if location == "A":
            df["A"] = 1
            df["B"] = 0
            df["C"] = 0

        if location == "B":
            df["A"] = 0
            df["B"] = 1
            df["C"] = 0

        if location == "C":
            df["A"] = 0
            df["B"] = 0
            df["C"] = 1

        return df
        

        


        






    
    
    

   

    





    








    



        


    

