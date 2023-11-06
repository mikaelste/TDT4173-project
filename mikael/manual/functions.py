import pandas as pd
import numpy as np


def get_days_sinse_beginning_of_year(datetime64_string: str):
    datetime_obj = np.datetime64(datetime64_string).astype(pd.Timestamp)
    year = datetime_obj.year
    beginning_of_year = pd.Timestamp(year=year, month=1, day=1)
    days_gone_by = (datetime_obj - beginning_of_year).days
    return days_gone_by


def get_hours_of_day(datetime64_string: str):
    datetime_obj = np.datetime64(datetime64_string).astype(pd.Timestamp)
    time_difference = datetime_obj - datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    time_passed_hours = time_difference.total_seconds() / (60**2)
    return time_passed_hours


def set_consecutive_to_nan(df: pd.DataFrame, consecutive_threshold: int = 48):
    column_to_check = "pv_measurement"
    mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

    df["consecutive_count"] = df.groupby(mask).transform("count")[column_to_check]
    mask = df["consecutive_count"] > consecutive_threshold
    df["consecutive_count"] = np.nan
    return df.loc[~mask]


def set_missing_with_mean_historic_data(df: pd.DataFrame, consecutive_threshold: int = 48):
    column_to_check = "pv_measurement"
    mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

    df["consecutive_count"] = df.groupby(mask).transform("count")[column_to_check]
    mask = df["consecutive_count"] > consecutive_threshold
    df["consecutive_count"] = np.nan
    df[["consecutive_count", column_to_check, "time"]].apply(lambda row: df[row["time"].df["time"]])

    return df.loc[~mask]


# @Mats
def remove_consecutive_measurments(df: pd.DataFrame, consecutive_threshold: int = 48):
    column_to_check = "pv_measurement"
    mask = (df[column_to_check] != df[column_to_check].shift(2)).cumsum()

    df["consecutive_count"] = df.groupby(mask).transform("count")[column_to_check]
    mask = df["consecutive_count"] > consecutive_threshold
    df.drop(columns=["consecutive_count"], inplace=True)
    return df.loc[~mask]
