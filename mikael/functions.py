import pandas as pd
import numpy as np


def get_days_sinse_beginning_of_year(datetime64_string: str):
    datetime_obj = np.datetime64(datetime64_string).astype(pd.Timestamp)
    year = datetime_obj.year
    beginning_of_year = pd.Timestamp(year=year, month=1, day=1)
    days_gone_by = (datetime_obj - beginning_of_year).days
    return days_gone_by


def get_seconds_of_day(datetime64_string: str):
    datetime_obj = np.datetime64(datetime64_string).astype(pd.Timestamp)
    time_difference = datetime_obj - datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
    time_passed_seconds = time_difference.total_seconds()
    return time_passed_seconds
