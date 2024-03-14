import os as os
import pandas as pandas
import numpy as np
from datetime import datetime

os.chdir("D:\\TU\\PEIN\\data")
dayFile = pandas.read_excel('day.xls',
                        sheet_name='pavlovo', 
                        header=1, 
                        usecols="A:C", 
                        dtype={"Date": 'string', "O3": float, "PM10": float})
hourFile = pandas.read_excel('hour.xls',
                        sheet_name='pavlovo', 
                        header=1, 
                        usecols={ "Date", "NO", "NO2", "AirTemp","Press", "UMR" }, 
                        dtype={"Date": 'string', "NO": float, "NO2": float, "AirTemp": float, "Press": float, "UMR": float})
merged_df = pandas.DataFrame()

for i in range(dayFile["Date"].size):
    date = dayFile.iloc[i]["Date"]
    O3 = dayFile.iloc[i]["O3"]
    PM10 = dayFile.iloc[i]["PM10"]
    
    filtered_hour = hourFile[hourFile["Date"].str.contains(date)]

    filtered_hour.insert(2, "O3", O3)
    filtered_hour.insert(3, "PM10", PM10)

    merged_df = merged_df._append(filtered_hour, ignore_index=True)

merged_df = merged_df.dropna()

rounded_merged_df = merged_df.round({"NO": 2, "NO2": 2, "O3": 2, "PM10": 2, "AirTemp": 1, "UMR": 1, "Press": 0})

rounded_merged_df.to_csv("D:\\TU\\task2.csv", columns = ["NO", "NO2", "AirTemp", "Press", "UMR", "O3", "PM10"])
