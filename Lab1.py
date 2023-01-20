import numpy as np
import pandas as pd
import os

canteens_dict = {"Name" : ["North Spine", "Koufu", "Canteen 9", "North Hill", "Canteen 11"],
                 "Stalls" : [20, 15, 10, 12, 8],
                 "Rating" : [4.5, 4.2, 4.0, 3.7, 4.2]
                }

canteens_df = pd.DataFrame(canteens_dict)
canteens_df

canteens_df["Name"]

canteens_df.iloc[0]

os.getcwd()
//to get current directory

csv_data = pd.read_csv('train.csv', header = None)
csv_data.head()
//if the file requested is in the current directory, otherwise need list out whole location
