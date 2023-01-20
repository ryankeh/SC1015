import numpy as np
import pandas as pd
import os

#creating dictionary
canteens_dict = {"Name" : ["North Spine", "Koufu", "Canteen 9", "North Hill", "Canteen 11"],
                 "Stalls" : [20, 15, 10, 12, 8],
                 "Rating" : [4.5, 4.2, 4.0, 3.7, 4.2]
                }

#expressing information in dictionary as a dataframe
canteens_df = pd.DataFrame(canteens_dict)
canteens_df

#accessing single column in dataframe 
canteens_df["Name"]

#accessing single row in dataframe, using iloc followed by the index
canteens_df.iloc[0]

#to get current directory
os.getcwd()

#if the file requested is in the current directory, otherwise need list out whole location
#if the dataset is in a standard CSV format (flat file), we may use the read_csv function from Pandas.
#CSV FORMAT
csv_data = pd.read_csv('train.csv', header = None)
csv_data.head()


#data type of whole set of data
print("Data type : ", type(csv_data))
#gives dimensions of the data, how many observations (rows) and variables (columns)
print("Data dims : ", csv_data.shape)




#FOR DIFFERENT FILE 
#TXT FORMAT
#If the dataset is in a standard TXT format (flat file), we may use the read_table function from Pandas.
  txt_data = pd.read_table('somedata.txt', sep = "\s+", header = None)
  txt_data.head()
  print("Data type : ", type(txt_data))
  print("Data dims : ", txt_data.shape)
#XLS FORMAT (EXCEL)
#If the dataset is in a Microsoft XLS or XLSX format, we may use the read_excel function from Pandas.
#However, to use the read_excel function, you will need to install the xlrd module using Anaconda.
  xls_data = pd.read_excel('somedata.xlsx', sheet_name = 'Sheet1', header = None)
  xls_data.head()
  print("Data type : ", type(xls_data))
  print("Data dims : ", xls_data.shape)
#JSON FORMAT
#If the dataset is in a standard JSON format, we may use the read_json function from Pandas.
  json_data = pd.read_json('somedata.json')
  json_data.head()
  print("Data type : ", type(json_data))
  print("Data dims : ", json_data.shape)
#HTML FORMAT
#If the dataset is in a table formal within an HTML website, we may use the read_html function from Pandas.
#Let's try to get the Cast of Kung-Fu Panda : http://www.imdb.com/title/tt0441773/fullcredits/?ref_=tt_ov_st_sm
  html_data = pd.read_html('http://www.imdb.com/title/tt0441773/fullcredits/?ref_=tt_ov_st_sm')
  print("Data type : ", type(html_data))
  print("HTML tables : ", len(html_data))
  html_data[2].head()
