# Basic Libraries
import numpy as np
import pandas as pd
import os

#extract data from weblink
html_data = pd.read_html('https://en.wikipedia.org/wiki/2016_Summer_Olympics_medal_table')
#basic info of table, use len to check for how many tables there are in the dataset
print("Data type : ", type(html_data))
print("HTML tables : ", len(html_data))

#identify actual 2016 Summer Olympics medal table
html_data[2].head()
#extract the main table and store it as a new Pandas DataFrame
medal_table = html_data[2]
#view table
medal_table

#Extract the TOP 20 countries from the medal table, as above, and store these rows as a new DataFrame.
top20countries_table = medal_table[0:20]
top20countries_table
