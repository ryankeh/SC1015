# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
os.getcwd()


# Obtain data from excel file
data = pd.read_csv('train.csv')
data.head()

# Obtain data types
print("Data type : ", type(data))
print(data.dtypes)

# Select all columns with int data
int_data=data.select_dtypes(include="int64")


# Display filtered data
pd.DataFrame(int_data)

# Dropping non-numeric variables by name
int_data2 = int_data.drop(['MSSubClass', 'OverallQual','OverallCond'], axis=1)
pd.DataFrame(int_data2)


# Find summary statistics
salePrice = pd.DataFrame(data['SalePrice'])
salePrice.describe()


# Visualize the summary statistics and distribution of SalePrice using standard Box-Plot, Histogram, KDE
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = salePrice, orient = "h")

f = plt.figure(figsize=(16, 8))
sb.histplot(data = salePrice)

f = plt.figure(figsize=(16, 8))
sb.kdeplot(data = salePrice)


# Find summary statistics for LotArea
lotArea = pd.DataFrame(data['LotArea'])
lotArea.describe()


# Visualize the summary statistics and distribution of LotArea using standard Box-Plot, Histogram, KDE.
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = lotArea, orient = "h")

f = plt.figure(figsize=(16, 8))
sb.histplot(data = lotArea)

f = plt.figure(figsize=(16, 8))
sb.kdeplot(data = lotArea)


# Plot SalePrice (y-axis) vs LotArea (x-axis) using jointplot
sb.jointplot(x = "LotArea", y = "SalePrice", data = data)


# Find the Correlation between the two.
jointDF = pd.concat([lotArea, salePrice], axis = 1).reindex(lotArea.index)
jointDF
jointDF.corr()


# Visualise correlation
sb.heatmap(jointDF.corr(), vmin = -1, vmax = 1, annot = True, fmt=".2f")


# Bonus
# Create a new Pandas DataFrame consisting of all variables (columns) of type Integer (int64) or Float (float64).
bonus_data=data.select_dtypes(include=["int64","float64"])
pd.DataFrame(bonus_data)

# Read the description for each variable carefully and try to identify the “actual” Numeric variables in the data.
# Drop non-Numeric variables from the DataFrame to have a clean DataFrame with only the Numeric variables.
bonus_data2 = bonus_data.drop(['MSSubClass', 'OverallQual','OverallCond'], axis=1)
pd.DataFrame(bonus_data2)

# Plot SalePrice vs each of the Numeric variables you identified to understand their correlation or dependence.
bonus_data2.corr()



# Dropping Data
# Using DataFrame.drop
df.drop(df.columns[[1, 2]], axis=1, inplace=True)
# drop by Name
df1 = df1.drop(['B', 'C'], axis=1)
# Select the ones you want
df1 = df[['a','d']]
# use this to drop columns instead
