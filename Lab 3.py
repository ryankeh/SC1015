# Basic Libraries
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot 
sb.set() # set the default Seaborn style for graphics
os.getcwd()

#Problem 1
data = pd.read_csv('train.csv')
data.head()

LotArea = pd.DataFrame(data['LotArea'])
GrLivArea = pd.DataFrame(data['GrLivArea'])
TotalBsmtSF = pd.DataFrame(data['TotalBsmtSF'])
GarageArea = pd.DataFrame(data['GarageArea'])

#viewing boxplot for all 4 variables
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = LotArea, orient = "h")
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = GrLivArea, orient = "h")
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = TotalBsmtSF, orient = "h")
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = GarageArea, orient = "h")

data.describe()[['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']]

#getting number of outliers for each variable
q1=LotArea.quantile(0.25)
q3=LotArea.quantile(0.75)
IQR=q3-q1
outliers = LotArea[((LotArea<(q1-1.5*IQR)) | (LotArea>(q3+1.5*IQR)))]
outliers.describe()
#69 outliers
q1=GrLivArea.quantile(0.25)
q3=GrLivArea.quantile(0.75)
IQR=q3-q1
outliers = GrLivArea[((GrLivArea<(q1-1.5*IQR)) | (GrLivArea>(q3+1.5*IQR)))]
outliers.describe()
#31 outliers
q1=TotalBsmtSF.quantile(0.25)
q3=TotalBsmtSF.quantile(0.75)
IQR=q3-q1
outliers = TotalBsmtSF[((TotalBsmtSF<(q1-1.5*IQR)) | (TotalBsmtSF>(q3+1.5*IQR)))]
outliers.describe()
#61 outliers
q1=GarageArea.quantile(0.25)
q3=GarageArea.quantile(0.75)
IQR=q3-q1
outliers = GarageArea[((GarageArea<(q1-1.5*IQR)) | (GarageArea>(q3+1.5*IQR)))]
outliers.describe()
#21 outliers

# 1a) LotArea has the maximum number of outliers with 69 outliers

#getting skew values
LotArea.skew()
GrLivArea.skew()
TotalBsmtSF.skew()
GarageArea.skew()

# 1b) Lot Area is the most skewed from a regular normal distribution and positively skewed

#obtaining correlation values data
jointData = pd.DataFrame(data[["SalePrice", "LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"]])
jointData.corr()
#general living area and garage area has the highest correlation with SalePrice

# 1c) General Living Area and Garage Area has the highest 2 correlations with SalePrice and hence will help us the most in predicting ‘SalePrice’ of houses in this data.


#Problem 2

data.head()[['MSSubClass', 'Neighborhood', 'BldgType', 'OverallQual']]

#gets number of categories (one value)
data.MSSubClass.value_counts().count()
data.Neighborhood.value_counts().count()
data.BldgType.value_counts().count()
data.OverallQual.value_counts().count()

#gets all categories and their distributions (how many houses for each category)
data.MSSubClass.value_counts()
data.Neighborhood.value_counts()
data.BldgType.value_counts()
data.OverallQual.value_counts()

# 2a) Neighborhood has the highest number of levels, and the NAmes level had the highest number of houses with 225 houses.

f = plt.figure(figsize=(24,10))
sb.boxplot(data=data, x="MSSubClass", y="SalePrice")

f = plt.figure(figsize=(24,10))
sb.boxplot(data=data, x="Neighborhood", y="SalePrice")

f = plt.figure(figsize=(24,10))
sb.boxplot(data=data, x="BldgType", y="SalePrice")

f = plt.figure(figsize=(24,10))
sb.boxplot(data=data, x="OverallQual", y="SalePrice")

# 2b) By observation, Neighborhood and OverallQual will help us the most in predicting ‘SalePrice’ of houses in this data.


#Problem 3

SalePrice = pd.DataFrame(data['SalePrice'])
GarageArea = pd.DataFrame(data['GarageArea'])

GarageArea2 = np.where(data["GarageArea"]!=0, "garage", "nogarage")

f = plt.figure(figsize=(24,10))
sb.boxplot(data=data, x = GarageArea2, y = "SalePrice")

# Yes, by comparing the respective means, we can see that the SalePrice of a house get affected by whether it has a Garage

