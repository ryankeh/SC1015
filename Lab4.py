# Basic Libraries
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot 

# Import essential models and functions from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sb.set() # set the default Seaborn style for graphics
os.getcwd()


# Problem 1

data = pd.read_csv('train.csv')
data.head()

SalePrice = pd.DataFrame(data['SalePrice']) #response
LotArea = pd.DataFrame(data['LotArea']) 
GrLivArea = pd.DataFrame(data['GrLivArea']) #predictor
TotalBsmtSF = pd.DataFrame(data['TotalBsmtSF'])
GarageArea = pd.DataFrame(data['GarageArea'])

print("Data dims : ", data.SalePrice)
#1460 total samples
#1168 train / 292 test

GrLivArea_train, GrLivArea_test, SalePrice_train, SalePrice_test = train_test_split(GrLivArea, SalePrice, test_size = 0.20)

# Check the sample sizes
print("Train Set :", GrLivArea_train.shape, SalePrice_train.shape)
print("Test Set  :", GrLivArea_test.shape, SalePrice_test.shape)

# Create a joint dataframe by concatenating the two variables
trainDF = pd.concat([GrLivArea_train, SalePrice_train], axis = 1).reindex(GrLivArea_train.index)

# Jointplot of SalePrice Train against GrLivArea Train
sb.jointplot(data = trainDF, x = "GrLivArea", y = "SalePrice", height = 12)

# Import LinearRegression model from Scikit-Learn
from sklearn.linear_model import LinearRegression

# Create a Linear Regression object
linreg = LinearRegression()

# Train the Linear Regression model
linreg.fit(GrLivArea_train, SalePrice_train)

# Coefficients of the Linear Regression line
print('Intercept \t: b = ', linreg.intercept_)
print('Coefficients \t: a = ', linreg.coef_)

# Formula for Regression Line
regline_x = GrLivArea_train
regline_y = linreg.intercept_ + linreg.coef_ * GrLivArea_train
# Plot the Linear Regression line
f, axes = plt.subplots(1,1, figsize=(16, 8))
plt.scatter(GrLivArea_train, SalePrice_train)
plt.plot(regline_x.squeeze(), regline_y.squeeze(), "r-", linewidth = 3)
plt.show()

# Predict Total values corresponding to GrLivArea Train
SalePrice_train_pred = linreg.predict(GrLivArea_train)

# Plot the Linear Regression line
f = plt.figure(figsize=(16, 8))
plt.scatter(GrLivArea_train, SalePrice_train)
plt.scatter(GrLivArea_train, SalePrice_train_pred, color = "r")
plt.show()

# Explained Variance (R^2)
print("Explained Variance (R^2) \t:", linreg.score(GrLivArea_train, SalePrice_train))

# Mean Squared Error (MSE)
def mean_sq_err(actual, predicted):
    '''Returns the Mean Squared Error of actual and predicted values'''
    return np.mean(np.square(np.array(actual) - np.array(predicted)))

mse = mean_sq_err(SalePrice_train, SalePrice_train_pred)
print("Mean Squared Error (MSE) \t:", mse)
print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mse))

# Predict Total values corresponding to HP Test
SalePrice_test_pred = linreg.predict(GrLivArea_test)

# Plot the Predictions
f = plt.figure(figsize=(16, 8))
plt.scatter(GrLivArea_test, SalePrice_test, color = "green")
plt.scatter(GrLivArea_test, SalePrice_test_pred, color = "red")
plt.show()

# Explained Variance (R^2)
print("Explained Variance (R^2) \t:", linreg.score(GrLivArea_test, SalePrice_test))

# Mean Squared Error (MSE)
def mean_sq_err(actual, predicted):
    '''Returns the Mean Squared Error of actual and predicted values'''
    return np.mean(np.square(np.array(actual) - np.array(predicted)))

mse = mean_sq_err(SalePrice_test, SalePrice_test_pred)
print("Mean Squared Error (MSE) \t:", mse)
print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mse))

# Import mean_squared_error from sklearn
from sklearn.metrics import mean_squared_error

# Check the Goodness of Fit (on Train Data)
print("Goodness of Fit of Model \tTrain Dataset")
print("Explained Variance (R^2) \t:", linreg.score(GrLivArea_train, SalePrice_train))
print("Mean Squared Error (MSE) \t:", mean_squared_error(SalePrice_train, SalePrice_train_pred))
print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mean_squared_error(SalePrice_train, SalePrice_train_pred)))
print()

# Check the Goodness of Fit (on Test Data)
print("Goodness of Fit of Model \tTest Dataset")
print("Explained Variance (R^2) \t:", linreg.score(GrLivArea_test, SalePrice_test))
print("Mean Squared Error (MSE) \t:", mean_squared_error(SalePrice_test, SalePrice_test_pred))
print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mean_squared_error(SalePrice_test, SalePrice_test_pred)))
print()


# Problem 2

def function(X,y):
    # Import train_test_split from sklearn
    from sklearn.model_selection import train_test_split

    # Split the Dataset into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # Check the sample sizes
    print("Train Set :", y_train.shape, X_train.shape)
    print("Test Set  :", y_test.shape, X_test.shape)

    # Relationship between Response and the Predictors
    sb.pairplot(data = trainDF)

    # Linear Regression using Train Data
    linreg = LinearRegression()         # create the linear regression object
    linreg.fit(X_train, y_train)        # train the linear regression model

    # Coefficients of the Linear Regression line
    print('Intercept of Regression \t: b = ', linreg.intercept_)
    print('Coefficients of Regression \t: a = ', linreg.coef_)
    print()
    
    # Formula for Regression Line
    regline_x = GrLivArea_train
    regline_y = linreg.intercept_ + linreg.coef_ * GrLivArea_train
    # Plot the Linear Regression line
    f, axes = plt.subplots(1,1, figsize=(16, 8))
    plt.scatter(GrLivArea_train, SalePrice_train)
    plt.plot(regline_x.squeeze(), regline_y.squeeze(), "r-", linewidth = 3)
    plt.show()

    # Print the Coefficients against Predictors
    pd.DataFrame(list(zip(X_train.columns, linreg.coef_[0])), columns = ["Predictors", "Coefficients"])

    # Predict the Total values from Predictors
    y_train_pred = linreg.predict(X_train)
    y_test_pred = linreg.predict(X_test)

    f = plt.figure(figsize=(16, 8))
    plt.scatter(X_test, y_test, color = "green")
    plt.scatter(X_test, y_test_pred, color = "red")
    plt.show()

    # Check the Goodness of Fit (on Train Data)
    print("Goodness of Fit of Model \tTrain Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_train, y_train))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_train, y_train_pred))
    print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
    print()

    # Check the Goodness of Fit (on Test Data)
    print("Goodness of Fit of Model \tTest Dataset")
    print("Explained Variance (R^2) \t:", linreg.score(X_test, y_test))
    print("Mean Squared Error (MSE) \t:", mean_squared_error(y_test, y_test_pred))
    print("Root Mean Squared Error (RMSE) \t:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Extract Response and Predictors
y = pd.DataFrame(data["SalePrice"])
X = pd.DataFrame(data["TotalBsmtSF"])
function(X,y)

# Extract Response and Predictors
y = pd.DataFrame(data["SalePrice"])
X = pd.DataFrame(data["GarageArea"])
function(X,y)


# The model with GrLivArea is the best as its Root Mean Squared Error is the lowest among the 3 models

# Problem 3

data.describe()[['SalePrice', 'GrLivArea']]

SalePrice_q1=SalePrice.quantile(0.25)
SalePrice_q3=SalePrice.quantile(0.75)
SalePrice_IQR=SalePrice_q3-SalePrice_q1

GrLivArea_q1=GrLivArea.quantile(0.25)
GrLivArea_q3=GrLivArea.quantile(0.75)
GrLivArea_IQR=GrLivArea_q3-GrLivArea_q1

data = pd.read_csv('train.csv')
SalePrice = pd.DataFrame(data['SalePrice']) #response
LotArea = pd.DataFrame(data['LotArea']) 
GrLivArea = pd.DataFrame(data['GrLivArea']) #predictor
TotalBsmtSF = pd.DataFrame(data['TotalBsmtSF'])
GarageArea = pd.DataFrame(data['GarageArea'])

cleandata = data[(data['SalePrice']>int(SalePrice_q1-1.5*SalePrice_IQR)) & (data['SalePrice']<int(SalePrice_q3+1.5*SalePrice_IQR))]
cleandata = cleandata[((cleandata["GrLivArea"]>int(GrLivArea_q1-1.5*GrLivArea_IQR)) & (cleandata["GrLivArea"]<int(GrLivArea_q3+1.5*GrLivArea_IQR)))]
print("Outliers removed:", len(data)-len(cleandata))

# Extract Response and Predictors
y = pd.DataFrame(cleandata["SalePrice"])
X = pd.DataFrame(cleandata["GrLivArea"])
function(X,y)


# This model is better than the model attained in Problem 1, the root mean squared error is noticeably lower compared to the model used in Problem 1.
