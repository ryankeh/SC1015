# Basic Libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics

#dataset is in CSV format; hence we use the read_csv function from Pandas.
pkmndata = pd.read_csv('pokemonData.csv')
#take a quick look at the data using the head function.
pkmndata.head()

#check vital statistics of the dataset using the `type` and `shape` attributes.
print("Data type : ", type(pkmndata))
print("Data dims : ", pkmndata.shape)

#check variables (and their types) in the dataset using the dtypes attribute.
print(pkmndata.dtypes)

#extracting a single variable
#start by analyzing a single variable from the dataset, HP, extract the variable and its associated data as a Pandas DataFrame.
hp = pd.DataFrame(pkmndata['HP'])
print("Data type : ", type(hp))
print("Data dims : ", hp.size)
hp.head()

#Uni-Variate Statistics
#check the Summary Statistics of Uni-Variate Series using describe.
hp.describe()
#gives 8 data
#count - The number of not-empty values.
#mean - The average (mean) value.
#std - The standard deviation.
#min - the minimum value.
#25% - The 25% percentile*.
#50% - The 50% percentile*.
#75% - The 75% percentile*.
#max - the maximum value.

#check the Summary Statistics visually using a standard boxplot.
f = plt.figure(figsize=(24, 4))
sb.boxplot(data = hp, orient = "h")
#Extend the summary to visualize the complete distribution of the Series.
#The first visualization is a simple Histogram with automatic bin sizes.
#Bar Graph
f = plt.figure(figsize=(16, 8))
sb.histplot(data = hp)
#The second visualization is a simple Kernel Density Estimate (KDE).
#Line Graph
f = plt.figure(figsize=(16, 8))
sb.kdeplot(data = hp)
#Combination of both
f = plt.figure(figsize=(16, 8))
sb.histplot(data = hp, kde = True)
#Violin Plot combines boxplot with kernel density estimate.
f = plt.figure(figsize=(16, 8))
sb.violinplot(data = hp, orient = "h")


#Extract 2 variables
#analyze two variables from the dataset, HP vs Attack. Extract the two variables and their associated data as a Pandas DataFrame.
hp = pd.DataFrame(pkmndata['HP'])
attack = pd.DataFrame(pkmndata['Attack'])
#check the uni-variate Summary Statistics for each variable.
hp.describe()
attack.describe()

#Visualize the uni-variate Distributions of each variable independently.
#Set up matplotlib figure with three subplots
f, axes = plt.subplots(2, 3, figsize=(24, 12))
# Plot the basic uni-variate figures for HP
sb.boxplot(data = hp, orient = "h", ax = axes[0,0])
sb.histplot(data = hp, ax = axes[0,1])
sb.violinplot(data = hp, orient = "h", ax = axes[0,2])
#Plot the basic uni-variate figures for Attack
sb.boxplot(data = attack, orient = "h", ax = axes[1,0])
sb.histplot(data = attack, ax = axes[1,1])
sb.violinplot(data = attack, orient = "h", ax = axes[1,2])

#Create a joint dataframe by concatenating the two variables
jointDF = pd.concat([attack, hp], axis = 1).reindex(attack.index)
jointDF
#Draw jointplot of the two variables in the joined dataframe
sb.jointplot(data = jointDF, x = "Attack", y = "HP", height = 12)
# Calculate the correlation between the two columns/variables
jointDF.corr()
#Visualize the correlation matrix as a heatmap to gain a better insight.
sb.heatmap(jointDF.corr(), vmin = -1, vmax = 1, annot = True, fmt=".2f")
