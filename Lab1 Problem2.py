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
