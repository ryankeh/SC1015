# Basic Libraries
import os
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest

sb.set() # set the default Seaborn style for graphics
os.getcwd()

data = pd.read_csv('winequalityN.csv')
data.head()
data.describe()

#dropping null data
data = data.dropna()
data.info()

data.head()

# training the model
rng = np.random.RandomState(42)
model = IsolationForest(contamination=float(0.1),random_state=42)

anomaly_inputs = ['fixed acidity','volatile acidity','citric acid','pH', 'quality']
model.fit(data[anomaly_inputs])

print(model.get_params())

data['anomaly_scores'] = model.decision_function(data[anomaly_inputs])
data['anomaly'] = model.predict(data[anomaly_inputs])

#to isolate rows to analyse
data.loc[:, ['pH', 'quality','anomaly_scores','anomaly'] ]

def outlier_plot(data, outlier_method_name, x_var, y_var, 
                 xaxis_limits=[0,1], yaxis_limits=[0,1]):
    
    print(f'Outlier Method: {outlier_method_name}')
    
    # Create a dynamic title based on the method
    method = f'{outlier_method_name}_anomaly'
    
    # Print out key statistics
    print(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
    print(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
    print(f'Total Number of Values: {len(data)}')
    
    # Create the chart using seaborn
    g = sb.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
    g.map(sb.scatterplot, x_var, y_var)
    g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
    g.set(xlim=xaxis_limits, ylim=yaxis_limits)
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
    axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
    return g
  
  
outlier_plot(data, 'Isolation Forest', 'alcohol', 'quality', [8, 15], [0, 10]);

#look at how each factor identifies outliers individually
palette = ['#ff7f0e', '#1f77b4']
sb.pairplot(data, vars=anomaly_inputs, hue='anomaly', palette=palette)
