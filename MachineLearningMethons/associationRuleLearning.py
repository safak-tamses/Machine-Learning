# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

from collections import Counter
import pandas as pd

c = ClusteringMethod
c.kMeansClustering()
# Data Preprocessing
dataset = pd.read_csv('../modelFiles/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

X = dataset.iloc[0:7500,0:20].values
y = dataset.iloc[:,]
# Training the Apriori model on the dataset
print(Counter(X).keys())
print(Counter(X).values())



# Visualising the results


## Displaying the results sorted by descending lifts
