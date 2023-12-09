
#importing basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


data0=pd.read_csv("C:\ML Model deployment\phishing.csv")
data0.head()



#Checking the shape of the dataset
#data0.shape

#Listing the features of the dataset
#data0.columns

#Information about the dataset
#data0.info()

"""## **4. Visualizing the data**


#Plotting the data distribution
#data0.hist(bins = 50,figsize = (15,15))
#plt.show()

#Correlation heatmap

plt.figure(figsize=(15,13))
sns.heatmap(data0.corr(), annot=True)
plt.show()

"""## **5. Data Preprocessing & EDA**


#data0.describe()

"""The above obtained result shows that the most of the data is made of 0's & 1's except 'Domain' & 'URL_Depth' columns. The Domain column doesnt have any significance to the machine learning model training. So dropping the *'Domain'* column from the dataset. """

#Dropping the Domain column
data = data0.drop(['class'], axis = 1).copy()

"""This leaves us with 16 features & a target column. The *'URL_Depth'* maximum value is 20. According to my understanding, there is no necessity to change this column."""

#checking the data for null or missing values
#data0.isnull().sum()



# shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
data = data0.sample(frac=1).reset_index(drop=True)
data.head()

data = data.drop('Index',axis=1)

#data.columns


## **6. Splitting the Data**


# Sepratating & assigning features and target columns to X & y
y = data['class']
X = data.drop('class',axis=1)
X.shape, y.shape

# Splitting the dataset into train and test sets: 80-20 split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 12)
X_train.shape, X_test.shape

## **7. Machine Learning Models & Training**


#importing packages
from sklearn.metrics import accuracy_score

# Creating holders to store the model performance results
ML_Model = []
acc_train = []
acc_test = []

#function to call for storing the results
def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))

### **7.1. Decision Tree Classifier**


# Decision Tree model 
from sklearn.tree import DecisionTreeClassifier

# instantiate the model 
tree = DecisionTreeClassifier(max_depth = 5)
# fit the model 
tree.fit(X_train, y_train)

#predicting the target value from the model for the samples
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)

"""**Performance Evaluation:**"""

#computing the accuracy of the model performance
acc_train_tree = accuracy_score(y_train,y_train_tree)
acc_test_tree = accuracy_score(y_test,y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))

#checking the feature improtance in the model
# plt.figure(figsize=(9,7))
# n_features = X_train.shape[1]
# plt.barh(range(n_features), tree.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), X_train.columns)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.show()

"""**Storing the results:**"""

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

### **7.2. Random Forest Classifier**


# Random Forest model
#from sklearn.ensemble import RandomForestClassifier

# instantiate the model
#forest = RandomForestClassifier(max_depth=5)

# fit the model 
#forest.fit(X_train, y_train)

#predicting the target value from the model for the samples
# y_test_forest = forest.predict(X_test)
# y_train_forest = forest.predict(X_train)

"""**Performance Evaluation:**"""

#computing the accuracy of the model performance
# acc_train_forest = accuracy_score(y_train,y_train_forest)
# acc_test_forest = accuracy_score(y_test,y_test_forest)

# print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
# print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))

#checking the feature improtance in the model
# plt.figure(figsize=(9,7))
# n_features = X_train.shape[1]
# plt.barh(range(n_features), forest.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), X_train.columns)
# plt.xlabel("Feature importance")
# plt.ylabel("Feature")
# plt.show()

"""**Storing the results:**"""

#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Random Forest', acc_train_forest, acc_test_forest)

## **8. Comparision of Models**


#creating dataframe
# results = pd.DataFrame({ 'ML Model': ML_Model,    
#     'Train Accuracy': acc_train,
#     'Test Accuracy': acc_test})
# results

# #Sorting the datafram on accuracy
# results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

"""For the above comparision, it is clear that the Random Forest Classifier works well with this dataset.

So, saving the model for future use.
"""

import pickle
path_to_file = "D:\ML Model deployment\model.pkl"
pickle.dump(tree, open(path_to_file, "wb"))

print("done")
print(X_train.shape)
print(X_test.shape)