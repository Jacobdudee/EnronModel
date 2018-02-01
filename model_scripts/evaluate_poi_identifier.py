#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_score

x_train,x_test, y_train,y_test = train_test_split(features,labels,random_state=42,test_size=0.3)

### it's all yours from here forward!  
from sklearn import tree

dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
print("Accuracy: ",dtree.score(x_test, y_test))

pred = dtree.predict(x_test)

print(confusion_matrix(y_test, pred))
print('Precision: ',recall_score(y_test, pred))
print("Recall: ",precision_score(y_test, pred))


