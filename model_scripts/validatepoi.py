#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(features,labels,random_state=42,test_size=0.3)

### it's all yours from here forward!  
from sklearn import tree

dtree = tree.DecisionTreeClassifier()
dtree.fit(x_train, y_train)
print(dtree.score(x_test, y_test))