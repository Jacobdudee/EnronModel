#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#import seaborn as sns

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments',
                                 'total_payments', 'loan_advances', 
                                 'bonus', 'restricted_stock_deferred',
                                 'deferred_income', 'total_stock_value',
                                 'expenses', 'exercised_stock_options', 
                                 'long_term_incentive', 'restricted_stock',
                                 'director_fees', 'from_poi_to_this_person',
                                 'from_this_person_to_poi',
                                 'shared_receipt_with_poi','to_messages',
                'from_messages'] # You will need to use more features
print(' ')   
print("Features in dataset: ")
print(features_list)
print("# of Features: ")
print(len(features_list))
print(' ')

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")

#### Any missing values?

# the code below prints the percentage of non-null values for each feature
print('Percentage of non-null values for each feature:') 
enron_df = pd.DataFrame.from_dict(data_dict,orient='index')
for col in enron_df.columns:
    if str(enron_df[col].dtype) != 'bool':
        print col,(enron_df[enron_df[col] !='NaN'][col].count()) \
        ,round((enron_df[enron_df[col] !='NaN'][col].count())/ \
        (float(enron_df.shape[0])),2)

#### Explore Dataset
print(' ')
print("Data Exploration")
print(' ')

# How many poi are there in this dataset?
print("number of poi: ",enron_df[enron_df['poi']==True]['poi'].count())

# how many people are there in this dataset?
print("Number of people:",len(data_dict.keys()))

# how many features per person?
print("Number of features/person:",len(data_dict.values()[1]))

#what is the relationship between salary and long term incentive?
print("what is the relationship between salary and long term incentive?")

#creating plot
fig, ax = plt.subplots(1,1)
c_map = {1: 'r', 0: 'b'}
ax.scatter(enron_df['salary'],enron_df['long_term_incentive'], c=[c_map[i] for i in enron_df['poi']])
plt.ylabel("Long Term Incentive ($)")
plt.xlabel("Salary ($)")
plt.title("Salary vs. Long Term Incentive")

not_poi = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=15, label='Not POI')
poi = mlines.Line2D([], [], color='red', marker='o',
                          markersize=15, label='POI')
plt.legend(handles=[not_poi,poi])

#ax.legend(c_map,['poi','Not poi'])
#legend = ax.get_legend()
plt.show()
print(' ')

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

# I will create 2 variables: a proportion of messages sent to POI, and the proportion of messages recieved from poi
for key in my_dataset:
    empl = my_dataset[key]
    perc_to_poi, perc_from_poi = [] , []
    if (empl['from_this_person_to_poi'] != 'NaN') & (empl['from_messages'] != 'NaN'):
        perc_to_poi = float(empl['from_this_person_to_poi'])/float(empl['from_messages'])
        empl['perc_to_poi'] = perc_to_poi
    else:
        empl['perc_to_poi'] = 0
    
    if (empl['from_poi_to_this_person'] != 'NaN') & (empl['to_messages'] != 'NaN'):
        perc_to_poi = float(empl['from_poi_to_this_person'])/float(empl['to_messages'])
        empl['perc_from_poi'] = perc_to_poi
    else:
        empl['perc_from_poi'] = 0

# Adding the created features
features_list = features_list + ['perc_from_poi',
                                 'perc_to_poi']
    
    
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Scaling features- not used in final model
#scaler = MinMaxScaler()
#from sklearn.preprocessing import RobustScaler
#scaler = RobustScaler()
#features = scaler.fit_transform(features)

# Select 10 best features using k-best
k_best = SelectKBest(k=11)
k_best.fit(features, labels)

#creating dictionary with feature names and scores
d = {'feature_names':features_list[1:],"scores":k_best.scores_}

#creating dataframe with scores for cleaner presentation
kscores = pd.DataFrame(data=d).sort_values("scores",ascending=False)
print(' ')
print("SelectKbest scores: ")
print(kscores)
print(' ')
# keeping only the k-best features
k_best = [features_list[i] for i in k_best.get_support(indices=True)]
#k_best = list(kscores['feature_names'][0:6])

#k_best = k_best.insert(0,'poi')
#k_best = ['poi'] + k_best
print(' ')
print('k-best features:',k_best)
print(' ')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

def scoreCLF(clf,features,labels,param_grid,cv,score):
    '''
    This function takes a classifier, features, target variable, paramter grid (for grid search) 
    and fits and scores the resulting model.
    A confusion matrix and classification report is returned 
    with the performance metrics.
    Takes inspiration from:  http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
    And the know how to tune gridsearch from:
    https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn
    '''
    #importing relevant libraries
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score
   
    skf = StratifiedKFold(n_splits=cv)
    
    # splitting the data
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
    if param_grid != None:
        #creating and initializing the grid search
        clf = GridSearchCV(clf, param_grid=param_grid, cv=skf,
                           scoring=score,
                           return_train_score=True, n_jobs=-1)
        clf.fit(features_train, labels_train)
     
        # printing best parameters
        print(' ')
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print(' ')
        bestP = clf.best_params_
    else:
        clf.fit(features_train, labels_train)
        bestP= []

    # getting predictions and confusion matrix
    pred = clf.predict(features_test)
    print(classification_report(labels_test, pred))
    print(' ')
    return bestP
    
#clf = LogisticRegression()
#params = {'C':[0.001,0.01,0.1,1,10,100]}

#Accuracy: 0.78447	Precision: 0.17467	Recall: 0.16550	F1: 0.16996	F2: 0.16726

#clf = GaussianNB()
# original performance (no tuning, or feature engineering):
#Accuracy: 0.75780	Precision: 0.25692	Recall: 0.43150	F1: 0.32208	F2: 0.37987

#clf = SVC()

clf = AdaBoostClassifier()
params = [{'algorithm':['SAMME'],
         'n_estimators':[25,35,40,50,75,100],
             'learning_rate':[0.1,0.25,0.5,0.75,1]},
          {'algorithm':['SAMME.R'],
         'n_estimators':[25,35,40,50,75,100],
             'learning_rate':[0.1,0.25,0.5,0.75,1]}]
#Accuracy: 0.84980	Precision: 0.41412	Recall: 0.30500	F1: 0.35128	F2: 0.32197

#clf = tree.DecisionTreeClassifier() not used

# Original results, no modification: 	Accuracy: 0.79727	Precision: 0.22905	Recall: 0.22000	F1: 0.22443	F2: 0.22175

#clf = RandomForestClassifier()
#params = params = {
#    'min_samples_split': [5, 10,15], 
#    'n_estimators' : [10,30,50,100],
#    'max_depth': [3, 5, 15, 25],
#    'max_features': [3, 5, 10, 20]}

#	Accuracy: 0.85580	Precision: 0.37364	Recall: 0.12050	F1: 0.18223	F2: 0.13939

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
    

#get final parameters and apply to classifier
bestParams = scoreCLF(clf,features,labels,params,10,"recall")
clf = AdaBoostClassifier(**bestParams)
#clf = RandomForestClassifier(**bestParams)
#clf = LogisticRegression(**bestParams)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, k_best)