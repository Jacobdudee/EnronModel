# Identifying Fraud in the Enron Dataset: Building a POI Classifier

## Introduction

### Background

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. As a result of the ensuing trial, many of the employees emails had their emails and financial information released. 

##### Full project code is the poi_id.py file, while the write-up is the EnronFraudClassifier jupyter notebook/html/markdown files.

### Project Goal

This project seeks to classify former Enron employees as "Persons of Interest",or people that the authorities should interview in the investigation, based on the dataset described above. 

I will be using Python and Python's data analysis and machine learning libraries to accomplish this task. Python is a flexible, general purpose programming language with libraries that allow mining of all data, including text data, easy and efficient.

Machine learning is great for this task, as there are many strong classification algorithms that can seek patterns in the data that someone manually looking over might not, while being quicker as well.

This project was completed as part of my Data Analyst Nanodegree from Udacity. 

##### I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.

## Results 
#### Best Model
The best results came from using an AdaBoost Classifier using 10 features, with a learning rate of 1 and 35 estimators:

    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
              n_estimators=35, random_state=None)

    Accuracy: 0.88040	Precision: 0.56527	Recall: 0.44600	F1: 0.49860	F2: 0.46565
	Total predictions: 15000	True positives:  892	False positives:  686	False negatives: 1108	True negatives: 12314
  
#### Variables Used
	['deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred', 'deferred_income', 'expenses', 'exercised_stock_options', 'long_term_incentive', 'from_this_person_to_poi', 'perc_from_poi'])

## Conlcusion
An AdaBoost Classifier was built that was able to predict a person of interest in the Enron dataset, with 88% accuracy, 56.5% precision, and 44.6% recall. 

We were able to predict whether an employee was a person of interest to an extent. The model was right 88% of the time, but when it guessed someone was a person of interest, it was right slightly less than half the time (45%).
