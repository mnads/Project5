#!/usr/bin/python

import sys
import pickle
import pprint
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','total_stock_value'] # You will need to use more features


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#pprint.pprint(data_dict) #preview the data uncomment if needed
print "Count of people in dataset: ", len(data_dict) 
print "Count of features for each person: ",len(data_dict.values()[0].keys())
pprint.pprint(data_dict.values()[0].keys())#View all features
'''
#Features for reference:
financial_features= ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] #all units are in US dollars

email_features= ['from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi','to_messages', 'email_address', 'from_messages', , 'poi'] #numbers of emails messages email_address is a text string
'''
### Task 2: Remove outliers
#pprint.pprint(data_dict.keys())#To examine names of outliers that did not belong by name. There might be some other ones that don't belong to be discovered later.
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')#not a person
data_dict.pop('TOTAL')#is just a summary
data_dict.pop('LOCKHART EUGENE E')#this person has all NaN features
#pprint.pprint(data_dict.keys())#recheck that names were removed from dictionary

### Task 3: Create new feature(s)
for k, i in data_dict.iteritems():
	if i['salary']=="NaN" and i['total_stock_value']=="NaN":
		i['xsalary_to_stockvalue']="NaN"	
	else :#New Feature xsalary_to_value
		i['xsalary_to_stockvalue']=float(i['salary'])/float(i['total_stock_value'])
		
	if i['from_poi_to_this_person']=="NaN" and i['from_this_person_to_poi']=="NaN":
		i['xfrom_to_poi']="NaN"	
	else :#New Feature xfrom_to_poi
		i['xfrom_to_poi']=float(i['from_poi_to_this_person'])/float(i['from_this_person_to_poi']+1)
		
	if i['from_this_person_to_poi']=="NaN" and i['shared_receipt_with_poi']=="NaN":
		i['xto_receipt_poi']="NaN"	
	else :#New Feature xto_receipt_poi
		i['xto_receipt_poi']=float(i['from_this_person_to_poi'])/float(i['shared_receipt_with_poi'])		

	if i['total_payments']=="NaN" and i['total_stock_value']=="NaN":
		i['xpayout']="NaN"	
	else :#New Feature xpayout
		i['xpayout']=float(i['total_payments'])+float(i['total_stock_value'])
		
nfeatures_list=features_list+['xsalary_to_stockvalue','xfrom_to_poi','xto_receipt_poi','xpayout']		
pprint.pprint(data_dict)#to see some example values

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
clf.fit(features, labels)
pred = clf.predict(features)
accuracy = accuracy_score(pred, labels)
print "\nGaussianNB ACCURACY=", accuracy, "\n"

'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pred=clf.predict(features)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels)
print "Decision Tree ACCURACY:",acc
#hold out 30 percent of data for testing and set random state number to 42. Decreased precision down to .24186
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
print "Decision Tree ACCURACY after removing 30%:", clf.score(features_test,labels_test),"\n"
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall using our testing script.
### Because of the small size of the dataset, the script uses stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)