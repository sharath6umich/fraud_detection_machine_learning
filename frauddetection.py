# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv(\creditcard.csv\)

# Success - Display the first record
display(data.head(n=1))

# Total number of records
n_records = len(data.index)

# Number of records with fraudulent transactions
n_fraud = len(data[data['Class'] != 0])

# Number of records with legitimate transactions
n_legitimate = len(data[data['Class'] == 0])

# Percentage of fraudulent transactions
fraud_percent = float(n_fraud)/n_records*100

# Print the results
print "Total number of transactions: {}".format(n_records)
print "Total number of fraudulent transactions: {}".format(n_fraud)
print "Total number of legitimate transactions: {}".format(n_legitimate)
print "Percentage of fraudulent transactions: {:.2f}%".format(fraud_percent)

# Extract labels out and separate from features
label_raw = data['Class']
features_raw = data.drop('Class', axis = 1)

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'Class' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

# Calculate accuracy

accuracy = greater_percent/100
precision = accuracy
recall = 1
# Calculate F-score using the formula above for beta = 0.5
fscore = (1 + 0.5**2) * precision *  recall / (0.5**2 * precision + recall)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
	'''
	inputs:
	   - learner: the learning algorithm to be trained and predicted on
	   - sample_size: the size of samples (number) to be drawn from training set
	   - X_train: features training set
	   - y_train: label training set
	   - X_test: features testing set
	   - y_test: label testing set
	'''
	
	results = {}
	
	# Fit the learner to the training data using slicing with 'sample_size'
	start = time() # Get start time
	learner.fit(X_train[:sample_size], y_train[:sample_size])
	end = time() # Get end time
	
	# Calculate the training time
	results['train_time'] = end - start
		
	# Get the predictions on the test set,
	#       then get predictions on the first 300 training samples
	start = time() # Get start time
	predictions_test = learner.predict(X_test)
	predictions_train = learner.predict(X_train[:300])
	end = time() # Get end time
	
	# Calculate the total prediction time
	results['pred_time'] = end - start
			
	# Compute accuracy on the first 300 training samples
	results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
		
	# Compute accuracy on test set
	results['acc_test'] = accuracy_score(y_test, predictions_test)
	
	# Compute F-score on the the first 300 training samples
	results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
		
	# Compute F-score on the test set
	results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
	   
	# Success
	print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
	
	# Return the results
	return results
	
# Import the supervised learning models from sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# Initialize the first iteration of three models
clf_B = KNeighborsClassifier(n_neighbors=3)
clf_C = LogisticRegression(random_state=0)
clf_D = AdaBoostClassifier()

# Initialize the second iteration of three models
clf_B = KNeighborsClassifier(n_neighbors=3)
clf_C = RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1)
clf_D = AdaBoostClassifier()
#RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)


# print 'Running AdaBoostClassifier to get reduced'

model_1 = clf_D.fit(X_train, y_train.values.ravel())
importances_1 = model_1.feature_importances_
print X_train.columns.values[(np.argsort(importances_1)[::-1])[:5]]
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances_1)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances_1)[::-1])[:5]]]


# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = len(X_train)/100
samples_10 = len(X_train)/10
samples_100 = len(X_train)

# Collect results on the learners
results = {}

print 'Running with all features'
for clf in [clf_D,clf_C,clf_B]:
	start_time = time()
	clf_name = clf.__class__.__name__
	results[clf_name] = {}
	for i, samples in enumerate([samples_1, samples_10, samples_100]):
		results[clf_name][i] = \\
		train_predict(clf, samples, X_train_reduced, y_train.values.ravel(), X_test_reduced, y_test.values.ravel())
		elapsed_time = time() - start_time
		print elapsed_time
		
# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

print 'Running AdaBoostClassifier to get reduced'

model_1 = clf_D.fit(X_train, y_train.values.ravel())
importances_1 = model_1.feature_importances_
print X_train.columns.values[(np.argsort(importances_1)[::-1])[:5]]
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances_1)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances_1)[::-1])[:5]]]


# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = len(X_train_reduced)/100
samples_10 = len(X_train_reduced)/10
samples_100 = len(X_train_reduced)

print 'Running with reduced'
for clf in [clf_B,clf_C,clf_D]:
    start_time = time()
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train_reduced, y_train.values.ravel(), X_test_reduced, y_test.values.ravel())
        elapsed_time = time() - start_time
        print elapsed_time
		
# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

		
# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = len(X_train_pca)/100
samples_10 = len(X_train_pca)/10
samples_100 = len(X_train_pca)

print 'Running PCA to get tranformed'
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(X_train)
print(pca.explained_variance_ratio_)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print 'Running with tranformed'
for clf in [clf_B,clf_C,clf_D]:
    start_time = time()
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train_pca, y_train.values.ravel(), X_test_pca, y_test.values.ravel())
        elapsed_time = time() - start_time
        print elapsed_time
		
# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# Initialize the classifier
clf = KNeighborsClassifier()

# Create the parameters list you wish to tune
parameters = { 'n_neighbors':[3,6,9], 'leaf_size': [10,30,60]}

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)
print X_train_reduced.shape
print y_train.values.ravel().shape
print X_test_reduced.shape
print y_test.values.ravel().shape

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train_reduced, y_train.values.ravel())

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train_reduced, y_train.values.ravel())).predict(X_test_reduced)
best_predictions = best_clf.predict(X_test_reduced)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test.values.ravel(), predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test.values.ravel(), predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test.values.ravel(), best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test.values.ravel(), best_predictions, beta = 0.5))

# Train the supervised model on the training set 
new_clf = AdaBoostClassifier()
model = new_clf.fit(X_train, y_train.values.ravel())

# Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train.values.ravel())