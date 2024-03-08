# Classifier - Support Vector Machine

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import argparse

#Added argument parser to get dataset from command line
parser = argparse.ArgumentParser(description='Comparative study of support vector machine')
parser.add_argument('-d', metavar='dataset', required=True, dest='dataset', action='store', help='eg: diabetes.csv')
args = parser.parse_args()

# Import dataset
data_set = pd.read_csv(args.dataset)
data_x = data_set.iloc[:, :-1].values
data_y = data_set.iloc[:,8].values
data_set.describe()

# Data histograms - uncomment below 2 lines
#data_set.hist(bins = 50, figsize=(25, 25))
#plt.show()

# Missing data (0s) imputed with mean
from sklearn.impute import SimpleImputer as Imputer
imp = Imputer(missing_values=0, strategy='mean')
imp = imp.fit(data_x[:, 2:6])
data_x[:, 2:6] = imp.transform(data_x[:, 2:6])

# Training set and Test set splitting of dataset
from sklearn.model_selection import train_test_split
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data_x, data_y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
data_x_train = standScaler.fit_transform(data_x_train)
data_x_test = standScaler.transform(data_x_test)

# Training set classification using SVM - kernel = 'rbf' by default
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(data_x_train, data_y_train)

# Predict results for test set
prediction_y = classifier.predict(data_x_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(data_y_test, prediction_y)
print("Confusion Matrix")
print(cm)
print("")

# Accuracy Score
from sklearn.metrics import accuracy_score
accScore = round(accuracy_score(data_y_test, prediction_y) * 100, 3)
print("Accuracy Score")
print(str(accScore) + "%")
print("")

# Classification Report
from sklearn.metrics import classification_report as CR
print("Classification Report - Support Vector Machine")
print("")
print(CR(data_y_test, prediction_y))
print("")

# Find best model and best parameters using K-Fold Cross Validation using Grid Search
from sklearn.model_selection import GridSearchCV as CV
params = [
    { 'C':[1, 10, 100, 1000], 'kernel':['linear']},
    { 'C':[1, 10, 100, 1000], 'kernel':['rbf'], 
    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.10]}
]
gridSearch = CV(estimator = classifier, param_grid = params, scoring = 'accuracy', cv = 5)

gridSearch = gridSearch.fit(data_x_train, data_y_train)
bestAccuracy = round(gridSearch.best_score_ * 100, 3)

standardDev = gridSearch.cv_results_['std_test_score'][gridSearch.best_index_]
standardDev = round(standardDev * 100, 3)
bestParams = gridSearch.best_params_

print("Mean Accuracy Score:- " + str(bestAccuracy) + "%")
print("Standard Deviation:- " + str(standardDev) + "%")
print("Best Model Parameters:- ")
print(bestParams)

# Plotting ROC Curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(data_y_test, prediction_y)
roc_auc=auc(fpr, tpr)

plt.title('Curve - Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='SVM Classifier AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.axis([0, 1, 0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()