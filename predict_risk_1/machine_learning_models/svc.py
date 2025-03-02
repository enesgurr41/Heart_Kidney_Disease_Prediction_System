# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
# Importing the dataset
dataset = pd.read_csv('kidney_disease2.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,24].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X = imputer.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50, random_state = 101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler
scaler_file = "standard_scalar_svc.pkl"
joblib.dump(sc, scaler_file)

# Exploring the Dataset
import seaborn as sn
sn.countplot(x='classification', data=dataset)
dataset['classification'].value_counts()

## Support Vector Classifications

## Checking for different kernels
from sklearn.svm import SVC
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svc_classifier = SVC(kernel=kernel, probability=True)
    svc_classifier.fit(X_train, y_train)
    svc_scores.append(svc_classifier.score(X_test, y_test))

# Plotting the scores
plt.bar(kernels, svc_scores)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')
plt.show()

# Fitting SVM to the Training set
classifier = SVC(kernel='rbf', random_state=0, probability=True)
classifier.fit(X_train, y_train)

# Save the model
filename = 'svc_model.pkl'
joblib.dump(classifier, filename)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Checking the accuracy for predicted results
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Interpretation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('SVC_ROC')
plt.show()

## PREDICTION FOR NEW DATASET
Newdataset = pd.read_csv('newdata.csv')
Newdataset = sc.transform(Newdataset)
y_new = classifier.predict(Newdataset)
print("Predicted values for new dataset:", y_new)
