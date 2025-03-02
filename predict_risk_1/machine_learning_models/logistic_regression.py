# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kidney_disease2.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,24].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:,:24] = imputer.fit_transform(X[:,:24])

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.50,random_state=101)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# Save the scaler
scaler_file = "standard_scalar_logistic.pkl"
joblib.dump(sc_X, scaler_file)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Save the model
filename = 'Logistic_regression_model.pkl'
joblib.dump(classifier, filename)

# Predict the test set results
y_Class_pred = classifier.predict(X_test)

# Checking the accuracy for predicted results
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(Y_test,y_Class_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_Class_pred)

# Interpretation:
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_Class_pred))

# ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

## PREDICTION FOR NEW DATASET

Newdataset = pd.read_csv('newdata.csv')
# Assuming you have already scaled the new dataset using the saved scaler
Newdataset = sc_X.transform(Newdataset)
y_new = classifier.predict(Newdataset)
print("Predicted values for new dataset:", y_new)
