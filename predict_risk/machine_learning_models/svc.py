# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('HealthData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

# Handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 11:13] = imputer.fit_transform(X[:, 11:13])

# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Save the scaler
import joblib
scaler_file = "standard_scalar.pkl"
joblib.dump(sc_X, scaler_file)

# Exploring the dataset
import seaborn as sns
sns.countplot(x='num', data=dataset)
dataset.num.value_counts()

# Support Vector Classification
from sklearn.svm import SVC

# Checking for different kernels
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    svc_classifier = SVC(kernel=kernel)
    svc_classifier.fit(X_train, Y_train)
    svc_scores.append(svc_classifier.score(X_test, Y_test))

colors = sns.color_palette('Set1', n_colors=len(kernels))
plt.bar(kernels, svc_scores, color=colors)
for i, score in enumerate(svc_scores):
    plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')

# Fitting SVM to the Training set
classifier = SVC(kernel='rbf', random_state=0, probability=True)
classifier.fit(X_train, Y_train)

# Save the model
filename = 'svc_model.pkl'
joblib.dump(classifier, filename)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

# Interpretation
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

# ROC Curve
from sklearn.metrics import roc_auc_score, roc_curve
logit_roc_auc = roc_auc_score(Y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='SVC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('SVC_ROC')
plt.show()

# Prediction for new dataset
Newdataset = pd.read_csv('newdata.csv')
y_new = classifier.predict(Newdataset)

print("Predictions for new dataset:", y_new)
