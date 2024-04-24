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

# Exploring the dataset
import seaborn as sns
def draw_histograms(dataframe, features, rows, cols):
    fig = plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i+1)
        dataframe[feature].hist(bins=20, ax=ax, facecolor='midnightblue')
        ax.set_title(feature + " Distribution", color='DarkRed')

    fig.tight_layout()
    plt.show()

draw_histograms(dataset, dataset.columns, 6, 3)

dataset['num'].value_counts()
sns.countplot(x='num', data=dataset)

# Visualizing relationships between attributes (examples)

pd.crosstab(dataset['thal'], dataset['num']).plot(kind='bar')
plt.title('Bar chart for thal vs num')
plt.xlabel('thal')
plt.ylabel('num')
plt.show()

# Fit Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Saving the model to disk
import joblib
filename = 'Logistic_regression_model.pkl'
joblib.dump(classifier, filename)

# Predict the test set results
Y_pred = classifier.predict(X_test)

# Checking the accuracy for predicted results
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# Interpretation
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# ROC Curve
from sklearn.metrics import roc_auc_score, roc_curve
logit_roc_auc = roc_auc_score(Y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Prediction for new dataset
Newdataset = pd.read_csv('newdata.csv')
y_new = classifier.predict(Newdataset)

print("Predictions for new dataset:", y_new)
