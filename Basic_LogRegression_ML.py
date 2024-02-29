import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Read in and preview the dataset
loans = pd.read_csv(r"C:\Users\Administrator\OneDrive\Datacamp_csv\loans.csv")
print(loans.head())

# Exploratory Data Analysis
loans.drop(columns=['loan_id'], inplace=True)
print(loans.head(3))
print(loans.info())

# Visualise distributions and relationships
sns.pairplot(data=loans, diag_kind='kde', hue='loan_status')
plt.show()

# Visualise correlation between variables
numeric_loans = loans[['applicant_income', 'coapplicant_income', 'loan_amount',
                       'loan_amount_term', 'credit_history', 'loan_status']]

plt.figure(figsize=(9, 5))
sns.heatmap(numeric_loans.corr(), annot=True)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Target frequency
loans['loan_status'].value_counts(normalize=True)

# Visualise class frequency by loan_status
for col in loans.columns[loans.dtypes == 'object']:
    sns.countplot(data=loans, x=col, hue='loan_status')
    plt.show()

# MODELING
# First model using loan amount
X = loans[['loan_amount']]
y = loans[['loan_status']]
y = np.array(y).reshape(-1)

# Train-test split and preview
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=77,
                                                    stratify=y)
print(X_train[:5], '\n', y_train[:5])

# Instantiate a log regression model
clf = LogisticRegression(random_state=77, max_iter=10000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred[:5])

# Accuracy
print(clf.score(X_test, y_test))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=['Rejected', 'Approved'])
disp.plot()
plt.show()

# FEATURE ENGINEERING
loans = pd.get_dummies(loans)
loans.head()

# Second model using loan_status
# Re-split into features and targets
X = loans.drop(columns=['loan_status'])
y = loans[['loan_status']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=77,
                                                    stratify=y)

clf = LogisticRegression(random_state=77)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(clf.score(X_test, y_test))

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=['Rejected', 'Approved'])
disp.plot()
plt.show()

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': clf.coef_.reshape(-1)
})
plt.figure(figsize=(9, 5))
sns.barplot(data=feature_importance, x='feature', y='importance')
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()
