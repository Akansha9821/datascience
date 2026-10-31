import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

np.random.seed(42)

data = {
    'Age': np.random.randint(21, 60, 1000),
    'Income': np.random.randint(20000, 150000, 1000),
    'Credit_Score': np.random.randint(300, 850, 1000),
    'Loan_Amount': np.random.randint(5000, 100000, 1000),
    'Employment': np.random.choice(
        ['Salaried', 'Self-Employed', 'Unemployed'], 1000
    ),
    'Education': np.random.choice(
        ['Graduate', 'Postgraduate', 'Undergraduate'], 1000
    )
}

df = pd.DataFrame(data)

df['Approved'] = (
    (df['Credit_Score'] > 650) &
    (df['Income'] > 40000) &
    (df['Loan_Amount'] < 70000)
).astype(int)

print(df.head())

# Dataset shape
print(df.shape)

# Dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Missing values
print(df.isnull().sum())

# Target distribution
print(df['Approved'].value_counts())

sns.countplot(data=df, x='Approved')

plt.title('Loan Approval Distribution')
plt.xlabel('Approval Status')
plt.ylabel('Count')

# plt.show()

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Matrix')
# plt.show()
X = df.drop('Approved', axis=1)

y = df['Approved']

numeric_features = [
    'Age',
    'Income',
    'Credit_Score',
    'Loan_Amount'
]

categorical_features = [
    'Employment',
    'Education'
]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])
print("Actual Values:", y_test.iloc[:10].values)

y_probability = model.predict_proba(X_test)

print(y_probability[:10])

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')

plt.show()

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc_score = roc_auc_score(y_test, y_prob)

plt.plot(
    fpr,
    tpr,
    label=f'Naive Bayes AUC = {auc_score:.3f}'
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()
plt.show()

print("ROC AUC:", auc_score)

import joblib

joblib.dump(model, 'naive_bayes_model.pkl')

loaded_model = joblib.load('naive_bayes_model.pkl')

prediction = loaded_model.predict(new_customer)

print(prediction)