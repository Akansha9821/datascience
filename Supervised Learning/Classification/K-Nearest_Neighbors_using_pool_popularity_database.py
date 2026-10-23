import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("pool_popularity.csv")

df["pop"] = pd.to_numeric(df["pop"], errors="coerce")

df = df.dropna(subset=["pop"])

df["popularity_class"] = pd.cut(
    df["pop"],
    bins=[-np.inf, 0.3, 0.7, np.inf],
    labels=["Low", "Medium", "High"]
)

features = [
    "monoisotopic_mass",
    "log1p_substances",
    "log1p_pubmed",
    "log1p_patents"
]

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["popularity_class"]

valid = y.notna()
X, y = X.loc[valid], y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))