# --------- IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import dash
from dash import dcc, html

# ------------------------------------
# 1. LOAD DATA
# ------------------------------------

df = pd.read_csv("car_insurance_premium_dataset.csv")
print("First 5 rows:")
print(df.head())
print("last 5 rows:")
print(df.tail())
print("describe:")
print(df.describe())
print(df.info())
# ------------------------------------
# 2. SAVE DATA
# ------------------------------------
df.to_csv("exported_original_data.csv", index=False)

# ------------------------------------
# 3. DATA CLEANING
# ------------------------------------
df = pd.read_csv("car_insurance_premium_dataset.csv")
print("Missing values (isnull) before dropna:")
print(df.isnull().sum())
print("\nNon-missing values (notnull) before dropna:")
print(df.notnull().sum())
car_drop = df.dropna()
# Drop duplicate rows
car_drop = car_drop.drop_duplicates()
print("\nMissing values (isnull) after dropna and duplicates removed:")
print(car_drop.isnull().sum())
print("\nNon-missing values (notnull) after dropna and duplicates removed:")
print(car_drop.notnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)
numeric_cols = df.select_dtypes(include=np.number).columns
print(numeric_cols)

# Outlier removal
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print("IQR Outlier :",outlier)
# ------------------------------------
# 4. TRANSFORMATION
# ------------------------------------
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
columns=numeric_cols)
print("\n---StandardScaler---")
print(df.head())

scalers = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]),
columns=numeric_cols)
print("\n---MinMaxScaler---")
print(df.head())

# ------------------------------------
# 5. DESCRIPTIVE STATS
# ------------------------------------
print("\n----Descriptive Statistics:----")
print("mean:",df.mean())
print("median:",df.median())
print("mode:",df.mode())
print("minimum:",df.min())
print("maximum:",df.max())
print("standard deviation:",df.std())

# ------------------------------------
# 6. VISUALIZATION
# ------------------------------------
plt.figure()
df[numeric_cols[0]].hist()
plt.title("Histogram")
plt.show()

plt.figure()
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------
# 7. PROBABILITY ANALYSIS
# ------------------------------------
sns.histplot(df[numeric_cols[0]], kde=True)
plt.title("Probability Distribution")
plt.show()

# ------------------------------------
# 8. MODELING – kNN
# ------------------------------------
target_col = df.columns[-1]

X = df.drop(target_col, axis=1)
y = pd.qcut(df[target_col], q=2, labels=[0, 1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\n---KNeighborsClassifier:---")
print("k-NN Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------------
# 9. CLUSTERING – kMeans
# ------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

sns.scatterplot(
    x=X.iloc[:, 0],
    y=X.iloc[:, 1],
    hue=df["Cluster"]
)
plt.title("K-Means Clustering")
plt.show()

# ------------------------------------
# 10. DASH DASHBOARD
# ------------------------------------
app = dash.Dash(__name__)

fig = px.scatter(
    df,
    x=df.columns[0],
    y=df.columns[1],
    color="Cluster",
    title="Interactive Car Insurance Dashboard"
)

app.layout = html.Div([
    html.H1("Car Insurance Premium Dashboard"),
    dcc.Graph(figure=fig)
])

# ------------------------------------
#  RUN DASH APP
# ------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
print("\nEDA Project Completed Successfully")