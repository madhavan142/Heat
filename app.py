import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Page title
st.title("🌸 Iris Dataset Analysis using Streamlit")

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
df = pd.concat([X, y], axis=1)

# Show dataset
st.subheader("📊 Iris Dataset")
st.dataframe(df)

# -------------------- Correlation Heatmap --------------------
st.subheader("🔥 Correlation Heatmap")

fig1, ax1 = plt.subplots(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
st.pyplot(fig1)

# -------------------- Model Training --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# -------------------- Confusion Matrix --------------------
st.subheader("📉 Confusion Matrix Heatmap")

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=ax2
)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# -------------------- Classification Report --------------------
st.subheader("📄 Classification Report")
st.text(classification_report(y_test, y_pred))
