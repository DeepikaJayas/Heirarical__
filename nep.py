import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------
# App Title
# -------------------------------
st.title("🌍 Nepal Earthquake Hierarchical Clustering & Prediction")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv("nepal_earthquakes_1990_2026.csv")
df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Select Numeric Features
# -------------------------------
X = df.select_dtypes(include=[np.number])
X = X.fillna(X.mean())

st.subheader("🔢 Selected Numeric Features")
st.write(list(X.columns))

# -------------------------------
# Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Dendrogram
# -------------------------------
st.subheader("🌳 Dendrogram")

fig, ax = plt.subplots(figsize=(10, 5))
linked = linkage(X_scaled, method="ward")
dendrogram(linked, ax=ax)
ax.set_title("Hierarchical Clustering Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")
st.pyplot(fig)

# -------------------------------
# Clustering
# -------------------------------
st.subheader("🎯 Clustering")

n_clusters = st.slider(
    "Select Number of Clusters", 2, 10, 3
)

hc = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage="ward"
)

clusters = hc.fit_predict(X_scaled)
df["Cluster"] = clusters

st.subheader("📊 Clustered Data Preview")
st.dataframe(df.head())

# -------------------------------
# Cluster Visualization
# -------------------------------
st.subheader("📈 Cluster Visualization")

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap="rainbow"
)
ax2.set_xlabel(X.columns[0])
ax2.set_ylabel(X.columns[1])
ax2.set_title("Hierarchical Clustering Result")
st.pyplot(fig2)

# =====================================================
# 🔮 PREDICTION SECTION (NEW EARTHQUAKE DATA)
# =====================================================
st.subheader("🔮 Predict Cluster for New Earthquake Data")

input_data = []

for col in X.columns:
    value = st.number_input(f"Enter {col}", value=float(X[col].mean()))
    input_data.append(value)

if st.button("Predict Cluster"):
    input_array = np.array(input_data).reshape(1, -1)

    # Scale new input
    input_scaled = scaler.transform(input_array)

    # Find nearest cluster (manual approach)
    distances = []
    for i in range(n_clusters):
        cluster_points = X_scaled[clusters == i]
        centroid = cluster_points.mean(axis=0)
        dist = np.linalg.norm(input_scaled - centroid)
        distances.append(dist)

    predicted_cluster = np.argmin(distances)

    st.success(f"✅ Predicted Cluster: {predicted_cluster}")
