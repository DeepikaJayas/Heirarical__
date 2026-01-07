# ---------- IMPORTS ----------
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------- CSV PATH (PASTE HERE) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "nepal_earthquakes_1990_2026.csv")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

# ---------- USE DATA ----------
df = load_data()
