# --------------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------------
# All necessary libraries are imported below, they provide the tools for data handling, visualization, ML, and using Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram 

# --------------------------------------------------------------------------------------------------------
# PAGE CONFIGURATION: TITLE 
# --------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title= "ML Unsupervised App",
    page_icon="📊",
    layout= "wide")

# --------------------------------------------------------------------------------------------------------
# FORMAT
# --------------------------------------------------------------------------------------------------------
# Styling is used below to create ...
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
h1, h2, h3 {
    color: #1f4e79;
}
div.stMetric {
    background-color: #f4f6f8;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------------------
# TITLE 
# --------------------------------------------------------------------------------------------------------
st.title("📊 Unsupervised Machine Learning: Interactive Application")
st.write("Upload a dataset ot use Titanic sample data to explore clustering techniques. Experiment with hyperparameters and visualize results.")

# --------------------------------------------------------------------------------------------------------
# LOAD THE DATASET
# --------------------------------------------------------------------------------------------------------
@st.cache_data
def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

# --------------------------------------------------------------------------------------------------------
# CREATE A SIDEBAR
# --------------------------------------------------------------------------------------------------------
st.sidebar.header("Settings")

dataset_choice = st.sidebar.radio(
    "Choose Dataset",
    ["Titanic Sample Dataset", "Upload Your Own CSV"]
)

if dataset_choice == "Titanic Sample Dataset":
    df = load_titanic 
else:
    uploaded_file = st.sidebar.fileuploader("Upload a CSV", type = ["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Upload a CSV file to continue.")
        st.stop()

# --------------------------------------------------------------------------------------------------------
# CLEANING AND PREPROCESSING
# --------------------------------------------------------------------------------------------------------                                                     
df_clean = df.copy()

for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].fillna("Missing")
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

df_clean = df_clean.fillna(df_clean.median(numeric_only = True))

numeric_cols = df_clean.slect.dtypes(include=np.number).columns.tolist()

st.sidebar.subheader("Features")
features = st.sidebar.multiselect(
    "Select columns for clustering",
    numeric_cols,
    default=numeric_cols[:4]
)

if len(features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

k = st.sidebar.slider("Number of Clusers", 2, 10, 3)
# Make sure to explain this

# --------------------------------------------------------------------------------------------------------
# MODEL PREP
# --------------------------------------------------------------------------------------------------------
X = df_clean[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------------------------------------------------------
# KMEANS
# --------------------------------------------------------------------------------------------------------
kmeans = KMeans(n_clusters=k, random_state = 42, n_init=10) # explain this
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

score = silhouette_score(X_scaled, clusters)

# --------------------------------------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------------------------------------
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": clusters.astype(str)
})