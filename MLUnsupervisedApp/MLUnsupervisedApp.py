# --------------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------------
# All necessary libraries are imported below, they provide the tools for:
# - data handling
# - visualization 
# - ML
# - Hierarchical clustering 

import streamlit as st                      # Builds the web app interface
import pandas as pd                         # Handles tabular data
import numpy as np                          # Numerical operations
import plotly.express as px                 # Interactive visualizations
import plotly.figure_factory as ff          # Advanced plots like dendrograms
import scipy.cluster.hierarchy as sch       # Imports hierarchical clustering 

# Machine learning tools
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram 

# --------------------------------------------------------------------------------------------------------
# PAGE CONFIGURATION: TITLE 
# --------------------------------------------------------------------------------------------------------
# Controls the app title, icon, and layout

st.set_page_config(
    page_title= "ML Unsupervised App",
    page_icon="📊",
    layout= "wide")

# --------------------------------------------------------------------------------------------------------
# FORMAT
# --------------------------------------------------------------------------------------------------------
# Styling is used below to improve visual appearance

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
# Explains what the app does (important for users)
st.title("📊 Unsupervised Machine Learning: Interactive Application")

st.write("""
        This app allows users to: 
         - Upload their own dataset or use the Titanic dataset
         - Select features (inputs)
         - Adjust model parameters (# of clusters)
         - Visualize clustering results
         
        Outputs: 
         - Cluster assignments
         - PCA visualization
         - Elbow plot
         - Silhouette score""")

# --------------------------------------------------------------------------------------------------------
# LOAD THE DATASET
# --------------------------------------------------------------------------------------------------------
# Input: none (functions fetches dataset from URL)
# Output: pandas DataFrame

@st.cache_data              # Speeds up app by caching dataset
def load_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

# --------------------------------------------------------------------------------------------------------
# CREATE A SIDEBAR: USER INPUTS
# --------------------------------------------------------------------------------------------------------
# The sidebar is where the user interacts with the app
st.sidebar.header("Settings")

# Create an expander that illustrates the interactive metrics in the sidebar 
with st.sidebar.expander("What do these parameters mean?"):
    st.markdown("""
                **Choose Dataset:**
                Select whether to use the built-in Titanic dataset or upload your own CSV file.
                
                **Features:**
                The columns are used as inputs for clustering. 
                Clustering results change significantly depending on which features you include.

                **PCA Components:**
                Number of principal components to compute.
                PCA reduces high-dimensional data into fewer dimensions.
                *Note:* The visualization always uses PC1 and PC2.

                **Number of Clusters (K):**
                Controls how many groups K-means will try to find in the data.
                Higher K = more, smaller clusters
                Lower K = fewer, broader clusters

                **Linkage Method (Hierarchical Clustering):**
                Determines how distances between clusters are calculated:
                - **Ward:** minimizes variance (most common)
                - **Single:** based on closest points (can chain)
                - **Complete:** based on farthest points (compact clusters)
                - **Average:** uses average distance between clusters

                **Explained Variance:**
                - Ratio of (variance captured by the specific principal component / Total variance in the dataset)
                - Tells you how much information each component keeps
                - You can choose the smallest number of components that explain the highest percentage of variance
                """)

# Button that reloads the entire site 
if st.sidebar.button("🔄 Refresh App"):
    st.rerun()

# User chooses a dataset source, either the Titanic dataset or uploads thier own
dataset_choice = st.sidebar.radio(
    "Choose Dataset",
    ["Titanic Sample Dataset", "Upload Your Own CSV"]
)

# Input: user selection
# Output: Dataframe (df)

if dataset_choice == "Titanic Sample Dataset":
    df = load_titanic()
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV", type = ["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Upload a CSV file to continue.")
        st.stop()

# --------------------------------------------------------------------------------------------------------
# CLEANING AND PREPROCESSING
# --------------------------------------------------------------------------------------------------------                                                     
# Goal: convert dataset into a numeric format usable for machine learning

df_clean = df.copy()

# Step 1: convert categorical columns to numeric, ML requires numbers, not text
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].fillna("Missing")
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Step 2: Fill missing numeric values with median 
df_clean = df_clean.fillna(df_clean.median(numeric_only = True))

# Step 3: Identify numeric columns 
numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

# --------------------------------------------------------------------------------------------------------
# FEATURE SELECTION (USER INPUT)
# --------------------------------------------------------------------------------------------------------
# User selects the columns to use for features 
st.sidebar.subheader("Features")
features = st.sidebar.multiselect(
    "Select columns for clustering",
    numeric_cols,
    default=numeric_cols[:4]
)

# User selects n_components 
n_components = st.sidebar.slider(
    "PCA Components",
    min_value = 2,
    max_value = 5,
    value = 2
)


# Ensure valid input of parameters
if n_components < 2:
    st.error("PCA requires at least 2 components")
    st.stop()

if len(features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

# Explain why n_components only changes explained variance
st.sidebar.info("PCA visualization uses only the first two components. Increasing n_components affects explained variance but does not change the 2D plot.")

# Input: number of clusters (k), k controls how many groups the algorithm will try to find in the data
k = st.sidebar.slider("Number of Clusters", 2, 10, 3)

linkage_method = st.sidebar.selectbox(
    "Linkage method (Hierarchical Clustering)",
    ["ward", "single", "complete", "average"]
)

# --------------------------------------------------------------------------------------------------------
# MODEL PREP
# --------------------------------------------------------------------------------------------------------
# Input: selected features
# Output: scaled numeric matrix

X = df_clean[features]

# StandardScaler: 
# - Centers data around the mean = 0
# - Scales variance to 1
# Why? Prevents features with large values from dominating 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------------------------------------------------------
# KMEANS
# --------------------------------------------------------------------------------------------------------
# Input: scaled data and number of clusters
# Output: cluster labels for each row

kmeans = KMeans(n_clusters=k,                   # number of clusters 
                random_state = 42,              # ensures reproductibilty (same result every run)
                n_init=10)                      # runs algorithm multiple times to find the best solution 
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels back to the orginal dataset
df_clean["Cluster"] = clusters

# Silhouette Score: 
# Measures how well separated clusters are
# Range: -1 to 1 (higher = better clustering)
if len(set(clusters)) > 1:
    score = silhouette_score(X_scaled, clusters)
else:
    score = 0

# --------------------------------------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------------------------------------
# Input: scaled data
# Output: 2D representation of high-dimensional data

pca = PCA(n_components = n_components)
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_

# Explained variance ratio
# Ratio: Variance captured by the specific principal component / Total variance in the dataset
st.sidebar.write("Explained Variance:", np.round(explained_var, 3))


# Create a DataFrame for plotting
plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": clusters.astype(str)
})

# --------------------------------------------------------------------------------------------------------
# CREATING TABS
# --------------------------------------------------------------------------------------------------------
# Separates content and visualizations into tabs 

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Dataset",
    "📈 KMeans Results",
    "📉 Elbow Plot",
    "🌳 Hierarchical",
    "⬇️ Download"
])

# --------------------------------------------------------------------------------------------------------
# TAB 1: DATASET
# --------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader('Dataset Preview')

    st.write("""
             This section displays the dataset currently being used for clustering.
             Each row represents an observation (e.g., a passenger in the Titanic dataset),
             and each column represents a feature (input variable).
             
             The selected features will be used as inputs into the clustering algorithm. 
             Choosing different features can significantly change the clustering results
             because the model groups data based on similarity across these variables.""")

    st.dataframe(df_clean.head(20), use_container_width=True)

    # Output: first 20 rows
    col1, col2, col3 = st.columns(3)

    # Summary metrics
    col1.metric("Rows", df_clean.shape[0])
    col2.metric("Columns", df_clean.shape[1])
    col3.metric("Selected Features", len(features))

# --------------------------------------------------------------------------------------------------------
# TAB 2: RESULTS
# --------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("KMeans Cluster Results")

    st.write("""
             This visualization shows the results of the K-Means clustering algorithm. 
             
             K-Means works by grouping data points in K clusters based on similarity.
             Each point is assigned to the nearest cluster center.
             
             The scatter plot below is a 2D representation created using PCA (Principal Component Analysis),
             which reduces high-dimensional data into two dimensions while preserving structure.
             
             - Each point represents an observation
             - Colors indicate cluster grouping
             - Points closer together are more similar
             
             The Silhouette Score measures how well-separated the clusters are:
             - Close to 1: well separated clusters
             - Around 0: overlapping clusters
             - Negative: poor clustering""")
    
    c1, c2 = st.columns(2)

    c1.metric("Clusters", k)
    c2.metric("Silhouette Score", round(score, 3))

    # Output: interactive scatter plot
    fig = px.scatter(
        plot_df, 
        x= "PC1",
        y = "PC2",
        color = "Cluster",
        title = "PCA Cluster Visualization",
        height = 600
    )

    st.plotly_chart(fig, use_container_width=True, key=f"pca_{n_components}_{k}")

# --------------------------------------------------------------------------------------------------------
# TAB 3: ELBOW
# --------------------------------------------------------------------------------------------------------
# Purpose: help user choose optimal K

with tab3: 
    st.subheader("Elbow Method")

    st.write("""
    The Elbow Method helps determine the optimal number of clusters (K).
             
    The plot shows: 
    - X-axis: number of clusters (K)
    - Y-axis: inertia (within-cluster distance)
             
    Inertia measures how tightly groups the data points are within each cluster.
    Lower values indicate better clustering. 
             
    The "elbow point" is where the rate of improvement simply decreases.
    This point shows a good balance between simplicity (fewer clusters) and accuracy (better grouping).
    
    Choosing K at the elbow avoids overfitting (too many clusters) and underfitting (too few clusters).
             """)
    inertia = []

    # Run KMeans for mulitple K values
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10) # explain this 
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    elbow_df = pd.DataFrame({
        "K": list(range(1, 11)),
        "Inertia": inertia
    })

    # Output: elbow plot
    fig2 = px.line(
        elbow_df,
        x="K",
        y="Inertia",
        markers = True, 
        title = "Elbow Plot"
    )

    st.plotly_chart(fig2, use_container_width=True, key=f"elbow_{k}")

# --------------------------------------------------------------------------------------------------------
# TAB 4: HIERACHICAL
# --------------------------------------------------------------------------------------------------------
# Alternative clustering method (tree-based)

with tab4: 
    st.subheader("Hierarchical Clustering Dendrogram")
    
    st.write("""
    This dendrogram represents hierarchical clustering, a different clustering approach compared to K-means.

    Instead of assigning points directly to clusters, hierarchical clustering: 
    - Starts by treating each data point as its own cluster
    - Iteratively merges the closest clusters together
    - Builds a tree-like structure (dendrogram)

    How to interpret the dendrogram:
    - Each merge represents clusters being combined
    - The height shows the distance between clusters
    - Larger vertical jumps suggest more distinct clusters   

    You can "cut" the tree at a certain height to decide how many clusters to form.""")

    # Limit size for performance
    sample_size = min(len(X_scaled), 100)
    sample_data = X_scaled[:sample_size]

    # Output: dendrogram (tree diagram)
    import scipy.cluster.hierarchy as sch       

    fig3 = ff.create_dendrogram(sample_data,
    linkagefun=lambda x: sch.linkage(x, method=linkage_method)
    )

    fig3.update_layout(width = 1000, height = 600)
   
    st.plotly_chart(fig3, use_container_width=True, key=f"hier_{linkage_method}")

# --------------------------------------------------------------------------------------------------------
# TAB 5: DOWNLOAD
# --------------------------------------------------------------------------------------------------------
with tab5: 
    st.subheader("Download Clustered Dataset")

    st.write("""
    This section allows you to download the dataset after clustering.
             
    A new column called "Cluster" has been added to the dataset:
    - Each value represents the cluster assignment for the observation
             
    This output can be used for: 
    - Further analysis
    - Reporting insights
    - Building downstream models
             
    Downloading the dataset allows you to work with the clustered results outside of this app.""")

    # Convert DataFrame to CSV
    csv = df_clean.to_csv(index=False).encode("utf-8")

    # Output: downloadable file
    st.download_button(
        label = "Download CSV",
        data= csv, 

        file_name= "clustered_dataset.csv",
        mime= "text/csv"
    )

    st.dataframe(df_clean.head(20), use_container_width=True)

# --------------------------------------------------------------------------------------------------------
# CONCLUSION
# --------------------------------------------------------------------------------------------------------
st.markdown("---")
st.caption("Thank you for interacting with the app.")
st.caption("Created with Streamlit | Data Science Portfolio Project")