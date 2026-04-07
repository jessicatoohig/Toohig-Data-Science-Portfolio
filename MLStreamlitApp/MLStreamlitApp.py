# Setting up the Streamlit App Structure and Importing Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Title
st.title(" 📊 Supervised Machine Learning Project")
st.subheader("By: Jessica Toohig")
st.markdown("Upload a dataset, choose a model, tune it, and interactively explore performance within your dataset.")

# Create Sidebar Controls for Organization 
st.sidebar.header("⚙️ Configuration")

# Upload any Dataset
st.subheader("Upload a Dataset")

# Choose to use sample or upload own
st.markdown("The Titanic Dataset can be used as a sample.")
dataset = st.sidebar.selectbox("Choose a Dataset", ["Titanic Dataset", "Upload Your Own"])

# Dataset: Titanic
if dataset == "Titanic Dataset":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Basic Data Cleansing
    df - df.drop(columns = ["Name", "Ticket", "Cabin"])
    df = df.dropna()

    st.success("Using Titanic dataset")

# Dataset: Choose your own
elif dataset == "Upload Your Own":
    csv_file = st.sidebar.file_uploader("Upload Your CSV", type = ["csv"])

if csv_file:
    try:
        df = pd.read_csv(csv_file)
        st.write("Here is a preview of your dataset:")
        st.dataframe(df.head)
    except: 
        st.error("There was an error reading the file. Please upload a valid CSV file.")
        st.stop()
else:
    st.info("Upload a dataset to begin.")
    st.stop()

# Choosing a Target Column
st.subheader("Choosing a Target Column")

if dataset == "Titanic Dataset":
    target = "Survived"
    st.write("Target Column: Survived.")
else:
    target = st.slectbox("Select a Target Column", df.columns)

# Spearate the Predictor and Target Variables
X = df.drop(columns = [target])
y = df[target]

# Convert to categorical variables 
X = pd.get_dummies(X)

# Train Test Split

# Choose a test size
test_size = st.sidebar.slide("Test Size", 0.1, 0.5, 0.2)

X_train, X_test, y_train = train_test_split(X, y, test_size=test_size, random_state = 42)

# Scale the Features
scale = st.sidebar.checkbox("Apply Feature Scaling")
    # If user checks box, code standardizes all numeric features
    #Training data computes scaling prarmeters
    # Test data is scaled using same parameters 
if scale:
    scaler = StandardScaler()
    X_train = scalar.fit-transform(X_train)
    X_test = scalar.transform(X_test)
    