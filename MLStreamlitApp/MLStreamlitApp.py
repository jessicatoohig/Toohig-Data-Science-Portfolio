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

# Creating a Background Color Gradient
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #a1c4fd, #c2e9fb);
        }
    </style>
    """,
    unsafe_allow_html=True
)

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

# Initialize the csv_file so it always exists
csv_file = None

# Dataset: Titanic
if dataset == "Titanic Dataset":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Basic Data Cleansing
    df = df.drop(columns = ["Name", "Ticket", "Cabin"])
    df = df.dropna()
    st.success("Using Titanic dataset")

# Dataset: Choose your own
else: 
    csv_file = st.sidebar.file_uploader("Upload Your CSV", type = ["csv"])

if dataset == "Upload Your Own":   
    if csv_file is None:
        st.info("Upload a dataset to begin.")
        st.stop()

    try:
        df = pd.read_csv(csv_file)
        st.write("Here is a preview of your dataset:")
        st.dataframe(df.head())
    except: 
        st.error("There was an error reading the file. Please upload a valid CSV file.")
        st.stop()

# Choosing a Target Column
st.subheader("Choosing a Target Column")

if dataset == "Titanic Dataset":
    target = "Survived"
    st.write("Target Column: Survived.")
else:
    target = st.selectbox("Select a Target Column", df.columns)

# Separate the Predictor and Target Variables
X = df.drop(columns = [target])
y = df[target]

# Convert to categorical variables 
X = pd.get_dummies(X)

# Train Test Split

# Choose a test size
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 42)

# Scale the Features
scale = st.sidebar.checkbox("Apply Feature Scaling")
# If user checks box, code standardizes all numeric features
#Training data computes scaling prarmeters
# Test data is scaled using same parameters 
if scale:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Selecting a Supervised Learning Model 
model = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"])

# Adjusting the Hyperparameters 
if model == "Logistic Regression":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    user_model = LogisticRegression(C=C, max_iter=1000)

elif model == "KNN":
    k = st.sidebar.slider("K", 1, 10, 5)
    user_model = KNeighborsClassifier(n_neighbors=k)

elif model == "Decision Tree":
    depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    user_model = DecisionTreeClassifier(max_depth = depth)

elif model == "Random Forest":
    trees = st.sidebar.slider("Number of Trees", 10, 200, 100)
    depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    user_model = RandomForestClassifier(n_estimators=trees, max_depth=depth)

# Training the model

if st.button("Train Model"):
    user_model.fit(X_train, y_train)
    predictions = user_model.predict(X_test)

    st.subheader("Model Performance")

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    st.metric("Accuracy", f"{acc:.2f}")

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))

    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt = 'd', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    if len(np.unique(y)) == 2:
        st.subheader("ROC Curve")
        y_probs = user_model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr,tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.legend()
        st.pyplot(fig2)

    # Feature Importance
    if model in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importance = user_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending = False)
        st.dataframe(feat_df)