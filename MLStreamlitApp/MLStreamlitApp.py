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
from sklearn.impute import SimpleImputer

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
    except Exception: 
        st.error("There was an error reading the file. Please upload a valid CSV file.")
        st.stop()

# Dataset Preview
st.subheader("Here is a preview of your dataset:")
st.dataframe(df.head())

# Adding a Correlation Heatmap
st.subheader("Correlation Heatmap")

# Use just the numeric columns
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 0:
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot = True, cmap = "coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
else:
    st.info("No numeric columns are available for a correlation heatmap.")

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

# Fixing NaN issues
mask = y.notna()
X = X[mask]
y = y[mask]

# Fill missing values
X = pd.get_dummies(X)
feature_names = X.columns                   # Save names before imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X = pd.DataFrame(X, columns=feature_names)

# Train Test Split

# Choose a test size
test_size = st.sidebar.slider("Test Size", 0.01, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 42)

# Scale the Features
# If user checks box, code standardizes all numeric features
# Training data computes scaling prarmeters
# Test data is scaled using same parameters 

scale = st.sidebar.checkbox("Apply Feature Scaling")

if scale:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Choosing a Model
model = st.sidebar.selectbox(
        "Choose a Model",
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    )
# Hyperparameters in sidebar
if model == "Logistic Regression":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    user_model = LogisticRegression(C=C, max_iter = 1000)

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

# Training and Evaluation

if st.button("Train Model"):

    # Training the Model
    user_model.fit(X_train, y_train)
    predictions = user_model.predict(X_test)

    # Model Performance 
    st.subheader("Model Performance")

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    st.metric("Accuracy", f"{acc:.2f}")

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt = 'd', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    if len(np.unique(y_test)) == 2 and hasattr(user_model, "predict_proba"):
        st.subheader("ROC Curve")
        y_probs = user_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("ROC Curve is available only for binary classification models with probability outputs.")

    # Feature Importance
    if model in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importance = user_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending = False)
        st.dataframe(feat_df)

        # selectdtypes for visulizations
        # show supervised learning visualizatioins