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
st.markdown("Welcome to an interactive supervised learning platform. Use the buttons and sliders below and to the left to upload a dataset, choose a model, tune it, and interactively explore performance within your dataset. The TItanic dataset is preloaded as an example.")

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
    st.markdown("Through the input of a numeric dataframe, the correlation heatmap provides a visual of the strength and direction of variable pairs within a dataset. The output heatmap is organized through color shading, giving a quick insight into what patterns might be occuring, and moreover, waht variables are best to dive deeper into in the supervised machine learning modeles below. Each cell represents the correlation coefficient; number clsoer to 1 indicate a strong, positive, linear relationship, and numbers clsoer to -1 indicate a strong, negative, linear realtionship. Values near 0 indicate little to no linear relationship.")
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

    # Model Visualizations
    st.subheader("Model Visualization")

    # Logistic Regression
    if model == "Logistic Regression":
        st.write("Feature Coefficients")

        coef = user_model.coef_[0]
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": coef
        }).sort_values(by="Coefficient", key=abs, ascending=False)
    
        fig, ax = plt.subplots()
        sns.barplot(
            data=coef_df.head(10),
            x="Coefficient",
            y="Feature",
            ax=ax
        )
        ax.set_title("Top Feature Coefficients")
        st.pyplot(fig)
        st.markdown("The graph displays feature coefficients from the Logisitc Regression Model. The size of the feature coefficients is crucial for logistic regression, as we are performing classification, not predicting a continuous value. In this model, the inputs are the feature variables (X values), and the output is a probability between 0 and 1 that represents the likelihood of belonging to a specific class. This coeficcient plot is helpful for multi-feature datasets, and tells us what features matter most and whether they increase or decrease in probablity by computing a weighted combinaiton of the inputs and passing the results througha sigmoid function. A positive coefcient increases the liklihood of class, and a negative coefficient decreases the liklihood. Feature importance signals what data is driving the prediciton.")
    
    # KNN
    elif model == "KNN":
        st.write("K vs Accuracy")

        k_values = range(1, 11)
        acc_scores = []

        for k_val in k_values:
            knn_temp = KNeighborsClassifier(n_neighbors=k_val)
            knn_temp.fit(X_train, y_train)
            preds = knn_temp.predict(X_test)
            acc_scores.append(accuracy_score(y_test, preds))

        fig, ax = plt.subplots()
        ax.plot(k_values, acc_scores, marker='o')
        ax.set_xlabel("K")
        ax.set_ylabel("Accuracy")
        ax.set_title("K vs Accuracy")
        st.pyplot(fig)
        st.markdown("K-Nearest Neighbors (KNN) is used for classification, and it makes predicitons based on similarity between data points. The inputs are the feature variables (X values), and the output is a predicted class label. Rather than using a matemathical formula, the KNN model interprets training data and looks at the k closest points (neighbors) to a new observations based on Euclidean distance. Classes are assigned based on neighbors. Feature sccaling is important here because KNN works based on distance, so smaller values are moer sensitive to noise, and larger values create more general outputs.")
    
    # Decision Tree
    elif model == "Decision Tree":
        st.write("Decision Tree Visualization")

        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            user_model,
            feature_names=X.columns,
            filled=True,
            max_depth=3,  # keeps it readable
            ax=ax
        )
        st.pyplot(fig)
        st.markdown("Decision tree models classify data by splitting it into smaller groups based on feauture values. Furthermore, the feature values are the inputs (X), and the putput is the predicted class. The model selects the feature that best separates the data at each step, and create a series of decison rules that forms the tree-like estructure. Each internal node represents a decison based on a feature, and eahc branch represents the outcome of that decsion. The ends, or leaf nodes, are the final decsions. THe tree splits until reaching a stopping ocndiotn, such as maximum depth. It is important to remeber that if a tree is too deep, iit can overfit the data.")
    
    # Random Forest
    elif model == "Random Forest":
        st.write("Feature Importance (Top 10)")

        importance = user_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(
            data=feat_df.head(10),
            x="Importance",
            y="Feature",
            ax=ax
        )
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)
        st.markdown("The graph displays feature importance from the Random Forest Model. This shows which input variables (feature varibles, x), have the greatest influence on the model's predicitons, which is the output of a predicted class label. Suggested by the name, the random forest model biilds multiple decsions trees and analyzes how much each features reduces error/imporves the splits across the trees. The importance scoe is a measue of how much that features helps with accurate predicitons. Higher importance values have a stronger impacts on the final decsion, so the model relies more on them when predicitng the output.")
    
    # Model Performance 
    st.subheader("Model Performance")

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    st.metric("Accuracy", f"{acc:.2f}")
    st.markdown(" Accuracy is the simplest and most common measure of classification models, measuring the percentage of times that the model is correct. It is calculated by creating a ratio of correctly predicted data points to the total number of data points in the set.")

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))
    st. markdown("The classification report shows how well a classification model performs by separating is predicitons into groups of key evalution metrics for each class. The inputs are the predicted labels, and the true labels, and the outputs is a table that displays precision, recall, F1-score, and support for every class. The F1-score, specifically, is when β = 1 and combines both recall and precision. It behaves like an average but is close to the minimum value of the two through the property of the harmonic mean. The harmonic mean is like the average of two numbers, but is always smaller or equal to the average. Furthermore, the F1-score is defined as the harmonic mean between precision and recall, and measures if either are high, and alerts us if one is low. Recall is the proportion of correct predictions with a positive label, or how well the model does with false negatives. Precision is similar, considering only the data points with a true label and measuring how well a model does with false positives. With both metrics, you must define a goal in order to best interpret the results. Lastly, support counts the number of true examples of each class that were present in the test data. ")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt = 'd', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("The confusion matrix shows how well a classification model's prediciotns relfect true labels by breaking down into four categories. The inputs are the model's predicted classes as well as actual classes from the test set, and teh output is a matrix (typically binary), that counts how many prections fall into each of the four categories. True positives mean the model correctly predicted thepositive class, and true negatives are correct prediciotns of the negative class. On the other hand, false postives are when the model predicts postive but the true label is negative, and false negatives occur when the model prects a negative but the true label is positive. This way, we can see not only the accuracy of the model but also the types of mistakes a model is making. ")

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
        st.markdown("The ROC Curve evaluates how well a binary classification model seprates positive and negative classes. It examines predicted probabilities, and the inputs are the probability estimates for the positive class (predict_proba) and the true labels form the test set. ")
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
        st.markdown("The feature importance dataframe explains which input variables have teh strongest influence on a tree-based model's predictions. The inputs are the trained model as well as the feature matrix used during training. The model assigns each feature a mumerical values through the feature_importances_ attribute, which represents the feature's contributions to reducing ipurity in the tree-splitting process. The output is a DataFrame that is sorted from most to least important feature, which giver further insight into the decision-making behavior.")