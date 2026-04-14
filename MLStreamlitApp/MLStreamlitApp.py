# --------------------------------------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------------------------------------
# All necessary libraries are imported below, they provide the tools for data handling, visualization, ML, and using Streamlit
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

# --------------------------------------------------------------------------------------------------------
# FORMAT
# --------------------------------------------------------------------------------------------------------
# Styling is used below to create a light blue background on the web application
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

# --------------------------------------------------------------------------------------------------------
# TITLE AND INTRODUCTION
# --------------------------------------------------------------------------------------------------------
st.title(" 📊 Supervised Machine Learning Project")            
st.subheader("By: Jessica Toohig")
st.markdown("Welcome to an interactive supervised learning platform. Use the buttons and sidebar controls to upload a dataset, choose a model, tune it, and interactively explore performance within your dataset. The Titanic dataset is preloaded as an example.")

# --------------------------------------------------------------------------------------------------------
# SIDEBAR CONFIG
# --------------------------------------------------------------------------------------------------------
# Users can use this sidebar to configure a dataset and model its settings
st.sidebar.header("⚙️ Configuration")

# --------------------------------------------------------------------------------------------------------
# DATASET SELECTION
# --------------------------------------------------------------------------------------------------------
# The dataset upload section allows the user to download and input a dataset onto the web applications, or use the Titanic Dataset as a sample
# The output is a variable storing the chosen dataset
st.subheader("Upload a Dataset")
st.markdown("The Titanic Dataset can be used as a sample.")
dataset = st.sidebar.selectbox("Choose a Dataset", ["Titanic Dataset", "Upload Your Own"])

# Dataset: Titanic
# If the user selects the Titanic Dataset, the CSV is loaded from a URL
if dataset == "Titanic Dataset":
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # Removes unnecessary columns, and outputs a cleaned DataFrame
    df = df.drop(columns = ["Name", "Ticket", "Cabin"])
    df = df.dropna()
    st.success("Using Titanic dataset")

# Dataset: Choose your own
# If the user decides to upload their own dataset, this block displays a file uploader for a CSV to be uploaded 
else: 
    csv_file = st.sidebar.file_uploader("Upload Your CSV", type = ["csv"])

# Reads the uploaded CSV file and puts it into a DataFrame, but if no file is uploaded, the app stops and prompts the user
if dataset == "Upload Your Own":   
    if csv_file is None:
        st.info("Upload a dataset to begin.")
        st.stop()
    try:
        df = pd.read_csv(csv_file)
    except Exception: 
        st.error("There was an error reading the file. Please upload a valid CSV file.")
        st.stop()

# --------------------------------------------------------------------------------------------------------
# INITIAL DATASET VISUALS
# --------------------------------------------------------------------------------------------------------
# A dataset preview to display the first five rows of the dataset, this way, the user can understand the structure
st.subheader("Here is a preview of your dataset:")
st.dataframe(df.head())
st.markdown("The above dataframe displays the first five rows of your dataset. This gives you a glance as to how the data is labeled and organized.")

# A correlation heatmap with only the numeric columns, the input is a numeric subset of the dataset, and the output is the heatmap visualization
st.subheader("Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 0:
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot = True, cmap = "coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
    st.markdown("Through the input of a numeric dataframe, the correlation heatmap provides a visual of the strength and direction of variable pairs within a dataset. The output heatmap is organized through color shading, giving a quick insight into what patterns might be occurring, and moreover, what variables are best to dive deeper into in the supervised machine learning models below. Each cell represents the correlation coefficient; number closer to 1 indicate a strong, positive, linear relationship, and numbers closer to -1 indicate a strong, negative, linear relationship. Values near 0 indicate little to no linear relationship.")
else:
    st.info("No numeric columns are available for a correlation heatmap.")

# --------------------------------------------------------------------------------------------------------
# TARGET COLUMNS
# --------------------------------------------------------------------------------------------------------
# The user can choose a target column for prediction using a dropdown selectbox
# The output is a variable that stores the selected target column
st.subheader("Choosing a Target Column")

if dataset == "Titanic Dataset":
    target = "Survived"
    st.write("Target Column: Survived.")
else:
    target = st.selectbox("Select a Target Column", df.columns)

# Separates the dataset into features (X) and the target variable (y)
X = df.drop(columns = [target])
y = df[target]

# This fixes NaN issues by removing rows where the target value is missing to provide clean training data
mask = y.notna()
X = X[mask]
y = y[mask]

# This converts categorical variables to numeric form, then fills missing values using mean imputation 
X = pd.get_dummies(X)
feature_names = X.columns                   # Save names before imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
X = pd.DataFrame(X, columns=feature_names)

# --------------------------------------------------------------------------------------------------------
# TRAINING, TESTING, AND SPLITTING DATA
# --------------------------------------------------------------------------------------------------------
# Choose a test size
test_size = st.sidebar.slider("Test Size", 0.01, 0.5, 0.2)

# Splits the data into training and testing sets; the input is the processed feature matrix and target vector, and the output is four datasets used for model training/evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 42)

# Scale the Features
# If user checks a box to apply feature scaling, the code standardizes the numeric features
# Training data computes scaling parameters, and test data is scaled using the same parameters 
# The output is scaled versions of X_train and X_test
scale = st.sidebar.checkbox("Apply Feature Scaling")

if scale:
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# --------------------------------------------------------------------------------------------------------
# SUPERVISED LEARNING MODELS
# --------------------------------------------------------------------------------------------------------
# The sidebar selectbox allows a user to choose a machine learning model
model = st.sidebar.selectbox(
        "Choose a Model",
        ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"]
    )

# --------------------------------------------------------------------------------------------------------
# CREATING HYPERPARAMETERS
# --------------------------------------------------------------------------------------------------------
# After choosing a model, sliders will appear that correlate with the model chosen to tune its hyperparameters
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

# --------------------------------------------------------------------------------------------------------
# TRAINING AND VISUALIZING THE MODELS
# --------------------------------------------------------------------------------------------------------

# When the user clicks the buttons, the selected model is trained using the training data, and outputs the trained model and its predictions

if st.button("Train Model"):

    # Training the Model
    user_model.fit(X_train, y_train)
    predictions = user_model.predict(X_test)

    # Model Visualizations
    st.subheader("Model Visualization")

    # Logistic Regression: The input is the learned coefficients, and the output is a bar chart showing the most influential features
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
        st.markdown("The graph displays feature coefficients from the Logistic Regression Model. The size of the feature coefficients is crucial for logistic regression, as we are performing classification, not predicting a continuous value. In this model, the inputs are the feature variables (X values), and the output is a probability between 0 and 1 that represents the likelihood of belonging to a specific class. This coefficient plot is helpful for multi-feature datasets, and tells us what features matter most and whether they increase or decrease in probability by computing a weighted combination of the inputs and passing the results through a sigmoid function. A positive coefficient increases the likelihood of class, and a negative coefficient decreases the likelihood. Feature importance signals what data is driving the prediction.")
    
    # KNN: the input is a range of k-values, and the output is a plot displaying how accuracy changes with different neighbor counts
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
        st.markdown("K-Nearest Neighbors (KNN) is used for classification, and it makes predictions based on similarity between data points. The inputs are the feature variables (X values), and the output is a predicted class label. Rather than using a mathematical formula, the KNN model interprets training data and looks at the k closest points (neighbors) to a new observations based on Euclidean distance. Classes are assigned based on neighbors. Feature scaling is important here because KNN works based on distance, so smaller values are more sensitive to noise, and larger values create more general outputs.")
    
    # Decision Tree: The input is the trained tree model, and the output is a visual diagram of the first few levels of the tree
    elif model == "Decision Tree":
        st.write("Decision Tree Visualization")

        from sklearn.tree import plot_tree

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(
            user_model,
            feature_names=X.columns,
            filled=True,
            max_depth=3,  
            ax=ax
        )
        st.pyplot(fig)
        st.markdown("Decision tree models classify data by splitting it into smaller groups based on feature values. Furthermore, the feature values are the inputs (X), and the output is the predicted class. The model selects the feature that best separates the data at each step, and create a series of decision rules that forms the tree-like structure. Each internal node represents a decision based on a feature, and each branch represents the outcome of that decision. The ends, or leaf nodes, are the final decisions. The tree splits until reaching a stopping condition, such as maximum depth. It is important to remember that if a tree is too deep, it can overfit the data.")
    
    # Random Forest: The input is the feature importance scores, and the output is a bar chart showing the top 10 most important features 
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
        st.markdown("The graph displays feature importance from the Random Forest Model. This shows which input variables (feature variables, x), have the greatest influence on the model's predictions, which is the output of a predicted class label. Suggested by the name, the random forest model builds multiple decisions trees and analyzes how much each feature reduces error/improves the splits across the trees. The importance score is a measure of how much that features helps with accurate predictions. Higher importance values have a stronger impacts on the final decision, so the model relies more on them when predicting the output.")

# --------------------------------------------------------------------------------------------------------
# PERFORMANCE METRICS AND VISUALIZATIONS
# --------------------------------------------------------------------------------------------------------
    # This section evaluates model performance through accuracy, a classification report, and a confusion matrix
    # The inputs are the true labels and predicted labels, and the outputs are performance metrics 
    st.subheader("Model Performance")

    # Accuracy
    acc = accuracy_score(y_test, predictions)
    st.metric("Accuracy", f"{acc:.2f}")
    st.markdown(" Accuracy is the simplest and most common measure of classification models, measuring the percentage of times that the model is correct. It is calculated by creating a ratio of correctly predicted data points to the total number of data points in the set.")

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, predictions))
    st. markdown("The classification report shows how well a classification model performs by separating its predictions into groups of key evaluation metrics for each class. The inputs are the predicted labels, and the true labels, and the outputs is a table that displays precision, recall, F1-score, and support for every class. The F1-score, specifically, is when β = 1 and combines both recall and precision. It behaves like an average but is close to the minimum value of the two through the property of the harmonic mean. The harmonic mean is like the average of two numbers, but is always smaller or equal to the average. Furthermore, the F1-score is defined as the harmonic mean between precision and recall, and measures if either are high, and alerts us if one is low. Recall is the proportion of correct predictions with a positive label, or how well the model does with false negatives. Precision is similar, considering only the data points with a true label and measuring how well a model does with false positives. With both metrics, you must define a goal in order to best interpret the results. Lastly, support counts the number of true examples of each class that were present in the test data. ")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.text(confusion_matrix(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt = 'd', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.markdown("The confusion matrix shows how well a classification model's predictions reflect true labels by breaking down into four categories. The inputs are the model's predicted classes as well as actual classes from the test set, and the output is a matrix (typically binary), that counts how many predictions fall into each of the four categories. True positives mean the model correctly predicted the positive class, and true negatives are correct predictions of the negative class. On the other hand, false positives are when the model predicts positive but the true label is negative, and false negatives occur when the model predicts a negative but the true label is positive. This way, we can see not only the accuracy of the model but also the types of mistakes a model is making. ")

    # If the model supports probability predictions and the task is binary classification, then an ROC curve is generated
    # The input is predicted probabilities, and the output is a plot showing the trade-off between true and false positive rates 
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
        st.markdown("The ROC Curve evaluates how well a binary classification model separates positive and negative classes. It examines predicted probabilities, and the inputs are the probability estimates for the positive class (predict_proba) and the true labels from the test set. ")
    else:
        st.info("ROC Curve is available only for binary classification models with probability outputs.")

    # For only the tree-based models, a table of feature importance values is displayed
    # The input is the model's internal importance scores, and the output is a sorted DataFrame 
    if model in ["Decision Tree", "Random Forest"]:
        st.subheader("Feature Importance")
        importance = user_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending = False)
        st.dataframe(feat_df)
        st.markdown("The feature importance dataframe explains which input variables have the strongest influence on a tree-based model's predictions. The inputs are the trained model as well as the feature matrix used during training. The model assigns each feature a numerical value through the feature_importances_ attribute, which represents the feature's contributions to reducing impurity in the tree-splitting process. The output is a DataFrame that is sorted from most to least important feature, which gives further insight into the decision-making behavior.")
# --------------------------------------------------------------------------------------------------------
# CONCLUSION
# --------------------------------------------------------------------------------------------------------
# Displays a closing message to the user
st.subheader("Conclusion")
st.markdown("Thank you for interacting with the project!")