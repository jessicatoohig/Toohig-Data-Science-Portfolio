import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("Data/titanic-1.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum())
# Create a matplotlib figure and axis for the heatmap
fig, ax = plt.subplots()
# Plots a heatmap with missing values
sns.heatmap(df.isnull(), cmap = "viridis", cbar = False)
# Render the heatmap with a pyplot
st.pyplot(fig)
# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.write("Heatmap of Missing Values")


# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
# Let our user select a numeric column to work with.
column = st.selectbox("Choose a column to fill", 
                      df.select_dtypes(include = ['number']).columns)

#Provide options of rhow to handle missing data
method = st.radio("Choose a method", 
                  ['Original DF', "Drop Rows", "Drop Columns (>50% Missing)", "Impute Mean", 'Impute Median', 'Impute Zero'])

# Work on a copy of the DataFrame so the original data remains unchanged.
df_clean = df.copy()

# Apply the selected method to handle missing data.
if method == "Original DF":
        pass # Keep the data unchanged.
elif method == "Drop Rows":
    # Remove all rows that contain any missing values.
    df_clean = df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    # Drop columns where more than 50% of the values are missing.
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() >
0.5])
elif method == "Impute Mean":
    # Replace missing values in the selected column with the column's mean.
    df_clean[column] = df_clean[column].fillna(df[column].mean())
elif method == "Impute Median":
    # Replace missing values in the selected column with the column's median.
    df_clean[column] = df_clean[column].fillna(df[column].median())
elif method == "Impute Zero":
    # Replace missing values in the selected column with zero.
    df_clean[column] = df_clean[column].fillna(0)


# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
# Side-by-Side Visualization: Original vs. Cleaned Data
# ------------------------------------------------------------------------------
# Create two columns in the Streamlit layout for side-by-side comparison.
col1, col2 = st.columns(2)
# --- Original Data Visualization ---
with col1:
    st.subheader("Original Data Distribution")
    # Plot a histogram (with a KDE) for the selected column from the original DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"Original Distribution of {column}")
    st.pyplot(fig)
    st.subheader(f"{column}'s Original Stats")
    # Display statistical summary for the selected column.
    st.write(df[column].describe())

# --- Cleaned Data Visualization ---
with col2:
    st.subheader("Cleaned Data Distribution")
    # Plot a histogram (with a KDE) for the selected column from the cleaned DataFrame.
    fig, ax = plt.subplots()
    sns.histplot(df_clean[column], kde=True)
    plt.title(f"Distribution of {column} after {method}")
    st.pyplot(fig)
    st.subheader(f"{column}'s New Stats")
    # Display statistical summary for the cleaned data.
    st.write(df_clean[column].describe())

# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------

