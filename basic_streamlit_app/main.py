# --------------------------------------------------------------------------------------------------------
# Project 1: Palmer's Penguins
# Streamlit App
# --------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# --------------------------------------------------------------------------------------------------------
import streamlit as st                  # Creates the streamlit wep app interface
import pandas as pd                     # Used for reading / analyzing data
import matplotlib.pyplot as plt         # Used for cusotm charts / visualizations 

# --------------------------------------------------------------------------------------------------------
# APP TITLE AND INTRODUCTION
# --------------------------------------------------------------------------------------------------------
# Main title at the top of page
st.title("Project 1: Palmer's Penguins")
# Subheader and author name        
st.subheader("By: Jessica Toohig")
# Introductory description of the project 
st.write("This project explores the Palmer's Penguins dataset through interactive visualizations and user-driven filtering. Using Streamlit, users can examine penguin characteristics across islands, species, and years to discern patterns in size and distribution. The project demonstrates exploratory data analysis, data visualization, and the use of interactivity to communicate insights effectively.")

# --------------------------------------------------------------------------------------------------------
# LOADING THE DATASET
# --------------------------------------------------------------------------------------------------------
# Importing the CSV file and Creating the Dataframe
st.subheader("Exploring the Dataset")                 # Title
df = pd.read_csv("data1/penguins.csv")                # Input: reads the csv file from folder data1/penguins.csv
st.write("Here's our data!")                          
st.dataframe(df)                                      # Output: creates a pandas dataframe named df

# --------------------------------------------------------------------------------------------------------
# FILTERING THE DATA BY ISLAND
# --------------------------------------------------------------------------------------------------------
st.subheader("Filter by an Island")
# Creating a Select Box
island = st.selectbox("Select an island", df["island"].unique(), index = None)
    # Input: user selects one island form the dropdown menu, options are taken from the names in the dataset
    # Output: user can select "Biscoe", "Dream", or "Torgersen"

# --------------------------------------------------------------------------------------------------------
# DISPLAYING THE FILTERED TABLE
# --------------------------------------------------------------------------------------------------------
# If the user clicks the button:
if st.button("Click me!"):
    # Filter rows where the isalnd matches user choice
    filtered_df = df[df["island"] == island]
    # Output: text and the filters table
    st.write(f"Penguins found on {island}.")
    st.dataframe(filtered_df)
else:
    # What is shown before the button is clicked 
    st.write("Click the button to display the filtered dataframe.")

# --------------------------------------------------------------------------------------------------------
# ADDING BAR CHARTS
# --------------------------------------------------------------------------------------------------------
# Filter the data using the slected island
filtered_df = df[df["island"] == island]
# Chart 1: counts the number of penguins by species 
st.bar_chart(filtered_df["species"].value_counts(), x_label= "Species", y_label = "# of Penguins")
    # Output: a bar chart showing species totals on the seclected island

# Chart 2: counts the number of penguins by sex
st.bar_chart(filtered_df["sex"].value_counts(), x_label= "Sex", y_label = "# of Penguins")
    # Output: a bar chart showing the count of male and female penguins on the selected island

# Key takeaway paragraph 
st.write("Male and female penguins are evenly distributed across all three islands. However, the Adelie penguin is the only species on all three islands. Use the select box below to find out more about the Adelie's metrics, as well as the other species.")

# --------------------------------------------------------------------------------------------------------
# FILTERING DATA BY SPECIES
# --------------------------------------------------------------------------------------------------------
st.subheader("Filter by a Species")

# Input: user slects a species from a drop down menu
species = st.selectbox("Select a species", df["species"].unique())          # Creates a selectbox 
filtered_df2 = df[df["species"] == species]                                 # Filters rows for the selected species 
# Output: a DataFrame with only the selcted species

# --------------------------------------------------------------------------------------------------------
# ADDING A BOXPLOT 
# --------------------------------------------------------------------------------------------------------
# Creating the measurement options
measurement = st.selectbox("Choose a measurement",["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    # Outputs: which ever option the user chooses out of "bill_length_mm", "bill_depth_mm", "flipper_length_mm", and "body_mass_g"

# Making the boxplot, by year 

# Create matplotlib figure and axis
fig, ax = plt.subplots()

# Boxplot: shows the spread of the selcted measurement grouped by year 
filtered_df2.boxplot(column=measurement, by="year", ax=ax)

# Axis labels
ax.set_xlabel("Year")
ax.set_ylabel(measurement.replace("_", " ").title())

# Dynamic chart title
ax.set_title(f"{measurement.replace('_', ' ').title()} by Year")

# Output: a boxplot displayed in Streamlit 
st.pyplot(fig)

# Key Takeaways and interprestation paragraphs of eaach box plot option
st.subheader("Key Takeaways from Average Species Metrics") # Title
# Adelie speices description 
st.markdown("**Adelie:** Bill length is steady throughout the years, with the most variety in 2008. Bill depth decreases slightly, whereas flipper length increases. Body mass is also relatively steady.")
# Gentoo species description
st.markdown("**Gentoo:** Bill length decreases then increases. Bill depth increases, and also grows in range. Flipper length has an initial increase, then decreases. Body mass is steady.")
# Chinstrap species description
st.markdown("**Chinstrap:** Bill length and depth are steady, with a small increase. Flipper length has a more significant increase. Again, body mass is steady.")

# --------------------------------------------------------------------------------------------------------
# ADDING A CORRELATION HEATMAP
# --------------------------------------------------------------------------------------------------------
st.subheader("Correlation Between Measurements")

# Selecting only the numeric measurement columns
corr = filtered_df2[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].corr()

# Compute the correlation matirx and create the heatmap figure
fig, ax = plt.subplots()
im = ax.imshow(corr)

# Set labels 
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.columns)

# Add a color scaler bar
fig.colorbar(im)

# Display the chart 
st.pyplot(fig)
    # Output:  Table of raltionships between variables 
        # values near: 
            # 1: strong positive correlation 
            # 0: no correlation
            # -1: strong negative correlation 

# Key takeawy and descrition of the correlation heatmap 
st.write("Above is a correlation map based on the variables visualized in the boxplots.")

# --------------------------------------------------------------------------------------------------------
# END OF THE APP
# --------------------------------------------------------------------------------------------------------
# Final thank you 
st.subheader("Thank you for reading!")