import streamlit as st

# Title and Info
st.title("Project 1: Palmer's Penguins ")
st.subheader("By: Jessica Toohig")
st.write("Include short description here, maybe make this a button")

# Importing CSV
import pandas as pd 

st.subheader("Exploring the Dataset")
df = pd.read_csv("data1/penguins.csv")
st.write("Here's our data!")
st.dataframe(df)

# Creating a Select Box
st.subheader("Filter the Dataframe by an Island")
island = st.selectbox("Select a island", df["island"].unique(), index = None)

#Adding a Button
if st.button("Click me!"):
    filtered_df = df[df["island"] == island]
    st.write(f"Penguins in the {island} island.")
    st.dataframe(filtered_df)
    st.write("Include description of filtered data frame here.")
else:
    st.write("Click the button to display the filtered dataframe.")

# Adding a bar chart
filtered_df = df[df["island"] == island]
st.bar_chart(filtered_df["species"].value_counts(), x_label= "Species", y_label = "# of Penguins")