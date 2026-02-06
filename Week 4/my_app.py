import streamlit as st

# Markdown Hashtag
st.title("Hello, Streamlit!")
st.markdown("# Hello, Streamlit!")

st.write("This is my first Streamlit app.")

if st.button("Click me!"):
    st.write("You clicked the button.")
else:
    st.write("Click the button to see what happens...")

### Loading our csv file
import pandas as pd 

st.subheader("Exploring Our Dataset")

# Load the CSV file
df = pd.read_csv("data/sample_data-1.csv")

st.write("Here's our data!")
st.dataframe(df)

city = st.selectbox("Select a city", df["City"].unique(), index = None)
filtered_df = df[df["City"] == city]

st.write(f"People in {city}")
st.dataframe(filtered_df)

## Add bar chart
st.bar_chart(df["Salary"], x_label= "Person", y_label = "Salary $")

import seaborn as sns

box_plot1 = sns.boxplot(x=df["City"], y=df["Salary"])

st.pyplot(box_plot1.get_figure())

