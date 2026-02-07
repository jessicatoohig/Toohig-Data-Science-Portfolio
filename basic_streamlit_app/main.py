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
st.subheader("Filter by an Island")
island = st.selectbox("Select a island", df["island"].unique(), index = None)

# Adding a Button
if st.button("Click me!"):
    filtered_df = df[df["island"] == island]
    st.write(f"Penguins in the {island} island.")
    st.dataframe(filtered_df)
    st.write("Include description of filtered data frame here.")
else:
    st.write("Click the button to display the filtered dataframe.")

# Adding Bar Charts
filtered_df = df[df["island"] == island]
st.bar_chart(filtered_df["species"].value_counts(), x_label= "Species", y_label = "# of Penguins")
st.bar_chart(filtered_df["sex"].value_counts(), x_label= "Sex", y_label = "# of Penguins")

st.write("Male and female penguins are evenly distributed across all three islands. However, the Adelie penguin is the only specie on all three islands. Use the selct box below to find out more about the Adelie's metrics, as well as the other species.")

# Creating a Second Select Box
st.subheader("Filter by a Species")
species = st.selectbox("Select a species", df["species"].unique(), index = None)
filtered_df2 = df[df["species"] == species]


#Adding a Box Plot with Filters 
import matplotlib.pyplot as plt
import streamlit as st

measurement = st.selectbox("Choose a measurement",["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

fig, ax = plt.subplots()
filtered_df2.boxplot(column=measurement, by="year", ax=ax)

ax.set_xlabel("Year")
ax.set_ylabel(measurement.replace("_", " ").title())
ax.set_title(f"{measurement.replace('_', ' ').title()} by Year")

st.pyplot(fig)

# Key Takeaways
st.subheader("Key Takeaways from Average Specie Metrics")
st.subheader("Adelie")
st.write("Bill length is steady thorughout the years, with the most variety in 2008. Bill depth decreases slightly, whereas flipper length increases. Body mass is also relatively steady.")
st.subheader("Gentoo")
st.write("Bill length decreases then increases. Bill depth increases, and also grows in range. Flipper lenght has an intial increase, then decrease. Body mass is steady.")
st.subheader("Chinstrap")
st.write("Bill lenngth and depth are steady, with a small increase. Flipper length has a more significant increase. Again, body mass in is steady.")

# Map 
st.subheader("Correlation Between Measurements")

corr = filtered_df2[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].corr()

fig, ax = plt.subplots()
im = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45)
ax.set_yticklabels(corr.columns)

fig.colorbar(im)
st.pyplot(fig)