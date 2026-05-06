# # Project 4: MLUnsupervised Streamlit App
## Project Overview ▶️
The Unsuperived ML Streamlit App is an interactive unsupervised machine learning web app built with Streamlit. SImilar to Poroject 3 except deploying unsuperivised learning models, the goal is to of this project is to allow users freedom and creativity in exploring, training, and visualizing unsuperivised learning models with a dataset of their choice. The Titanic Dataset is uploaded as an example, but users can also upload a dataset in the form od a csv file of their choosing. They can then select featuers for analysis, and adjusst model hyperparameters. The app provides visualizaitons and performance metrics to help users underestnad how clustering behaves under differnt conditons. 

## How to Run the App on Your Machine 🧩
**1. Clone the repository** 
```bash
$ git clone https://github.com/Toohig-Data-Science-Portfolio/MLUnsupervisedApp.git
cd MLUnsupervisedApp
```
**2. Start a virtual environment (using uv) and install libraries:**
```bash
pip install uv
uv venv

# Activate environment
# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

uv sync
```
**3. Start the app locally**
```bash
$ streamlit run MLUnsupervisedApp.py
```
### Required Libraries 📦
**Core libraries:**
- streamlit (1.32+)
- pandas (2.0+)
- numpy (1.24+)
- scikit-learn (1.3+)
- plotly (5.0+)
- scipy (1.10+)

**Link to the deployed version in Streamlit Cloud**

[Streamlit Cloud]()


## App Features ⭐
**The app includes mulitple unsupervised learning models:** 

**1. K-Means Clustering**
- Groups data into K clusters based on similarity
- Hyperparameter: K (numer of clsuters)
- Visualization: PCA cluster scatter plot
        - Reducues high-dimensional data into 2D for visualizaiton
        - Helps visualize how clusers are separated

**2. Elbow Plot**
- Used to determine optimal number of clusters (K)
- Plots inertia vs. number of clusters
- Helps identify the "elbow point" where improvement slows
- Supports better hyperparameter selction for K-means

**3. Hierarchical Clustering**
- Builds a tree-like structure of clusters (dendrogram)
- No predefined number of clusters required 
- Shows how data points merge step-by-step based on similarity 
- Visualization: Dendrogram 

**Hyperparameter Tuning:**
The primary hyperparameter is the number of clusters (K), which can be adjusted using the sidebar slider. As the k value changes, the clustering model updates automatically and all visualizations refresh in real time. This allows users to experiment and observe how clustering structure changes without manually retraining the model. 

## Graphs and Interactions Involved 📊
All graphs below are derived from the Titanic Dataset. 


## References 📂
[Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
[Tidy Data Cheat Sheet](https://vita.had.co.nz/papers/tidy-data.pdf)