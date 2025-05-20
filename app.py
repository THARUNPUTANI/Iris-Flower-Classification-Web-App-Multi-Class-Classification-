import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
iris = load_iris()

# Train the model
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

# Streamlit App UI
st.title("🌸 Iris Flower Classification")
st.markdown("Predict the species of Iris flower based on sepal and petal measurements.")

# Sidebar for user input
st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Convert input into dataframe
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)
predicted_species = iris.target_names[prediction[0]]

# Output the prediction
st.subheader("🌼 Predicted Species")
st.success(f"The predicted Iris species is: **{predicted_species}**")

# Show raw data
if st.checkbox("Show Training Dataset"):
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Series(iris.target).map(dict(enumerate(iris.target_names)))
    st.write(df)
