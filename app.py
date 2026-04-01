import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student-mat.csv", sep=';')
data = data[['studytime', 'absences', 'G1', 'G2', 'G3']]

# Prepare data
X = data[['studytime', 'absences', 'G1', 'G2']]
y = data['G3']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🎓 Student Performance Predictor")

studytime = st.slider("Study Time", 1, 4)
absences = st.slider("Absences", 0, 30)
G1 = st.slider("G1 Marks", 0, 20)
G2 = st.slider("G2 Marks", 0, 20)

if st.button("Predict"):
    prediction = model.predict([[studytime, absences, G1, G2]])
    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")