import streamlit as st
import pickle
import numpy as np

st.markdown("""
### 📊 AI-Powered Academic Performance Analysis

This machine learning model predicts final GPA using:

- 📚 Study Hours
- 🏫 Attendance
- 😴 Sleep Schedule
- 📝 Assignment Completion
- 🎯 Previous Academic Performance
""")
# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# Title
st.title("🎓 Student Performance Predictor")

st.write("Enter student details below to predict the final GPA.")


# User inputs
study_hours = st.slider(
    "Study Hours per Day",
    min_value=0,
    max_value=12,
    value=5
)

attendance = st.slider(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=80
)

sleep_hours = st.slider(
    "Sleep Hours",
    min_value=3,
    max_value=10,
    value=7
)

assignments_completed = st.slider(
    "Assignments Completed",
    min_value=0,
    max_value=10,
    value=7
)

previous_gpa = st.slider(
    "Previous GPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5
)


# Predict button
if st.button("Predict GPA"):

    student_data = np.array([[
        study_hours,
        attendance,
        sleep_hours,
        assignments_completed,
        previous_gpa
    ]])

    prediction = model.predict(student_data)

    gpa = prediction[0]

    st.success(f"Predicted Final GPA: {gpa:.2f}")

    if gpa >= 9:
        st.balloons()
        st.success("🏆 Performance Level: Excellent")

    elif gpa >= 8:
        st.info("🔥 Performance Level: Good")

    elif gpa >= 6:
        st.warning("📚 Performance Level: Average")

    else:
        st.error("⚠️ Performance Level: Needs Improvement")