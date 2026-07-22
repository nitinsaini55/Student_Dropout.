import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_dropout_model.pkl")


def predict_dropout(
    age,
    gender,
    year_of_study,
    attendance,
    study_hours,
    previous_gpa,
    backlogs,
    financial_stress,
    stress_level,
    burnout_level
):

    # Encoding
    gender = 1 if gender == "Male" else 0

    # if gender=="Male":
    # gender=1
    # else:
    # gender=0

    burnout_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    burnout = burnout_mapping[burnout_level]

    input_data = np.array([[
        age,
        gender,
        year_of_study,
        attendance,
        study_hours,
        previous_gpa,
        backlogs,
        financial_stress,
        stress_level,
        burnout
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Student Will Drop Out"
    else:
        return "Student Will Not Drop Out"


student_dropout_app = gr.Interface(
    fn=predict_dropout,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown([1, 2, 3, 4], label="Year of Study"),
        gr.Slider(0, 100, label="Attendance Percentage"),
        gr.Number(label="Study Hours Per Day"),
        gr.Number(label="Previous GPA"),
        gr.Number(label="Backlogs"),
        gr.Slider(1, 10, step=1, label="Financial Stress Score"),
        gr.Slider(1, 10, step=1, label="Stress Level"),
        gr.Dropdown(["Low", "Medium", "High"], label="Burnout Level"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Student Dropout Risk Prediction",
    description="Predict whether a student has High or Low Dropout Risk."
)

student_dropout_app.launch(
    server_name="0.0.0.0",
    server_port=7860
)
