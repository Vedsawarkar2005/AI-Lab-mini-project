"""
=============================================================================
test_predictions.py
=============================================================================
Automated test script to verify model inference on the three required
student test profiles without requiring the web browser.
=============================================================================
"""

import os
import joblib
import pandas as pd

def test_student_examples():
    model_path = os.path.join("model", "student_performance_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Run train_model.py first.")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    
    test_cases = [
        {
            "name": "Example 1 (High Performer)",
            "data": {
                "attendance": [90],
                "internal_marks": [27],
                "assignment_completion": [95],
                "previous_semester_percentage": [82],
                "study_hours": [5]
            }
        },
        {
            "name": "Example 2 (Average Performer)",
            "data": {
                "attendance": [75],
                "internal_marks": [19],
                "assignment_completion": [70],
                "previous_semester_percentage": [60],
                "study_hours": [3]
            }
        },
        {
            "name": "Example 3 (At Risk Performer)",
            "data": {
                "attendance": [50],
                "internal_marks": [10],
                "assignment_completion": [40],
                "previous_semester_percentage": [42],
                "study_hours": [1]
            }
        }
    ]

    print("=" * 70)
    print("TESTING TRAINED MODEL ON REQUIRED STUDENT PROFILES")
    print("=" * 70)

    for tc in test_cases:
        df = pd.DataFrame(tc["data"])[feature_names]
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        classes = model.classes_
        
        prob_summary = ", ".join([f"{c}: {p*100:.1f}%" for c, p in zip(classes, probabilities)])
        
        print(f"\nProfile: {tc['name']}")
        print(f"Inputs: {tc['data']}")
        print(f"-> Predicted Category: {prediction}")
        print(f"-> Probabilities:      {prob_summary}")

    print("\n" + "=" * 70)
    print("All test cases evaluated successfully by ML Model!")
    print("=" * 70)

if __name__ == "__main__":
    test_student_examples()
