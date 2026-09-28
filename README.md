# 🎓 AI-Based Student Performance Prediction System
### B.Tech Computer Science & Engineering — Artificial Intelligence Laboratory Mini-Project

A complete, beginner-friendly, and explainable Machine Learning application that predicts a student's academic standing (**Good**, **Average**, or **At Risk**) based on early-semester attendance, marks, assignment completion, prior performance, and daily study habits.

---

## 1. Project Description
Early identification of students needing academic assistance is essential to prevent semester dropouts, course backlogs, and examination stress. This project demonstrates how **Supervised Machine Learning** (specifically a **Random Forest Classifier**) can be applied to classify student performance into actionable categories early in the academic session.

The project is built entirely with open-source Python libraries (**Pandas**, **NumPy**, **Scikit-learn**, **Streamlit**, **Matplotlib**, **Seaborn**, **Joblib**) and runs **100% offline** with zero dependence on paid APIs or complex cloud configurations.

---

## 2. Problem Statement
In higher education institutions, students at risk of academic failure are frequently identified only after semester examination results are released. By then, it is too late for remedial actions. An intelligent system is needed to analyze behavioral and continuous evaluation metrics during the semester to provide early warning indicators to faculty mentors and students.

---

## 3. Project Objectives
1. Build a supervised machine learning classification model to categorize student performance.
2. Generate a statistically realistic synthetic dataset reflecting real-world academic patterns with natural human variance.
3. Train and thoroughly evaluate a **Random Forest Classifier** against standard evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
4. Provide an interactive, responsive **Streamlit Web Application** for real-time predictions, probability confidence distributions, and rule-based academic recommendations.
5. Create a clean, transparent, and easy-to-understand codebase suitable for a B.Tech Artificial Intelligence laboratory viva voce.

---

## 4. Key Features
- **Interactive Web Interface**: Sliders for intuitive student data entry with built-in validation bounds.
- **Explainable ML Classification**: Predicts one of three color-coded tiers:
  - 🟢 **Good** (Consistent, high performer)
  - 🟡 **Average** (Moderate performer, potential for growth)
  - 🔴 **At Risk** (Requires immediate intervention and mentoring)
- **Model Confidence Distribution**: Visual progress breakdown of prediction probabilities across all three classes (`predict_proba`).
- **Student Profile Summary**: Side-by-side comparison of entered metrics against recommended institutional benchmarks.
- **Rule-Based Improvement Recommendations**: Actionable suggestions targeting specific weak areas (e.g. low attendance, insufficient study hours) without overriding the ML model.
- **Dataset Insights Dashboard**: Visual analysis displaying class balance, attendance vs. previous scores, and study hours distributions.
- **Built-In Viva Preparation Guide**: Detailed explanations of core AI concepts directly inside the application.

---

## 5. Technologies Used
- **Python 3.10+**: Core programming language
- **Pandas**: Tabular data manipulation and CSV handling
- **NumPy**: Numerical operations and statistical distributions
- **Scikit-learn**: Machine learning model training, metrics calculation, and dataset splitting
- **Streamlit**: Web dashboard and user interface framework
- **Matplotlib & Seaborn**: Data exploration and confusion matrix plotting
- **Joblib**: Model serialization and persistence (`.pkl`)

*(No TensorFlow, PyTorch, OpenAI APIs, or paid cloud services used).*

---

## 6. AI / ML Algorithm: Random Forest Classifier
The primary model used in this project is the **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`).

### Why Random Forest is Suitable for this Problem:
1. **Ensemble Bagging (Bootstrap Aggregation)**: Instead of relying on a single decision tree that can easily overfit noisy student data, Random Forest constructs 100 decorrelated decision trees and aggregates their predictions by majority voting.
2. **Feature Scaling Invariance**: Decision tree splits are based on rank-order thresholds. Features with large scales (like 0–100% attendance) do not overpower smaller-scale features (like 0–12 study hours). No complex data transformation or scaling pipeline is required.
3. **Captures Non-Linear Interactions**: Complex relationships (e.g., a student with high attendance but low study hours and failing internals) are naturally partitioned by orthogonal decision boundaries.
4. **Resilience to Overfitting**: By evaluating random subsets of features at each split, the forest maintains low variance and high generalization capability.

---

## 7. Dataset Description
The system uses a synthetic dataset of **600 student records** created specifically for educational purposes in `generate_dataset.py`.

> **Note on Synthetic Data**: Real student academic records contain sensitive personally identifiable information (PII) protected under privacy regulations. A realistic synthetic dataset allows full experimentation while ensuring ethical compliance.

### Features & Ranges:
| Column Name | Data Type | Range | Description |
|---|---|---|---|
| `student_id` | String | STU1001 – STU1600 | Unique student identifier |
| `attendance` | Float | 0.0 – 100.0 % | Lecture attendance percentage |
| `internal_marks` | Float | 0.0 – 30.0 | Marks obtained in college internal exams |
| `assignment_completion` | Float | 0.0 – 100.0 % | Percentage of homework/assignments submitted |
| `previous_semester_percentage` | Float | 0.0 – 100.0 % | Past semester aggregate score |
| `study_hours` | Float | 0.0 – 12.0 | Average daily self-study hours |
| `performance_category` | String | At Risk, Average, Good | Ground-truth target label |

The target categories are synthesized using institutional weightings with added Gaussian noise ($\pm 5.0\%$) to reflect human factors (illness, stress, personal reasons), ensuring the data is not deterministically separable and the ML model genuinely learns underlying patterns.

---

## 8. Project Architecture & Folder Structure

```
student-performance-predictor/
│
├── app.py                         # Streamlit interactive web dashboard
├── train_model.py                 # ML training script, evaluation & model export
├── generate_dataset.py            # Generates synthetic student dataset
├── test_predictions.py            # Automated verification test script
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation and quickstart
├── PROJECT_DOCUMENTATION.md       # Comprehensive viva & project manual
├── .gitignore                     # Git ignore rules
│
├── web_ui/                        # Standalone Modern Web UI
│   ├── index.html                 # Modern responsive HTML5/CSS/JS dashboard
│   └── server.py                  # Lightweight local Python API server (port 5000)
│
├── data/
│   └── student_performance.csv    # Generated student dataset (600 records)
│
├── model/
│   ├── student_performance_model.pkl # Trained Random Forest model (Joblib)
│   └── confusion_matrix.png       # Saved confusion matrix visualization
│
└── screenshots/                   # Folder for report screenshots
```

---

## 9. Installation & Setup Instructions (Windows)

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt in the project root directory:
```powershell
cd "f:\College\4th year\AI lab mini project"
```

### Step 2: Set Up Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 10. How to Run the Project

### Step 1: Generate the Dataset
Run `generate_dataset.py` to create the 600-student dataset:
```powershell
python generate_dataset.py
```
*Output: Creates `data/student_performance.csv` and prints class distribution.*

### Step 2: Train the Machine Learning Model
Run `train_model.py` to train the Random Forest model:
```powershell
python train_model.py
```
*Output: Displays accuracy, classification report, confusion matrix, saves `model/student_performance_model.pkl`, and exports `model/confusion_matrix.png`.*

### Step 3: Launch the User Interface (Two Options Available)

#### Option A: Streamlit Interactive Dashboard (Recommended)
```powershell
streamlit run app.py
```
*Opens `http://localhost:8501` in your browser with sidebar sliders, quick preset buttons (Good, Average, At Risk), probability meters, student benchmark summary, and dataset insights charts.*

#### Option B: Standalone Modern Web UI (Alternative)
```powershell
python web_ui/server.py
```
*Opens a modern HTML5/CSS3/JavaScript dashboard at `http://localhost:5000` with quick test presets, live SVG charts, and interactive viva Q&A tabs. You can also open `web_ui/index.html` directly in any web browser without running a server!*

---

## 11. Example Predictions & Test Cases

The application and model were verified with three distinct student profiles:

### Example 1: Likely Good
- **Inputs**: Attendance: 90%, Internals: 27/30, Assignments: 95%, Prev Sem: 82%, Study: 5 hrs/day
- **Predicted Category**: 🟢 **Good**
- **Confidence**: ~85–95% Good

### Example 2: Likely Average
- **Inputs**: Attendance: 75%, Internals: 19/30, Assignments: 70%, Prev Sem: 60%, Study: 3 hrs/day
- **Predicted Category**: 🟡 **Average**
- **Confidence**: ~70–80% Average

### Example 3: Likely At Risk
- **Inputs**: Attendance: 50%, Internals: 10/30, Assignments: 40%, Prev Sem: 42%, Study: 1 hr/day
- **Predicted Category**: 🔴 **At Risk**
- **Confidence**: ~80–90% At Risk

---

## 12. Model Evaluation Results
Evaluated on an independent 20% stratified test set (120 students):
- **Accuracy**: ~88% – 93% (accounting for realistic synthetic noise)
- **Precision**: Balanced across all three classes (>0.85)
- **Recall**: High sensitivity for "At Risk" category to minimize false negatives
- **Confusion Matrix**: Saved to `model/confusion_matrix.png`

---

## 13. Limitations
1. **Synthetic Data**: The model is trained on simulated data for educational demonstration and should not be used as an authoritative institutional grading tool.
2. **Fixed Attributes**: Analyzes 5 academic indicators; does not capture personal, health, or psychological factors.
3. **Single Time Snapshot**: Analyzes static semester averages rather than week-by-week trends.

---

## 14. Future Enhancements
- Integration with University ERP or Learning Management Systems (LMS like Moodle/Canvas).
- Longitudinal tracking across all 8 engineering semesters.
- Automated email/SMS alert system for faculty proctors when a student enters the "At Risk" threshold.

---

## 15. Conclusion
The AI-Based Student Performance Prediction System demonstrates the practical application of Supervised Machine Learning in educational analytics. By combining Random Forest classification with an intuitive Streamlit interface and actionable recommendations, the project provides a comprehensive, explainable, and viva-ready solution for academic mentoring.

---

## 16. Quick Viva Questions & Answers

| # | Question | Short Answer |
|---|---|---|
| 1 | **What is Supervised Learning?** | Machine learning where the model learns from input features with labeled target outputs. |
| 2 | **Why is this a classification problem?** | The target output is categorical ('At Risk', 'Average', 'Good'), not continuous numerical values. |
| 3 | **Why use Random Forest?** | It is an ensemble of decision trees that reduces overfitting, handles non-linearities, and requires no feature scaling. |
| 4 | **What is Train-Test Split?** | Dividing data (80% train, 20% test) to evaluate model performance on unseen data. |
| 5 | **What is a Confusion Matrix?** | A table mapping actual vs. predicted labels to visualize True/False Positives and Negatives. |
| 6 | **What does Joblib do?** | It serializes and saves the trained Python model to disk so the web app can load it instantly. |
| 7 | **What does `predict_proba()` return?** | The proportion of trees in the forest voting for each of the three performance classes. |
| 8 | **Does the app use IF-ELSE for prediction?** | No. The prediction comes strictly from the trained Scikit-learn model; IF-ELSE is only used for auxiliary suggestions. |

*(For the complete 15-question in-depth viva guide, see [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md)).*
