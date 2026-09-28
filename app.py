"""
=============================================================================
app.py
=============================================================================
Streamlit Interactive Web Application for the AI-Based Student Performance
Prediction System.

Features:
1. Sidebar: Student input parameters via sliders.
2. Prediction: Real-time inference using trained RandomForestClassifier.
3. Probability Distribution: Confidence scores across all 3 classes.
4. Student Summary: Clean tabular overview of inputs against benchmarks.
5. Rule-Based Academic Recommendations: Actionable tips for improvement.
6. Dataset Insights: Summary metrics and exploratory visualization charts.
7. Educational Section: Viva-ready explanation of ML concepts.
=============================================================================
"""

import os
import streamlit as pd_st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean college-lab dashboard styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #E5E7EB;
    }
    .metric-container {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .recommendation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Paths and Data Loading Functions
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "student_performance_model.pkl")
DATASET_PATH = os.path.join("data", "student_performance.csv")

@st.cache_resource
def load_trained_model():
    """Load the serialized Random Forest model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        payload = joblib.load(MODEL_PATH)
        return payload
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_student_dataset():
    """Load the synthetic student dataset for insights."""
    if not os.path.exists(DATASET_PATH):
        return None
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as e:
        st.error(f"Error reading dataset: {e}")
        return None

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🎓 AI Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict student academic performance using Supervised Machine Learning (Random Forest Classifier)</div>', unsafe_allow_html=True)

# Check system state
model_bundle = load_trained_model()
dataset_df = load_student_dataset()

# ---------------------------------------------------------------------------
# Sidebar: Student Input Form & Quick Presets
# ---------------------------------------------------------------------------
if "att_val" not in st.session_state:
    st.session_state.att_val = 75
if "int_val" not in st.session_state:
    st.session_state.int_val = 20
if "assign_val" not in st.session_state:
    st.session_state.assign_val = 75
if "prev_val" not in st.session_state:
    st.session_state.prev_val = 65
if "study_val" not in st.session_state:
    st.session_state.study_val = 3

st.sidebar.header("📋 Student Information")
st.sidebar.markdown("**Quick Presets (For Viva Demo):**")
p_col1, p_col2, p_col3 = st.sidebar.columns(3)
if p_col1.button("🟢 Good", use_container_width=True):
    st.session_state.att_val = 90
    st.session_state.int_val = 27
    st.session_state.assign_val = 95
    st.session_state.prev_val = 82
    st.session_state.study_val = 5
if p_col2.button("🟡 Average", use_container_width=True):
    st.session_state.att_val = 75
    st.session_state.int_val = 19
    st.session_state.assign_val = 70
    st.session_state.prev_val = 60
    st.session_state.study_val = 3
if p_col3.button("🔴 At Risk", use_container_width=True):
    st.session_state.att_val = 50
    st.session_state.int_val = 10
    st.session_state.assign_val = 40
    st.session_state.prev_val = 42
    st.session_state.study_val = 1

st.sidebar.markdown("---")
st.sidebar.markdown("Adjust individual sliders to customize metrics:")

attendance = st.sidebar.slider(
    "Attendance Percentage (%)",
    min_value=0,
    max_value=100,
    value=st.session_state.att_val,
    step=1,
    help="Overall class attendance in the current semester."
)

internal_marks = st.sidebar.slider(
    "Internal Marks (out of 30)",
    min_value=0,
    max_value=30,
    value=st.session_state.int_val,
    step=1,
    help="Scores obtained in internal mid-term examinations and continuous evaluations."
)

assignment_completion = st.sidebar.slider(
    "Assignment Completion (%)",
    min_value=0,
    max_value=100,
    value=st.session_state.assign_val,
    step=1,
    help="Percentage of submitted lab tasks, quizzes, and homework assignments."
)

prev_sem_percentage = st.sidebar.slider(
    "Previous Semester Percentage (%)",
    min_value=0,
    max_value=100,
    value=st.session_state.prev_val,
    step=1,
    help="Cumulative marks percentage scored in the previous semester."
)

study_hours = st.sidebar.slider(
    "Average Study Hours per Day",
    min_value=0,
    max_value=12,
    value=st.session_state.study_val,
    step=1,
    help="Self-study hours outside regular college lecture hours."
)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔍 Predict Performance", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main Interface: Two Tabs (Prediction & Insights)
# ---------------------------------------------------------------------------
tab_prediction, tab_insights, tab_about = st.tabs([
    "🎯 Prediction & Assessment",
    "📊 Dataset Insights & Analytics",
    "💡 About the AI Model (Viva Guide)"
])

# ===========================================================================
# TAB 1: Prediction & Assessment
# ===========================================================================
with tab_prediction:
    if model_bundle is None:
        st.warning("⚠️ **Trained Model Not Found!**")
        st.info(
            "Please initialize and train the machine learning model first by running the following commands in your terminal:\n\n"
            "```bash\n"
            "python generate_dataset.py\n"
            "python train_model.py\n"
            "```\n"
            "Once `model/student_performance_model.pkl` is generated, refresh this page."
        )
    else:
        model = model_bundle["model"]
        feature_names = model_bundle["feature_names"]
        target_classes = model_bundle["target_classes"]
        model_acc = model_bundle.get("accuracy", 0.0)

        # Prepare input data in exact DataFrame format used during model training
        input_dict = {
            "attendance": [attendance],
            "internal_marks": [internal_marks],
            "assignment_completion": [assignment_completion],
            "previous_semester_percentage": [prev_sem_percentage],
            "study_hours": [study_hours]
        }
        input_df = pd.DataFrame(input_dict)

        col1, col2 = st.columns([1.1, 0.9])

        with col1:
            st.subheader("Assessment Result")

            if predict_clicked:
                # 1. Run Machine Learning Inference
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                # Model class ordering
                model_classes = list(model.classes_)
                prob_dict = {cls: prob for cls, prob in zip(model_classes, probabilities)}

                # Display category badge
                if prediction == "Good":
                    st.success("### 🟢 Predicted Category: **Good**")
                    st.markdown("The student exhibits consistent academic commitment, regular attendance, and steady study habits.")
                elif prediction == "Average":
                    st.warning("### 🟡 Predicted Category: **Average**")
                    st.markdown("The student is performing satisfactorily, but targeted focus on key weak areas can elevate them to the 'Good' tier.")
                else:  # At Risk
                    st.error("### 🔴 Predicted Category: **At Risk**")
                    st.markdown("The student requires timely academic mentoring and intervention to prevent semester backlogs or poor grades.")

                st.markdown("---")

                # 2. Prediction Probabilities
                st.write("#### 📈 Model Probability Estimate")
                st.caption("Confidence scores predicted by the Random Forest ensemble (sum = 100%):")

                prob_cols = st.columns(3)
                colors = {"Good": "#10B981", "Average": "#F59E0B", "At Risk": "#EF4444"}
                icons = {"Good": "🟢", "Average": "🟡", "At Risk": "🔴"}

                for idx, cls_name in enumerate(["Good", "Average", "At Risk"]):
                    p_val = prob_dict.get(cls_name, 0.0) * 100
                    with prob_cols[idx]:
                        st.metric(
                            label=f"{icons[cls_name]} {cls_name}",
                            value=f"{p_val:.1f}%"
                        )
                        st.progress(min(int(p_val), 100))

                # Educational Disclaimer
                st.caption("ℹ️ *Note: Probabilities represent model confidence estimates based on synthetic training data, not absolute certainty.*")

            else:
                st.info("👈 Set student values in the sidebar and click **'Predict Performance'** to view the ML assessment.")

        with col2:
            st.subheader("Student Profile Summary")
            
            summary_data = {
                "Feature": [
                    "Attendance",
                    "Internal Marks",
                    "Assignment Completion",
                    "Previous Semester",
                    "Daily Study Hours"
                ],
                "Student Value": [
                    f"{attendance}%",
                    f"{internal_marks} / 30",
                    f"{assignment_completion}%",
                    f"{prev_sem_percentage}%",
                    f"{study_hours} hrs/day"
                ],
                "Institutional Benchmark": [
                    "≥ 75% required",
                    "≥ 18 / 30 target",
                    "≥ 80% submission",
                    "≥ 60% passing standard",
                    "≥ 3 hrs/day recommended"
                ]
            }
            summary_table = pd.DataFrame(summary_data)
            st.table(summary_table)

        # -------------------------------------------------------------------
        # Rule-Based Academic Recommendations
        # -------------------------------------------------------------------
        if predict_clicked:
            st.markdown("---")
            st.subheader("📌 Academic Improvement Recommendations")
            st.markdown(
                "*These rule-based suggestions are tailored based on individual metrics "
                "to provide actionable feedback alongside the ML classification.*"
            )

            recs = []
            if attendance < 75:
                recs.append("⚠️ **Attendance below 75%:** Improve daily lecture attendance to avoid semester condonation penalties and ensure continuous learning.")
            if internal_marks < 15:
                recs.append("📝 **Internal marks below 15/30:** Review internal test mistakes, attend tutorial classes, and revise high-weightage topics.")
            if assignment_completion < 70:
                recs.append("📚 **Assignment completion below 70%:** Submit lab records and weekly assignments promptly to secure continuous assessment marks.")
            if prev_sem_percentage < 50:
                recs.append("🔍 **Previous semester percentage below 50%:** Strengthen prerequisite fundamentals and schedule dedicated revision for difficult subjects.")
            if study_hours < 2:
                recs.append("⏰ **Study hours under 2 hrs/day:** Establish a regular, distraction-free study timetable with at least 2 to 3 focused hours daily.")

            if not recs:
                st.success("🌟 **Outstanding Profile:** All metrics satisfy or exceed recommended college benchmarks. Maintain this consistency through final exams!")
            else:
                for rec in recs:
                    st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 2: Dataset Insights & Analytics
# ===========================================================================
with tab_insights:
    st.subheader("📊 Dataset Exploration & Institutional Insights")

    if dataset_df is None:
        st.warning("Dataset not found at `data/student_performance.csv`. Run `python generate_dataset.py` to create it.")
    else:
        # High-level Metrics Row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Students", f"{len(dataset_df)}")
        with m_col2:
            st.metric("Average Attendance", f"{dataset_df['attendance'].mean():.1f}%")
        with m_col3:
            st.metric("Avg Internal Marks", f"{dataset_df['internal_marks'].mean():.1f} / 30")
        with m_col4:
            st.metric("Avg Study Hours", f"{dataset_df['study_hours'].mean():.1f} hrs/day")

        st.markdown("---")

        # Visualization Row 1
        v_col1, v_col2 = st.columns(2)

        palette = {"Good": "#10B981", "Average": "#F59E0B", "At Risk": "#EF4444"}

        with v_col1:
            st.write("#### 1. Distribution of Performance Categories")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            cat_counts = dataset_df["performance_category"].value_counts()[["Good", "Average", "At Risk"]]
            colors = [palette[c] for c in cat_counts.index]
            bars = ax1.bar(cat_counts.index, cat_counts.values, color=colors, edgecolor="black", linewidth=0.8)
            ax1.set_ylabel("Number of Students")
            ax1.set_title("Student Distribution Across Performance Tiers")
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f"{height}",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig1)
            plt.close(fig1)

        with v_col2:
            st.write("#### 2. Attendance vs Previous Semester Marks")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            for cat, color in palette.items():
                subset = dataset_df[dataset_df["performance_category"] == cat]
                ax2.scatter(subset["attendance"], subset["previous_semester_percentage"],
                            label=cat, color=color, alpha=0.6, edgecolors="none", s=30)
            ax2.set_xlabel("Attendance (%)")
            ax2.set_ylabel("Previous Semester Percentage (%)")
            ax2.set_title("Attendance vs Previous Performance by Category")
            ax2.legend(title="Category")
            ax2.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig2)
            plt.close(fig2)

        # Visualization Row 2: Study Hours vs Category Boxplot
        st.write("#### 3. Daily Study Hours across Performance Categories")
        fig3, ax3 = plt.subplots(figsize=(8, 3.8))
        order = ["At Risk", "Average", "Good"]
        sns.boxplot(
            data=dataset_df,
            x="performance_category",
            y="study_hours",
            order=order,
            palette=palette,
            ax=ax3,
            boxprops=dict(alpha=0.8)
        )
        ax3.set_xlabel("Performance Category")
        ax3.set_ylabel("Study Hours per Day")
        ax3.set_title("Study Hours Distribution Across Categories")
        st.pyplot(fig3)
        plt.close(fig3)

        # Dataset Preview
        with st.expander("👀 View Raw Dataset Sample (First 10 records)"):
            st.dataframe(dataset_df.head(10), use_container_width=True)

# ===========================================================================
# TAB 3: About the AI Model (Viva Guide)
# ===========================================================================
with tab_about:
    st.subheader("💡 About the AI Model & Viva Concepts")
    st.markdown("""
This section summarizes the fundamental Artificial Intelligence and Machine Learning concepts 
used in this mini-project in simple, clear language for your laboratory viva examination.
""")

    with st.expander("🤖 1. What type of Machine Learning is used here?", expanded=True):
        st.markdown("""
- **Supervised Learning**: The algorithm learns from labeled training data containing both input features and corresponding target categories (`performance_category`).
- **Classification Problem**: The output is discrete/categorical ('At Risk', 'Average', 'Good'), unlike regression which outputs a continuous number.
""")

    with st.expander("🌲 2. Why is Random Forest Classifier chosen over a single Decision Tree?"):
        st.markdown("""
- **Ensemble Learning (Bagging)**: Random Forest builds 100 individual decision trees on random subsets of data and merges their predictions through majority voting.
- **Overfitting Prevention**: Single decision trees often overfit noisy student data; Random Forest reduces variance significantly.
- **No Feature Scaling Required**: Unlike KNN or Neural Networks, Decision Trees make partition splits based on thresholds, so normalization (e.g. Min-Max) is not required.
- **Handles Non-Linear Relationships**: Accurately captures complex dependencies between attendance, internal marks, and study hours.
""")

    with st.expander("📋 3. What are the 5 Input Features and the Target?"):
        st.markdown("""
- **Input Features (X)**:
  1. `attendance` (0–100 %)
  2. `internal_marks` (0–30 marks)
  3. `assignment_completion` (0–100 %)
  4. `previous_semester_percentage` (0–100 %)
  5. `study_hours` (0–12 hours/day)
- **Target Label (y)**: `performance_category` with 3 discrete classes:
  - 🟢 **Good**
  - 🟡 **Average**
  - 🔴 **At Risk**
""")

    with st.expander("🧪 4. Why is Synthetic Data used?"):
        st.markdown("""
- Real student educational records contain sensitive personal data protected by privacy guidelines (FERPA/institutional policies).
- A synthetic dataset allows controlled modeling of realistic academic dynamics (with bell curves and human noise) for educational learning without privacy concerns.
""")

    with st.expander("📊 5. How is the model evaluated?"):
        st.markdown("""
- **Train-Test Split**: 80% of data is used for training and 20% for testing.
- **Accuracy**: Percentage of correct predictions out of all test cases.
- **Precision**: Out of all students predicted as 'At Risk', how many were actually 'At Risk'.
- **Recall**: Out of all actual 'At Risk' students, how many were correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: A table showing actual vs. predicted classifications to spot misclassifications.
""")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("🎓 B.Tech Computer Science Artificial Intelligence Lab Mini-Project | Powered by Python, Scikit-learn & Streamlit")
