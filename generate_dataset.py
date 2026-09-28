"""
=============================================================================
generate_dataset.py
=============================================================================
Generates a realistic synthetic dataset for the AI-Based Student Performance
Prediction System.

Features:
- student_id: Unique identifier for each student (e.g., STU1001)
- attendance: Student attendance percentage (0 to 100 %)
- internal_marks: College internal exam score (0 to 30 marks)
- assignment_completion: Percentage of homework/assignments submitted (0 to 100 %)
- previous_semester_percentage: Past semester examination score (0 to 100 %)
- study_hours: Self-study hours per day (0 to 12 hours)

Target:
- performance_category: 'At Risk', 'Average', or 'Good'

The categories are generated using an academic weighted score plus controlled
random noise (to reflect realistic academic unpredictability like exam stress
or illness), allowing the machine learning model to genuinely learn real patterns.
=============================================================================
"""

import os
import numpy as np
import pandas as pd

def generate_student_data(num_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student performance data with realistic academic correlations.
    
    Parameters:
        num_samples (int): Number of student records to generate (default 600).
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Formatted dataset ready for training and analysis.
    """
    # Set seed for reproducible results across runs
    np.random.seed(random_seed)
    
    # 1. Generate unique student IDs
    student_ids = [f"STU{1001 + i}" for i in range(num_samples)]
    
    # 2. Generate academic baseline latent ability (0 to 1) representing student capability
    # Using a beta distribution to model typical classroom bell-curve spread
    academic_latent = np.random.beta(a=3.0, b=2.8, size=num_samples)
    
    # 3. Generate feature values strongly influenced by academic ability, with realistic variation
    
    # Attendance (%): typically between 40% and 98%, with lower outliers
    attendance_noise = np.random.normal(0, 7, size=num_samples)
    attendance = (academic_latent * 60 + 35) + attendance_noise
    attendance = np.clip(np.round(attendance, 1), 10.0, 100.0)
    
    # Internal Marks (0 - 30): heavily correlated with attendance and preparation
    internal_noise = np.random.normal(0, 2.5, size=num_samples)
    internal_marks = (academic_latent * 22 + 6) + internal_noise
    internal_marks = np.clip(np.round(internal_marks, 1), 0.0, 30.0)
    
    # Assignment Completion (%): 0 - 100%
    assignment_noise = np.random.normal(0, 8, size=num_samples)
    assignment_completion = (academic_latent * 65 + 30) + assignment_noise
    assignment_completion = np.clip(np.round(assignment_completion, 1), 10.0, 100.0)
    
    # Previous Semester Percentage (%): 0 - 100%
    prev_sem_noise = np.random.normal(0, 6, size=num_samples)
    prev_sem = (academic_latent * 55 + 40) + prev_sem_noise
    prev_sem = np.clip(np.round(prev_sem, 1), 25.0, 98.0)
    
    # Average Study Hours per day: 0 - 12 hours (typical 1 to 7 hours)
    hours_noise = np.random.normal(0, 0.8, size=num_samples)
    study_hours = (academic_latent * 5.0 + 1.2) + hours_noise
    study_hours = np.clip(np.round(study_hours, 1), 0.5, 12.0)
    
    # 4. Composite Academic Indicator for Ground Truth labeling
    # Weighted calculation reflecting institutional evaluation importance:
    # - Attendance: 20%
    # - Internal Marks: 25% (converted to 100 scale: marks / 30 * 100)
    # - Assignment Completion: 15%
    # - Previous Semester Performance: 25%
    # - Daily Study Hours: 15% (normalized against 8 hrs benchmark: hours / 8 * 100 capped at 100)
    
    norm_internal = (internal_marks / 30.0) * 100.0
    norm_study = np.clip((study_hours / 8.0) * 100.0, 0.0, 100.0)
    
    composite_score = (
        0.20 * attendance +
        0.25 * norm_internal +
        0.15 * assignment_completion +
        0.25 * prev_sem +
        0.15 * norm_study
    )
    
    # Add human unpredictability noise (+/- 5%) so data is not deterministically separable
    human_factor_noise = np.random.normal(0, 5.0, size=num_samples)
    effective_score = composite_score + human_factor_noise
    
    # 5. Classify into realistic academic performance tiers:
    # Score < 50%  -> At Risk
    # 50% to 73%   -> Average
    # >= 73%       -> Good
    categories = []
    for score in effective_score:
        if score < 50.0:
            categories.append("At Risk")
        elif score < 73.0:
            categories.append("Average")
        else:
            categories.append("Good")
            
    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_completion": assignment_completion,
        "previous_semester_percentage": prev_sem,
        "study_hours": study_hours,
        "performance_category": categories
    })
    
    return df

def main():
    print("=" * 60)
    print("AI-Based Student Performance Prediction System: Data Generator")
    print("=" * 60)
    
    # Ensure data directory exists
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "student_performance.csv")
    
    # Generate 600 records
    df = generate_student_data(num_samples=600, random_seed=42)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary output as required
    print("Dataset generated successfully.")
    print(f"Number of records: {len(df)}")
    print("Saved to:", output_file)
    print("\nColumns:")
    for col in df.columns:
        print(f" - {col}")
        
    print("\nPerformance Category Distribution:")
    counts = df["performance_category"].value_counts()
    for category, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f" - {category:<10}: {count} students ({percentage:.1f}%)")
        
    print("\nFeature Summary Statistics:")
    print(df.drop(columns=["student_id", "performance_category"]).describe().round(2).to_string())
    print("=" * 60)

if __name__ == "__main__":
    main()
