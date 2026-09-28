"""
=============================================================================
web_ui/server.py
=============================================================================
Lightweight local web server for the AI Student Performance Prediction System.
Uses Python's built-in http.server to provide an offline REST API and serve
the standalone HTML5 web interface.

Endpoints:
- GET  /              : Serves web_ui/index.html
- POST /api/predict   : Generates ML prediction using trained Random Forest
- GET  /api/stats     : Returns dataset summary metrics and class distributions
=============================================================================
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import joblib
import pandas as pd

PORT = 5000
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "student_performance_model.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
HTML_PATH = os.path.join(BASE_DIR, "web_ui", "index.html")

# Global model bundle
MODEL_BUNDLE = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
        print(f"[OK] Random Forest model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load model: {e}")
else:
    print(f"[WARN] Model file not found at {MODEL_PATH}. Run 'python train_model.py' first.")

class MLPredictionHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local testing
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/" or parsed.path == "/index.html":
            if os.path.exists(HTML_PATH):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(HTML_PATH, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "index.html not found")
                return

        elif parsed.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            stats = {
                "total_records": 0,
                "avg_attendance": 0,
                "avg_internal": 0,
                "avg_study_hours": 0,
                "category_counts": {}
            }
            if os.path.exists(DATASET_PATH):
                try:
                    df = pd.read_csv(DATASET_PATH)
                    stats["total_records"] = len(df)
                    stats["avg_attendance"] = round(float(df["attendance"].mean()), 1)
                    stats["avg_internal"] = round(float(df["internal_marks"].mean()), 1)
                    stats["avg_study_hours"] = round(float(df["study_hours"].mean()), 1)
                    stats["category_counts"] = df["performance_category"].value_counts().to_dict()
                except Exception as e:
                    stats["error"] = str(e)

            self.wfile.write(json.dumps(stats).encode("utf-8"))
            return

        # Fallback to default static file serving from web_ui
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_error(400, f"Invalid JSON payload: {e}")
                return

            if MODEL_BUNDLE is None:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Model not loaded. Please execute 'python train_model.py' first."
                }).encode("utf-8"))
                return

            try:
                model = MODEL_BUNDLE["model"]
                feature_names = MODEL_BUNDLE["feature_names"]
                
                # Format into Pandas DataFrame
                input_df = pd.DataFrame([{
                    "attendance": float(data.get("attendance", 75)),
                    "internal_marks": float(data.get("internal_marks", 20)),
                    "assignment_completion": float(data.get("assignment_completion", 75)),
                    "previous_semester_percentage": float(data.get("previous_semester_percentage", 65)),
                    "study_hours": float(data.get("study_hours", 3))
                }])[feature_names]

                # Run inference
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                
                prob_dict = {cls: round(float(prob) * 100, 1) for cls, prob in zip(classes, probabilities)}

                # Rule-based suggestions
                att = float(data.get("attendance", 75))
                marks = float(data.get("internal_marks", 20))
                assign = float(data.get("assignment_completion", 75))
                prev = float(data.get("previous_semester_percentage", 65))
                hours = float(data.get("study_hours", 3))

                recommendations = []
                if att < 75:
                    recommendations.append("Attendance is below 75%: Attend all remaining lecture hours to avoid condonation issues.")
                if marks < 15:
                    recommendations.append("Internal marks are below 15/30: Focus on core theory concepts and attend tutorial sessions.")
                if assign < 70:
                    recommendations.append("Assignment completion is below 70%: Submit pending lab records and weekly assignments promptly.")
                if prev < 50:
                    recommendations.append("Previous semester score below 50%: Dedicate revision blocks to clear foundational weak concepts.")
                if hours < 2:
                    recommendations.append("Study hours below 2 hrs/day: Increase daily self-study to at least 2.5 to 3 hours.")

                if not recommendations:
                    recommendations.append("Excellent academic standing! All parameters meet or exceed the college benchmarks.")

                response_data = {
                    "prediction": prediction,
                    "probabilities": prob_dict,
                    "recommendations": recommendations,
                    "inputs": data
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

def run_server():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, MLPredictionHandler)
    print("=" * 65)
    print("[AI SYSTEM] Student Performance Prediction System - Web Server")
    print(f"Server running at: http://localhost:{PORT}")
    print("Open your browser and navigate to the URL above.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
