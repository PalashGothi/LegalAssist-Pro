from flask import Blueprint, render_template, jsonify
import time
import psutil

dashboard_bp = Blueprint('dashboard', __name__)

metrics_data = {
    "requests": [],
    "summarization_times": [],
    "accuracy_scores": []
}

@dashboard_bp.route("/metrics")
def metrics():
    return jsonify(metrics_data)

@dashboard_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=metrics_data)

def record_metrics(response_time, accuracy=None):
    metrics_data["requests"].append(time.time())
    metrics_data["summarization_times"].append(response_time)
    if accuracy:
        metrics_data["accuracy_scores"].append(accuracy)
