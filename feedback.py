from flask import Blueprint, request, jsonify
import json

feedback_bp = Blueprint('feedback', __name__)
feedback_store = "feedback.json"

@feedback_bp.route('/feedback', methods=['POST'])
def save_feedback():
    feedback = request.json
    try:
        with open(feedback_store, 'a') as f:
            f.write(json.dumps(feedback) + '\n')
        return jsonify({"message": "Feedback saved successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
