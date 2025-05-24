from flask import Flask
from dashboard.routes import dashboard_bp 
from feedback import feedback_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(feedback_bp)
    return app