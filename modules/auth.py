
import bcrypt
import jwt
import datetime
import os
import json
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")

class AuthManager:
    def __init__(self):
        self.users_file = "data/users.json"
        self.users = self._load_users_from_file()

    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def _load_users_from_file(self) -> Dict[str, dict]:
        """Load users from JSON file."""
        try:
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            return {
                "admin": {
                    "password": self._hash_password("admin123"),
                    "role": "admin"
                },
                "user": {
                    "password": self._hash_password("user123"),
                    "role": "user"
                }
            }
        except Exception:
            return {}

    def _save_users_to_file(self):
        """Save users to JSON file."""
        os.makedirs("data", exist_ok=True)
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """Authenticate a user and return (success, token) tuple"""
        user = self.users.get(username)
        if user and bcrypt.checkpw(password.encode(), user['password'].encode()):
            token = self.generate_token(username)
            return True, token
        return False, None

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user account"""
        if username in self.users:
            return False
        self.users[username] = {
            "password": self._hash_password(password),
            "role": role
        }
        self._save_users_to_file()
        return True

    def generate_token(self, username: str) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            "sub": username,
            "role": self.users[username]["role"],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    def decode_token(self, token: str) -> Optional[dict]:
        """Decode and verify JWT token"""
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

# Standalone functions for backward compatibility
def hash_password(password: str) -> str:
    """Standalone password hashing function"""
    return AuthManager()._hash_password(password)

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[str]]:
    """Standalone authentication function"""
    return AuthManager().authenticate_user(username, password)

def create_user(username: str, password: str, role: str = "user") -> bool:
    """Standalone user creation function"""
    return AuthManager().create_user(username, password, role)

def load_users() -> Dict[str, dict]:
    """Standalone function to load users"""
    return AuthManager()._load_users_from_file()

