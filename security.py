import os
import base64
import json
import uuid
import datetime
import hashlib
import jwt
from passlib.hash import bcrypt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

class SecurityManager:
    """Manager for authentication and security."""
    
    def __init__(self, secret_key=None):
        """Initialize the security manager."""
        # Use provided secret key or generate one
        self.secret_key = secret_key or os.environ.get('SECRET_KEY') or self._generate_secret()
        
        # Create the users directory if it doesn't exist
        self.users_dir = Path("./users")
        self.users_dir.mkdir(exist_ok=True)
        
        # Initialize roles and permissions
        self.roles = {
            "admin": {
                "permissions": ["read", "write", "edit", "delete", "manage_users"]
            },
            "legal_professional": {
                "permissions": ["read", "write", "edit", "delete"]
            },
            "paralegal": {
                "permissions": ["read", "write"]
            },
            "client": {
                "permissions": ["read"]
            }
        }
        
        # Save roles to file if it doesn't exist
        roles_file = self.users_dir / "roles.json"
        if not roles_file.exists():
            with open(roles_file, 'w') as f:
                json.dump(self.roles, f, indent=2)
        
        # Initialize demo user if no users exist
        if not list(self.users_dir.glob("*.user")):
            self.create_user(
                username="admin",
                password="admin123",
                email="admin@legalassist.com",
                full_name="Admin User",
                role="admin"
            )
    
    def _generate_secret(self) -> str:
        """Generate a secure random secret key."""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def create_user(self, username: str, password: str, email: str, 
                   full_name: str, role: str = "client") -> bool:
        """Create a new user."""
        # Validate role
        if role not in self.roles:
            return False
        
        # Check if username already exists
        user_path = self.users_dir / f"{username}.user"
        if user_path.exists():
            return False
        
        # Hash the password
        password_hash = bcrypt.hash(password)
        
        # Create user object
        user = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "full_name": full_name,
            "role": role,
            "created_at": datetime.datetime.now().isoformat(),
            "last_login": None,
            "active": True
        }
        
        # Save user to file
        try:
            with open(user_path, 'w') as f:
                json.dump(user, f, indent=2)
            return True
        except Exception:
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate a user and return a JWT token if successful."""
        user = self._get_user(username)
        if not user:
            return None
        
        if not user["active"]:
            return None
            
        # Verify password
        if not bcrypt.verify(password, user["password_hash"]):
            return None
        
        # Update last login
        user["last_login"] = datetime.datetime.now().isoformat()
        user_path = self.users_dir / f"{username}.user"
        with open(user_path, 'w') as f:
            json.dump(user, f, indent=2)
        
        # Generate token
        expiration = datetime.datetime.now() + datetime.timedelta(hours=12)
        payload = {
            "sub": username,
            "role": user["role"],
            "exp": expiration.timestamp()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify a JWT token and return a tuple of (is_valid, payload)."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check if user still exists and is active
            user = self._get_user(payload["sub"])
            if not user or not user["active"]:
                return False, None
                
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except (jwt.InvalidTokenError, Exception):
            return False, {"error": "Invalid token"}
    
    def has_permission(self, username: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        user = self._get_user(username)
        if not user or not user["active"]:
            return False
            
        user_role = user["role"]
        if user_role not in self.roles:
            return False
            
        return permission in self.roles[user_role]["permissions"]
    
    def change_password(self, username: str, current_password: str, new_password: str) -> bool:
        """Change a user's password."""
        user = self._get_user(username)
        if not user:
            return False
            
        # Verify current password
        if not bcrypt.verify(current_password, user["password_hash"]):
            return False
            
        # Update password
        user["password_hash"] = bcrypt.hash(new_password)
        
        # Save updated user
        user_path = self.users_dir / f"{username}.user"
        try:
            with open(user_path, 'w') as f:
                json.dump(user, f, indent=2)
            return True
        except Exception:
            return False
    
    def get_users(self) -> List[Dict]:
        """Get a list of all users (without password hashes)."""
        users = []
        for user_file in self.users_dir.glob("*.user"):
            try:
                with open(user_file, 'r') as f:
                    user = json.load(f)
                    # Remove sensitive information
                    if "password_hash" in user:
                        del user["password_hash"]
                    users.append(user)
            except Exception:
                continue
        return users
    
    def _get_user(self, username: str) -> Optional[Dict]:
        """Get a user by username."""
        user_path = self.users_dir / f"{username}.user"
        if not user_path.exists():
            return None
            
        try:
            with open(user_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def update_user(self, username: str, updates: Dict, admin_override: bool = False) -> bool:
        """Update a user's information."""
        user = self._get_user(username)
        if not user:
            return False
        
        # Don't allow updating certain fields
        protected_fields = ["username", "password_hash", "created_at"]
        for field in protected_fields:
            if field in updates:
                del updates[field]
        
        # Update the user object
        user.update(updates)
        
        # Save updated user
        user_path = self.users_dir / f"{username}.user"
        try:
            with open(user_path, 'w') as f:
                json.dump(user, f, indent=2)
            return True
        except Exception:
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        user_path = self.users_dir / f"{username}.user"
        if not user_path.exists():
            return False
            
        try:
            user_path.unlink()
            return True
        except Exception:
            return False
    
    def get_activity_log(self, username: Optional[str] = None) -> List[Dict]:
        """Get activity log for a user or all users."""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        logs = []
        if username:
            log_path = log_dir / f"{username}_activity.log"
            if log_path.exists():
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line))
                        except:
                            pass
        else:
            for log_file in log_dir.glob("*_activity.log"):
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line))
                        except:
                            pass
        
        # Sort logs by timestamp
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs
    
    def log_activity(self, username: str, action: str, details: Dict = None) -> bool:
        """Log a user activity."""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "username": username,
            "action": action,
            "details": details or {}
        }
        
        log_path = log_dir / f"{username}_activity.log"
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            return True
        except Exception:
            return False