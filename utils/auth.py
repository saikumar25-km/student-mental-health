import json
import os
import hashlib
import streamlit as st

USER_DB_PATH = 'data/users.json'

def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def sign_up(username, password):
    init_db()
    with open(USER_DB_PATH, 'r') as f:
        users = json.load(f)
    
    if username in users:
        return False, "Username already exists."
    
    users[username] = hash_password(password)
    with open(USER_DB_PATH, 'w') as f:
        json.dump(users, f, indent=4)
    return True, "Account created successfully!"

def login(username, password):
    init_db()
    with open(USER_DB_PATH, 'r') as f:
        users = json.load(f)
    
    if username not in users:
        return False, "Username not found."
    
    if users[username] == hash_password(password):
        return True, "Login successful!"
    else:
        return False, "Incorrect password."

def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    return st.session_state['authenticated']

def logout():
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()
