import sqlite3
import hashlib
import os

DB_PATH = "health_app.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            height_cm REAL DEFAULT 170.0,
            target_weight REAL DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS health_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            weight_kg REAL NOT NULL,
            bmi REAL NOT NULL,
            notes TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, height_cm=170.0):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password_hash, height_cm) VALUES (?, ?, ?)",
            (username.strip().lower(), hash_password(password), height_cm)
        )
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username.strip().lower(), hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def update_user_settings(user_id, height_cm, target_weight):
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE users SET height_cm = ?, target_weight = ? WHERE id = ?",
        (height_cm, target_weight, user_id)
    )
    conn.commit()
    conn.close()

def add_health_entry(user_id, date, weight_kg, height_cm, notes=""):
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR REPLACE INTO health_data (user_id, date, weight_kg, bmi, notes) VALUES (?, ?, ?, ?, ?)",
            (user_id, date, weight_kg, bmi, notes)
        )
        conn.commit()
        return True, bmi
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def get_health_data(user_id, start_date=None, end_date=None):
    conn = get_connection()
    c = conn.cursor()
    query = "SELECT * FROM health_data WHERE user_id = ?"
    params = [user_id]
    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))
    query += " ORDER BY date ASC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_entry(user_id, date):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM health_data WHERE user_id = ? AND date = ?", (user_id, date))
    conn.commit()
    conn.close()
