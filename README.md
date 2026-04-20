# 💪 HealthTrack Pro — Personal Health Analytics & Prediction System

A multi-user Streamlit web app for tracking weight, BMI, and predicting future health trends using Machine Learning.

---

## 🚀 Quick Start

### 1. Clone / Download the project
```bash
git clone https://github.com/YOUR_USERNAME/healthtrack-pro.git
cd healthtrack-pro
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
healthtrack-pro/
│
├── app.py              # Main Streamlit app (UI + routing)
├── database.py         # SQLite database setup & queries
├── ml_model.py         # Machine Learning predictions & insights
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── health_app.db       # SQLite database (auto-created on first run)
```

---

## 🔑 Features

| Feature | Description |
|---|---|
| 👤 Auth | Register/Login with hashed passwords (SHA-256) |
| ➕ Log Entry | Daily weight + height → auto BMI calculation |
| 📊 Dashboard | KPI cards, weight/BMI charts, smart insights |
| 📈 Analytics | Rolling average, daily change, BMI gauge |
| 🤖 ML Predictions | Linear Regression to predict next 7–90 days |
| 📜 History | Filter by date, export CSV, delete entries |
| ⚙️ Settings | Update height, set goal weight |

---

## 🧠 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.10+
- **Database:** SQLite (via built-in `sqlite3`)
- **ML:** scikit-learn (LinearRegression)
- **Visualization:** Plotly
- **Data:** pandas, numpy

---

## 📝 Description

> "Developed a multi-user health analytics web application using Streamlit, enabling daily tracking of weight and BMI with interactive dashboards and machine learning-based predictions for future weight trends. Implemented SHA-256 authentication, SQLite storage, and Plotly visualizations to provide actionable health insights."

---

## 🔮 Future Enhancements

- [ ] Calorie & nutrition tracking
- [ ] Export PDF report
- [ ] Email reminders
- [ ] ARIMA / time-series model
- [ ] Mobile-friendly PWA version
