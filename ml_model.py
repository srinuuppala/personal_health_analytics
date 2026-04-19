import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score

def prepare_data(df):
    """Convert health data dataframe into ML-ready format."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['day_index'] = (df['date'] - df['date'].min()).dt.days
    return df

def predict_weight(df, days_ahead=30):
    """
    Predict future weight using Linear Regression.
    Returns prediction dataframe and model metrics.
    """
    if len(df) < 3:
        return None, None, "Need at least 3 data points for prediction."

    df = prepare_data(df)
    X = df[['day_index']].values
    y = df['weight_kg'].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred_train = model.predict(X)
    mae = round(mean_absolute_error(y, y_pred_train), 3)
    r2 = round(r2_score(y, y_pred_train), 3)

    last_day = df['day_index'].max()
    last_date = df['date'].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_dates = [last_date + pd.Timedelta(days=int(i)) for i in range(1, days_ahead + 1)]

    future_weights = model.predict(future_days)

    pred_df = pd.DataFrame({
        'date': future_dates,
        'predicted_weight': np.round(future_weights, 2)
    })

    metrics = {
        'mae': mae,
        'r2': r2,
        'slope_per_day': round(model.coef_[0], 4),
        'trend': 'gaining' if model.coef_[0] > 0 else 'losing'
    }

    return pred_df, metrics, None

def rolling_average(df, window=7):
    """Calculate rolling average weight."""
    df = prepare_data(df)
    df['rolling_avg'] = df['weight_kg'].rolling(window=window, min_periods=1).mean().round(2)
    return df

def detect_insights(df):
    """Detect patterns and generate health insights."""
    insights = []
    if len(df) < 2:
        return insights

    df = prepare_data(df)
    weights = df['weight_kg'].values
    bmis = df['bmi'].values

    # Check last 3 days trend
    if len(weights) >= 3:
        last3 = weights[-3:]
        if all(last3[i] < last3[i+1] for i in range(2)):
            insights.append(("⚠️", "warning", "Weight has been increasing for 3+ consecutive days. Consider reviewing your diet."))
        elif all(last3[i] > last3[i+1] for i in range(2)):
            insights.append(("✅", "success", "Great progress! Weight has been decreasing consistently for 3 days."))

    # Sudden change detection
    if len(weights) >= 2:
        recent_change = weights[-1] - weights[-2]
        if abs(recent_change) > 1.5:
            direction = "increased" if recent_change > 0 else "decreased"
            insights.append(("⚠️", "warning", f"Sudden weight {direction} by {abs(recent_change):.1f} kg detected. Could be water retention or measurement error."))

    # Weekly comparison
    if len(df) >= 7:
        last_7 = weights[-7:]
        prev_7 = weights[-14:-7] if len(weights) >= 14 else weights[:7]
        avg_last = np.mean(last_7)
        avg_prev = np.mean(prev_7)
        diff = round(avg_last - avg_prev, 2)
        if diff < 0:
            insights.append(("🎉", "success", f"This week's average is {abs(diff)} kg lower than the previous period. Keep it up!"))
        elif diff > 0:
            insights.append(("📈", "info", f"This week's average is {abs(diff)} kg higher than the previous period."))

    # BMI category
    current_bmi = bmis[-1]
    if current_bmi < 18.5:
        insights.append(("⚠️", "warning", f"Current BMI ({current_bmi}) is in the Underweight range. Consider consulting a nutritionist."))
    elif 18.5 <= current_bmi < 25:
        insights.append(("✅", "success", f"Current BMI ({current_bmi}) is in the Normal range. Great job maintaining a healthy weight!"))
    elif 25 <= current_bmi < 30:
        insights.append(("📋", "info", f"Current BMI ({current_bmi}) is in the Overweight range."))
    else:
        insights.append(("⚠️", "warning", f"Current BMI ({current_bmi}) is in the Obese range. Consider seeking medical advice."))

    return insights

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3498db"
    elif bmi < 25:
        return "Normal", "#2ecc71"
    elif bmi < 30:
        return "Overweight", "#f39c12"
    else:
        return "Obese", "#e74c3c"
