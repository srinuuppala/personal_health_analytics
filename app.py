import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
import io

from database import (
    init_db, register_user, login_user, get_user,
    add_health_entry, get_health_data, update_user_settings,
    delete_entry
)
from ml_model import (
    predict_weight, rolling_average, detect_insights, get_bmi_category
)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HealthTrack Pro",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; color: white;
        text-align: center; margin: 5px 0;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.85; }
    .insight-success {
        background: #d4edda; border-left: 4px solid #28a745;
        padding: 10px 15px; border-radius: 6px; margin: 6px 0; color: #155724;
    }
    .insight-warning {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 10px 15px; border-radius: 6px; margin: 6px 0; color: #856404;
    }
    .insight-info {
        background: #cce5ff; border-left: 4px solid #0066cc;
        padding: 10px 15px; border-radius: 6px; margin: 6px 0; color: #004085;
    }
    .stButton > button {
        border-radius: 8px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Init DB ──────────────────────────────────────────────────
init_db()

# ─── Session State ────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ══════════════════════════════════════════════════════════════
# AUTH SCREENS
# ══════════════════════════════════════════════════════════════

def show_auth():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 💪 HealthTrack Pro")
        st.markdown("*Your personal health analytics & prediction system*")
        st.divider()

        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields.")
                    else:
                        user = login_user(username, password)
                        if user:
                            st.session_state.user = user
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Choose a Username")
                new_pass = st.text_input("Choose a Password", type="password")
                confirm  = st.text_input("Confirm Password", type="password")
                height   = st.number_input("Your Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submitted:
                    if not new_user or not new_pass:
                        st.error("Please fill in all fields.")
                    elif new_pass != confirm:
                        st.error("Passwords do not match.")
                    elif len(new_pass) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        ok, msg = register_user(new_user, new_pass, height)
                        if ok:
                            st.success(msg + " Please log in.")
                        else:
                            st.error(msg)

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def show_app():
    user = st.session_state.user
    # Refresh user data
    user = get_user(user['id'])
    st.session_state.user = user

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👋 Hello, {user['username'].title()}!")
        st.divider()
        page = st.radio("Navigate", [
            "📊 Dashboard",
            "➕ Log Entry",
            "📈 Analytics",
            "🤖 ML Predictions",
            "📜 History",
            "⚙️ Settings"
        ])
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # ── Pages ────────────────────────────────────────────────
    if page == "📊 Dashboard":
        show_dashboard(user)
    elif page == "➕ Log Entry":
        show_log_entry(user)
    elif page == "📈 Analytics":
        show_analytics(user)
    elif page == "🤖 ML Predictions":
        show_predictions(user)
    elif page == "📜 History":
        show_history(user)
    elif page == "⚙️ Settings":
        show_settings(user)


# ── Dashboard ─────────────────────────────────────────────────
def show_dashboard(user):
    st.title("📊 Dashboard")
    records = get_health_data(user['id'])

    if not records:
        st.info("No data yet! Go to **➕ Log Entry** to add your first entry.")
        return

    df = pd.DataFrame(records)

    # ── KPI Cards ─────────────────────────────────────────────
    latest = df.iloc[-1]
    bmi_cat, bmi_color = get_bmi_category(latest['bmi'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{latest['weight_kg']} kg</div>
            <div class="metric-label">Current Weight</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{latest['bmi']}</div>
            <div class="metric-label">Current BMI · {bmi_cat}</div></div>""", unsafe_allow_html=True)

    # 7-day change
    if len(df) >= 2:
        week_ago = df.iloc[max(0, len(df)-8)]['weight_kg']
        change7 = round(latest['weight_kg'] - week_ago, 2)
        sign = "+" if change7 > 0 else ""
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{sign}{change7} kg</div>
                <div class="metric-label">7-Day Change</div></div>""", unsafe_allow_html=True)
    else:
        with col3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">—</div>
                <div class="metric-label">7-Day Change</div></div>""", unsafe_allow_html=True)

    # Goal progress
    target = user.get('target_weight')
    if target:
        if len(df) > 1:
            start_w = df.iloc[0]['weight_kg']
            total_to_lose = abs(start_w - target)
            done = abs(latest['weight_kg'] - start_w)
            pct = min(100, round((done / total_to_lose) * 100, 1)) if total_to_lose > 0 else 100
        else:
            pct = 0
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{pct}%</div>
                <div class="metric-label">Goal Progress ({target} kg)</div></div>""", unsafe_allow_html=True)
    else:
        with col4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">No Goal</div>
                <div class="metric-label">Set one in ⚙️ Settings</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.line(df, x='date', y='weight_kg', title='⚖️ Weight Trend',
                      markers=True, line_shape='spline',
                      color_discrete_sequence=['#667eea'])
        if target:
            fig.add_hline(y=target, line_dash="dash", line_color="green",
                          annotation_text=f"Goal: {target} kg")
        fig.update_layout(xaxis_title="Date", yaxis_title="Weight (kg)",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.line(df, x='date', y='bmi', title='📏 BMI Trend',
                       markers=True, line_shape='spline',
                       color_discrete_sequence=['#764ba2'])
        fig2.add_hline(y=18.5, line_dash="dot", line_color="#3498db", annotation_text="18.5")
        fig2.add_hline(y=25, line_dash="dot", line_color="#2ecc71", annotation_text="25")
        fig2.add_hline(y=30, line_dash="dot", line_color="#e74c3c", annotation_text="30")
        fig2.update_layout(xaxis_title="Date", yaxis_title="BMI",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Smart Insights ────────────────────────────────────────
    st.subheader("💡 Smart Insights")
    insights = detect_insights(df)
    if insights:
        for icon, level, msg in insights:
            st.markdown(f'<div class="insight-{level}">{icon} {msg}</div>', unsafe_allow_html=True)
    else:
        st.info("Add more data to unlock smart insights.")


# ── Log Entry ─────────────────────────────────────────────────
def show_log_entry(user):
    st.title("➕ Log Today's Health Data")
    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("log_form"):
            entry_date = st.date_input("Date", value=date.today(), max_value=date.today())
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0,
                                     value=float(user.get('height_cm', 170) * 0.4),  # rough default
                                     step=0.1)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0,
                                     value=float(user.get('height_cm', 170.0)))
            notes = st.text_area("Notes (optional)", placeholder="e.g. After workout, morning weight...")
            submitted = st.form_submit_button("💾 Save Entry", use_container_width=True, type="primary")

            if submitted:
                ok, result = add_health_entry(user['id'], str(entry_date), weight, height, notes)
                if ok:
                    bmi = result
                    cat, _ = get_bmi_category(bmi)
                    st.success(f"✅ Entry saved! BMI: **{bmi}** ({cat})")
                    # Update height in settings too
                    update_user_settings(user['id'], height, user.get('target_weight'))
                else:
                    st.error(f"Error: {result}")

    with col2:
        st.subheader("📖 BMI Reference Chart")
        bmi_data = pd.DataFrame({
            'Category': ['Underweight', 'Normal', 'Overweight', 'Obese'],
            'BMI Range': ['< 18.5', '18.5 – 24.9', '25 – 29.9', '≥ 30'],
            'Color': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        })
        for _, row in bmi_data.iterrows():
            st.markdown(
                f"<div style='background:{row['Color']};color:white;padding:8px 14px;"
                f"border-radius:6px;margin:4px 0;font-weight:600'>"
                f"{row['Category']} &nbsp;|&nbsp; BMI {row['BMI Range']}</div>",
                unsafe_allow_html=True
            )
        st.markdown("")
        st.info("💡 Tip: Weigh yourself in the morning, before eating, for most consistent results.")


# ── Analytics ─────────────────────────────────────────────────
def show_analytics(user):
    st.title("📈 Advanced Analytics")
    records = get_health_data(user['id'])
    if len(records) < 2:
        st.info("Add at least 2 entries to see analytics.")
        return

    df = pd.DataFrame(records)
    df = rolling_average(df)
    df['date'] = pd.to_datetime(df['date'])
    df['weight_diff'] = df['weight_kg'].diff()

    # Rolling average chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['weight_kg'],
                             mode='lines+markers', name='Daily Weight',
                             line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['rolling_avg'],
                             mode='lines', name='7-Day Average',
                             line=dict(color='#e74c3c', width=2, dash='dash')))
    fig.update_layout(title='⚖️ Weight with 7-Day Rolling Average',
                      xaxis_title='Date', yaxis_title='Weight (kg)',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Daily change bar chart
        fig2 = px.bar(df.dropna(subset=['weight_diff']), x='date', y='weight_diff',
                      title='📊 Daily Weight Change',
                      color='weight_diff',
                      color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
                      labels={'weight_diff': 'Change (kg)'})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # BMI gauge
        latest_bmi = df.iloc[-1]['bmi']
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_bmi,
            title={'text': "Current BMI"},
            delta={'reference': df.iloc[-2]['bmi'] if len(df) > 1 else latest_bmi},
            gauge={
                'axis': {'range': [10, 40]},
                'steps': [
                    {'range': [10, 18.5], 'color': '#3498db'},
                    {'range': [18.5, 25], 'color': '#2ecc71'},
                    {'range': [25, 30], 'color': '#f39c12'},
                    {'range': [30, 40], 'color': '#e74c3c'},
                ],
                'bar': {'color': '#333'},
            }
        ))
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    # Stats table
    st.subheader("📋 Summary Statistics")
    stats = {
        "Metric": ["Min Weight", "Max Weight", "Avg Weight", "Min BMI", "Max BMI", "Avg BMI", "Total Entries"],
        "Value": [
            f"{df['weight_kg'].min():.2f} kg",
            f"{df['weight_kg'].max():.2f} kg",
            f"{df['weight_kg'].mean():.2f} kg",
            f"{df['bmi'].min():.2f}",
            f"{df['bmi'].max():.2f}",
            f"{df['bmi'].mean():.2f}",
            str(len(df))
        ]
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)


# ── ML Predictions ────────────────────────────────────────────
def show_predictions(user):
    st.title("🤖 ML Weight Predictions")
    records = get_health_data(user['id'])

    if len(records) < 3:
        st.warning("Please log at least 3 entries to generate predictions.")
        return

    df = pd.DataFrame(records)
    days_ahead = st.slider("Predict how many days ahead?", 7, 90, 30)
    pred_df, metrics, error = predict_weight(df, days_ahead)

    if error:
        st.error(error)
        return

    # Chart
    df['date'] = pd.to_datetime(df['date'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['weight_kg'],
                             mode='lines+markers', name='Actual Weight',
                             line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['predicted_weight'],
                             mode='lines', name='Predicted Weight',
                             line=dict(color='#e74c3c', width=2, dash='dot')))

    target = user.get('target_weight')
    if target:
        fig.add_hline(y=target, line_dash="dash", line_color="green",
                      annotation_text=f"🎯 Goal: {target} kg")

    fig.update_layout(title=f'📈 Weight Prediction — Next {days_ahead} Days',
                      xaxis_title='Date', yaxis_title='Weight (kg)',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Prediction insight
    final_pred = pred_df.iloc[-1]['predicted_weight']
    current_w = df.iloc[-1]['weight_kg']
    trend_word = "increase" if final_pred > current_w else "decrease"
    st.info(f"🤖 **ML Insight:** At your current trend, your weight will **{trend_word}** "
            f"from **{current_w} kg** to **{final_pred} kg** in {days_ahead} days.")

    if target:
        remaining = round(final_pred - target, 2)
        if remaining > 0:
            st.warning(f"📋 You will still be **{remaining} kg above** your goal after {days_ahead} days.")
        else:
            st.success(f"🎉 At this rate, you will **reach your goal** within {days_ahead} days!")

    # Model metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Model R² Score", metrics['r2'], help="Closer to 1.0 = better fit")
    col2.metric("Mean Abs Error", f"±{metrics['mae']} kg")
    col3.metric("Daily Trend", f"{metrics['slope_per_day']:+.3f} kg/day",
                delta=metrics['trend'])


# ── History ───────────────────────────────────────────────────
def show_history(user):
    st.title("📜 Entry History")

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("From", value=date.today() - timedelta(days=30))
    with col2:
        end = st.date_input("To", value=date.today())

    records = get_health_data(user['id'], start_date=start, end_date=end)

    if not records:
        st.info("No entries found for the selected date range.")
        return

    df = pd.DataFrame(records)[['date', 'weight_kg', 'bmi', 'notes']]
    df.columns = ['Date', 'Weight (kg)', 'BMI', 'Notes']
    df = df.sort_values('Date', ascending=False).reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Export as CSV",
        data=csv_buf.getvalue(),
        file_name=f"health_data_{user['username']}.csv",
        mime="text/csv",
        type="primary"
    )

    # Delete entry
    st.divider()
    st.subheader("🗑️ Delete an Entry")
    del_date = st.date_input("Select date to delete")
    if st.button("Delete Entry", type="secondary"):
        delete_entry(user['id'], str(del_date))
        st.success(f"Entry for {del_date} deleted.")
        st.rerun()


# ── Settings ──────────────────────────────────────────────────
def show_settings(user):
    st.title("⚙️ Settings")
    col1, _ = st.columns([1, 1])
    with col1:
        with st.form("settings_form"):
            st.subheader("👤 Profile Settings")
            height = st.number_input("Your Height (cm)", min_value=100.0, max_value=250.0,
                                     value=float(user.get('height_cm', 170.0)))
            target = st.number_input("Target Weight (kg) — set 0 to clear",
                                     min_value=0.0, max_value=300.0,
                                     value=float(user.get('target_weight') or 0.0))
            submitted = st.form_submit_button("💾 Save Settings", type="primary", use_container_width=True)
            if submitted:
                update_user_settings(user['id'], height, target if target > 0 else None)
                st.success("Settings saved!")
                st.rerun()

        st.divider()
        st.subheader("ℹ️ Account Info")
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Member since:** {user['created_at'][:10]}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if st.session_state.user is None:
    show_auth()
else:
    show_app()
