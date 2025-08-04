import streamlit as st
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import numpy as np
import streamlit.components.v1 as components

# -------------------------
# 1. Load Model
# -------------------------
import cloudpickle

with open("tuned_personality_model.pk2.0", "rb") as f:
    model = cloudpickle.load(f)


# -------------------------
# 2. Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Personality Prediction Dashboard")
st.write("Predict if you are an **Introvert** or **Extrovert** using AI!")

# -------------------------
# 3. Sidebar Input
# -------------------------

def is_mobile_device():
    # Use st.query_params instead of deprecated st.experimental_get_query_params
    user_agent = st.query_params.get('user-agent', [''])[0] if hasattr(st, 'query_params') else ''
    if not user_agent:
        user_agent = ''
    ua = user_agent.lower()
    return any(x in ua for x in ['android', 'iphone', 'ipad', 'mobile'])

is_mobile = is_mobile_device()

# Move the input form to the main page for all users
st.header("🔧 Input Your Data")
with st.form("input_form"):
    time_spent_alone = st.slider("⏳ Time spent alone (hours/day)", 0, 24, 5)
    friends_circle_size = st.slider("👥 Close friends count", 0, 50, 10)
    social_event_attendance = st.slider("🎉 Social events per month", 0, 20, 2)
    post_frequency = st.slider("📱 Social media posts per month", 0, 100, 7)
    going_outside = st.slider("🚶 Times you go outside per week", 0, 30, 3)
    stage_fear = st.radio("🎤 Stage fear?", ["Yes", "No"])
    drained_after_socializing = st.radio("😓 Drained after socializing?", ["Yes", "No"])
    submitted = st.form_submit_button("🔮 Predict Personality")

input_data = pd.DataFrame([{
    "Time_spent_Alone": time_spent_alone,
    "Friends_circle_size": friends_circle_size,
    "Social_event_attendance": social_event_attendance,
    "Going_outside": going_outside,
    "Post_frequency": post_frequency,
    "Stage_fear": stage_fear,
    "Drained_after_socializing": drained_after_socializing
}])

# -------------------------
# 4. Prediction
# -------------------------
if submitted:
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    class_labels = model.classes_

    confidence = max(proba) * 100
    # Only predict between Introvert and Extrovert
    # Filter to only those two classes if present
    valid_classes = [c for c in class_labels if c in ["Introvert", "Extrovert"]]
    # If model has more classes, filter proba and class_labels
    if len(valid_classes) == 2:
        idxs = [list(class_labels).index(c) for c in valid_classes]
        proba = [proba[i] for i in idxs]
        class_labels = valid_classes
        # If prediction is not in valid_classes, pick highest proba
        if prediction not in valid_classes:
            personality_label = valid_classes[int(np.argmax(proba))]
        else:
            personality_label = prediction
    else:
        personality_label = prediction

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"""
        <div style="padding:20px; background:white; border-radius:15px; text-align:center; box-shadow:0px 4px 10px rgba(0,0,0,0.1);">
            <h2>✅ Predicted Personality</h2>
            <h1 style='color:#4CAF50;'>{personality_label}</h1>
            <h3>Confidence: {confidence:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#4CAF50"}},
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # Probability Bar Chart
        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({"Personality": class_labels, "Probability": proba})
        fig_bar = px.bar(prob_df, x="Personality", y="Probability", color="Personality",
                         text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%"),
                         color_discrete_sequence=["#4CAF50", "#2196F3"])
        fig_bar.update_layout(yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------
    # 5. Combine Visuals into Single Report for Download
    # -------------------------
    # Create a single figure with Gauge + Probability Bar as subplots
    fig_report = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Confidence Gauge", "Prediction Probabilities"),
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )

    # Add gauge indicator to subplot 1
    fig_report.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#4CAF50"}},
        ),
        row=1, col=1
    )

    # Add probability bar chart to subplot 2
    fig_report.add_trace(
        go.Bar(
            x=prob_df["Personality"],
            y=prob_df["Probability"],
            text=prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%"),
            marker_color=["#4CAF50" if p == personality_label else "#2196F3" for p in prob_df["Personality"]],
        ),
        row=1, col=2
    )
    fig_report.update_yaxes(range=[0, 1], row=1, col=2)
    fig_report.update_layout(height=500, width=1200)

    # Show the combined figure interactively
    st.plotly_chart(fig_report, use_container_width=True)

    # Optionally, add CSV download for probabilities
    st.download_button(
        label="📥 Download Probabilities as CSV",
        data=prob_df.to_csv(index=False).encode(),
        file_name="personality_probabilities.csv",
        mime="text/csv"
    )
