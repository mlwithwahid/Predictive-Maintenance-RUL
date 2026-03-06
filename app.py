import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from backend.predict import load_model, predict_rul, get_shap_values, get_global_shap, get_risk_level

st.set_page_config(
    page_title = "PredictiveMX — Turbofan Health Monitor",
    page_icon  = "✈️",
    layout     = "wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@700;800&display=swap');
:root {
    --bg-primary:#0a0e1a; --bg-card:#131d35; --bg-dark:#0f1629;
    --blue:#3b82f6; --cyan:#06b6d4; --green:#22c55e;
    --red:#ef4444; --amber:#f59e0b;
    --text:#e2e8f0; --muted:#94a3b8; --border:#1e3a5f;
}
html,body,[class*="css"]{font-family:'Barlow',sans-serif;background:var(--bg-primary)!important;color:var(--text)!important;}
.stApp{background:var(--bg-primary)!important;}
header[data-testid="stHeader"]{display:none;}
[data-testid="stSidebar"]{display:none;}
[data-testid="metric-container"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:16px!important;}
[data-testid="stMetricValue"]{font-family:'Barlow Condensed',sans-serif!important;font-size:2rem!important;color:var(--cyan)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}
.stButton>button{background:linear-gradient(135deg,var(--blue),var(--cyan))!important;color:white!important;border:none!important;border-radius:6px!important;font-family:'Barlow Condensed',sans-serif!important;font-size:1rem!important;font-weight:700!important;letter-spacing:1px!important;padding:10px 24px!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 4px 20px rgba(59,130,246,0.4)!important;}
.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg-card)!important;border-radius:8px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'Barlow Condensed',sans-serif!important;font-weight:700!important;}
.stTabs [aria-selected="true"]{background:var(--blue)!important;color:white!important;border-radius:6px!important;}
hr{border-color:var(--border)!important;opacity:0.5!important;}
</style>
""", unsafe_allow_html=True)

# ── TOP NAVIGATION BAR ────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0f1629;border-bottom:1px solid #1e3a5f;
            padding:12px 24px;margin:-1rem -1rem 2rem -1rem;
            display:flex;align-items:center;gap:32px;">
    <div style="display:flex;align-items:center;gap:10px;min-width:200px;">
        <span style="font-size:1.6rem">✈️</span>
        <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                        font-weight:800;color:#06b6d4;letter-spacing:2px;">PREDICTIVE MX</div>
            <div style="font-size:0.65rem;color:#475569;letter-spacing:1px;">TURBOFAN HEALTH MONITOR</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation using columns with buttons
nav_cols = st.columns(5)
pages    = ["🏠 Overview", "🔮 RUL Prediction", "📊 SHAP Explainability", "📈 Sensor Trends", "🏆 Model Comparison"]

if 'page' not in st.session_state:
    st.session_state['page'] = "🏠 Overview"

for i, (col, p) in enumerate(zip(nav_cols, pages)):
    with col:
        is_active = st.session_state['page'] == p
        btn_style = f"""
        <style>
        div[data-testid="column"]:nth-child({i+1}) .stButton>button {{
            background: {'linear-gradient(135deg,#3b82f6,#06b6d4)' if is_active else '#131d35'} !important;
            border: 1px solid {'#3b82f6' if is_active else '#1e3a5f'} !important;
            width: 100% !important;
            font-size: 0.85rem !important;
            padding: 8px 4px !important;
        }}
        </style>
        """
        st.markdown(btn_style, unsafe_allow_html=True)
        if st.button(p, key=f"nav_{i}", use_container_width=True):
            st.session_state['page'] = p
            st.rerun()

page = st.session_state['page']
st.markdown("<hr style='margin:0 0 24px 0;'>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <span style="font-size:1.6rem">{icon}</span>
            <h2 style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
                       font-weight:800;color:#e2e8f0;margin:0;letter-spacing:1px;">
                {title.upper()}
            </h2>
        </div>
        {"" if not subtitle else f'<p style="color:#94a3b8;margin:0;font-size:0.9rem;padding-left:52px;">{subtitle}</p>'}
        <div style="height:2px;background:linear-gradient(90deg,#3b82f6,transparent);margin-top:12px;border-radius:2px;"></div>
    </div>
    """, unsafe_allow_html=True)

PLOT_TEMPLATE = dict(layout=go.Layout(
    paper_bgcolor="#131d35", plot_bgcolor="#0f1629",
    font=dict(family="Barlow, sans-serif", color="#e2e8f0"),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    colorway=["#3b82f6","#06b6d4","#22c55e","#f59e0b","#ef4444","#8b5cf6","#ec4899"],
))

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    return load_model()

@st.cache_data
def load_data():
    train_df = pd.read_csv("models/train_df.csv")
    test_df  = pd.read_csv("models/test_df.csv")
    rul_df   = pd.read_csv("models/rul_df.csv")
    return train_df, test_df, rul_df

model, feature_cols = load()
train_df, test_df, rul_df = load_data()
test_last = test_df.groupby('unit_id').last().reset_index()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    section_header("🏠", "System Overview", "NASA Turbofan FD001 — Explainable Predictive Maintenance")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🚀 Engines",      "100")
    c2.metric("📡 Sensors Used", str(len(feature_cols)))
    c3.metric("🎯 Test R²",      "0.81")
    c4.metric("📉 Test RMSE",    "18.12")
    c5.metric("📏 Test MAE",     "12.93")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div style="background:#131d35;border:1px solid #1e3a5f;border-radius:10px;padding:24px;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                        font-weight:700;color:#06b6d4;letter-spacing:1px;margin-bottom:16px;">
                ABOUT THIS PROJECT
            </div>
            <p style="color:#94a3b8;line-height:1.8;font-size:0.95rem;">
                This dashboard predicts the <b style="color:#e2e8f0">Remaining Useful Life (RUL)</b>
                of jet engines using real NASA sensor data and explains every prediction
                using <b style="color:#e2e8f0">SHAP (SHapley Additive exPlanations)</b>.
            </p>
            <div style="margin-top:16px;display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div style="background:#0f1629;border-radius:6px;padding:12px;border-left:3px solid #3b82f6;">
                    <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">DATASET</div>
                    <div style="font-weight:600;color:#e2e8f0;">NASA CMAPSS FD001</div>
                </div>
                <div style="background:#0f1629;border-radius:6px;padding:12px;border-left:3px solid #06b6d4;">
                    <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">FAULT MODE</div>
                    <div style="font-weight:600;color:#e2e8f0;">HPC Degradation</div>
                </div>
                <div style="background:#0f1629;border-radius:6px;padding:12px;border-left:3px solid #22c55e;">
                    <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">MODEL</div>
                    <div style="font-weight:600;color:#e2e8f0;">XGBoost Regressor</div>
                </div>
                <div style="background:#0f1629;border-radius:6px;padding:12px;border-left:3px solid #f59e0b;">
                    <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">EXPLAINABILITY</div>
                    <div style="font-weight:600;color:#e2e8f0;">SHAP Values</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=train_df['RUL'], nbinsx=50,
            marker=dict(color="rgba(59,130,246,0.7)", line=dict(color="#06b6d4", width=0.5))))
        fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
            title="RUL Distribution — Training Set", xaxis_title="RUL (cycles)",
            yaxis_title="Count", height=280, margin=dict(l=40,r=20,t=40,b=40), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    engine_lives = train_df.groupby('unit_id')['cycle'].max().reset_index()
    fig2 = go.Figure(go.Bar(x=engine_lives['unit_id'], y=engine_lives['cycle'],
        marker_color="rgba(6,182,212,0.6)", marker_line=dict(color="#3b82f6", width=0.5)))
    fig2.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
        xaxis_title="Engine Unit ID", yaxis_title="Total Cycles",
        height=220, margin=dict(l=40,r=20,t=10,b=40), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RUL PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif "RUL Prediction" in page:
    section_header("🔮", "RUL Prediction", "Predict remaining useful life from sensor readings")

    input_mode = st.radio("Input Mode", ["📋 Preloaded Engine", "✏️ Manual Sensor Input"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "Preloaded" in input_mode:
        col1, col2 = st.columns([1, 2])
        with col1:
            engine_id     = st.selectbox("Select Engine ID", options=range(1, 101))
            true_rul      = int(rul_df.iloc[engine_id - 1]['RUL'])
            engine_row    = test_last[test_last['unit_id'] == engine_id].iloc[0]
            sensor_values = {col: engine_row[col] for col in feature_cols}
            st.markdown(f"""
            <div style="background:#131d35;border:1px solid #1e3a5f;border-radius:8px;padding:16px;margin-top:12px;">
                <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">TRUE RUL</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.5rem;font-weight:800;color:#06b6d4;">
                    {true_rul} <span style="font-size:1rem;color:#64748b;">cycles</span>
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            sensor_df = pd.DataFrame({'Sensor': list(sensor_values.keys()),
                                      'Value': [round(v,4) for v in sensor_values.values()]})
            st.markdown("**Current Sensor Readings (last cycle):**")
            st.dataframe(sensor_df, use_container_width=True, height=200)
    else:
        sensor_values = {}
        cols = st.columns(3)
        for i, col in enumerate(feature_cols):
            col_min  = float(train_df[col].min())
            col_max  = float(train_df[col].max())
            col_mean = float(train_df[col].mean())
            with cols[i % 3]:
                sensor_values[col] = st.slider(col, min_value=round(col_min,4),
                    max_value=round(col_max,4), value=round(col_mean,4),
                    step=round((col_max-col_min)/200,4))
        true_rul = None

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀  PREDICT REMAINING USEFUL LIFE", use_container_width=True):
        rul_pred = predict_rul(model, feature_cols, sensor_values)
        risk_label, risk_color, risk_msg = get_risk_level(rul_pred)

        st.markdown(f"""
        <div style="background:{risk_color}22;border:2px solid {risk_color};border-radius:10px;
                    padding:20px;text-align:center;margin:16px 0;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:800;
                        color:{risk_color};letter-spacing:2px;">● {risk_label}</div>
            <div style="color:#e2e8f0;margin-top:6px;">{risk_msg}</div>
        </div>""", unsafe_allow_html=True)

        col1,col2,col3 = st.columns(3)
        col1.metric("⏱️ Predicted RUL", f"{rul_pred} cycles")
        col2.metric("⚠️ Risk Status",   risk_label)
        if true_rul is not None:
            col3.metric("✅ True RUL", f"{true_rul} cycles", delta=f"{round(rul_pred-true_rul,1):+.1f} cycles")

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=rul_pred,
            title={'text':"Predicted RUL (cycles)",'font':{'color':'#e2e8f0','size':16}},
            number={'font':{'color':'#06b6d4','size':48}},
            gauge={'axis':{'range':[0,125],'tickcolor':'#64748b'},
                   'bar':{'color':risk_color,'thickness':0.3},'bgcolor':'#0f1629',
                   'steps':[{'range':[0,30],'color':'rgba(239,68,68,0.15)'},
                             {'range':[30,70],'color':'rgba(245,158,11,0.15)'},
                             {'range':[70,125],'color':'rgba(34,197,94,0.15)'}]}))
        fig.update_layout(paper_bgcolor="#131d35", font=dict(color="#e2e8f0"),
                          height=300, margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    section_header("📊", "SHAP Explainability", "Understand why the model makes each prediction")
    tab1, tab2 = st.tabs(["🌍 Global Feature Importance", "🔍 Local Engine Explanation"])

    with tab1:
        st.markdown("**Which sensors drive RUL predictions across all 100 engines?**")
        if st.button("Compute Global SHAP", type="primary"):
            with st.spinner("Computing SHAP values..."):
                shap_df = get_global_shap(model, feature_cols, test_last[feature_cols])
            col1, col2 = st.columns([1.5, 1])
            with col1:
                fig = go.Figure(go.Bar(x=shap_df['Mean SHAP Value'], y=shap_df['Feature'],
                    orientation='h', marker=dict(color=shap_df['Mean SHAP Value'],
                    colorscale=[[0,"#1e3a5f"],[0.5,"#3b82f6"],[1,"#06b6d4"]], showscale=False)))
                fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                    title="Global Feature Importance (Mean |SHAP|)", xaxis_title="Mean |SHAP Value|",
                    height=420, margin=dict(l=120,r=30,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(shap_df.reset_index(drop=True), use_container_width=True, height=400)
            top3 = shap_df['Feature'].iloc[:3].tolist()
            st.markdown(f"""
            <div style="background:#131d35;border:1px solid #1e3a5f;border-left:4px solid #06b6d4;
                        border-radius:8px;padding:20px;margin-top:16px;">
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;color:#06b6d4;
                            letter-spacing:1px;margin-bottom:10px;">🧠 KEY INSIGHT</div>
                <p style="color:#94a3b8;margin:0;line-height:1.7;">
                    Top 3 sensors: <b style="color:#e2e8f0">{top3[0]}</b>,
                    <b style="color:#e2e8f0">{top3[1]}</b>, <b style="color:#e2e8f0">{top3[2]}</b>.
                    These are direct indicators of HPC (High Pressure Compressor) degradation.
                </p>
            </div>""", unsafe_allow_html=True)

    with tab2:
        engine_id   = st.selectbox("Select Engine", options=range(1, 101), key="shap_engine")
        engine_row  = test_last[test_last['unit_id'] == engine_id].iloc[0]
        sensor_vals = {col: engine_row[col] for col in feature_cols}
        true_rul    = int(rul_df.iloc[engine_id - 1]['RUL'])

        if st.button("Explain This Engine's Prediction", type="primary"):
            with st.spinner("Computing local SHAP..."):
                rul_pred = predict_rul(model, feature_cols, sensor_vals)
                shap_df  = get_shap_values(model, feature_cols, sensor_vals)
                risk_label, risk_color, _ = get_risk_level(rul_pred)

            col1,col2,col3 = st.columns(3)
            col1.metric("Predicted RUL", f"{rul_pred} cycles")
            col2.metric("True RUL",      f"{true_rul} cycles")
            col3.metric("Error",         f"{abs(rul_pred-true_rul):.1f} cycles")

            colors = ['#22c55e' if v > 0 else '#ef4444' for v in shap_df['SHAP Value']]
            fig = go.Figure(go.Bar(x=shap_df['SHAP Value'], y=shap_df['Feature'],
                orientation='h', marker=dict(color=colors),
                text=[f"{v:+.3f}" for v in shap_df['SHAP Value']],
                textposition='outside', textfont=dict(color='#94a3b8', size=11)))
            fig.add_vline(x=0, line_color="#64748b", line_width=1)
            fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                title=f"Engine #{engine_id} — Local SHAP Explanation",
                xaxis_title="SHAP Value", height=420, margin=dict(l=120,r=80,t=50,b=40))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(shap_df.reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SENSOR TRENDS
# ════════════════════════════════════════════════════════════════════════════════
elif "Sensor Trends" in page:
    section_header("📈", "Sensor Degradation Trends", "Visualize how sensors change as engines wear out")

    col1, col2 = st.columns([1, 2])
    with col1:
        engine_id   = st.selectbox("Select Engine", options=range(1, 101))
        engine_data = train_df[train_df['unit_id'] == engine_id]
        total_life  = int(engine_data['cycle'].max())
        st.markdown(f"""
        <div style="background:#131d35;border:1px solid #1e3a5f;border-radius:8px;padding:16px;margin:12px 0;">
            <div style="font-size:0.75rem;color:#64748b;letter-spacing:1px;">TOTAL LIFETIME</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.5rem;font-weight:800;color:#06b6d4;">
                {total_life} <span style="font-size:1rem;color:#64748b;">cycles</span>
            </div>
        </div>""", unsafe_allow_html=True)
        sensor_choice = st.multiselect("Select Sensors", options=feature_cols,
                                       default=['sensor_11','sensor_4','sensor_9'])

    with col2:
        if sensor_choice:
            fig = go.Figure()
            clrs = ["#3b82f6","#06b6d4","#22c55e","#f59e0b","#ef4444","#8b5cf6"]
            for i, sensor in enumerate(sensor_choice):
                fig.add_trace(go.Scatter(x=engine_data['cycle'], y=engine_data[sensor],
                    mode='lines', name=sensor, line=dict(color=clrs[i%len(clrs)], width=2)))
            fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                title=f"Engine #{engine_id} — Sensor Readings Over Time",
                xaxis_title="Cycle", yaxis_title="Sensor Value",
                height=320, margin=dict(l=50,r=20,t=50,b=40), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=engine_data['cycle'], y=engine_data['RUL'],
        mode='lines', fill='tozeroy', fillcolor='rgba(59,130,246,0.1)',
        line=dict(color='#3b82f6', width=2.5)))
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(239,68,68,0.08)",  line_width=0, annotation_text="CRITICAL", annotation_position="left")
    fig2.add_hrect(y0=30, y1=70,  fillcolor="rgba(245,158,11,0.08)", line_width=0, annotation_text="WARNING",  annotation_position="left")
    fig2.add_hrect(y0=70, y1=125, fillcolor="rgba(34,197,94,0.08)",  line_width=0, annotation_text="HEALTHY",  annotation_position="left")
    fig2.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
        title=f"Engine #{engine_id} — RUL Declining Over Lifetime",
        xaxis_title="Cycle", yaxis_title="Remaining Useful Life",
        height=300, margin=dict(l=80,r=20,t=50,b=40), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in page:
    import json, pickle, time
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    section_header("🏆", "Model Comparison", "Compare all models on FD001 test set")

    def load_saved_results():
        try:
            with open("models/model_results.json", "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df if df['Test R²'].sum() != 0 else None
        except:
            return None

    saved_results = load_saved_results()

    tab1, tab2, tab3 = st.tabs(["📊 Results & Charts", "🔄 Live Train & Compare", "🔍 Per-Engine Prediction"])

    with tab1:
        if saved_results is not None:
            results_df = saved_results.sort_values('Test R²', ascending=False)
            st.success("✅ Loaded from models/model_results.json")
        elif 'comparison_results' in st.session_state:
            results_df = pd.DataFrame(st.session_state['comparison_results']).sort_values('Test R²', ascending=False)
            st.success("✅ Using results from Live Train tab")
        else:
            st.warning("⚠️ No results yet — use the Live Train & Compare tab below!")
            results_df = None

        if results_df is not None:
            best = results_df.iloc[0]
            st.markdown(f"""
            <div style="background:#22c55e22;border:1px solid #22c55e;border-radius:8px;
                        padding:14px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.8rem">🏆</span>
                <div>
                    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                                 font-weight:800;color:#22c55e;letter-spacing:1px;">
                        BEST MODEL: {best['Model'].upper()}
                    </span>
                    <span style="color:#94a3b8;margin-left:16px;font-size:0.9rem;">
                        R² = {best['Test R²']:.4f} · RMSE = {best['Test RMSE']:.2f} · MAE = {best['Test MAE']:.2f}
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("🥇 Best R²",   f"{best['Test R²']:.4f}", best['Model'])
            c2.metric("📉 Best RMSE", f"{best['Test RMSE']:.2f}", best['Model'])
            c3.metric("📏 Best MAE",  f"{best['Test MAE']:.2f}", best['Model'])
            c4.metric("📊 Models",    str(len(results_df)))

            st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                sr = results_df.sort_values('Test RMSE')
                fig = go.Figure(go.Bar(x=sr['Model'], y=sr['Test RMSE'],
                    marker_color=['#22c55e' if m==best['Model'] else '#3b82f6' for m in sr['Model']],
                    text=[f"{v:.2f}" for v in sr['Test RMSE']], textposition='outside',
                    textfont=dict(color='#94a3b8', size=11)))
                fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                    title="Test RMSE — lower is better", height=350, margin=dict(l=40,r=20,t=50,b=80))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                sr2 = results_df.sort_values('Test R²', ascending=False)
                fig2 = go.Figure(go.Bar(x=sr2['Model'], y=sr2['Test R²'],
                    marker_color=['#22c55e' if m==best['Model'] else '#06b6d4' for m in sr2['Model']],
                    text=[f"{v:.4f}" for v in sr2['Test R²']], textposition='outside',
                    textfont=dict(color='#94a3b8', size=11)))
                fig2.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                    title="Test R² — higher is better", height=350, margin=dict(l=40,r=20,t=50,b=80))
                st.plotly_chart(fig2, use_container_width=True)

            # Radar chart
            df_r = results_df.copy().reset_index(drop=True)
            df_r['RMSE Score'] = 1-(df_r['Test RMSE']-df_r['Test RMSE'].min())/(df_r['Test RMSE'].max()-df_r['Test RMSE'].min()+1e-9)
            df_r['MAE Score']  = 1-(df_r['Test MAE'] -df_r['Test MAE'].min()) /(df_r['Test MAE'].max() -df_r['Test MAE'].min() +1e-9)
            df_r['R² Score']   =   (df_r['Test R²']  -df_r['Test R²'].min())  /(df_r['Test R²'].max()  -df_r['Test R²'].min()  +1e-9)

            rcolors = ["#3b82f6","#06b6d4","#22c55e","#f59e0b","#ef4444","#8b5cf6","#ec4899"]
            rfills  = ["rgba(59,130,246,0.1)","rgba(6,182,212,0.1)","rgba(34,197,94,0.1)",
                       "rgba(245,158,11,0.1)","rgba(239,68,68,0.1)","rgba(139,92,246,0.1)","rgba(236,72,153,0.1)"]
            cats = ['RMSE Score','MAE Score','R² Score']
            fig3 = go.Figure()
            for i, row in df_r.iterrows():
                vals = [row['RMSE Score'], row['MAE Score'], row['R² Score'], row['RMSE Score']]
                fig3.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]], fill='toself',
                    name=row['Model'], line=dict(color=rcolors[i%len(rcolors)], width=2),
                    fillcolor=rfills[i%len(rfills)]))
            fig3.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                polar=dict(bgcolor='#0f1629',
                    radialaxis=dict(visible=True,range=[0,1],gridcolor='#1e3a5f',color='#64748b'),
                    angularaxis=dict(gridcolor='#1e3a5f',color='#94a3b8')),
                showlegend=True, height=420, margin=dict(l=60,r=60,t=40,b=40),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')))
            st.plotly_chart(fig3, use_container_width=True)

            if 'Time (s)' in results_df.columns:
                st_time = results_df.sort_values('Time (s)')
                fig4 = go.Figure(go.Bar(x=st_time['Model'], y=st_time['Time (s)'],
                    marker_color='rgba(139,92,246,0.7)',
                    text=[f"{v:.1f}s" for v in st_time['Time (s)']], textposition='outside',
                    textfont=dict(color='#94a3b8', size=11)))
                fig4.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                    title="Training Time — lower is faster", height=300, margin=dict(l=40,r=20,t=50,b=80))
                st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        model_choices = st.multiselect("Choose models",
            options=["XGBoost","LightGBM","Random Forest","Gradient Boosting","Ridge","Lasso","KNN"],
            default=["XGBoost","LightGBM","Random Forest"])
        c1,c2 = st.columns(2)
        n_est = c1.slider("n_estimators", 50, 300, 200, 50)
        lr    = c2.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)

        if st.button("🚀 TRAIN & COMPARE", type="primary", use_container_width=True):
            fcmp = [c for c in feature_cols if c != 'op_setting_3']
            from sklearn.model_selection import train_test_split
            X_tr,X_v,y_tr,y_v = train_test_split(train_df[fcmp], train_df['RUL'], test_size=0.2, random_state=42)
            X_te = test_last[fcmp]
            y_te = rul_df['RUL'].values

            mmap = {
                "XGBoost"          : XGBRegressor(n_estimators=n_est,learning_rate=lr,max_depth=6,random_state=42,n_jobs=-1,verbosity=0),
                "LightGBM"         : LGBMRegressor(n_estimators=n_est,learning_rate=lr,max_depth=6,random_state=42,n_jobs=-1,verbose=-1),
                "Random Forest"    : RandomForestRegressor(n_estimators=n_est,random_state=42,n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=n_est,learning_rate=lr,max_depth=6,random_state=42),
                "Ridge": Ridge(), "Lasso": Lasso(), "KNN": KNeighborsRegressor(n_neighbors=5,n_jobs=-1),
            }
            live, prog, stat = [], st.progress(0), st.empty()
            for idx, name in enumerate(model_choices):
                stat.markdown(f"⚙️ Training **{name}**...")
                m = mmap[name]; t0 = time.time(); m.fit(X_tr,y_tr); elapsed=round(time.time()-t0,2)
                yp = m.predict(X_te)
                live.append({'Model':name,
                    'Test RMSE':round(float(np.sqrt(mean_squared_error(y_te,yp))),2),
                    'Test MAE' :round(float(mean_absolute_error(y_te,yp)),2),
                    'Test R²'  :round(float(r2_score(y_te,yp)),4),
                    'Time (s)' :elapsed})
                prog.progress((idx+1)/len(model_choices))
                stat.markdown(f"✅ **{name}** — R²: `{live[-1]['Test R²']}`")
            st.session_state['comparison_results'] = live
            st.dataframe(pd.DataFrame(live).sort_values('Test R²',ascending=False), use_container_width=True)
            st.info("✅ Switch to Results & Charts tab to see visualizations!")

    with tab3:
        engine_id  = st.selectbox("Select Engine", options=range(1,101), key="cmp_eng")
        true_rul   = int(rul_df.iloc[engine_id-1]['RUL'])
        engine_row = test_last[test_last['unit_id']==engine_id].iloc[0]
        fcmp       = [c for c in feature_cols if c != 'op_setting_3']

        if st.button("Compare All Models on This Engine", type="primary"):
            try:
                with open("models/all_models.pkl","rb") as f:
                    all_models = pickle.load(f)
                inp = pd.DataFrame([{col: engine_row[col] for col in fcmp}])
                rows = []
                for name, m in all_models.items():
                    pred = round(float(m.predict(inp)[0]),2)
                    rl,_,_ = get_risk_level(pred)
                    rows.append({'Model':name,'Predicted RUL':pred,'True RUL':true_rul,
                                 'Error':round(abs(pred-true_rul),2),'Risk':rl})
                per_df = pd.DataFrame(rows).sort_values('Error')
                st.dataframe(per_df, use_container_width=True)
                fig = go.Figure()
                fig.add_hline(y=true_rul, line_color="#22c55e", line_dash="dash",
                              annotation_text=f"True RUL={true_rul}", annotation_font_color="#22c55e")
                fig.add_trace(go.Bar(x=per_df['Model'], y=per_df['Predicted RUL'],
                    marker_color=['#22c55e' if e==per_df['Error'].min() else '#3b82f6' for e in per_df['Error']],
                    text=per_df['Predicted RUL'], textposition='outside'))
                fig.update_layout(**PLOT_TEMPLATE['layout'].to_plotly_json(),
                    title=f"Engine #{engine_id} — All Models vs True RUL={true_rul}",
                    height=380, margin=dict(l=40,r=20,t=60,b=80))
                st.plotly_chart(fig, use_container_width=True)
            except FileNotFoundError:
                st.error("❌ models/all_models.pkl not found. Use Live Train tab first.")
