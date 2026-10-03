import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# PROJECT IMPORTS
# =========================================================

from Agent.agent import DQNAgent
from Environment.actions import Action, ACTION_NAMES
from Environment.cyber_env import CyberEnvironment
from Environment.reward import get_threat_category

# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = PROJECT_ROOT / "Data" / "Processed" / "dqn_cyber_agent.pt"
DATA_PATH = PROJECT_ROOT / "Data" / "Deployment" / "demo_processed.csv"
ENCODER_PATH = PROJECT_ROOT / "Data" / "Processed" / "label_encoder.pkl"

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Autonomous Cyber Threat Hunter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CYBERSECURITY UI
# =========================================================

st.markdown(
    """
    <style>
    /* ---------- GLOBAL ---------- */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 180, 255, 0.07), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(0, 255, 170, 0.05), transparent 25%),
            #07111f;
        color: #e7eef7;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #091522;
        border-right: 1px solid rgba(74, 144, 226, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #dce8f5;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- HEADER ---------- */
    .hero {
        padding: 1.25rem 1.45rem;
        border: 1px solid rgba(0, 214, 255, 0.18);
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(9, 30, 49, 0.98),
            rgba(8, 22, 37, 0.96)
        );
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.22);
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #f5f9ff;
        margin: 0;
    }

    .hero-subtitle {
        color: #8fa9bf;
        font-size: 0.94rem;
        margin-top: 0.25rem;
    }

    .online {
        float: right;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(0, 255, 160, 0.28);
        background: rgba(0, 255, 160, 0.08);
        color: #54f0ad;
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.1rem;
    }

    /* ---------- SECTION TITLES ---------- */
    .section {
        margin: 1.15rem 0 0.65rem;
        font-size: 1.05rem;
        font-weight: 750;
        color: #dcecff;
        letter-spacing: 0.02em;
    }

    .section::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 17px;
        background: #20d9ff;
        border-radius: 5px;
        margin-right: 8px;
        vertical-align: -2px;
    }

    /* ---------- KPI CARDS ---------- */
    .kpi {
        min-height: 112px;
        padding: 1rem 1.05rem;
        border-radius: 15px;
        background: rgba(11, 28, 45, 0.96);
        border: 1px solid rgba(115, 158, 196, 0.16);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
    }

    .kpi-label {
        color: #8ca5ba;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .kpi-value {
        color: #f3f8ff;
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: 0.28rem;
    }

    .kpi-note {
        color: #6f899e;
        font-size: 0.72rem;
        margin-top: 0.15rem;
    }

    /* ---------- PANELS ---------- */
    .panel {
        padding: 1.05rem 1.1rem;
        border-radius: 16px;
        background: rgba(9, 25, 41, 0.94);
        border: 1px solid rgba(115, 158, 196, 0.15);
        min-height: 185px;
    }

    .panel-title {
        color: #91a9bf;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        font-weight: 750;
    }

    .panel-main {
        color: #f4f8fd;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.45rem 0;
    }

    .muted {
        color: #7891a5;
        font-size: 0.8rem;
    }

    /* ---------- THREAT STATUS ---------- */
    .threat-card {
        padding: 1.2rem;
        border-radius: 16px;
        min-height: 205px;
        background: linear-gradient(145deg, rgba(16, 30, 46, 0.98), rgba(8, 20, 34, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .risk-high {
        border-color: rgba(255, 71, 87, 0.55);
        box-shadow: 0 0 28px rgba(255, 71, 87, 0.08);
    }

    .risk-medium {
        border-color: rgba(255, 179, 71, 0.45);
    }

    .risk-low {
        border-color: rgba(55, 220, 150, 0.4);
    }

    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .badge-high {
        background: rgba(255, 71, 87, 0.13);
        color: #ff6674;
    }

    .badge-medium {
        background: rgba(255, 179, 71, 0.13);
        color: #ffbd62;
    }

    .badge-low {
        background: rgba(55, 220, 150, 0.12);
        color: #5ee4a7;
    }

    .confidence-track {
        height: 8px;
        background: #17283a;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 0.55rem;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #19b8ff, #36e6a1);
        border-radius: 999px;
    }

    /* ---------- ACTION CARD ---------- */
    .action-card {
        padding: 1.2rem;
        border-radius: 16px;
        min-height: 205px;
        background: linear-gradient(145deg, rgba(8, 33, 48, 0.98), rgba(8, 22, 36, 0.98));
        border: 1px solid rgba(32, 217, 255, 0.22);
    }

    .action-value {
        font-size: 1.65rem;
        font-weight: 850;
        color: #46e4ff;
        margin: 0.55rem 0;
    }

    /* ---------- FOOTER ---------- */
    .footer {
        text-align: center;
        color: #526a7e;
        font-size: 0.72rem;
        padding-top: 1.5rem;
    }

    /* ---------- STREAMLIT COMPONENTS ---------- */
    div[data-testid="stMetric"] {
        background: rgba(11, 28, 45, 0.96);
        border: 1px solid rgba(115, 158, 196, 0.15);
        padding: 0.75rem;
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DQN MODEL
# =========================================================

@st.cache_resource
def load_agent():
    agent = DQNAgent(
        state_size=8,
        action_size=6,
    )
    agent.load(str(MODEL_PATH))
    agent.epsilon = 0.0
    return agent


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

@st.cache_resource
def load_environment():
    return CyberEnvironment(
        data_path=str(DATA_PATH),
        max_steps=1000,
    )


# =========================================================
# LOAD LABEL MAPPING
# =========================================================

@st.cache_resource
def load_label_mapping():
    encoder = joblib.load(ENCODER_PATH)
    return encoder["id_to_label"]


# =========================================================
# SESSION STATE
# =========================================================

def initialise_session():
    if "environment" not in st.session_state:
        st.session_state.environment = load_environment()

    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()

    if "label_mapping" not in st.session_state:
        st.session_state.label_mapping = load_label_mapping()

    if "state" not in st.session_state:
        st.session_state.state = st.session_state.environment.reset()

    if "history" not in st.session_state:
        st.session_state.history = []

    if "samples_processed" not in st.session_state:
        st.session_state.samples_processed = 0

    if "threats_detected" not in st.session_state:
        st.session_state.threats_detected = 0

    if "total_reward" not in st.session_state:
        st.session_state.total_reward = 0.0

    if "confidence_sum" not in st.session_state:
        st.session_state.confidence_sum = 0.0


# =========================================================
# DQN Q-VALUES
# =========================================================

def get_q_values(agent, state):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)

    return q_values.squeeze(0).cpu().numpy()


# =========================================================
# PROCESS ONE SAMPLE
# =========================================================

def process_next_observation():
    environment = st.session_state.environment
    agent = st.session_state.agent
    state = st.session_state.state
    label_mapping = st.session_state.label_mapping

    q_values = get_q_values(agent, state)
    action_index = int(np.argmax(q_values))
    action = Action(action_index)

    next_state, reward, done, info = environment.step(action)

    probability = float(info["current_probability"])
    predicted_class = int(info["current_predicted_class"])

    predicted_label = label_mapping.get(
        predicted_class,
        "Unknown",
    )

    true_label = info["current_true_label"]
    category = get_threat_category(true_label)

    st.session_state.samples_processed += 1
    st.session_state.confidence_sum += probability
    st.session_state.total_reward += reward

    if predicted_label != "BenignTraffic":
        st.session_state.threats_detected += 1

    history_item = {
        "Step": info["step"],
        "Threat": predicted_label,
        "Confidence": probability,
        "Risk": category,
        "RL Action": ACTION_NAMES[action],
        "Reward": reward,
    }

    st.session_state.history.append(history_item)

    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

    st.session_state.state = next_state

    return done


# =========================================================
# RESET
# =========================================================

def reset_simulation():
    st.session_state.state = st.session_state.environment.reset()
    st.session_state.history = []
    st.session_state.samples_processed = 0
    st.session_state.threats_detected = 0
    st.session_state.total_reward = 0.0
    st.session_state.confidence_sum = 0.0


# =========================================================
# INITIALISE
# =========================================================

initialise_session()

state = st.session_state.state
environment = st.session_state.environment
agent = st.session_state.agent
label_mapping = st.session_state.label_mapping

q_values = get_q_values(agent, state)
selected_action_index = int(np.argmax(q_values))
selected_action = Action(selected_action_index)

current_probability = float(state[1])
current_class = int(state[0])
current_label = label_mapping.get(current_class, "Unknown")

if current_probability >= 0.7:
    risk_display = "HIGH"
    risk_class = "high"
elif current_probability >= 0.4:
    risk_display = "MEDIUM"
    risk_class = "medium"
else:
    risk_display = "LOW"
    risk_class = "low"

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    st.caption("Autonomous threat-hunting simulation")

    st.markdown("---")

    if st.button("▶  Process Next Sample", width="stretch"):
        done = process_next_observation()
        if done:
            st.warning("Simulation reached the end of the configured episode.")
        st.rerun()

    if st.button("↻  Reset Simulation", width="stretch"):
        reset_simulation()
        st.rerun()

    st.markdown("---")

    st.markdown("### 🧠 AI Model")
    st.write(f"**Algorithm:** DQN")
    st.write(f"**Device:** `{agent.device}`")
    st.write(f"**Exploration:** `ε = {agent.epsilon:.1f}`")

    st.markdown("### 📡 Data Source")
    st.write(f"**Dataset:** `{DATA_PATH.name}`")
    st.write(f"**Samples:** `{len(environment.data):,}`")

    st.markdown("---")
    st.caption(
        "Dataset-based simulation. Ground-truth labels are used internally "
        "for evaluation and are not available to a real-time agent."
    )

# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="online">● SYSTEM ONLINE</div>
        <div class="hero-title">🛡️ Autonomous Cyber Threat Hunter</div>
        <div class="hero-subtitle">
            AI-driven network threat detection with Deep Q-Network autonomous response
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TOP KPIs
# =========================================================

samples = st.session_state.samples_processed

if samples > 0:
    average_confidence = st.session_state.confidence_sum / samples
    average_reward = st.session_state.total_reward / samples
else:
    average_confidence = 0.0
    average_reward = 0.0

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Samples Processed</div>
            <div class="kpi-value">{samples:,}</div>
            <div class="kpi-note">Current simulation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Threats Detected</div>
            <div class="kpi-value">{st.session_state.threats_detected:,}</div>
            <div class="kpi-note">Non-benign predictions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Avg Confidence</div>
            <div class="kpi-value">{average_confidence:.1%}</div>
            <div class="kpi-note">Model prediction confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">Avg Reward</div>
            <div class="kpi-value">{average_reward:.3f}</div>
            <div class="kpi-note">RL feedback signal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">RL Algorithm</div>
            <div class="kpi-value">DQN</div>
            <div class="kpi-note">{len(list(Action))} available actions</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# CURRENT THREAT + RESPONSE
# =========================================================

st.markdown('<div class="section">🚨 Live Threat Assessment</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown(
        f"""
        <div class="threat-card risk-{risk_class}">
            <div class="panel-title">Current Threat Classification</div>
            <div class="panel-main">{current_label}</div>
            <span class="risk-badge badge-{risk_class}">
                {risk_display} RISK
            </span>
            <div class="muted" style="margin-top:0.8rem;">
                Detection confidence: {current_probability:.2%}
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{min(current_probability * 100, 100):.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div class="action-card">
            <div class="panel-title">Autonomous RL Response</div>
            <div class="action-value">{ACTION_NAMES[selected_action]}</div>
            <div class="muted">
                DQN selected the action with the highest estimated Q-value.
            </div>
            <div style="margin-top:1rem;">
                <span class="panel-title">Selected Q-Value</span>
                <div class="panel-main">{float(q_values[selected_action_index]):.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# RL DETAILS
# =========================================================

st.markdown('<div class="section">🤖 Reinforcement Learning Intelligence</div>', unsafe_allow_html=True)

rl1, rl2 = st.columns(2)

with rl1:
    st.markdown("#### DQN Q-Values")

    q_value_df = pd.DataFrame(
        {
            "Action": [ACTION_NAMES[action] for action in Action],
            "Q-Value": [float(value) for value in q_values],
        }
    )

    q_value_df["Q-Value"] = q_value_df["Q-Value"].round(4)

    st.bar_chart(
        q_value_df.set_index("Action"),
        width="stretch",
    )

with rl2:
    st.markdown("#### Environment State")

    state_df = pd.DataFrame(
        {
            "State Variable": [
                "Threat class ID",
                "Threat probability",
                "Host risk",
                "Connection risk",
                "Monitoring level",
                "Host isolated",
                "IP blocked",
                "Investigation active",
            ],
            "Value": [
                f"{state[0]:.0f}",
                f"{state[1]:.4f}",
                f"{state[2]:.4f}",
                f"{state[3]:.4f}",
                f"{state[4]:.2f}",
                f"{state[5]:.0f}",
                f"{state[6]:.0f}",
                f"{state[7]:.0f}",
            ],
        }
    )

    st.dataframe(
        state_df,
        width="stretch",
        hide_index=True,
    )

# =========================================================
# ACTION DISTRIBUTION
# =========================================================

st.markdown('<div class="section">📈 Agent Activity</div>', unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    chart_col, summary_col = st.columns([1.5, 1])

    with chart_col:
        st.markdown("#### RL Action Distribution")
        action_distribution = (
            history_df["RL Action"]
            .value_counts()
            .rename("Count")
        )
        st.bar_chart(action_distribution, width="stretch")

    with summary_col:
        st.markdown("#### Current Episode")
        st.metric("Current Step", int(state[0]) if False else samples)
        st.metric("Total Reward", f"{st.session_state.total_reward:.3f}")
        st.metric("Latest Action", ACTION_NAMES[selected_action])
else:
    st.info("Process samples to generate agent activity analytics.")

# =========================================================
# THREAT HISTORY
# =========================================================

st.markdown('<div class="section">📋 Threat Event Timeline</div>', unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    display_df = history_df.copy()

    display_df["Confidence"] = display_df["Confidence"].map(
        lambda value: f"{value:.2%}"
    )
    display_df["Reward"] = display_df["Reward"].map(
        lambda value: f"{value:.2f}"
    )

    st.dataframe(
        display_df.iloc[::-1],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No observations processed yet.")

# =========================================================
# EVALUATION INFORMATION
# =========================================================

with st.expander("ℹ️ About this dashboard"):
    st.write(
        "This dashboard uses the processed test dataset as simulated network traffic."
    )
    st.write(
        "The DQN receives the same 8-dimensional cyber state representation used during training."
    )
    st.write(
        "The trained DQN model is loaded from dqn_cyber_agent.pt and runs with "
        "epsilon = 0 for deterministic inference."
    )
    st.write(
        "Ground-truth labels are available internally because this is a dataset-based "
        "evaluation simulation. In a real deployment, ground-truth labels would not "
        "be available to the agent."
    )

st.markdown(
    """
    <div class="footer">
        Autonomous Cyber Threat Hunting • DQN-based detection & response prototype
    </div>
    """,
    unsafe_allow_html=True,
)
