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

MODEL_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "dqn_cyber_agent.pt"
)

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Deployment"
    / "demo_processed.csv"
)

ENCODER_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Processed"
    / "label_encoder.pkl"
)


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Autonomous Cyber Threat Hunter",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #777;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
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

    agent.load(
        str(MODEL_PATH)
    )

    # Deterministic inference.
    # No random exploration in the dashboard.
    agent.epsilon = 0.0

    return agent


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

@st.cache_resource
def load_environment():

    environment = CyberEnvironment(
        data_path=str(DATA_PATH),
        max_steps=1000,
    )

    return environment


# =========================================================
# LOAD LABEL ENCODER
# =========================================================

@st.cache_resource
def load_label_mapping():

    encoder = joblib.load(
        ENCODER_PATH
    )

    return encoder["id_to_label"]


# =========================================================
# INITIALISE SESSION STATE
# =========================================================

def initialise_session():

    if "environment" not in st.session_state:

        st.session_state.environment = (
            load_environment()
        )

    if "agent" not in st.session_state:

        st.session_state.agent = (
            load_agent()
        )

    if "label_mapping" not in st.session_state:

        st.session_state.label_mapping = (
            load_label_mapping()
        )

    if "state" not in st.session_state:

        st.session_state.state = (
            st.session_state.environment.reset()
        )

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
# GET DQN Q-VALUES
# =========================================================

def get_q_values(
    agent,
    state,
):

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)

    with torch.no_grad():

        q_values = (
            agent.policy_network(
                state_tensor
            )
        )

    return (
        q_values
        .squeeze(0)
        .cpu()
        .numpy()
    )


# =========================================================
# PROCESS ONE SAMPLE
# =========================================================

def process_next_observation():

    environment = (
        st.session_state.environment
    )

    agent = (
        st.session_state.agent
    )

    state = (
        st.session_state.state
    )

    label_mapping = (
        st.session_state.label_mapping
    )

    # -----------------------------------------------------
    # Get Q-values
    # -----------------------------------------------------

    q_values = get_q_values(
        agent,
        state,
    )

    # -----------------------------------------------------
    # Select action with highest Q-value
    # -----------------------------------------------------

    action_index = int(
        np.argmax(q_values)
    )

    action = Action(
        action_index
    )

    # -----------------------------------------------------
    # Execute action
    # -----------------------------------------------------

    (
        next_state,
        reward,
        done,
        info,
    ) = environment.step(
        action
    )

    # -----------------------------------------------------
    # Current prediction
    # -----------------------------------------------------

    probability = float(
        info["current_probability"]
    )

    predicted_class = int(
        info["current_predicted_class"]
    )

    predicted_label = label_mapping.get(
        predicted_class,
        "Unknown",
    )

    # -----------------------------------------------------
    # Ground truth
    #
    # Available only because this is a dataset simulation.
    # It would not be available in real deployment.
    # -----------------------------------------------------

    true_label = info[
        "current_true_label"
    ]

    category = get_threat_category(
        true_label
    )

    # -----------------------------------------------------
    # Update statistics
    # -----------------------------------------------------

    st.session_state.samples_processed += 1

    st.session_state.confidence_sum += (
        probability
    )

    st.session_state.total_reward += (
        reward
    )

    # Count non-benign predictions as detected threats.
    if predicted_label != "BenignTraffic":

        st.session_state.threats_detected += 1

    # -----------------------------------------------------
    # Add history record
    # -----------------------------------------------------

    history_item = {
        "Step": info["step"],
        "Threat": predicted_label,
        "Confidence": probability,
        "Risk": category,
        "RL Action": ACTION_NAMES[action],
        "Reward": reward,
    }

    st.session_state.history.append(
        history_item
    )

    # Keep only the latest 100 observations.
    if len(st.session_state.history) > 100:

        st.session_state.history = (
            st.session_state.history[-100:]
        )

    # -----------------------------------------------------
    # Update current state
    # -----------------------------------------------------

    st.session_state.state = (
        next_state
    )

    return done


# =========================================================
# RESET SIMULATION
# =========================================================

def reset_simulation():

    environment = (
        st.session_state.environment
    )

    st.session_state.state = (
        environment.reset()
    )

    st.session_state.history = []

    st.session_state.samples_processed = 0

    st.session_state.threats_detected = 0

    st.session_state.total_reward = 0.0

    st.session_state.confidence_sum = 0.0


# =========================================================
# INITIALISE
# =========================================================

initialise_session()


# =========================================================
# LOAD CURRENT VALUES
# =========================================================

state = (
    st.session_state.state
)

environment = (
    st.session_state.environment
)

agent = (
    st.session_state.agent
)

label_mapping = (
    st.session_state.label_mapping
)


# =========================================================
# CURRENT DQN Q-VALUES
# =========================================================

q_values = get_q_values(
    agent,
    state,
)

selected_action_index = int(
    np.argmax(q_values)
)

selected_action = Action(
    selected_action_index
)


# =========================================================
# CURRENT THREAT INFORMATION
# =========================================================

current_probability = float(
    state[1]
)

current_class = int(
    state[0]
)

current_label = label_mapping.get(
    current_class,
    "Unknown",
)


# =========================================================
# RISK LEVEL
# =========================================================

if current_probability >= 0.7:

    risk_display = "HIGH"

elif current_probability >= 0.4:

    risk_display = "MEDIUM"

else:

    risk_display = "LOW"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🛡️ Autonomous Cyber Threat Hunter
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        AI-powered threat detection and Deep Q-Network
        autonomous response simulation
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🎛️ Control Panel")

    if st.button(
        "▶ Process Next Sample",
        width="stretch",
    ):

        done = process_next_observation()

        if done:

            st.warning(
                "Simulation reached the end of the configured episode."
            )

        st.rerun()

    if st.button(
        "🔄 Reset Simulation",
        width="stretch",
    ):

        reset_simulation()

        st.rerun()

    st.divider()

    st.subheader("System Information")

    st.write(
        f"**DQN Model:** "
        f"{MODEL_PATH.name}"
    )

    st.write(
        f"**Device:** "
        f"{agent.device}"
    )

    st.write(
        f"**Dataset:** "
        f"{DATA_PATH.name}"
    )

    st.write(
        f"**Available Samples:** "
        f"{len(environment.data):,}"
    )

    st.divider()

    st.caption(
        "The dashboard uses the processed test dataset "
        "as simulated network traffic."
    )


# =========================================================
# CURRENT THREAT DETECTION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        🚨 Current Threat Detection
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Threat Class",
        current_label,
    )


with col2:

    st.metric(
        "Confidence",
        f"{current_probability:.2%}",
    )


with col3:

    st.metric(
        "Risk Level",
        risk_display,
    )


with col4:

    st.metric(
        "RL Response",
        ACTION_NAMES[selected_action],
    )


# =========================================================
# RL RESPONSE
# =========================================================

st.markdown(
    """
    <div class="section-title">
        🤖 Reinforcement Learning Response
    </div>
    """,
    unsafe_allow_html=True,
)

response_col1, response_col2 = st.columns(
    [1, 1]
)


# =========================================================
# Q-VALUE PANEL
# =========================================================

with response_col1:

    st.subheader(
        "Selected Action"
    )

    st.success(
        ACTION_NAMES[selected_action]
    )

    st.subheader(
        "DQN Q-Values"
    )

    q_value_df = pd.DataFrame(
        {
            "Action": [
                ACTION_NAMES[action]
                for action in Action
            ],
            "Q-Value": [
                float(value)
                for value in q_values
            ],
        }
    )

    q_value_df["Q-Value"] = (
        q_value_df["Q-Value"]
        .round(4)
    )

    st.dataframe(
        q_value_df,
        width="stretch",
        hide_index=True,
    )


# =========================================================
# ENVIRONMENT STATE PANEL
# =========================================================

with response_col2:

    st.subheader(
        "Environment State"
    )

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
# SYSTEM STATISTICS
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📊 System Statistics
    </div>
    """,
    unsafe_allow_html=True,
)

samples = (
    st.session_state.samples_processed
)

if samples > 0:

    average_confidence = (
        st.session_state.confidence_sum
        / samples
    )

    average_reward = (
        st.session_state.total_reward
        / samples
    )

else:

    average_confidence = 0.0

    average_reward = 0.0


stat1, stat2, stat3, stat4 = st.columns(4)


with stat1:

    st.metric(
        "Samples Processed",
        samples,
    )


with stat2:

    st.metric(
        "Threats Detected",
        st.session_state.threats_detected,
    )


with stat3:

    st.metric(
        "Average Confidence",
        f"{average_confidence:.2%}",
    )


with stat4:

    st.metric(
        "Average Reward",
        f"{average_reward:.3f}",
    )


# =========================================================
# ACTION DISTRIBUTION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📈 RL Action Distribution
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    action_distribution = (
        history_df[
            "RL Action"
        ]
        .value_counts()
        .rename("Count")
    )

    st.bar_chart(
        action_distribution,
        width="stretch",
    )

else:

    st.info(
        "Process samples to generate "
        "the RL action distribution."
    )


# =========================================================
# THREAT HISTORY
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📋 Threat Detection History
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    display_df = history_df.copy()

    display_df["Confidence"] = (
        display_df["Confidence"]
        .map(
            lambda value:
            f"{value:.2%}"
        )
    )

    display_df["Reward"] = (
        display_df["Reward"]
        .map(
            lambda value:
            f"{value:.2f}"
        )
    )

    st.dataframe(
        display_df.iloc[::-1],
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No observations processed yet."
    )


# =========================================================
# EVALUATION INFORMATION
# =========================================================

with st.expander(
    "ℹ️ Evaluation Information"
):

    st.write(
        "This dashboard uses the processed test dataset "
        "as simulated network traffic."
    )

    st.write(
        "The DQN receives the same 8-dimensional cyber "
        "state representation used during training."
    )

    st.write(
        "The trained DQN model is loaded from "
        "dqn_cyber_agent.pt and runs with epsilon = 0 "
        "so the dashboard uses deterministic inference."
    )

    st.write(
        "Ground-truth labels are available internally "
        "because this is a dataset-based evaluation "
        "simulation. In a real deployment, ground-truth "
        "labels would not be available to the agent."
    )