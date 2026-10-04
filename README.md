# 🛡️ Autonomous Cyber Threat Hunting Agent Using Reinforcement Learning

An autonomous cybersecurity framework that uses **Deep Reinforcement Learning (DRL)** to detect, investigate, and respond to cyber threats dynamically.

The proposed system models cyber threat hunting as a **Markov Decision Process (MDP)** and trains reinforcement learning agents to select appropriate defensive actions based on network telemetry, security events, and threat severity.

---

## 📌 Overview

Traditional cybersecurity solutions such as rule-based IDS and SIEM systems primarily depend on predefined signatures and static rules. While effective against known threats, they can struggle with evolving attack patterns and generate large volumes of alerts, leading to **alert fatigue** and increased **Mean Time to Detect (MTTD)** and **Mean Time to Respond (MTTR)**.

This project explores a proactive approach using **Reinforcement Learning (RL)**, where an agent continuously observes the security environment, selects defensive actions, and learns from reward feedback.

The research presents a DRL-based framework using algorithms such as:

* **Deep Q-Network (DQN)**
* **Proximal Policy Optimization (PPO)**
* Markov Decision Process (MDP)
* Cybersecurity telemetry analysis
* Automated threat response
* SOC-oriented visualization

The overall architecture follows the pipeline:

**Data Sources → Preprocessing → State Modeling → RL Agent → Action Selection → Reward Feedback → SOC Dashboard & Mitigation**

---

## 🎯 Objectives

* Develop an autonomous cyber threat hunting framework.
* Model cybersecurity threat hunting as a reinforcement learning problem.
* Enable continuous analysis of network and security telemetry.
* Learn appropriate defensive actions using reward-based learning.
* Reduce unnecessary security alerts and analyst workload.
* Compare reinforcement learning with traditional rule-based approaches.
* Evaluate the effectiveness of DQN for automated threat response.

---

## 🏗️ System Architecture

```text
                ┌─────────────────────────┐
                │       Data Sources      │
                │ Traffic / Logs / CTI    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Preprocessing & Feature │
                │      Extraction         │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │   RL Environment /      │
                │    State Modeling       │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │      RL Agent           │
                │      DQN / PPO          │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    Action Selection     │
                │ Ignore / Investigate    │
                │ Alert / Block Source    │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    Reward Feedback      │
                │   Agent Policy Update   │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ SOC Dashboard &         │
                │ Automated Mitigation    │
                └─────────────────────────┘
```

The architecture is based on the research framework described in the paper.

---

## ⚙️ Methodology

The proposed system treats cyber threat hunting as a **Markov Decision Process**:

```text
MDP = (S, A, P, R, γ)
```

### State Space

The state represents security telemetry such as:

* Network traffic characteristics
* Packet information
* Port activity
* Login activity
* Security events
* MITRE ATT&CK mappings

### Action Space

The agent can select from four primary actions:

| Action         | Description                                          |
| -------------- | ---------------------------------------------------- |
| `Ignore`       | Treat the activity as benign and continue monitoring |
| `Investigate`  | Collect additional context and investigate the event |
| `Raise Alert`  | Generate a prioritized security notification         |
| `Block Source` | Isolate or block the malicious source                |

### Reward Function

The reward mechanism encourages correct threat handling while penalizing false positives, missed attacks, and unnecessary investigation.

```text
Correct threat response     → Positive reward
False positive              → Negative reward
Missed attack               → Negative reward
Unnecessary investigation   → Small negative cost
```

This reward feedback allows the agent to gradually learn a more effective threat-response policy.

---

## 🧠 Reinforcement Learning Algorithms

### Deep Q-Network (DQN)

DQN approximates the optimal action-value function:

```text
Q*(s, a)
```

The model uses experience replay and a target network to learn which defensive action should be selected for a particular security state.

### Proximal Policy Optimization (PPO)

PPO is considered as a second DRL approach for learning stable policies in the cybersecurity environment.

The research framework is designed to support both **DQN and PPO** for threat-response decision making.

---

## 🧪 Experimental Configuration

The reported DQN experiment used the following configuration:

| Parameter                  |  Value |
| -------------------------- | -----: |
| RL Algorithm               |    DQN |
| Training Episodes          |     20 |
| Steps per Episode          |    500 |
| Total Training Steps       | 10,000 |
| Evaluation Samples         |  1,000 |
| Evaluation Exploration (ε) |      0 |
| Best Episode Reward        |  1,413 |
| Best Average Reward/Step   |  2.826 |
| Final Episode Reward       |  1,335 |

---

## 📊 Results

The trained DQN agent was evaluated on **1,000 samples**.

| Metric                        |     Result |
| ----------------------------- | ---------: |
| Classification Accuracy       | **78.80%** |
| Action Appropriateness        | **98.60%** |
| Missed Threats                |      **0** |
| Unnecessary Defensive Actions |  **1.40%** |
| Average Reward                |  **8.549** |

The DQN achieved an average reward of **8.549**, compared with **5.238** for the rule-based baseline.

This corresponds to an improvement of approximately:

**63.21%**

under the conditions of the reported experiment.

---

## 🔍 Performance by Threat Category

| Threat Category | Samples | Appropriate Response | Average Reward |
| --------------- | ------: | -------------------: | -------------: |
| HIGH            |     715 |                 100% |         9.3958 |
| MEDIUM          |     255 |                 100% |         7.1765 |
| INVESTIGATION   |      10 |                 100% |         3.5000 |
| LOW             |      20 |                  30% |        -1.7000 |

The results show strong performance on **HIGH** and **MEDIUM** severity threats, while handling of **LOW** severity events remains an important area for improvement.

---

## 🛡️ Action Distribution

The trained DQN showed the following action-selection distribution:

| Defensive Action    | Selection Rate |
| ------------------- | -------------: |
| Isolate Host        |          54.1% |
| Block IP            |          44.7% |
| Investigate         |           0.6% |
| Allow               |           0.6% |
| Monitor             |           0.0% |
| Increase Monitoring |           0.0% |

The agent selected **Isolate Host** and **Block IP** for 98.8% of its decisions, indicating a strongly defensive policy. This also highlights a limitation of the current reward design: the model can become overly aggressive when stronger defensive actions receive greater reward.

---

## 🛠️ Technology Stack

* **Python**
* **PyTorch**
* **Gymnasium**
* Deep Reinforcement Learning
* DQN
* PPO
* NumPy
* Pandas
* Matplotlib
* Cybersecurity datasets / telemetry
* Streamlit *(for dashboard implementation)*

The research methodology specifies Python, PyTorch, and Gymnasium for the RL implementation.

---

## 📁 Project Structure

```text
Autonomous-Cyber-Threat-Hunting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── environment/
│   └── threat_hunting_env.py
│
├── models/
│   ├── dqn/
│   └── ppo/
│
├── training/
│   ├── train_dqn.py
│   └── train_ppo.py
│
├── evaluation/
│   ├── evaluate_dqn.py
│   ├── evaluate_ppo.py
│   └── metrics.csv
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Workflow

```text
1. Collect cybersecurity telemetry
             ↓
2. Preprocess and extract features
             ↓
3. Convert features into RL states
             ↓
4. Initialize RL environment
             ↓
5. Train DQN / PPO agent
             ↓
6. Generate reward from selected actions
             ↓
7. Update the agent's policy
             ↓
8. Evaluate the trained model
             ↓
9. Visualize results
             ↓
10. Recommend / perform defensive actions
```

---

## 📈 Key Findings

* The DQN successfully learned a threat-response policy.
* **98.60%** of selected responses were considered appropriate in the evaluation.
* No missed threats were recorded in the reported evaluation sample.
* DQN achieved an average reward of **8.549** compared with **5.238** for the rule-based baseline.
* The resulting reward improvement was approximately **63.21%**.
* HIGH and MEDIUM severity threats achieved **100% appropriate responses**.
* LOW-severity events remain a significant weakness.
* The current policy strongly favors aggressive actions such as host isolation and IP blocking.

These findings indicate that reinforcement learning can provide an adaptive approach to automated cyber threat response, while also demonstrating the need for better reward balancing and handling of low-severity events.

---

## 🔮 Future Scope

Future development can focus on:

* Improving detection and response for low-severity events.
* Designing a more balanced reward function.
* Increasing the diversity of defensive actions.
* Reducing over-reliance on blocking and isolation.
* Training PPO alongside DQN for comparative evaluation.
* Incorporating richer attack-chain context.
* Integrating additional cybersecurity telemetry.
* Improving explainability of agent decisions.
* Developing a real-time SOC dashboard.
* Evaluating the framework against more diverse attack scenarios.

---

## 👨‍💻 Authors

**Kalash Singh**
Department of Data Science, IoT and Cyber Security
G. H. Raisoni College of Engineering, Nagpur

**Riya Dhopre**
Department of Data Science, IoT and Cyber Security
G. H. Raisoni College of Engineering, Nagpur

**Ayush Wanjari**
Department of Data Science, IoT and Cyber Security
G. H. Raisoni College of Engineering, Nagpur

### Supervisor

**Dr. Nekita Chavhan Morris**
Head of Department
Department of Data Science, IoT and Cyber Security
G. H. Raisoni College of Engineering, Nagpur

---

## 📄 Research Paper

This project is associated with the research paper:

> **A Reinforcement Learning-Based Approach for Autonomous Cyber-Threat Detection and Hunting**

The paper presents the methodology, experimental configuration, evaluation results, and discussion of the proposed autonomous threat-hunting framework.

---

## ⚠️ Disclaimer

This project is developed for **research and educational purposes**. Any cybersecurity testing or automated response functionality should only be performed in controlled environments and on systems for which appropriate authorization has been obtained.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and sharing it with others interested in:

* Cybersecurity
* Reinforcement Learning
* Autonomous Threat Hunting
* SOC Automation
* Deep Reinforcement Learning
* AI for Cybersecurity
