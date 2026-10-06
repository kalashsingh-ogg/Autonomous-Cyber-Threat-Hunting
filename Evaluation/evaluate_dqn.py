import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch

# Make project root importable
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Environment.cyber_env import CyberEnvironment
from Environment.actions import Action
from Environment.reward import calculate_reward
from Agent.agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed",
    "test_processed.csv"
)

DQN_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "Processed",
    "dqn_cyber_agent.pt"
)

EVALUATION_SAMPLES = 50_000
RANDOM_STATE = 42

STATE_SIZE = 8
ACTION_SIZE = 6

# Batch size for the tree classifier.
# Smaller = lower memory usage.
CLASSIFIER_BATCH_SIZE = 5_000


# ============================================================
# THREAT CATEGORIES
# ============================================================

HIGH_THREATS = {
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation",
}

MEDIUM_THREATS = {
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
}

LOW_THREATS = {
    "BenignTraffic",
}


def get_threat_category(label):
    if label in HIGH_THREATS:
        return "HIGH"

    if label in MEDIUM_THREATS:
        return "MEDIUM"

    if label in LOW_THREATS:
        return "LOW"

    return "INVESTIGATION"


def is_appropriate_action(category, action):
    """
    Project-defined action appropriateness criteria.
    """

    if category == "HIGH":
        return action in {
            Action.BLOCK_IP,
            Action.ISOLATE_HOST,
            Action.INCREASE_MONITORING,
            Action.INVESTIGATE,
        }

    if category == "MEDIUM":
        return action in {
            Action.BLOCK_IP,
            Action.ISOLATE_HOST,
            Action.INCREASE_MONITORING,
            Action.INVESTIGATE,
        }

    if category == "INVESTIGATION":
        return action in {
            Action.BLOCK_IP,
            Action.ISOLATE_HOST,
            Action.INCREASE_MONITORING,
            Action.INVESTIGATE,
        }

    if category == "LOW":
        return action in {
            Action.ALLOW,
            Action.MONITOR,
        }

    return action == Action.MONITOR


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DQN AGENT EVALUATION")
    print("=" * 70)

    print(f"Evaluation samples: {EVALUATION_SAMPLES:,}")
    print(f"Test data: {TEST_DATA_PATH}")
    print(f"DQN model: {DQN_MODEL_PATH}")
    print()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("Loading test dataset...")

    test_data = pd.read_csv(TEST_DATA_PATH)

    print(
        f"Test dataset rows available: {len(test_data):,}"
    )

    if len(test_data) < EVALUATION_SAMPLES:
        raise ValueError(
            f"Test dataset contains only {len(test_data):,} rows, "
            f"but {EVALUATION_SAMPLES:,} were requested."
        )

    evaluation_data = test_data.sample(
        n=EVALUATION_SAMPLES,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    del test_data

    print(
        f"Evaluation rows selected: "
        f"{len(evaluation_data):,}"
    )

    print()

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    print("Loading threat detector...")

    # Import here so we use the exact production detector.
    from Detection.threat_detector import ThreatDetector

    detector = ThreatDetector()

    print("Threat detector loaded.")
    print(
        f"Classifier features: "
        f"{detector.expected_features}"
    )
    print(
        f"Classifier classes: "
        f"{len(detector.id_to_label)}"
    )
    print()

    # --------------------------------------------------------
    # Batched classifier prediction
    # --------------------------------------------------------

    print("=" * 70)
    print("BATCHED THREAT CLASSIFICATION")
    print("=" * 70)

    feature_columns = [
        column
        for column in evaluation_data.columns
        if column != "label"
    ]

    if len(feature_columns) != 46:
        raise ValueError(
            f"Expected 46 feature columns, "
            f"found {len(feature_columns)}"
        )

    X_eval = evaluation_data[
        feature_columns
    ]

    y_true = evaluation_data[
        "label"
    ].to_numpy()

    predicted_classes = []
    predicted_probabilities = []

    total_rows = len(X_eval)

    for start in range(
        0,
        total_rows,
        CLASSIFIER_BATCH_SIZE
    ):

        end = min(
            start + CLASSIFIER_BATCH_SIZE,
            total_rows
        )

        batch = X_eval.iloc[start:end]

        predictions = detector.model.predict(
            batch
        )

        probabilities = detector.model.predict_proba(
            batch
        )

        confidences = np.max(
            probabilities,
            axis=1
        )

        predicted_classes.extend(
            predictions.tolist()
        )

        predicted_probabilities.extend(
            confidences.tolist()
        )

        print(
            f"Classified "
            f"{end:,}/{total_rows:,} samples"
        )

    predicted_classes = np.asarray(
        predicted_classes,
        dtype=int
    )

    predicted_probabilities = np.asarray(
        predicted_probabilities,
        dtype=float
    )

    predicted_labels = np.array([
        detector.id_to_label.get(
            int(class_id),
            "Unknown"
        )
        for class_id in predicted_classes
    ])

    classification_correct = (
        predicted_labels == y_true
    )

    classification_accuracy = float(
        np.mean(classification_correct)
    )

    print()
    print(
        f"Classification accuracy: "
        f"{classification_accuracy:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------
    #
    # The environment normally performs detector.predict()
    # internally. To avoid doing another 50,000 expensive
    # classifier calls, we use a lightweight evaluation
    # environment and inject the already-computed predictions.
    # --------------------------------------------------------

    print("=" * 70)
    print("CREATING EVALUATION ENVIRONMENT")
    print("=" * 70)

    class BatchedEvaluationEnvironment:

        def __init__(
            self,
            data,
            predicted_classes,
            predicted_probabilities,
            predicted_labels
        ):

            self.data = data
            self.predicted_classes = predicted_classes
            self.predicted_probabilities = (
                predicted_probabilities
            )
            self.predicted_labels = predicted_labels

            self.current_index = 0
            self.current_step = 0
            self.max_steps = len(data)

            self.state = None
            self.current_true_label = None

            self.total_reward = 0.0

            self.action_counts = {
                action.name: 0
                for action in Action
            }

        def reset(self):

            self.current_index = 0
            self.current_step = 0
            self.total_reward = 0.0

            self.action_counts = {
                action.name: 0
                for action in Action
            }

            return self._load_current_observation()

        def _load_current_observation(self):

            prediction_class = int(
                self.predicted_classes[
                    self.current_index
                ]
            )

            probability = float(
                self.predicted_probabilities[
                    self.current_index
                ]
            )

            self.current_true_label = (
                self.data.iloc[
                    self.current_index
                ]["label"]
            )

            from Environment.state import CyberState

            self.state = CyberState(
                threat_class=prediction_class,
                threat_probability=probability,
                host_risk=probability,
                connection_risk=probability,
                monitoring_level=0.5,
                host_isolated=0.0,
                ip_blocked=0.0,
                investigation_active=0.0,
            )

            return np.array(
                self.state.to_vector(),
                dtype=np.float32
            )

        def step(self, action):

            if not isinstance(action, Action):
                action = Action(action)

            current_class = (
                self.state.threat_class
            )

            current_probability = (
                self.state.threat_probability
            )

            current_true_label = (
                self.current_true_label
            )

            reward = calculate_reward(
                threat_probability=current_probability,
                action=action,
                true_label=current_true_label,
                predicted_label=current_class,
            )

            self.total_reward += reward

            self.action_counts[
                action.name
            ] += 1

            self._apply_action(action)

            self.current_step += 1
            self.current_index += 1

            done = (
                self.current_step >= self.max_steps
                or self.current_index >= len(self.data)
            )

            if not done:

                next_state = (
                    self._load_current_observation()
                )

            else:

                next_state = np.array(
                    self.state.to_vector(),
                    dtype=np.float32
                )

            next_class = (
                self.state.threat_class
            )

            next_probability = (
                self.state.threat_probability
            )

            info = {
                "step": self.current_step,
                "action": action.name,
                "reward": reward,
                "current_predicted_class": current_class,
                "current_probability": current_probability,
                "current_true_label": current_true_label,
                "next_predicted_class": next_class,
                "next_probability": next_probability,
                "done": done,
            }

            return (
                next_state,
                reward,
                done,
                info
            )

        def _apply_action(self, action):

            if action == Action.BLOCK_IP:
                self.state.ip_blocked = 1.0

            elif action == Action.ISOLATE_HOST:
                self.state.host_isolated = 1.0

            elif action == Action.INCREASE_MONITORING:
                self.state.monitoring_level = min(
                    1.0,
                    self.state.monitoring_level + 0.2
                )

            elif action == Action.INVESTIGATE:
                self.state.investigation_active = 1.0

            elif action == Action.ALLOW:
                self.state.ip_blocked = 0.0

    env = BatchedEvaluationEnvironment(
        data=evaluation_data,
        predicted_classes=predicted_classes,
        predicted_probabilities=predicted_probabilities,
        predicted_labels=predicted_labels,
    )

    print("Evaluation environment ready.")
    print()

    # --------------------------------------------------------
    # Load DQN
    # --------------------------------------------------------

    print("=" * 70)
    print("LOADING DQN AGENT")
    print("=" * 70)

    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
    )

    agent.load(DQN_MODEL_PATH)

    # Deterministic evaluation
    agent.epsilon = 0.0

    print(
        f"Evaluation epsilon: "
        f"{agent.epsilon}"
    )

    print()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("=" * 70)
    print("STARTING DQN EVALUATION")
    print("=" * 70)
    print()

    state = env.reset()

    total_reward = 0.0
    appropriate_actions = 0
    missed_threats = 0
    unnecessary_actions = 0

    category_stats = {
        "HIGH": {
            "samples": 0,
            "appropriate": 0,
            "reward": 0.0,
        },
        "MEDIUM": {
            "samples": 0,
            "appropriate": 0,
            "reward": 0.0,
        },
        "INVESTIGATION": {
            "samples": 0,
            "appropriate": 0,
            "reward": 0.0,
        },
        "LOW": {
            "samples": 0,
            "appropriate": 0,
            "reward": 0.0,
        },
        "UNKNOWN": {
            "samples": 0,
            "appropriate": 0,
            "reward": 0.0,
        },
    }

    action_counts = {
        action.name: 0
        for action in Action
    }

    progress_interval = 5_000

    for step in range(EVALUATION_SAMPLES):

        action_value = agent.choose_action(
            state
        )

        action = Action(
            int(action_value)
        )

        next_state, reward, done, info = (
            env.step(action)
        )

        true_label = info[
            "current_true_label"
        ]

        category = get_threat_category(
            true_label
        )

        appropriate = is_appropriate_action(
            category,
            action
        )

        total_reward += reward

        action_counts[
            action.name
        ] += 1

        category_stats[
            category
        ]["samples"] += 1

        category_stats[
            category
        ]["reward"] += reward

        if appropriate:
            appropriate_actions += 1
            category_stats[
                category
            ]["appropriate"] += 1

        if category != "LOW":
            if not appropriate:
                missed_threats += 1

        else:
            if not appropriate:
                unnecessary_actions += 1

        state = next_state

        if (
            (step + 1) % progress_interval == 0
            or step + 1 == EVALUATION_SAMPLES
        ):

            print(
                f"Evaluated "
                f"{step + 1:,}/"
                f"{EVALUATION_SAMPLES:,} samples"
            )

        if done:
            break

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    samples_processed = step + 1

    action_appropriateness = (
        appropriate_actions
        / samples_processed
    )

    average_reward = (
        total_reward
        / samples_processed
    )

    missed_threat_rate = (
        missed_threats
        / samples_processed
    )

    unnecessary_action_rate = (
        unnecessary_actions
        / samples_processed
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DQN AGENT EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Samples processed: "
        f"{samples_processed:,}"
    )

    print(
        f"Classification accuracy: "
        f"{classification_accuracy:.4f}"
    )

    print(
        f"Action appropriateness: "
        f"{action_appropriateness:.4f}"
    )

    print(
        f"Total reward: "
        f"{total_reward:.2f}"
    )

    print(
        f"Average reward: "
        f"{average_reward:.4f}"
    )

    print(
        f"Missed threats: "
        f"{missed_threats:,}"
    )

    print(
        f"Missed threat rate: "
        f"{missed_threat_rate:.4f}"
    )

    print(
        f"Unnecessary defensive actions: "
        f"{unnecessary_actions:,}"
    )

    print(
        f"Unnecessary action rate: "
        f"{unnecessary_action_rate:.4f}"
    )

    # --------------------------------------------------------
    # Category results
    # --------------------------------------------------------

    print()
    print("Threat-category response:")

    for category in [
        "HIGH",
        "MEDIUM",
        "INVESTIGATION",
        "LOW",
        "UNKNOWN",
    ]:

        stats = category_stats[
            category
        ]

        if stats["samples"] == 0:
            continue

        category_appropriateness = (
            stats["appropriate"]
            / stats["samples"]
        )

        category_average_reward = (
            stats["reward"]
            / stats["samples"]
        )

        print(
            f"{category}: "
            f"{stats['samples']:,} samples | "
            f"Appropriate "
            f"{category_appropriateness:.4f} | "
            f"Avg reward "
            f"{category_average_reward:.4f}"
        )

    # --------------------------------------------------------
    # Action distribution
    # --------------------------------------------------------

    print()
    print("Action distribution:")

    for action_name, count in action_counts.items():

        percentage = (
            count
            / samples_processed
            * 100
        )

        print(
            f"{action_name}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
