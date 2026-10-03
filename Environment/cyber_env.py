import numpy as np
import pandas as pd

from .actions import Action
from .state import CyberState
from .reward import calculate_reward
from .config import MAX_STEPS, NUM_ACTIONS, STATE_SIZE
from Detection.threat_detector import ThreatDetector


class CyberEnvironment:

    def __init__(
        self,
        data_path="Data/Processed/test_processed.csv",
        max_steps=None
    ):
        self.data_path = data_path

        print("Loading dataset...")
        self.data = pd.read_csv(self.data_path)

        print(
            f"Dataset rows loaded: {len(self.data)}"
        )

        self.detector = ThreatDetector()

        self.feature_columns = [
            column
            for column in self.data.columns
            if column != "label"
        ]

        if len(self.feature_columns) != 46:
            raise ValueError(
                f"Expected 46 features, found "
                f"{len(self.feature_columns)}"
            )

        self.current_index = 0
        self.current_step = 0

        # Allow evaluation scripts to override MAX_STEPS.
        # If no value is provided, use the normal environment limit.
        if max_steps is None:
            self.max_steps = min(
                MAX_STEPS,
                len(self.data)
            )
        else:
            if max_steps <= 0:
                raise ValueError(
                    "max_steps must be greater than 0"
                )

            self.max_steps = min(
                max_steps,
                len(self.data)
            )

        self.num_actions = NUM_ACTIONS
        self.state_size = STATE_SIZE

        self.state = None
        self.current_true_label = None

        self.total_reward = 0.0

        self.action_counts = {
            action.name: 0
            for action in Action
        }

    def reset(self):

        self.current_step = 0
        self.current_index = 0

        self.total_reward = 0.0

        self.action_counts = {
            action.name: 0
            for action in Action
        }

        return self._load_current_observation()

    def _load_current_observation(self):

        row = self.data.iloc[self.current_index]

        features = row[self.feature_columns]

        prediction = self.detector.predict(features)

        self.current_true_label = row["label"]

        threat_probability = prediction["probability"]

        self.state = CyberState(
            threat_class=prediction["class_id"],
            threat_probability=threat_probability,
            host_risk=threat_probability,
            connection_risk=threat_probability,
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

        current_class = self.state.threat_class
        current_probability = self.state.threat_probability
        current_true_label = self.current_true_label

        reward = calculate_reward(
            threat_probability=current_probability,
            action=action,
            true_label=current_true_label,
            predicted_label=current_class,
        )

        self.total_reward += reward
        self.action_counts[action.name] += 1

        self._apply_action(action)

        self.current_step += 1
        self.current_index += 1

        done = (
            self.current_step >= self.max_steps
            or self.current_index >= len(self.data)
        )

        if not done:
            next_state = self._load_current_observation()
        else:
            next_state = np.array(
                self.state.to_vector(),
                dtype=np.float32
            )

        info = {
            "step": self.current_step,
            "action": action.name,
            "reward": reward,
            "current_predicted_class": current_class,
            "current_probability": current_probability,
            "current_true_label": current_true_label,
            "next_predicted_class": self.state.threat_class,
            "next_probability": self.state.threat_probability,
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

    def get_statistics(self):

        return {
            "steps": self.current_step,
            "total_reward": self.total_reward,
            "action_counts": self.action_counts.copy(),
        }

    def render(self):

        print(
            "\n========== CYBER ENVIRONMENT =========="
        )

        print(
            f"Step: "
            f"{self.current_step}/{self.max_steps}"
        )

        print(
            f"Threat class: "
            f"{self.state.threat_class}"
        )

        print(
            f"Threat probability: "
            f"{self.state.threat_probability:.4f}"
        )

        print(
            f"Host risk: "
            f"{self.state.host_risk:.4f}"
        )

        print(
            f"Connection risk: "
            f"{self.state.connection_risk:.4f}"
        )

        print(
            f"Monitoring level: "
            f"{self.state.monitoring_level:.2f}"
        )

        print(
            f"Host isolated: "
            f"{self.state.host_isolated}"
        )

        print(
            f"IP blocked: "
            f"{self.state.ip_blocked}"
        )

        print(
            f"Investigation active: "
            f"{self.state.investigation_active}"
        )

        print(
            f"True label: "
            f"{self.current_true_label}"
        )

        print(
            "========================================"
        )