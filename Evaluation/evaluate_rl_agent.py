import numpy as np

from Environment.cyber_env import CyberEnvironment
from Environment.actions import Action
from Environment.reward import get_threat_category

from Agent.agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

NUM_STEPS = 1000

MODEL_PATH = (
    "Data/Processed/dqn_cyber_agent.pt"
)


# ============================================================
# ACTION APPROPRIATENESS
# ============================================================

def is_appropriate_action(
    category,
    action,
):

    # HIGH threats require a defensive response.
    if category == "HIGH":

        return action in {
            Action.ISOLATE_HOST,
            Action.BLOCK_IP,
            Action.INVESTIGATE,
            Action.INCREASE_MONITORING,
        }

    # MEDIUM threats require active mitigation
    # or investigation.
    if category == "MEDIUM":

        return action in {
            Action.BLOCK_IP,
            Action.ISOLATE_HOST,
            Action.INVESTIGATE,
            Action.INCREASE_MONITORING,
        }

    # Investigation-oriented attacks.
    if category == "INVESTIGATION":

        return action in {
            Action.INVESTIGATE,
            Action.INCREASE_MONITORING,
            Action.BLOCK_IP,
            Action.ISOLATE_HOST,
        }

    # Benign traffic.
    if category == "LOW":

        return action in {
            Action.ALLOW,
            Action.MONITOR,
        }

    # Unknown category.
    return action == Action.MONITOR


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "DQN AGENT EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Evaluation samples: {NUM_STEPS}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = CyberEnvironment(
        max_steps=NUM_STEPS
    )

    # --------------------------------------------------------
    # DQN
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.num_actions,
        batch_size=64,
    )

    agent.load(
        MODEL_PATH
    )

    # Disable exploration during evaluation.
    agent.epsilon = 0.0

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_reward = 0.0

    completed_steps = 0

    correct_predictions = 0

    appropriate_actions = 0

    missed_threats = 0

    unnecessary_defensive_actions = 0

    category_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "INVESTIGATION": 0,
        "LOW": 0,
        "UNKNOWN": 0,
    }

    category_appropriate = {
        "HIGH": 0,
        "MEDIUM": 0,
        "INVESTIGATION": 0,
        "LOW": 0,
        "UNKNOWN": 0,
    }

    category_rewards = {
        "HIGH": [],
        "MEDIUM": [],
        "INVESTIGATION": [],
        "LOW": [],
        "UNKNOWN": [],
    }

    action_counts = {
        action.name: 0
        for action in Action
    }

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    for step in range(
        NUM_STEPS
    ):

        action = agent.choose_action(
            state,
            training=False,
        )

        (
            next_state,
            reward,
            done,
            info,
        ) = env.step(
            action
        )

        total_reward += reward

        completed_steps += 1

        action_counts[
            action.name
        ] += 1

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        predicted_class = info[
            "current_predicted_class"
        ]

        true_label = info[
            "current_true_label"
        ]

        predicted_label = (
            env.detector.id_to_label.get(
                predicted_class,
                "Unknown"
            )
        )

        if predicted_label == true_label:

            correct_predictions += 1

        # ----------------------------------------------------
        # Threat category
        # ----------------------------------------------------

        category = get_threat_category(
            true_label
        )

        category_counts[
            category
        ] += 1

        category_rewards[
            category
        ].append(
            reward
        )

        # ----------------------------------------------------
        # Action appropriateness
        # ----------------------------------------------------

        appropriate = is_appropriate_action(
            category,
            action,
        )

        if appropriate:

            appropriate_actions += 1

            category_appropriate[
                category
            ] += 1

        # ----------------------------------------------------
        # Missed threats
        # ----------------------------------------------------

        if (
            category
            in {
                "HIGH",
                "MEDIUM",
                "INVESTIGATION",
            }
            and action
            in {
                Action.MONITOR,
                Action.ALLOW,
            }
        ):

            missed_threats += 1

        # ----------------------------------------------------
        # Unnecessary defensive action
        # ----------------------------------------------------

        if (
            category == "LOW"
            and action
            in {
                Action.BLOCK_IP,
                Action.ISOLATE_HOST,
                Action.INVESTIGATE,
                Action.INCREASE_MONITORING,
            }
        ):

            unnecessary_defensive_actions += 1

        state = next_state

        if done:

            break

        if (
            (step + 1) % 100 == 0
        ):

            print(
                f"Processed "
                f"{step + 1} samples..."
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    classification_accuracy = (
        correct_predictions
        / completed_steps
        if completed_steps > 0
        else 0.0
    )

    action_appropriateness = (
        appropriate_actions
        / completed_steps
        if completed_steps > 0
        else 0.0
    )

    average_reward = (
        total_reward
        / completed_steps
        if completed_steps > 0
        else 0.0
    )

    missed_threat_rate = (
        missed_threats
        / completed_steps
        if completed_steps > 0
        else 0.0
    )

    unnecessary_action_rate = (
        unnecessary_defensive_actions
        / completed_steps
        if completed_steps > 0
        else 0.0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL DQN RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Samples processed: "
        f"{completed_steps}"
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
        f"{missed_threats}"
    )

    print(
        f"Missed threat rate: "
        f"{missed_threat_rate:.4f}"
    )

    print(
        f"Unnecessary defensive actions: "
        f"{unnecessary_defensive_actions}"
    )

    print(
        f"Unnecessary action rate: "
        f"{unnecessary_action_rate:.4f}"
    )

    # ========================================================
    # THREAT CATEGORY RESPONSE
    # ========================================================

    print(
        "\nThreat-category response:"
    )

    for category in category_counts:

        count = category_counts[
            category
        ]

        appropriate = category_appropriate[
            category
        ]

        rate = (
            appropriate / count
            if count > 0
            else 0.0
        )

        average_category_reward = (
            np.mean(
                category_rewards[
                    category
                ]
            )
            if category_rewards[
                category
            ]
            else 0.0
        )

        print(
            f"  {category}: "
            f"{count} samples | "
            f"Appropriate: "
            f"{appropriate / count:.4f} | "
            f"Avg reward: "
            f"{average_category_reward:.4f}"
            if count > 0
            else
            f"  {category}: 0 samples"
        )

    # ========================================================
    # ACTION DISTRIBUTION
    # ========================================================

    print(
        "\nAction distribution:"
    )

    for action_name, count in action_counts.items():

        percentage = (
            count
            / completed_steps
            * 100
            if completed_steps > 0
            else 0.0
        )

        print(
            f"  {action_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()