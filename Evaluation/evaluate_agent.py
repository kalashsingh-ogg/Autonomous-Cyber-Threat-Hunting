from Environment.cyber_env import CyberEnvironment
from Agent.agent import CyberAgent


# Number of test samples used for baseline evaluation.
NUM_STEPS = 1000


def main():

    print(
        "========================================"
    )
    print(
        "BASELINE AGENT EVALUATION"
    )
    print(
        "========================================"
    )

    # Explicitly request 1,000 evaluation steps.
    env = CyberEnvironment(
        max_steps=NUM_STEPS
    )

    agent = CyberAgent()

    state = env.reset()

    total_reward = 0.0
    completed_steps = 0

    correct_predictions = 0

    for step in range(NUM_STEPS):

        # Agent selects an action from the current state.
        action = agent.choose_action(state)

        # Environment applies the action.
        next_state, reward, done, info = env.step(
            action
        )

        total_reward += reward
        completed_steps += 1

        # Get predicted class ID.
        predicted_class = info[
            "current_predicted_class"
        ]

        # Get actual dataset label.
        true_label = info[
            "current_true_label"
        ]

        # Convert predicted class ID back to label.
        predicted_label = (
            env.detector.id_to_label.get(
                predicted_class,
                "Unknown"
            )
        )

        # Calculate classifier accuracy.
        if predicted_label == true_label:
            correct_predictions += 1

        state = next_state

        # Progress information every 100 samples.
        if (step + 1) % 100 == 0:

            print(
                f"Processed "
                f"{step + 1} samples..."
            )

        if done:
            break

    # Get environment statistics.
    statistics = env.get_statistics()

    # Calculate classification accuracy.
    accuracy = (
        correct_predictions / completed_steps
        if completed_steps > 0
        else 0.0
    )

    # Calculate average reward.
    average_reward = (
        total_reward / completed_steps
        if completed_steps > 0
        else 0.0
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL BASELINE RESULTS"
    )

    print(
        "========================================"
    )

    print(
        f"Samples processed: "
        f"{completed_steps}"
    )

    print(
        f"Threat classification accuracy: "
        f"{accuracy:.4f}"
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
        "\nAction distribution:"
    )

    for action, count in statistics[
        "action_counts"
    ].items():

        percentage = (
            count / completed_steps * 100
            if completed_steps > 0
            else 0.0
        )

        print(
            f"  {action}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()