from Environment.cyber_env import CyberEnvironment
from Agent.agent import CyberAgent


def main():

    print(
        "Starting Autonomous Cyber Threat "
        "Hunting Agent..."
    )

    env = CyberEnvironment()
    agent = CyberAgent()

    state = env.reset()

    print("\nInitial state:")
    print(state)

    total_reward = 0.0

    for step in range(10):

        # Agent chooses action from current state
        action = agent.choose_action(state)

        # Environment applies action
        next_state, reward, done, info = env.step(
            action
        )

        total_reward += reward

        print(
            f"\n========== STEP {step + 1} =========="
        )

        print(
            f"Current predicted class: "
            f"{info['current_predicted_class']}"
        )

        print(
            f"Current confidence: "
            f"{info['current_probability']:.4f}"
        )

        print(
            f"True label: "
            f"{info['current_true_label']}"
        )

        print(
            f"Action: {info['action']}"
        )

        print(
            f"Reward: {info['reward']:.2f}"
        )

        print(
            f"Next predicted class: "
            f"{info['next_predicted_class']}"
        )

        print(
            f"Next confidence: "
            f"{info['next_probability']:.4f}"
        )

        state = next_state

        if done:
            break

    print(
        "\n========================================"
    )

    print(
        f"Total reward: {total_reward:.2f}"
    )

    statistics = env.get_statistics()

    print(
        f"Steps completed: "
        f"{statistics['steps']}"
    )

    print(
        "Action counts:"
    )

    for action, count in statistics[
        "action_counts"
    ].items():

        print(
            f"  {action}: {count}"
        )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()