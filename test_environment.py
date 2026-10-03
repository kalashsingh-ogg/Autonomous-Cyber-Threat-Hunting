from Environment.cyber_env import CyberEnvironment
from Environment.actions import Action


def main():
    print("Starting Cyber Environment...")

    env = CyberEnvironment()

    state = env.reset()

    print("\nInitial state:")
    print(state)

    actions = [
        Action.MONITOR,
        Action.INVESTIGATE,
        Action.INCREASE_MONITORING,
        Action.BLOCK_IP,
        Action.ISOLATE_HOST,
    ]

    for step in range(5):

        action = actions[step]

        next_state, reward, done, info = env.step(action)

        print(f"\n========== STEP {step + 1} ==========")
        print(f"Action: {action.name}")
        print(f"Reward: {reward}")
        print(f"Next state: {next_state}")
        print(f"Predicted class: {info['predicted_class']}")
        print(
            f"Predicted probability: "
            f"{info['predicted_probability']:.4f}"
        )
        print(f"True label: {info['true_label']}")

        if done:
            print("\nEnvironment episode finished.")
            break


if __name__ == "__main__":
    main()