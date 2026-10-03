import os
import random

import numpy as np
import torch

from Environment.cyber_env import CyberEnvironment
from Agent.agent import DQNAgent


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPISODES = 20

STEPS_PER_EPISODE = 500

BATCH_SIZE = 64

TARGET_UPDATE_FREQUENCY = 5

MODEL_PATH = (
    "Data/Processed/dqn_cyber_agent.pt"
)

CHECKPOINT_DIR = (
    "Data/Processed/dqn_checkpoints"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print(
        "\n"
        "========================================\n"
        "DQN CYBER THREAT-HUNTING TRAINING\n"
        "========================================"
    )

    print(
        f"Episodes: {EPISODES}"
    )

    print(
        f"Steps per episode: "
        f"{STEPS_PER_EPISODE}"
    )

    print(
        f"Maximum training steps: "
        f"{EPISODES * STEPS_PER_EPISODE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        "========================================\n"
    )

    # --------------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = CyberEnvironment(
        max_steps=STEPS_PER_EPISODE
    )

    # --------------------------------------------------------
    # Create DQN agent
    # --------------------------------------------------------

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.num_actions,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Training statistics
    # --------------------------------------------------------

    episode_rewards = []
    episode_losses = []

    total_environment_steps = 0

    best_reward = float(
        "-inf"
    )

    # ========================================================
    # EPISODE LOOP
    # ========================================================

    for episode in range(
        1,
        EPISODES + 1
    ):

        state = env.reset()

        episode_reward = 0.0

        episode_loss_values = []

        completed_steps = 0

        # ----------------------------------------------------
        # Step loop
        # ----------------------------------------------------

        for step in range(
            STEPS_PER_EPISODE
        ):

            action = agent.choose_action(
                state,
                training=True,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(
                action
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.train_step()

            if loss is not None:

                episode_loss_values.append(
                    loss
                )

            state = next_state

            episode_reward += reward

            completed_steps += 1

            total_environment_steps += 1

            # ------------------------------------------------
            # Episode termination
            # ------------------------------------------------

            if done:
                break

        # ----------------------------------------------------
        # Exploration decay
        # ----------------------------------------------------

        agent.decay_epsilon()

        # ----------------------------------------------------
        # Target network update
        # ----------------------------------------------------

        if (
            episode
            % TARGET_UPDATE_FREQUENCY
            == 0
        ):

            agent.update_target_network()

        # ----------------------------------------------------
        # Calculate episode metrics
        # ----------------------------------------------------

        average_loss = (
            float(
                np.mean(
                    episode_loss_values
                )
            )
            if episode_loss_values
            else 0.0
        )

        average_reward = (
            episode_reward
            / completed_steps
            if completed_steps > 0
            else 0.0
        )

        episode_rewards.append(
            episode_reward
        )

        episode_losses.append(
            average_loss
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if episode_reward > best_reward:

            best_reward = episode_reward

            best_model_path = (
                "Data/Processed/"
                "dqn_cyber_agent_best.pt"
            )

            agent.save(
                best_model_path
            )

        # ----------------------------------------------------
        # Save periodic checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"dqn_episode_{episode}.pt"
        )

        agent.save(
            checkpoint_path
        )

        # ----------------------------------------------------
        # Print episode information
        # ----------------------------------------------------

        print(
            "\n----------------------------------------"
        )

        print(
            f"Episode: "
            f"{episode}/{EPISODES}"
        )

        print(
            f"Steps: "
            f"{completed_steps}"
        )

        print(
            f"Episode reward: "
            f"{episode_reward:.2f}"
        )

        print(
            f"Average reward: "
            f"{average_reward:.4f}"
        )

        print(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        print(
            f"Epsilon: "
            f"{agent.epsilon:.4f}"
        )

        print(
            f"Replay buffer: "
            f"{len(agent.replay_buffer)}"
        )

        print(
            f"Total environment steps: "
            f"{total_environment_steps}"
        )

        print(
            "----------------------------------------"
        )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    agent.save(
        MODEL_PATH
    )

    # ========================================================
    # TRAINING SUMMARY
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "DQN TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Episodes completed: "
        f"{EPISODES}"
    )

    print(
        f"Total environment steps: "
        f"{total_environment_steps}"
    )

    print(
        f"Best episode reward: "
        f"{max(episode_rewards):.2f}"
    )

    print(
        f"Final episode reward: "
        f"{episode_rewards[-1]:.2f}"
    )

    print(
        f"Final epsilon: "
        f"{agent.epsilon:.4f}"
    )

    print(
        f"Final replay buffer size: "
        f"{len(agent.replay_buffer)}"
    )

    print(
        f"Final model: "
        f"{MODEL_PATH}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()