import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Environment.actions import Action
from Agent.dqn_network import DQNNetwork
from Agent.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        state_size=8,
        action_size=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=50000,
        batch_size=64,
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"DQN device: {self.device}"
        )

        self.policy_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(self.device)

        self.target_network = DQNNetwork(
            state_size=state_size,
            action_size=action_size,
        ).to(self.device)

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity
        )

        self.training_steps = 0

    # ========================================================
    # ACTION SELECTION
    # ========================================================

    def choose_action(
        self,
        state,
        training=True,
    ):

        if training and random.random() < self.epsilon:

            return random.choice(
                list(Action)
            )

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_network(
                state_tensor
            )

        action_index = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

        return Action(action_index)

    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.replay_buffer.push(
            state,
            int(action),
            reward,
            next_state,
            done,
        )

    # ========================================================
    # TRAIN FROM REPLAY BUFFER
    # ========================================================

    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:

            return None

        batch = self.replay_buffer.sample(
            self.batch_size
        )

        states = np.array(
            [experience[0] for experience in batch],
            dtype=np.float32,
        )

        actions = np.array(
            [experience[1] for experience in batch],
            dtype=np.int64,
        )

        rewards = np.array(
            [experience[2] for experience in batch],
            dtype=np.float32,
        )

        next_states = np.array(
            [experience[3] for experience in batch],
            dtype=np.float32,
        )

        dones = np.array(
            [experience[4] for experience in batch],
            dtype=np.float32,
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        )

        # Current Q-values
        current_q_values = self.policy_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values
        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = next_q_values.max(
                dim=1
            )[0]

            target_q_values = (
                rewards
                + self.gamma
                * max_next_q_values
                * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.training_steps += 1

        return float(
            loss.item()
        )

    # ========================================================
    # UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    # ========================================================
    # DECAY EXPLORATION
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    def save(
        self,
        path
    ):

        torch.save(
            {
                "policy_network": (
                    self.policy_network.state_dict()
                ),
                "target_network": (
                    self.target_network.state_dict()
                ),
                "optimizer": (
                    self.optimizer.state_dict()
                ),
                "epsilon": self.epsilon,
                "training_steps": self.training_steps,
            },
            path,
        )

        print(
            f"DQN model saved to: {path}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.policy_network.load_state_dict(
            checkpoint["policy_network"]
        )

        self.target_network.load_state_dict(
            checkpoint["target_network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon_min,
        )

        self.training_steps = checkpoint.get(
            "training_steps",
            0,
        )

        print(
            f"DQN model loaded from: {path}"
        )