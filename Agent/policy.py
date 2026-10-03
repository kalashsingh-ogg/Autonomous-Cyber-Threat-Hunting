import random

from Environment.actions import Action


class RandomPolicy:

    def choose_action(self, state):

        return random.choice(list(Action))