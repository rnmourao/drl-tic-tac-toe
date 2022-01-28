import numpy as np
import gym
from gym import spaces
from utils import check_outcome


class TicTacToeEnv(gym.Env):
    """
    Custom Environment that follows gym interface. 
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, shape=(3, 3)):
        super(TicTacToeEnv, self).__init__()

        dim = set(shape)
        if len(dim) != 1:
            raise Exception("Board must have same dimensions.")

        dim = dim.pop()
        if dim < 3:
            raise Exception("Dimension size has to be at least 3.")

        self.row_size = dim
        self.shape = shape
        self.size = np.multiply(*self.shape)

        # the observation space is an 1D-array where each element represents a place in the
        # tic-tac-toe board and can have 3 possible values: -1, 0, and 1, which represent,
        # respectively, enemy, blank, and player.
        self.observation_space = spaces.MultiDiscrete([3] * self.size)

        # the action space is the agent's move in the board
        self.action_space = spaces.Discrete(self.size)

    def add_players(self, players):
        self.players = players

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        return np.array([0] * self.size).astype(np.float32)

    def step(self, current_player, state, action):
        outcome, next_state, _ = check_outcome(
            state, action, self.shape, self.row_size)
        self.notify_players(current_player, state, action, next_state, outcome)
        return outcome, next_state

    def notify_players(self, current_player, state, action, next_state, outcome):
        if outcome:
            done = True
            for player in self.players:
                if player is current_player:
                    out = outcome
                else:
                    if outcome == "win":
                        out = "other_win"
                    elif outcome == "mistake":
                        out = "other_mistake"
                    else:
                        out = "withdraw"
                reward = self.calculate_reward(out)
                player.step(state, action, reward,
                            next_state, done, out)
        else:
            done = False
            reward = self.calculate_reward(outcome)
            current_player.step(state, action, reward,
                                next_state, done, outcome)

    def calculate_reward(self, outcome):
        if outcome == "win":
            reward = 1
        elif outcome == "other_win":
            reward = -1
        elif outcome == "mistake":
            reward = -10
        elif outcome == "other_mistake":
            reward = 0
        elif outcome == "withdraw":
            reward = 0.5
        else:
            reward = -0.1
        return reward

    def render(self, state, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        symbols = {-1: "O", 0: " ", 1: "P"}

        grid = ""
        for i, value in enumerate(state):
            grid += symbols[value]
            if (i+1) % self.row_size == 0:
                grid += "\n"
            else:
                grid += " "
        print(grid)

    def close(self):
        pass
