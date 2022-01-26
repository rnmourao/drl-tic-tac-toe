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


  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    return np.array([0] * self.size).astype(np.float32)


  def step(self, state, action, agent):
    
    outcome, next_state, row = check_outcome(state, action, self.shape, self.row_size)

    if outcome == "win":
      reward = 1
      done = True
    elif outcome == "mistake":
      reward = -1
      done = True
    elif outcome == "withdraw":
      reward = 0
      done = True
    else: # game has't finished yet
      reward = -0.1
      done = False

    info = {"outcome": outcome, "row": row}

    return next_state, reward, done, info


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