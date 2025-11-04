import collections
import gymnasium as gym
import numpy as np
import cv2
import torch

class PinPad(gym.Env):

  # hardcoded_actions = [2,4,2,4,2,4,2,4,2,4,2,4,2,4,2,4,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

  COLORS = {
      '1': (255,   0,   0),
      '2': (  0, 255,   0),
      '3': (  0,   0, 255),
      '4': (255, 255,   0),
      '5': (255,   0, 255),
      '6': (  0, 255, 255),
      '7': (128,   0, 128),
      '8': (  0, 128, 128),
  }

  CUSTOM_TARGETS = {
      # 'three': ('1', '2', '3', '1', '3'),
  }

  def __init__(self, task, length=10000, extra_obs=False, size=[64, 64], random_starting_pos=True, device='cpu'):
    assert length > 0
    self.task = task
    self.device = device
    self.size = size
    layout = {
        'three': LAYOUT_THREE,
        'four': LAYOUT_FOUR,
        'five': LAYOUT_FIVE,
        'six': LAYOUT_SIX,
        'seven': LAYOUT_SEVEN,
        'eight': LAYOUT_EIGHT,
    }[task]
    self.layout = np.array([list(line) for line in layout.split('\n')]).T
    assert self.layout.shape == (16, 14), self.layout.shape
    self.length = length
    self.random = np.random.RandomState()
    self.pads = set(self.layout.flatten().tolist()) - set('* #\n')
    if task in self.CUSTOM_TARGETS:
      self.target = self.CUSTOM_TARGETS[task]
      print(f"Running custom target for task {task}")
    else:
      self.target = tuple(sorted(self.pads))
    print(f"pads {self.pads} target {self.target}")
    self.spawns = []
    for (x, y), char in np.ndenumerate(self.layout):
      if char != '#':
        self.spawns.append((x, y))
    self.sequence = collections.deque(maxlen=len(self.target))
    self.player = None
    self.steps = None
    self.done = None
    self.countdown = None

    self.next_tile_in_sequence = None; self.use_dense_reward = True

    self.EXTRA_OBS = extra_obs
    if self.EXTRA_OBS: 
      assert task == 'five', "Extra obs only supported for task five"
    self.random_starting_pos = random_starting_pos
    
    print(f'Created PinPad env with sequence: {"->".join(self.target)}' + f' and extra obs' if self.EXTRA_OBS else '')

    self.automated_action_idx = 0

  def gen_extra_obs(self):
    '''
    Generate a one-hot vector with the next target pad.
    '''
    onehot = np.zeros(len(self.target), dtype=np.float32)
    # ugh, hardcoding this shit
    correct = 0
    for entry in list(self.sequence):
      if entry == '1': correct = 1
      elif entry == str(correct+1): correct += 1
      else: correct = 0

    if correct < 5:
      onehot[correct] = 1.0

    # print(f"{self.sequence} next -> {np.argmax(onehot)}")

    return onehot
  
  def get_dense_reward(self, tile):
    # small negative for breaking the sequence + small positive for correct next tile
    if self.next_tile_in_sequence is None: self.next_tile_in_sequence = 0

    if self.next_tile_in_sequence >= len(self.target):
      print(f"WARNING: next_tile_in_sequence {self.next_tile_in_sequence} >= len(target) {len(self.target)}")
      from IPython import embed; embed()
    
    if tile != self.target[self.next_tile_in_sequence]: 
      self.next_tile_in_sequence = None
      return 0.0
    else:
      self.next_tile_in_sequence += 1
      return 0.01 * self.next_tile_in_sequence

  def reset(self):
    if self.random_starting_pos:
      self.player = self.spawns[self.random.randint(len(self.spawns))]
    else:
      print(f"WARN: hardcoded, nonrandom starting position.")
      self.player = self.spawns[20]

    self.sequence.clear()
    self.steps = 0
    self.done = False
    self.countdown = 0

    self.next_tile_in_sequence = None

    return self._obs(reward=0.0, is_first=True) # just return the observation
  
  @property
  def action_space(self):
    return gym.spaces.Discrete(5)

  @property
  def observation_space(self):
    # if self.EXTRA_OBS:
    #   return gym.spaces.Dict({
    #       'image': gym.spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8),
    #     #   'pixels': gym.spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8), # cover for compatability
    #       'dummy_state': gym.spaces.Box(low=0, high=1, shape=(len(self.target),), dtype=np.uint8),
    #       'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
    #       'is_first': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #       'is_last': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #       'is_terminal': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #   })
    # else:
    #   return gym.spaces.Dict({
    #       'image': gym.spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8),
    #     #   'pixels': gym.spaces.Box(low=0, high=255, shape=(*self.size, 3), dtype=np.uint8), # cover for compatability
    #       'reward': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
    #       'is_first': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #       'is_last': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #       'is_terminal': gym.spaces.Box(low=0, high=1, shape=(), dtype=bool),
    #   })
    return gym.spaces.Box(low=0, high=255, shape=(3, *self.size), dtype=np.uint8)

  def step(self, action, info=None):
    if type(action) is tuple:
      action = action[0].item()
    
    # print(action)
    if self.done:
      return self.reset()
    if self.countdown:
      self.countdown -= 1
      if self.countdown == 0:
        # self.player = self.spawns[self.random.randint(len(self.spawns))]
        self.sequence.clear(); self.next_tile_in_sequence = None

    reward = 0.0
    # move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
    move = [(0, 0), (0, 2), (0, -2), (2, 0), (-2, 0)][action]
    # print(f"pinpad move {move}")
    x = np.clip(self.player[0] + move[0], 0, 15)
    y = np.clip(self.player[1] + move[1], 0, 13)
    tile = self.layout[x][y]
    if tile != '#':
      self.player = (x, y)
    if tile in self.pads:
      if not self.sequence or self.sequence[-1] != tile:
        if self.use_dense_reward:
          reward += self.get_dense_reward(tile)
        self.sequence.append(tile)
    if tuple(self.sequence) == self.target and not self.countdown:
      info = {'success': True}
      reward += 10.0
      print(f"!!!!SUCCESS!!!")
      self.countdown = 4
    else:
      info = {'success': False}
    self.steps += 1
    self.done = self.done or (self.steps >= self.length)
    obs = self._obs(reward=reward, is_first=(self.steps==1), is_last=self.done)
    # self.render()

    ## pass the goal through
    # if info is not None:
    #   obs.update(info)



    return obs, torch.tensor(reward, device=self.device), torch.tensor(self.done).to(self.device), False, info

  def render(self):
    grid = np.zeros((16, 16, 3), np.uint8) + 255
    white = np.array([255, 255, 255])
    if self.countdown:
      grid[:] = (223, 255, 223)
    current = self.layout[self.player[0]][self.player[1]]
    for (x, y), char in np.ndenumerate(self.layout):
      if char == '#':
        grid[x, y] = (192, 192, 192)
      elif char in self.pads:
        color = np.array(self.COLORS[char])
        color = color if char == current else (10 * color + 90 * white) / 100
        grid[x, y] = color
    grid[self.player] = (0, 0, 0)
    grid[:, -2:] = (192, 192, 192)
    for i, char in enumerate(self.sequence):
      grid[2 * i + 1, -2] = self.COLORS[char]
    image = np.repeat(np.repeat(grid, 4, 0), 4, 1).transpose((1, 0, 2))

    # if the image is the wrong size, resize it
    if image.shape[0] != self.size[0] or image.shape[1] != self.size[1]:
      image = cv2.resize(image, tuple(reversed(self.size)), interpolation=cv2.INTER_NEAREST)

    # show the image
    # cv2.imshow("pinpad", image)
    # cv2.waitKey(50)
    return image

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):

    img = self.render()
    # if self.EXTRA_OBS:
    #   return dict(
    #       image=img, reward=reward, is_first=is_first, is_last=is_last,
    #       is_terminal=is_terminal, dummy_state=self.gen_extra_obs())

    # return dict(
    #     image=img, reward=reward, is_first=is_first, is_last=is_last,
    #     is_terminal=is_terminal)
    return torch.Tensor(img.transpose((2, 0, 1))).to(self.device)


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip('\n')

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip('\n')

LAYOUT_FIVE = """
################
#          4444#
#111       4444#
#111       4444#
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333       2222#
#333       2222#
#          2222#
################
""".strip('\n')

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#444        222#
#444        222#
#444        222#
################
""".strip('\n')

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

if __name__ == '__main__':
  import time
  import cv2
  l = 2000
  env = PinPad('three', length=l)
  env.reset()
  total_r = 0.0
  for steps in range(l):
    img = env.render()
    cv2.imshow("pinpad", img)
    key = cv2.waitKey(0)
    if key == ord('w'):
      cmd = 2
    elif key == ord('s'):
      cmd = 1
    elif key == ord('a'):
      cmd = 4
    elif key == ord('d'):
      cmd = 3
    elif key == ord('q'):
      break
    else:
      cmd = 0
    # env.step(env.action_space.sample())
    _, r, done, *other = env.step(cmd)
    total_r += r
    if done:
      print(f"total {total_r}")
      break
    print(f"step {steps} reward {r} done {done}")