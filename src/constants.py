from enum import IntEnum

# Pygame window constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_NAME = "Snake"

# Pygame font constants
DEFAULT_FONT_SIZE = 16

# Grid control parameters
GRID_SIZE = 20

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_GRAY = (45, 45, 45)
COLOR_BLACK = (0, 0, 0)

# Game control parameters
GAME_FPS = 30

# Enums
class Direction(IntEnum):
    LEFT = ord("a")
    UP = ord("w")
    RIGHT = ord("d")
    DOWN = ord("s")

DIRECTION_LIST = list(Direction)

# RL params
REWARD_EAT = 5
REWARD_COLLIDE = -5
REWARD_DO_NOTHING = -0.05
REWARD_DISCOUNT_GAMMA = 0.9

# Downsampling params
RESIZE_WINDOW_WIDTH = 128
RESIZE_WINDOW_HEIGHT = 128

# Network parameters
NETWORK_LEARNING_RATE = 0.001
NETWORK_OUTPUT_SIZE = 4
NETWORK_MODEL_PATH = "./model"
NETWORK_MODEL_NAME = "snake_agent_state.pth"

REPLAY_BUFFER_SIZE = 1000
BATCH_SIZE = 100