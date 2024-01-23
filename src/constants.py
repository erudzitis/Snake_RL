from enum import IntEnum

# Pygame window constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "Snake"

# Pygame font constants
DEFAULT_FONT_SIZE = 16

# Grid control parameters
GRID_SIZE = 30

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_GRAY = (45, 45, 45)
COLOR_BLACK = (0, 0, 0)

# Game control parameters
GAME_FPS = 10

# Enums
class Direction(IntEnum):
    LEFT = ord("a")
    UP = ord("w")
    RIGHT = ord("d")
    DOWN = ord("s")

DIRECTION_LIST = list(Direction)