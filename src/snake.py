"""
Snake class that represents moving snake object in game
"""

import random, pygame, numpy as np
from constants import COLOR_YELLOW, COLOR_ORANGE, Direction, DIRECTION_LIST
from utils import is_opposite_direction

class Snake:
    def __init__(self, surface_dimensions, segment_size=20, head_color=COLOR_ORANGE, body_color=COLOR_YELLOW) -> None:
        """
        Initializes snake object

        Parameters
        ----------
        surface_dimensions : (int, int)    
            Tuple that represents the width and height of the to be drawn surface
        segment_size : int, default = 20
            Controls the width and height of each snake's segment 
        head_color: (int, int, int), default = COLOR_ORANGE
            Controls the snakes color, expected RGB tuple with values ranging from 0 - 255
        body_color: (int, int, int), default = COLOR_YELLOW
            Controls the snakes color, expected RGB tuple with values ranging from 0 - 255
        """

        self.surface_dimensions = surface_dimensions
        self.segment_size = segment_size
        self.head_color = head_color
        self.body_color = body_color
        self.direction = random.choice(DIRECTION_LIST) # Assign a random direction by default
        self.segments = [(random.randrange(round((surface_dimensions[0] // 2 - self.segment_size * 4) / self.segment_size) * self.segment_size, round((surface_dimensions[0] // 2 + self.segment_size * 4) / self.segment_size) * self.segment_size, self.segment_size),
                          random.randrange(round((surface_dimensions[1] // 2 - self.segment_size * 4) / self.segment_size) * self.segment_size, round((surface_dimensions[1] // 2 + self.segment_size * 4) / self.segment_size) * self.segment_size, self.segment_size))] # Keeps track of (x, y) coordinates of each snakes body segment

    def update(self, key_down, grow=False) -> None:
        """
        Performs single move update

        Parameters
        ----------
        key_down : int, optional
            Sets the current moving direction of the snake. Not passing or not passing appropriate key_down will make snake preserve its current trajectory.

        grow : boolean, default = false
            Indicates whether or not to remove the last stored segment
        """
        # Optional / invalid value check
        if (key_down == None or key_down not in DIRECTION_LIST):
            key_down = self.direction

        # Update direction, but make sure the snake can't turn instantaniously
        if (not is_opposite_direction(self.direction, key_down)):
            self.direction = key_down
        else:
            self.direction = self.direction

        # Retrieve old head position
        curr_head_pos = np.array(self.segments[0])

        # Calculate the direction matrix, results in 1D array with 2 values indicating move direction in x-axis, y-axis
        direction_matrix = np.array([self.direction == Direction.LEFT, self.direction == Direction.UP, self.direction == Direction.RIGHT, self.direction == Direction.DOWN], dtype=int)
        direction_matrix = np.multiply(direction_matrix, [-1, -1, 1, 1])
        direction_matrix = np.array([direction_matrix[::2].sum(), direction_matrix[1::2].sum()])

        # Update body segments
        new_head_pos = curr_head_pos + direction_matrix * self.segment_size
        self.segments.insert(0, tuple(new_head_pos))

        # Remove tail if required
        if (not grow):
            self.segments.pop()

    def check_collision(self, pt=None) -> bool:
        """
        Performs a check to test whether snake has collided with itself or the surround surface,
        if pt is not provided, otherwise with the provided point.

        Parameters
        ----------
        pt : (int, int), optional
            Tuple coordinates of point to check collision for. 
            If not passed, collision will be checked against the snakes head and its body segments, screen area.
        """

        # Retrieve the current snakes head position
        head_pos = self.segments[0]

        # Perform regular check against surface borders and snakes body segments
        if (pt is None):
            return (head_pos[0] < 0 or head_pos[0] >= self.surface_dimensions[0] 
                    or head_pos[1] < 0 or head_pos[1] >= self.surface_dimensions[1]) or (head_pos in self.segments[1:])
        else:
            return head_pos == pt
        
    def get_apples_eaten(self):
        return len(self.segments) - 1
    
    def get_head_position(self) -> tuple[int, int]:
        return self.segments[0]

    def render(self, screen) -> None:
        # Render head
        pygame.draw.rect(screen, self.head_color, (*(self.segments[0]), self.segment_size, self.segment_size))

        # Render body
        for segment in self.segments[1:]:
            pygame.draw.rect(screen, self.body_color, (*segment, self.segment_size, self.segment_size))