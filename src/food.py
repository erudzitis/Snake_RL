"""
Food class that represents static, edible food object in game
"""

import random, pygame
from constants import COLOR_RED

class Food:
    def __init__(self, surface_dimensions, size=20, color=COLOR_RED) -> None:
        """
        Initializes food object and assigns a random location on surface

        Parameters
        ----------
        surface_dimensions : (int, int)    
            Tuple that represents the width and height of the to be drawn surface
        size : int, default = 20
            Width and height of the food object 
        color: (int, int, int), default = COLOR_RED
            Controls the food color, expected RGB tuple with values ranging from 0 - 255
        """

        self.surface_dimensions = surface_dimensions    
        self.size = size
        self.color = color
        self.position = (random.randrange(0, surface_dimensions[0], self.size),
                          random.randrange(0, surface_dimensions[1], self.size))

    def render(self, screen):
        pygame.draw.rect(screen, self.color, (*self.position, self.size, self.size))

    def get_position(self) -> (int, int):
        return self.position

