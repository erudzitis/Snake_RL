import pygame, sys

from constants import (WINDOW_WIDTH, 
                       WINDOW_HEIGHT, 
                       WINDOW_NAME,
                       DEFAULT_FONT_SIZE, 
                       GRID_SIZE, 
                       COLOR_WHITE,
                       COLOR_GRAY, 
                       COLOR_BLACK, 
                       GAME_FPS)

from snake import Snake
from food import Food

class Game:
    def __init__(self) -> None:
        self._init_screen()
        self.score = 0

    def _init_screen(self) -> None:
        """
        Internal helper method that initializes pygame window
        """

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), DEFAULT_FONT_SIZE)
        pygame.display.set_caption(WINDOW_NAME)

    def action_loop(self) -> None:
        """
        Continious game loop, performs input handling, processing, re-rendering
        """

        snake_ai = Snake((WINDOW_WIDTH, WINDOW_HEIGHT), segment_size=GRID_SIZE)
        snake_food = Food((WINDOW_WIDTH, WINDOW_HEIGHT), size=GRID_SIZE)

        while True:
            last_key_down = None

            # Retrieve all processable game events
            for event in pygame.event.get():
                # Check if user is intending to close the application
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # Store latest key inputs
                if event.type == pygame.KEYDOWN:
                    last_key_down = event.key

            # Check snakes collision with itself and surrounding space
            if (snake_ai.check_collision()):
                pygame.quit()
                sys.exit()

            # Determine snake collision with food
            snake_food_collision = snake_ai.check_collision(snake_food.get_position())

            if (snake_food_collision):
                snake_food = Food((WINDOW_WIDTH, WINDOW_HEIGHT), size=GRID_SIZE)
                self.score += 1

            # Run update
            snake_ai.update(last_key_down, snake_food_collision)

            # Fill background
            self.screen.fill(COLOR_BLACK)

            # Draw grids
            """
            for x in range(0, WINDOW_WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, COLOR_GRAY, (x, 0), (x, WINDOW_HEIGHT))
            for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, COLOR_GRAY, (0, y), (WINDOW_WIDTH, y))
            """

            # Render objects
            snake_food.render(self.screen)
            snake_ai.render(self.screen)

            # Render score text
            score_text = self.default_font.render(f"Score: {self.score}", True, COLOR_WHITE)
            self.screen.blit(score_text, (10, 10)) 

            # Update the display
            pygame.display.flip()

            # Test
            # screen_pixels = pygame.surfarray.pixels3d(self.screen)
                
            # Limit framerate
            self.clock.tick(GAME_FPS)


if __name__ == "__main__":
    Game().action_loop()