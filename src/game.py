import pygame, sys, numpy as np

from constants import (WINDOW_WIDTH, 
                       WINDOW_HEIGHT, 
                       WINDOW_NAME,
                       DEFAULT_FONT_SIZE, 
                       GRID_SIZE,
                       COLOR_GRAY, 
                       COLOR_BLACK, 
                       GAME_FPS,
                       REWARD_EAT,
                       REWARD_COLLIDE,
                       REWARD_DO_NOTHING, 
                       NETWORK_OUTPUT_SIZE)

from utils import euclidean_distance

from snake import Snake
from food import Food
from agent import Agent

class Game:
    def __init__(self) -> None:
        self._init_screen()
        self._init_objects()

    def _init_screen(self) -> None:
        """
        Internal helper method that initializes pygame window
        """

        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.default_font = pygame.font.Font(pygame.font.get_default_font(), DEFAULT_FONT_SIZE)
        pygame.display.set_caption(WINDOW_NAME)
    
    def _init_objects(self) -> None:
        self.snake_ai = Snake((WINDOW_WIDTH, WINDOW_HEIGHT), segment_size=GRID_SIZE)
        self.snake_food = Food((WINDOW_WIDTH, WINDOW_HEIGHT), size=GRID_SIZE)

    def game_observation(self) -> np.ndarray:
        return pygame.surfarray.pixels3d(self.screen)

    def game_step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        """
        Encompases the neccessary modifications to adapt the game objects to be operable by RL model

        Parameters
        ----------
        action : int 
            The action that the RL model has determined to take 

        Returns
        -------
        observation : ?
            ?
        reward : int 
            Reward received for taking the chosen action 
        done : bool 
            States whether an episode has finished
        """

        # Determine snake's collisions
        snake_surface_collision = self.snake_ai.check_collision()
        snake_food_collision = self.snake_ai.check_collision(self.snake_food.get_position())

        if (snake_food_collision):
            self.snake_food = Food((WINDOW_WIDTH, WINDOW_HEIGHT), size=GRID_SIZE)

        # This was my first proposal of reward determination, however
        # calculating the REWARD_DO_NOTHING dynamically to be increasing as the snake closes the distance between fruit
        # could be beneficial, again this needs to be tested and confirmed
        # reward = REWARD_EAT if snake_food_collision else (REWARD_COLLIDE if snake_surface_collision else REWARD_DO_NOTHING)
        reward = 0

        if (snake_food_collision):
            reward = REWARD_EAT
        elif (snake_surface_collision):
            reward = REWARD_COLLIDE
        else:
            distance_to_fruit = euclidean_distance(self.snake_ai.get_head_position(), self.snake_food.get_position())
            reward = 1.0 / (distance_to_fruit + 1e-8)

        # Run update
        self.snake_ai.update(action, snake_food_collision)

        return self.game_observation(), reward, snake_surface_collision, snake_food_collision

    def action_loop(self) -> None:
        """
        Continious game loop, performs input handling, processing, re-rendering
        """

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

            """
            # Check snakes collision with itself and surrounding space
            if (self.snake_ai.check_collision()):
                pygame.quit()
                sys.exit()

            # Determine snake collision with food
            snake_food_collision = self.snake_ai.check_collision(self.snake_food.get_position())

            if (snake_food_collision):
                self.snake_food = Food((WINDOW_WIDTH, WINDOW_HEIGHT), size=GRID_SIZE)
                self.score += 1

            # Run update
            self.snake_ai.update(last_key_down, snake_food_collision)
            """

            # Fill background
            self.screen.fill(COLOR_BLACK)

            # Draw grids
            for x in range(0, WINDOW_WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, COLOR_GRAY, (x, 0), (x, WINDOW_HEIGHT))
            for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, COLOR_GRAY, (0, y), (WINDOW_WIDTH, y))
            

            # Render objects
            self.snake_food.render(self.screen)
            self.snake_ai.render(self.screen)

            # Render score text
            """
            score_text = self.default_font.render(f"Score: {self.snake_ai.get_apples_eaten()}", True, COLOR_WHITE)
            self.screen.blit(score_text, (10, 10))             
            """

            # Update the display
            pygame.display.flip()
                
            # Limit framerate
            self.clock.tick(GAME_FPS)

    def train_agent(self) -> None:
        """
        Continious simulated game loop used for training the agent
        """

        # Initialize the agent and load model
        agent = Agent(input_channels=1, output_size=NETWORK_OUTPUT_SIZE)
        agent.load_model()

        # Store the previous observation, initialized to zeros
        previous_raw_observation = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3))

        # Store the count of total finished episodes
        finished_episodes = 0

        while True:
            # Retrieve all processable game events
            for event in pygame.event.get():
                # Check if user is intending to close the application
                if event.type == pygame.QUIT:
                    print("Saved model on close!")
                    agent.save_model()
                    pygame.quit()
                    sys.exit()

            # Fill background
            self.screen.fill(COLOR_BLACK)

            # Render objects
            self.snake_food.render(self.screen)
            self.snake_ai.render(self.screen)

            # Update the display
            pygame.display.flip()

            # Limit framerate
            self.clock.tick(GAME_FPS)

            # Get current environment observation
            current_raw_observation = self.game_observation()

            # Subtract previous observation from current, to 'display' motion
            # We need to clip the rgb channel values between 0 and 255 because negative values might occur
            raw_observation = np.clip(previous_raw_observation - current_raw_observation, 0, 255)

            # Pre-process the final observation
            observation = agent.pre_process(raw_observation)

            # Pass the final observation through network
            action_probabilities = agent(observation)

            # Select action based on the probability distribution
            # action = agent.best_action(action_probabilities)

            # Sample an action based on the probability distribution
            action_idx, action = agent.sample_action(action_probabilities)

            # Take action, observe reward and next state
            _, reward, collided, ate_food = self.game_step(action)

            # Store the experience
            agent.store_experience((observation, action_idx, reward))

            # Update the previous observation
            previous_raw_observation = current_raw_observation

            # Episode has ended, we have collected training data. Update the network parameters
            if (collided):
                # Increment the total number of episodes finished
                finished_episodes += 1

                # Train the agent based on observations
                agent.train()

                # Re-initialize state again
                self._init_objects()

                # After every hundredth finished episode, we automatically save the model
                if (finished_episodes % 100 == 0):
                    print(f"Saved model after {finished_episodes} episodes!")
                    agent.save_model()

            # For display purposes, show the score
            if (ate_food):
                pygame.display.set_caption(WINDOW_NAME + f" - Score: {self.snake_ai.get_apples_eaten()}")

        # Calculate loss
        # Backpropogate the loss and update the models parameters

if __name__ == "__main__":
    # Game().action_loop()
    Game().train_agent()