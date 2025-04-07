from gym import spaces
import numpy as np
import random
import pygame
import gym
import math
import sys

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, grid_size=10, render_mode="human", cell_size=20):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        
        # Direction constants: 0: RIGHT, 1: DOWN, 2: LEFT, 3: UP
        self.DIRECTIONS = {
            0: (0, 1),   # RIGHT
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (-1, 0)   # UP
        }
        
        self.action_space = spaces.Discrete(3)  # 0: STRAIGHT, 1: RIGHT, 2: LEFT
        
        # Improved observation space with more information
        # [danger_straight, danger_right, danger_left, 
        #  direction_right, direction_down, direction_left, direction_up,
        #  food_right, food_down, food_left, food_up]
        self.observation_space = self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        self.window = None
        self.clock = None

        self.max_steps = 200
        self.step_count = 0

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Snake starts in the middle with a single segment
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice([0, 1, 2, 3])  # Random start direction
        self.spawn_food()
        self.done = False
        self.score = 0
        self.step_count = 0
        self.last_distance = self._get_food_distance()
        
        if self.render_mode == "human":
            self.render()
            
        return self.get_observation(), {}

    def spawn_food(self):
        """Spawn food at a random empty position"""
        while True:
            # Consistent coordinate system - (row, col)
            self.food = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if self.food not in self.snake:
                break

    def _get_food_distance(self):
        """Calculate Manhattan distance to food"""
        head_row, head_col = self.snake[0]
        food_row, food_col = self.food
        return abs(head_row - food_row) + abs(head_col - food_col)

    def step(self, action):
        self.step_count += 1
        reward = 0
        new_direction = 0
        
        if action == 0:
            new_direction = self.direction
        elif action == 1:
            new_direction = (self.direction + 1) % 4
        elif action == 2:
            new_direction = (self.direction - 1) % 4
        
        self.direction = new_direction
        
        dr, dc = self.DIRECTIONS[self.direction]
        head_row, head_col = self.snake[0]
        new_head = (head_row + dr, head_col + dc)
        new_row, new_col = new_head

        if (
            new_row < 0 or new_row >= self.grid_size or
            new_col < 0 or new_col >= self.grid_size or
            new_head in self.snake
        ):
            self.done = True
            reward = -1.0
            if self.render_mode == "human":
                self.render()
            return self.get_observation(), reward, self.done, False, {"score": self.score}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self.step_count = 0
            self.spawn_food()
        else:
            self.snake.pop()
            reward = -0.001
            

        
        # reachable = self.bfs_reachable_empty_tiles()
        # max_possible = self.grid_size * self.grid_size - len(self.snake)
        # if reachable != max_possible:
        #     reward -= 1  # punish for cutting off space

        if self.step_count >= self.max_steps:
            reward = -1.0
            self.done = True

        if self.render_mode == "human":
            self.render()

        return self.get_observation(), reward, self.done, False, {"score": self.score}

# ... [rest of code remains the same] ...

    def bfs_reachable_empty_tiles(self):
        visited = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for r, c in self.snake:
            visited[r][c] = True

        found = False
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if not visited[r][c]:
                    start = (r, c)
                    found = True
                    break
            if found:
                break

        if not found:
            return 0

        queue = [start]
        visited[start[0]][start[1]] = True
        count = 1

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        count += 1
        return count

    def get_observation(self):
        """
        Create an improved observation with:
        - Danger in three directions
        - Current direction one-hot encoded
        - Food direction
        """
        head_row, head_col = self.snake[0]
        food_row, food_col = self.food

        # Check danger in each relative direction
        # danger_straight = self._is_collision_relative(0)  # straight
        # danger_right = self._is_collision_relative(1)     # right
        # danger_left = self._is_collision_relative(2)      # left

        # One-hot encode current direction
        # dir_right = int(self.direction == 0)
        # dir_down = int(self.direction == 1) 
        # dir_left = int(self.direction == 2)
        # dir_up = int(self.direction == 3)

        # Food direction relative to head
        # food_right = int(food_col > head_col)
        # food_down = int(food_row > head_row)
        # food_left = int(food_col < head_col)
        # food_up = int(food_row < head_row)

        head = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        body = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        food = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        head_row, head_col = self.snake[0]
        head[head_row, head_col] = 1

        for segment in self.snake[1:]:
            body[segment[0], segment[1]] = 1

        food[self.food[0], self.food[1]] = 1

        obs = np.stack([head, body, food], axis=0)

        
        return obs

    def _is_collision_relative(self, turn_direction):
        """
        Check if moving in a relative direction would result in collision
        turn_direction: 0 (straight), 1 (right), 2 (left)
        """
        # Calculate absolute direction after turn
        if turn_direction == 0:  # straight
            abs_direction = self.direction
        elif turn_direction == 1:  # right
            abs_direction = (self.direction + 1) % 4
        elif turn_direction == 2:  # left
            abs_direction = (self.direction - 1) % 4
        
        # Get position delta for that direction
        dr, dc = self.DIRECTIONS[abs_direction]
        head_row, head_col = self.snake[0]
        check_row, check_col = head_row + dr, head_col + dc
        
        # Check if this position would be a collision
        return (
            check_row < 0 or check_row >= self.grid_size or
            check_col < 0 or check_col >= self.grid_size or
            (check_row, check_col) in self.snake
        )

    def render(self):
        """Render the game state"""
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Snake Game")
            self.window = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))  # Black background

        # Draw snake body
        for row, col in self.snake[1:]:
            pygame.draw.rect(
                self.window, 
                (0, 255, 0),  # Green body
                (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
            )
        
        # Draw snake head
        head_row, head_col = self.snake[0]
        pygame.draw.rect(
            self.window, 
            (0, 100, 255),  # Blue head
            (head_col * self.cell_size, head_row * self.cell_size, self.cell_size, self.cell_size)
        )
        
        # Draw food
        food_row, food_col = self.food
        pygame.draw.rect(
            self.window, 
            (255, 0, 0),  # Red food
            (food_col * self.cell_size, food_row * self.cell_size, self.cell_size, self.cell_size)
        )
        
        # Display score
        font = pygame.font.SysFont('arial', 20)
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.window.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None



# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Parallel Parking Simulator')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Clock to control game speed
clock = pygame.time.Clock()
FPS = 60

# Car class with realistic physics
class Car:
    def __init__(self, x, y, width, height, color, is_player=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.is_player = is_player
        
        # Movement properties
        self.angle = 0  # Car's orientation in degrees
        self.speed = 0  # Current speed
        self.velocity_x = 0  # Velocity components
        self.velocity_y = 0
        
        # Physics constants
        self.max_speed_forward = 3.0
        self.max_speed_reverse = 1.5
        self.acceleration = 0.07
        self.braking = 0.15
        self.drag = 0.02      # Resistance when moving
        self.drift_factor = 0.95  # How much the car maintains direction when turning
        
        # Steering properties
        self.steering_angle = 0  # Current steering angle
        self.max_steering_angle = 30  # Maximum steering angle
        self.steering_speed = 1.0  # How quickly steering changes
        self.steering_return_speed = 1.2  # How quickly steering returns to center
        
        # Car dimensions
        self.wheel_base = self.width * 0.7  # Distance between front and rear axles
        
    def draw(self, surface):
        # Create a rotated surface for the car
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, self.color, (0, 0, self.width, self.height), border_radius=5)
        
        # Draw windows
        windshield_width = self.width * 0.15
        windshield_height = self.height * 0.6
        windshield_x = self.width * 0.2
        windshield_y = (self.height - windshield_height) / 2
        pygame.draw.rect(car_surface, (170, 220, 255), 
                         (windshield_x, windshield_y, windshield_width, windshield_height), 
                         border_radius=2)
        
        # Rear window
        pygame.draw.rect(car_surface, (170, 220, 255), 
                         (self.width - windshield_x - windshield_width, windshield_y, 
                          windshield_width, windshield_height), 
                         border_radius=2)
        
        # Wheels - show steering for front wheels
        wheel_width = self.width * 0.1
        wheel_height = self.height * 0.2
        front_wheel_y = self.height - wheel_height / 2
        
        # Front wheels with steering angle
        front_wheel_x = self.width * 0.15
        front_wheel_surface = pygame.Surface((wheel_width, wheel_height), pygame.SRCALPHA)
        pygame.draw.rect(front_wheel_surface, BLACK, (0, 0, wheel_width, wheel_height))
        
        rotated_wheel = pygame.transform.rotate(front_wheel_surface, -self.steering_angle)
        wheel_rect = rotated_wheel.get_rect(center=(front_wheel_x + wheel_width/2, front_wheel_y + wheel_height/2))
        car_surface.blit(rotated_wheel, wheel_rect.topleft)
        
        wheel_rect = rotated_wheel.get_rect(center=(self.width - front_wheel_x - wheel_width/2, front_wheel_y + wheel_height/2))
        car_surface.blit(rotated_wheel, wheel_rect.topleft)
        
        # Rear wheels
        pygame.draw.rect(car_surface, BLACK, 
                         (self.width * 0.2, front_wheel_y, wheel_width, wheel_height))
        pygame.draw.rect(car_surface, BLACK, 
                         (self.width - self.width * 0.2 - wheel_width, front_wheel_y, wheel_width, wheel_height))
        
        # Headlights and taillights
        light_width = self.width * 0.05
        light_height = self.height * 0.15
        light_y = (self.height - light_height) / 2
        
        # Front lights
        pygame.draw.rect(car_surface, YELLOW, (0, light_y, light_width, light_height))
        
        # Rear lights
        pygame.draw.rect(car_surface, RED, (self.width - light_width, light_y, light_width, light_height))
        
        # Rotate car surface
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        
        # Get new rect and position it
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the car
        surface.blit(rotated_car, car_rect.topleft)
        
        # Draw steering indicator for player car (optional)
        if self.is_player:
            direction_line_length = 40
            end_x = self.x + direction_line_length * math.cos(math.radians(self.angle))
            end_y = self.y - direction_line_length * math.sin(math.radians(self.angle))
            pygame.draw.line(surface, GREEN, (self.x, self.y), (end_x, end_y), 2)
    
    def update(self):
        if not self.is_player:
            return
        
        # Car faces in the direction of angle
        angle_rad = math.radians(self.angle)
        
        # Calculate direction vector (forward direction of the car)
        forward_x = math.cos(angle_rad)
        forward_y = -math.sin(angle_rad)  # Negative because pygame y increases downward
        
        # Apply acceleration or deceleration in the car's forward direction
        # This is the key to realistic car movement - we accelerate in the car's forward direction
        if abs(self.speed) < 0.01:
            # If almost stopped, set to zero to prevent drift
            self.speed = 0
            self.velocity_x = 0
            self.velocity_y = 0
        else:
            # Calculate new velocity based on car's forward direction
            self.velocity_x = self.speed * forward_x
            self.velocity_y = self.speed * forward_y
            
            # Apply drag to slow down
            self.speed *= (1 - self.drag)
        
        # Apply steering effect only when moving
        if abs(self.speed) > 0.1:
            # The critical part: turning radius depends on steering angle and wheelbase
            # Turn more sharply at lower speeds
            turn_rate = (self.steering_angle / 10.0) * (self.speed / self.max_speed_forward)
            
            # Reverse steering when going backward (just like a real car)
            if self.speed < 0:
                turn_rate *= -1
                
            # Update the car's angle based on steering and speed
            self.angle += turn_rate
            
            # Keep angle in reasonable range
            self.angle %= 360
        
        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Keep car on screen
        self.x = max(self.width/2, min(self.x, SCREEN_WIDTH - self.width/2))
        self.y = max(self.height/2, min(self.y, SCREEN_HEIGHT - self.height/2))
    
    def handle_keys(self):
        keys = pygame.key.get_pressed()
        
        # Steering control - gradually return to center when not steering
        if not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
            # Return steering to center gradually
            if self.steering_angle > 0:
                self.steering_angle = max(0, self.steering_angle - self.steering_return_speed)
            elif self.steering_angle < 0:
                self.steering_angle = min(0, self.steering_angle + self.steering_return_speed)
        
        # Apply steering with sensitivity
        if keys[pygame.K_LEFT]:
            self.steering_angle = max(-self.max_steering_angle, self.steering_angle - self.steering_speed)
        if keys[pygame.K_RIGHT]:
            self.steering_angle = min(self.max_steering_angle, self.steering_angle + self.steering_speed)
        
        # Acceleration control
        if keys[pygame.K_UP]:
            # Forward acceleration
            if self.speed < 0:
                # If going backward, apply stronger braking
                self.speed = min(0, self.speed + self.braking)
            else:
                # If going forward, accelerate up to max speed
                self.speed = min(self.max_speed_forward, self.speed + self.acceleration)
        elif keys[pygame.K_DOWN]:
            # Reverse/braking
            if self.speed > 0:
                # If going forward, apply stronger braking
                self.speed = max(0, self.speed - self.braking)
            else:
                # If in reverse, accelerate backward up to max reverse speed
                self.speed = max(-self.max_speed_reverse, self.speed - self.acceleration)
        else:
            # Natural slowdown when not accelerating (coast to a stop)
            if abs(self.speed) < self.drag:
                self.speed = 0
            elif self.speed > 0:
                self.speed -= self.drag
            elif self.speed < 0:
                self.speed += self.drag
        
        # Brake (space bar)
        if keys[pygame.K_SPACE]:
            # Strong braking in current direction
            if self.speed > 0:
                self.speed = max(0, self.speed - self.braking * 3)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.braking * 3)
    
    def get_rect(self):
        # Get corners of the car for collision detection
        half_width = self.width / 2
        half_height = self.height / 2
        
        # Calculate the four corners of the car based on angle
        corners = []
        for wx, wy in [(-half_width, -half_height), 
                       (half_width, -half_height), 
                       (half_width, half_height), 
                       (-half_width, half_height)]:
            # Rotate the point around the center
            angle_rad = math.radians(self.angle)
            rotated_x = wx * math.cos(angle_rad) - wy * math.sin(angle_rad)
            rotated_y = wx * math.sin(angle_rad) + wy * math.cos(angle_rad)
            
            # Translate to car's position
            corners.append((self.x + rotated_x, self.y + rotated_y))
        
        return corners

def check_collision(car1, car2):
    """Improved collision detection using Separating Axis Theorem"""
    # Get the corner points of both cars
    car1_corners = car1.get_rect()
    car2_corners = car2.get_rect()
    
    # Check for collision using the Separating Axis Theorem (SAT)
    for polygon in [car1_corners, car2_corners]:
        for i in range(len(polygon)):
            edge_start = polygon[i]
            edge_end = polygon[(i + 1) % len(polygon)]
            
            # Calculate the normal to this edge (perpendicular)
            edge_x = edge_end[0] - edge_start[0]
            edge_y = edge_end[1] - edge_start[1]
            normal_x = -edge_y
            normal_y = edge_x
            
            # Normalize the normal vector
            length = math.sqrt(normal_x**2 + normal_y**2)
            if length > 0:
                normal_x /= length
                normal_y /= length
            
            # Project both polygons onto this normal
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            
            # Project the first polygon
            for point in car1_corners:
                projection = normal_x * point[0] + normal_y * point[1]
                min1 = min(min1, projection)
                max1 = max(max1, projection)
            
            # Project the second polygon
            for point in car2_corners:
                projection = normal_x * point[0] + normal_y * point[1]
                min2 = min(min2, projection)
                max2 = max(max2, projection)
            
            # Check if projections overlap
            if max1 < min2 or max2 < min1:
                # Found a separating axis, no collision
                return False
    
    # No separating axis found, there is a collision
    return True

def draw_parking_environment(surface):
    # Draw road
    pygame.draw.rect(surface, DARK_GRAY, (0, 200, SCREEN_WIDTH, 300))
    
    # Draw sidewalk
    pygame.draw.rect(surface, GRAY, (0, 500, SCREEN_WIDTH, 100))
    
    # Draw center line
    for i in range(0, SCREEN_WIDTH, 40):
        pygame.draw.line(surface, YELLOW, (i, 350), (i + 20, 350), 4)
    
    # Draw solid side lines
    pygame.draw.line(surface, WHITE, (0, 200), (SCREEN_WIDTH, 200), 4)
    pygame.draw.line(surface, WHITE, (0, 500), (SCREEN_WIDTH, 500), 4)
    
    # Draw parking spaces
    for i in range(100, SCREEN_WIDTH - 100, 150):
        pygame.draw.line(surface, WHITE, (i, 500), (i, 470), 3)
        pygame.draw.line(surface, WHITE, (i + 100, 500), (i + 100, 470), 3)

def draw_game_over(surface, success=False):
    # Overlay for game over
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))  # Semi-transparent black
    surface.blit(overlay, (0, 0))
    
    # Game over text
    font = pygame.font.SysFont(None, 72)
    if success:
        text = font.render("PARKING SUCCESSFUL!", True, GREEN)
    else:
        text = font.render("COLLISION! GAME OVER", True, RED)
    
    text_rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
    surface.blit(text, text_rect)
    
    # Restart instructions
    font = pygame.font.SysFont(None, 36)
    restart_text = font.render("Press SPACE to restart", True, WHITE)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 30))
    surface.blit(restart_text, restart_rect)

def is_successfully_parked(player_car, parking_spot):
    # Check if the car is correctly parked in the designated spot
    car_corners = player_car.get_rect()
    
    # Calculate car's center point
    center_x = sum(corner[0] for corner in car_corners) / 4
    center_y = sum(corner[1] for corner in car_corners) / 4
    
    # Check if car is within parking spot
    in_spot = (parking_spot['x1'] <= center_x <= parking_spot['x2'] and 
               parking_spot['y1'] <= center_y <= parking_spot['y2'])
    
    # Check if car is roughly parallel to the road (angle close to 0 or 180 degrees)
    angle_mod = player_car.angle % 180
    parallel = (angle_mod < 25 or angle_mod > 155)
    
    # Speed needs to be very low (car stopped)
    stopped = abs(player_car.speed) < 0.1
    
    return in_spot and parallel and stopped

def draw_debug_info(surface, player_car):
    font = pygame.font.SysFont(None, 24)
    
    # Display speed and steering information
    speed_text = font.render(f"Speed: {player_car.speed:.2f}", True, BLACK)
    angle_text = font.render(f"Angle: {player_car.angle:.1f}°", True, BLACK)
    steering_text = font.render(f"Steering: {player_car.steering_angle:.1f}°", True, BLACK)
    
    surface.blit(speed_text, (SCREEN_WIDTH - 150, 20))
    surface.blit(angle_text, (SCREEN_WIDTH - 150, 50))
    surface.blit(steering_text, (SCREEN_WIDTH - 150, 80))

def main():
    # Game state
    game_active = True
    game_over = False
    parking_success = False
    
    # Create player car - start a bit higher up on the road
    player_car = Car(150, 300, 60, 30, BLUE, is_player=True)
    
    # Define parking spot
    parking_spot = {
        'x1': 350,
        'y1': 440,
        'x2': 500,
        'y2': 490
    }
    
    # Create parked cars
    parked_cars = [
        Car(250, 480, 80, 40, RED),
        Car(600, 480, 80, 40, GREEN)
    ]
    
    # Font for text
    font = pygame.font.SysFont(None, 36)
    
    # Game loop
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle restart on space press when game is over
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not game_active:
                    # Reset the game
                    player_car = Car(150, 300, 60, 30, BLUE, is_player=True)
                    game_active = True
                    game_over = False
                    parking_success = False
                
        # Clear screen
        screen.fill(WHITE)
        
        # Draw environment
        draw_parking_environment(screen)
        
        # Draw parking target spot
        pygame.draw.rect(screen, (0, 100, 0, 128), 
                        (parking_spot['x1'], parking_spot['y1'], 
                         parking_spot['x2'] - parking_spot['x1'], 
                         parking_spot['y2'] - parking_spot['y1']), 2)
        
        if game_active:
            # Handle player input
            player_car.handle_keys()
            
            # Update player car
            player_car.update()
            
            # Check for collisions
            for car in parked_cars:
                if check_collision(player_car, car):
                    game_active = False
                    game_over = True
                    break
                    
            # Check for successful parking
            if is_successfully_parked(player_car, parking_spot):
                game_active = False
                parking_success = True
        
        # Draw all cars
        for car in parked_cars:
            car.draw(screen)
        player_car.draw(screen)
        
        # Display debug information
        draw_debug_info(screen, player_car)
        
        # Display instructions if game is active
        if game_active:
            instructions = [
                "Arrow Keys: Drive car",
                "↑/↓: Accelerate/Reverse",
                "←/→: Steer Left/Right",
                "SPACE: Brake",
                "Park in the green outline!"
            ]
            
            for i, text in enumerate(instructions):
                text_surface = font.render(text, True, BLACK)
                screen.blit(text_surface, (20, 20 + i * 30))
        
        # Display game over or success screen
        if not game_active:
            draw_game_over(screen, parking_success)
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()