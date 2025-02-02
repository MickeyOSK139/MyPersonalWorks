import pygame
import random
import os

os.chdir("ML_projects")

# Initialize Pygame modules
pygame.init()

# Constants for screen dimensions and game settings
WIDTH, HEIGHT = 400, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird Clone with Scoring")

FPS = 60              # Frames per second for smooth animation
GRAVITY = 0.5         # Acceleration due to gravity (pixels per frame squared)
JUMP_VELOCITY = -10   # Upward velocity applied when the bird jumps
PIPE_VELOCITY = -4    # Horizontal velocity of pipes (moving leftwards)
PIPE_GAP = 150        # Vertical gap between the upper and lower pipes
PIPE_FREQUENCY = 1500 # Frequency (in milliseconds) at which new pipes spawn

# Initialize a font for rendering the score on screen
FONT = pygame.font.SysFont("comicsans", 40)

# # Load the actual bird sprite from disk.
# try:
#     bird_img = pygame.image.load("birb.png").convert_alpha()
# except pygame.error as e:
#     print("Error loading bird.png:", e)
#     raise SystemExit

# # Optionally, scale the sprite to the desired dimensions.
# desired_width, desired_height = 34, 24  # Adjust as needed for your game
# bird_img = pygame.transform.scale(bird_img, (desired_width, desired_height))

pipe_img = pygame.Surface((52, 400))
pipe_img.fill((0, 255, 0))    # Green pipe representation

# Create a clock object to regulate FPS
clock = pygame.time.Clock()

# Define the Bird class, which encapsulates the player-controlled sprite and its physics.
class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        # Load the bird sprite image from file and convert for transparency.
        try:
            original_image = pygame.image.load("birb.png").convert_alpha()
        except pygame.error as e:
            print("Error loading bird.png:", e)
            raise SystemExit
        
        # Optionally, scale the image to the desired dimensions.
        desired_width, desired_height = 34, 24  # Adjust as needed
        self.original_image = pygame.transform.scale(original_image, (desired_width, desired_height))
        self.image = self.original_image
        
        # Set the initial position and get the rect for collision/positioning.
        self.rect = self.image.get_rect(center=(50, HEIGHT // 2))
        self.velocity = 0
        
        # Define rotation parameters.
        # scale_factor determines how much the bird tilts per unit of velocity.
        self.rotation_scale = 3  # Adjust to taste.
        # Define maximum upward and downward tilt angles.
        self.max_up_tilt = 30    # Maximum tilt upward (in degrees).
        self.max_down_tilt = -60  # Maximum tilt downward (in degrees).

    def update(self):
        # Apply gravity to the velocity and update the vertical position.
        self.velocity += GRAVITY
        self.rect.y += self.velocity

        # Constrain the bird within vertical screen bounds.
        if self.rect.top < 0:
            self.rect.top = 0
            self.velocity = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            self.velocity = 0

        # --- Rotation Based on Y Velocity ---
        # Calculate the rotation angle from velocity.
        # A positive y-velocity (falling) should tilt the bird downward.
        # We use a negative multiplier so that falling (velocity > 0) results in a negative angle.
        angle = -self.velocity * self.rotation_scale
        
        # Clamp the angle within the specified range.
        # When rising (velocity negative), angle will be positive (upward tilt)
        # When falling (velocity positive), angle will be negative (downward tilt)
        angle = max(min(angle, self.max_up_tilt), self.max_down_tilt)
        
        # Rotate the original image by the computed angle.
        # Using the original image prevents degradation from repeated rotations.
        rotated_image = pygame.transform.rotate(self.original_image, angle)
        
        # Update the sprite image and re-center the rectangle.
        # It is essential to re-center the rotated image, since rotating changes its size.
        old_center = self.rect.center
        self.image = rotated_image
        self.rect = self.image.get_rect(center=old_center)
        # --- End Rotation ---

    def jump(self):
        # Impart an upward impulse to the bird.
        self.velocity = JUMP_VELOCITY


# Define the Pipe class for obstacle representation. Each pipe sprite can be either a top (flipped)
# or bottom pipe. We add a scoring attribute to the bottom pipes to ensure that each pair is scored once.
class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, flipped=False):
        super().__init__()
        self.image = pipe_img
        if flipped:
            # For the top pipe, flip the image vertically.
            self.image = pygame.transform.flip(pipe_img, False, True)
            # Position the bottom of the top pipe at y minus half the gap.
            self.rect = self.image.get_rect(midbottom=(x, y - PIPE_GAP // 2))
        else:
            # For the bottom pipe, position the top at y plus half the gap.
            self.rect = self.image.get_rect(midtop=(x, y + PIPE_GAP // 2))
            self.scored = False  # Indicates whether this pipe has contributed to the score.
        self.flipped = flipped

    def update(self):
        # Move the pipe leftwards across the screen.
        self.rect.x += PIPE_VELOCITY
        # Remove the pipe from all groups if it moves off-screen to the left.
        if self.rect.right < 0:
            self.kill()

# Function to render all game elements to the window, including the dynamically updated score.
def draw_window(bird, pipes, score):
    WIN.fill((135, 206, 250))  # Fill background with a sky-blue color.
    pipes.draw(WIN)            # Draw all pipe sprites.
    WIN.blit(bird.image, bird.rect)  # Draw the bird sprite.

    # Render the score text and display it at the upper-right corner of the screen.
    score_text = FONT.render(f"Score: {score}", True, (255, 255, 255))
    WIN.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))

    pygame.display.update()    # Update the full display surface to the screen.

# The main game loop encapsulated in the main() function.
def main():
    run = True
    score = 0  # Initialize the player's score.
    bird = Bird()
    # Using GroupSingle for a single sprite (the bird) ensures efficient updates and rendering.
    bird_group = pygame.sprite.GroupSingle(bird)
    pipe_group = pygame.sprite.Group()

    # Define a custom user event to spawn pipes at regular intervals.
    SPAWNPIPE = pygame.USEREVENT
    pygame.time.set_timer(SPAWNPIPE, PIPE_FREQUENCY)

    while run:
        clock.tick(FPS)  # Maintain a consistent frame rate.
        for event in pygame.event.get():
            # Allow graceful exit when the user closes the window.
            if event.type == pygame.QUIT:
                run = False

            # Process key events: the space bar triggers the bird's jump.
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()

            # Spawn new pipes when the custom timer event fires.
            if event.type == SPAWNPIPE:
                # Randomly select a vertical position for the pipe pair.
                pipe_height = random.randint(HEIGHT//2 - 200, HEIGHT//2 + 200)
                # Create both a top (flipped) and a bottom pipe.
                top_pipe = Pipe(WIDTH, pipe_height, flipped=True)
                bottom_pipe = Pipe(WIDTH, pipe_height, flipped=False)
                pipe_group.add(top_pipe)
                pipe_group.add(bottom_pipe)

        # Update sprite groups: this includes physics updates for the bird and positional updates for the pipes.
        bird_group.update()
        pipe_group.update()

        # --- Scoring Mechanism Implementation ---
        # For each bottom pipe (non-flipped) that has not yet been scored, check if it has been passed by the bird.
        # The bird is considered to have passed the pipe if its left edge is greater than the right edge of the pipe.
        for pipe in pipe_group:
            if not pipe.flipped and not pipe.scored and pipe.rect.right < bird.rect.left:
                score += 1       # Increment the score.
                pipe.scored = True  # Mark this pipe as having been scored to prevent duplicate scoring.
        # --- End of Scoring Mechanism ---

        # Collision detection: if the bird collides with any pipe, the game ends.
        if pygame.sprite.spritecollide(bird, pipe_group, False):
            print(f"Game Over! Final Score: {score}")
            run = False

        # Render all game elements along with the updated score.
        draw_window(bird, pipe_group, score)

    pygame.quit()

# Entry point of the program.
if __name__ == "__main__":
    main()


