import pygame
import sys
import json
import time
import numpy as np
import math
import os
from environment import DroneGridEnvironment

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GREY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)

# Create a list of distinct colors for different drones
DRONE_COLORS = [
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 0),  # Dark Green
    (128, 128, 255),  # Light Blue
    (255, 128, 128),  # Pink
    (128, 255, 128),  # Light Green
    (255, 255, 128),  # Light Yellow
    (255, 128, 255),  # Light Magenta
    (128, 255, 255),  # Light Cyan
    (165, 42, 42),  # Brown
    (0, 128, 128),  # Teal
    (128, 0, 0),  # Maroon
]


class DroneVisualization:
    def __init__(self, grid_size=10, cell_size=50):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_width = grid_size * cell_size + 400  # Extra space for info panel
        self.screen_height = grid_size * cell_size + 100  # Extra padding

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Drone Placement Visualization")

        # Font for text
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Load drone image (we'll use a simple circle for now)
        self.drone_radius = int(self.cell_size * 0.35)

        # Track coverage state for animation
        self.coverage_grid = np.zeros((grid_size, grid_size), dtype=int)
        self.current_drone_index = -1
        self.drone_positions = []
        self.drone_sizes = []
        self.coverage_history = [0]

        # Animation state
        self.animation_state = "idle"  # "idle", "placing", "expanding"
        self.animation_timer = 0
        self.animation_speed = 1.0  # seconds between drone placements
        self.animation_duration = 1.0  # seconds for animation
        self.current_expansion_radius = 0

        # Coverage cells for each drone - for visualizing individual drone coverages
        self.drone_coverage_cells = []

    def load_results(self, filename="drone_placement_results.json"):
        """Load results from training phase."""
        try:
            with open(filename, 'r') as f:
                results = json.load(f)

            # Validate the results
            if results["grid_size"] != self.grid_size:
                print(
                    f"Warning: Grid size in results ({results['grid_size']}) doesn't match visualization ({self.grid_size})")

            # Load drone positions and sizes
            self.drone_positions = [tuple(pos) for pos in results["drone_positions"]]
            self.drone_sizes = results["drone_sizes"]

            print(f"Loaded {len(self.drone_positions)} drone positions from {filename}")
            print(f"  {self.drone_sizes.count(1)} small drones (3x3)")
            print(f"  {self.drone_sizes.count(2)} large drones (5x5)")

            # Precompute coverage cells for each drone
            self._precompute_coverage_cells()

            return True
        except Exception as e:
            print(f"Error loading results: {e}")
            return False

    def _get_coverage_cells(self, pos, size):
        """Get all cells covered by a drone at position with given size."""
        x, y = pos
        coverage_cells = []
        radius = size  # size 1 = radius 1 (3x3), size 2 = radius 2 (5x5)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    coverage_cells.append((nx, ny))

        return coverage_cells

    def _precompute_coverage_cells(self):
        """Precompute coverage cells for each drone."""
        self.drone_coverage_cells = []
        for i, pos in enumerate(self.drone_positions):
            size = self.drone_sizes[i] if i < len(self.drone_sizes) else 1
            cells = self._get_coverage_cells(pos, size)
            self.drone_coverage_cells.append(cells)

    def draw_grid(self):
        """Draw the grid with current coverage."""
        # Draw background
        self.screen.fill(WHITE)

        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Vertical lines
            pygame.draw.line(self.screen, BLACK,
                             (i * self.cell_size, 0),
                             (i * self.cell_size, self.grid_size * self.cell_size), 1)
            # Horizontal lines
            pygame.draw.line(self.screen, BLACK,
                             (0, i * self.cell_size),
                             (self.grid_size * self.cell_size, i * self.cell_size), 1)

        # Draw coverage areas with transparency
        for i in range(self.current_drone_index + 1):
            if i >= len(self.drone_positions) or i >= len(self.drone_coverage_cells):
                continue

            pos = self.drone_positions[i]
            size = self.drone_sizes[i] if i < len(self.drone_sizes) else 1
            coverage_cells = self.drone_coverage_cells[i]
            color = DRONE_COLORS[i % len(DRONE_COLORS)]

            # Create a transparent surface for the coverage area
            coverage_color = (*color, 100)  # Add alpha for transparency

            # If we're in the expanding animation for the current drone
            if i == self.current_drone_index and self.animation_state == "expanding":
                # Only draw cells within the current expansion radius
                x, y = pos
                for cx, cy in coverage_cells:
                    dx, dy = cx - x, cy - y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= self.current_expansion_radius:
                        rect = pygame.Rect(cx * self.cell_size, cy * self.cell_size,
                                           self.cell_size, self.cell_size)
                        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        s.fill(coverage_color)
                        self.screen.blit(s, rect)
            else:
                # Draw all coverage cells for this drone
                for cx, cy in coverage_cells:
                    rect = pygame.Rect(cx * self.cell_size, cy * self.cell_size,
                                       self.cell_size, self.cell_size)
                    s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    s.fill(coverage_color)
                    self.screen.blit(s, rect)

        # Draw drone positions (already placed)
        for i in range(self.current_drone_index + 1):
            if i >= len(self.drone_positions):
                continue

            x, y = self.drone_positions[i]
            size = self.drone_sizes[i] if i < len(self.drone_sizes) else 1
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2

            # Use a unique color for each drone
            color = DRONE_COLORS[i % len(DRONE_COLORS)]

            # Draw drone - size affects the drone visual size
            drone_size = self.drone_radius * (1.0 + 0.5 * (size - 1))
            pygame.draw.circle(self.screen, color, (center_x, center_y), drone_size)

            # Draw drone index
            text = self.font.render(str(i + 1), True, WHITE)
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)

    def draw_info_panel(self):
        """Draw panel with information about the current state."""
        panel_x = self.grid_size * self.cell_size + 20
        panel_y = 20
        panel_width = 360
        panel_height = self.grid_size * self.cell_size

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, GREY, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)

        # Title
        title = self.title_font.render("Drone Placement Results", True, BLACK)
        self.screen.blit(title, (panel_x + 20, panel_y + 20))

        # Count drones by size
        small_drones = sum(1 for i in range(self.current_drone_index + 1)
                           if i < len(self.drone_sizes) and self.drone_sizes[i] == 1)
        large_drones = sum(1 for i in range(self.current_drone_index + 1)
                           if i < len(self.drone_sizes) and self.drone_sizes[i] == 2)

        # Information items
        info_items = [
            f"Grid Size: {self.grid_size}x{self.grid_size}",
            f"Drones Placed: {self.current_drone_index + 1}/{len(self.drone_positions)}",
            f"  Small drones (3x3): {small_drones}",
            f"  Large drones (5x5): {large_drones}",
            f"Current Coverage: {np.sum(self.coverage_grid)}/{self.grid_size * self.grid_size} cells",
            f"Coverage Percentage: {100 * np.sum(self.coverage_grid) / (self.grid_size * self.grid_size):.2f}%"
        ]

        for i, item in enumerate(info_items):
            text = self.font.render(item, True, BLACK)
            self.screen.blit(text, (panel_x + 20, panel_y + 70 + i * 25))

        # Coverage history graph (if we have multiple drones placed)
        if self.current_drone_index >= 0:
            graph_y = panel_y + 240
            graph_height = 150
            graph_width = panel_width - 40

            # Draw graph background
            graph_rect = pygame.Rect(panel_x + 20, graph_y, graph_width, graph_height)
            pygame.draw.rect(self.screen, WHITE, graph_rect)
            pygame.draw.rect(self.screen, BLACK, graph_rect, 1)

            # Draw graph title
            graph_title = self.font.render("Coverage Progress", True, BLACK)
            self.screen.blit(graph_title, (panel_x + 20, graph_y - 25))

            # Calculate coverage history based on current drone index
            if len(self.coverage_history) > self.current_drone_index + 2:
                coverage_data = self.coverage_history[:self.current_drone_index + 2]
            else:
                # If we don't have enough historical data, just use what we have
                coverage_data = self.coverage_history

            # Draw coverage history line
            if len(coverage_data) > 1:
                max_coverage = self.grid_size * self.grid_size
                x_step = graph_width / max(len(coverage_data) - 1, 1)

                points = []
                for i, coverage in enumerate(coverage_data):
                    x = panel_x + 20 + i * x_step
                    y = graph_y + graph_height - (coverage / max_coverage) * graph_height
                    points.append((x, y))

                if len(points) > 1:
                    pygame.draw.lines(self.screen, RED, False, points, 2)

                # Add points for each drone
                for i, point in enumerate(points):
                    if i > 0 and i - 1 < len(self.drone_sizes):
                        # Use color based on drone size
                        color = BLUE if self.drone_sizes[i - 1] == 1 else RED
                        size = 4 if self.drone_sizes[i - 1] == 1 else 6
                    else:
                        color = BLUE
                        size = 5
                    pygame.draw.circle(self.screen, color, point, size)

            # Draw legend
            legend_y = graph_y + graph_height + 10
            # Small drone legend
            pygame.draw.circle(self.screen, BLUE, (panel_x + 30, legend_y), 4)
            legend_text = self.font.render("Small drone (3x3)", True, BLACK)
            self.screen.blit(legend_text, (panel_x + 40, legend_y - 8))
            # Large drone legend
            pygame.draw.circle(self.screen, RED, (panel_x + 180, legend_y), 6)
            legend_text = self.font.render("Large drone (5x5)", True, BLACK)
            self.screen.blit(legend_text, (panel_x + 190, legend_y - 8))

            # Draw instructions
            instructions = [
                "Press Space to start/pause animation",
                "Press R to reset animation",
                "Press + to speed up, - to slow down",
                "Press Esc to exit"
            ]

            for i, instruction in enumerate(instructions):
                text = self.font.render(instruction, True, BLACK)
                self.screen.blit(text, (panel_x + 20, panel_y + panel_height - 120 + i * 25))

    def place_next_drone(self):
        """Place the next drone in the sequence and start its animation."""
        if self.current_drone_index + 1 < len(self.drone_positions):
            self.current_drone_index += 1
            pos = self.drone_positions[self.current_drone_index]
            size = self.drone_sizes[self.current_drone_index] if self.current_drone_index < len(self.drone_sizes) else 1

            # Update the coverage grid (create a new grid to track current coverage)
            new_coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

            # Apply coverage from all drones placed so far
            for i in range(self.current_drone_index + 1):
                if i < len(self.drone_coverage_cells):
                    cells = self.drone_coverage_cells[i]
                    for cx, cy in cells:
                        new_coverage_grid[cx, cy] = 1

            self.coverage_grid = new_coverage_grid

            # Update coverage history
            covered_cells = np.sum(self.coverage_grid)
            if len(self.coverage_history) <= self.current_drone_index + 1:
                self.coverage_history.append(covered_cells)

            # Start expansion animation
            self.animation_state = "placing"
            self.animation_timer = 0

            return True
        return False

    def update_animation(self, dt):
        """Update the animation state."""
        if self.animation_state == "idle":
            return

        # Update animation timer
        self.animation_timer += dt

        if self.animation_state == "placing":
            # Drone placement animation
            if self.animation_timer >= self.animation_duration * 0.3:
                # Transition to expansion animation
                self.animation_state = "expanding"
                self.animation_timer = 0
                self.current_expansion_radius = 0

        elif self.animation_state == "expanding":
            # Coverage expansion animation
            size = self.drone_sizes[self.current_drone_index] if self.current_drone_index < len(self.drone_sizes) else 1
            progress = min(1.0, self.animation_timer / self.animation_duration)
            self.current_expansion_radius = progress * (size + 0.5)

            if self.animation_timer >= self.animation_duration:
                # Expansion complete
                self.animation_state = "idle"

    def reset_animation(self):
        """Reset the animation to the beginning."""
        self.current_drone_index = -1
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.coverage_history = [0]
        self.animation_state = "idle"

    def run_animation(self):
        """Run the animation loop."""
        if not self.drone_positions:
            print("No drone positions loaded. Run training first or specify a results file.")
            return

        clock = pygame.time.Clock()
        running = True
        auto_advance = False
        last_advance_time = 0

        while running:
            current_time = time.time()
            dt = clock.tick(60) / 1000.0  # Time since last frame in seconds

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Start/pause auto-advance
                        auto_advance = not auto_advance
                        last_advance_time = current_time
                    elif event.key == pygame.K_r:
                        # Reset animation
                        self.reset_animation()
                        auto_advance = False
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        # Speed up
                        self.animation_speed = max(0.2, self.animation_speed - 0.2)
                        self.animation_duration = max(0.2, self.animation_duration - 0.2)
                    elif event.key in [pygame.K_MINUS, pygame.K_UNDERSCORE]:
                        # Slow down
                        self.animation_speed = min(3.0, self.animation_speed + 0.2)
                        self.animation_duration = min(3.0, self.animation_duration + 0.2)

            # Update animation
            self.update_animation(dt)

            # Auto-advance if enabled and current animation is complete
            if auto_advance and self.animation_state == "idle" and current_time - last_advance_time > self.animation_speed:
                advanced = self.place_next_drone()
                last_advance_time = current_time
                # If we've reached the end, stop auto-advance
                if not advanced:
                    auto_advance = False

            # Draw everything
            self.draw_grid()
            self.draw_info_panel()

            # Update display
            pygame.display.flip()

        pygame.quit()


def run_visualization(results_file="drone_placement_results.json", grid_size=None):
    """Run the visualization with the given parameters."""

    # First try to load parameters from results file
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)

        if grid_size is None:
            grid_size = results["grid_size"]
    except Exception as e:
        print(f"Error loading parameters from results file: {e}")
        if grid_size is None:
            grid_size = 10  # Default

    # Initialize visualization
    viz = DroneVisualization(grid_size=grid_size, cell_size=50)

    # Load results
    if viz.load_results(results_file):
        # Run animation
        viz.run_animation()
    else:
        print("Failed to load results. Please run training first.")


if __name__ == "__main__":
    # Check if a results file is specified
    results_file = "drone_placement_results.json"

    if len(sys.argv) > 1:
        results_file = sys.argv[1]

    run_visualization(results_file)