#!/usr/bin/env python
"""
testing_3.py

A comprehensive script for visualizing drone coverage and performing an
emergency pathfinding demonstration within a grid environment. This file
utilizes PyGame for a graphical animation showcasing how multiple drones
scan a grid, gather local obstacle information, stitch this data into a
global map, and finally run an A* pathfinding routine to locate an emergency
location from the grid's origin (0, 0).

Main Features:
--------------
1. **Random Environment Generation**: Creates a 2D grid of size `grid_size × grid_size`
   with randomly distributed obstacles based on `obstacle_prob`. Cells can be:
   - SAFE (1)
   - OBSTACLE (0)
   - UNEXPLORED (-1)

2. **Drone Simulation**:
   - Multiple drones (each has a center `(x, y)` and a scanning size).
   - Each drone "scans" a local area, revealing SAFE vs. OBSTACLE states.
   - Drone data merges into a global map, marking cells as SAFE or OBSTACLE
     if not already known.

3. **Animated Visualization**:
   - Iteratively reveals drone coverage areas with semi-transparent highlights.
   - Provides an info panel to display coverage statistics, exploration progress,
     and the discovered path to an emergency location.

4. **Emergency Pathfinding**:
   - Identifies or randomly selects an "emergency location" from the known SAFE cells.
   - Runs an A* algorithm from start (0, 0) to this location, if possible.
   - Visualizes the resulting path on the grid with colored cells.

5. **Controls**:
   - Spacebar: Toggle play/pause of drone coverage animation.
   - 'R': Reset the entire simulation's coverage data and pathfinding state.
   - 'P': Attempt pathfinding to an emergency location if enough coverage is available.
   - '+' / '-': Adjust the animation speed.
   - 'Esc': Quit the visualization.

Usage:
------
1. Generate or obtain a JSON file containing drone positions and sizes (e.g.,
   "drone_coverage_results_20.json").
2. Run: `python testing_3.py <grid_size>`
   - e.g. `python testing_3.py 20`
3. The script attempts to load "drone_coverage_results_<grid_size>.json". If found,
   it animates coverage and pathfinding with the specified drones.

Dependencies:
-------------
- Python 3.x
- pygame (for graphical display)
- numpy (for array operations)
- json, random, time, sys, os (standard library modules)
"""

import pygame
import sys
import json
import numpy as np
import os
import random
import time

# --------------------------------
# Constants
# --------------------------------

# Global cell states
UNEXPLORED = -1  # Cell not yet explored by any drone
OBSTACLE = 0     # Cell is blocked/obstacle
SAFE = 1         # Cell is passable / safe

# Some basic RGB color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Drone highlight colors (cyclic use for multiple drones)
DRONE_COLORS = [
    (255, 0, 0),      # Red
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 0),      # Dark Green
    (128, 128, 255),  # Light Blue
]


class DronePathfinder:
    """
    DronePathfinder
    ---------------
    Responsible for:
    1) Generating or loading a 2D environment with obstacles.
    2) Managing a global occupancy grid that merges each drone's local scan data.
    3) Visualizing the grid, drones, and pathfinding in PyGame.
    4) Computing a path from (0,0) to an emergency location using A*.

    Attributes:
    -----------
    grid_size : int
        The dimension of the square grid, e.g. 20 => 20x20.
    cell_size : int
        Pixel size of each cell for the PyGame display.
    screen_w : int
        Width of the PyGame screen (includes extra panel).
    screen_h : int
        Height of the PyGame screen.
    screen : Surface
        PyGame surface for rendering.
    actual_env : ndarray
        The ground-truth environment with SAFE/OBSTACLE states.
    global_grid : ndarray
        Global map representation, starts as UNEXPLORED and gets updated by drone scans.
    drone_positions : list of (int, int)
        Centers for each drone used to animate coverage.
    drone_sizes : list of int
        Square coverage radius for each drone's scan area.
    drone_coverage_cells : list of list
        Precomputed coverage cells for each drone, if available.
    animation_speed : float
        Time between placing subsequent drones in the animation.
    path : list of (int, int) or None
        The final path from start to emergency location (if found).
    path_found : bool
        Whether a path has been found in the environment.

    Usage:
    ------
    Typically, you instantiate this class with a certain grid size, optionally
    load drone positions from a JSON file, then call `.run()` to begin the
    coverage + pathfinding visualization.
    """

    def __init__(self, grid_size=20, cell_size=30):
        """
        Initialize DronePathfinder with the specified grid size (for environment)
        and cell size (for rendering).

        Parameters:
        -----------
        grid_size : int, optional
            The dimension of the environment grid. Default is 20.
        cell_size : int, optional
            The pixel size for each grid cell in PyGame. Default is 30.
        """
        # Initialize PyGame
        pygame.init()

        # Basic config
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_w = self.grid_size * self.cell_size + 400
        self.screen_h = self.grid_size * self.cell_size + 100
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Drone Coverage with Emergency Pathfinding")

        # Fonts
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.clock = pygame.time.Clock()

        # Generate random environment as ground-truth
        self.actual_env = self.generate_environment(self.grid_size)

        # The global map initially is all UNEXPLORED
        self.global_grid = np.full((self.grid_size, self.grid_size), UNEXPLORED, dtype=np.int32)
        self.global_grid[0, 0] = SAFE  # We assume the start location is known to be safe

        # Drone / animation states
        self.drone_positions = []
        self.drone_sizes = []
        self.drone_coverage_cells = []
        self.current_drone = -1       # Index of the drone we are currently animating
        self.animation_speed = 0.5
        self.animation_duration = 0.4
        self.animation_active = False
        self.expanding = False
        self.expansion_timer = 0.0
        self.current_expanded = set()  # Cells newly revealed in the current step
        self.coverage_history = [0]

        # Pathfinding states
        self.emergency_location = None
        self.path = None
        self.path_found = False

    def generate_environment(self, grid_size, obstacle_prob=0.2):
        """
        Generate a random environment, marking cells as SAFE or OBSTACLE.

        Parameters:
        -----------
        grid_size : int
            Size of the NxN environment.
        obstacle_prob : float
            Probability that any cell is an obstacle (0.2 default).

        Returns:
        --------
        env : np.ndarray of shape (grid_size, grid_size)
            The environment matrix with SAFE=1 or OBSTACLE=0.
        """
        env = np.random.choice(
            [OBSTACLE, SAFE],
            size=(grid_size, grid_size),
            p=[obstacle_prob, 1 - obstacle_prob]
        ).astype(np.int32)
        # Ensure (0,0) is always safe
        env[0, 0] = SAFE
        return env

    def drone_scan(self, position, size):
        """
        Simulate a drone scanning the environment. The drone has a center (position)
        and scans a square of side 'size'.

        Parameters:
        -----------
        position : (int, int)
            The center (x, y) of the drone's coverage.
        size : int
            Side length of the coverage square.

        Returns:
        --------
        local_info : np.ndarray
            A 'size x size' matrix with SAFE / OBSTACLE / UNEXPLORED representing
            what the drone sees in that sub-area.
        top_left : (int, int)
            The top-left coordinate of the sub-area in the environment.
        """
        x, y = position
        half_size = (size - 1) // 2
        top_left = (max(0, x - half_size), max(0, y - half_size))

        # local_info is the slice of the actual environment
        local_info = np.full((size, size), UNEXPLORED)
        for dx in range(size):
            for dy in range(size):
                nx, ny = top_left[0] + dx, top_left[1] + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    local_info[dx, dy] = self.actual_env[nx, ny]
        return local_info, top_left

    def stitch_information(self, local_info, top_left):
        """
        Merge the local scan results from one drone into the global_grid.

        Parameters:
        -----------
        local_info : np.ndarray
            Local map of shape (size, size) from a single drone scan.
        top_left : (int, int)
            The top-left coordinate where local_info should be placed in the global grid.
        """
        x_offset, y_offset = top_left
        for i in range(local_info.shape[0]):
            for j in range(local_info.shape[1]):
                gx, gy = x_offset + i, y_offset + j
                # Check boundary
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    if self.global_grid[gx, gy] == UNEXPLORED:
                        self.global_grid[gx, gy] = local_info[i, j]
                    else:
                        # If there's a mismatch, prefer SAFE over OBSTACLE
                        if local_info[i, j] == SAFE:
                            self.global_grid[gx, gy] = SAFE

    def find_emergency_location(self):
        """
        Randomly select a safe cell as the 'emergency location' that is a bit away
        from the start. Weighted towards farthest cells from (0,0).

        Returns:
        --------
        (x, y) : The chosen safe cell, or None if none was found.
        """
        safe_cells = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.global_grid[x, y] == SAFE:
                    dist = abs(x) + abs(y)  # Manhattan distance from (0,0)
                    if dist > 5:
                        safe_cells.append((x, y, dist))

        if not safe_cells:
            print("No suitable emergency location found!")
            return None

        # Sort by distance descending
        safe_cells.sort(key=lambda c: -c[2])
        # Take top 5 or so, pick randomly among them
        top_candidates = safe_cells[:5] if len(safe_cells) > 5 else safe_cells
        chosen = random.choice(top_candidates)
        return (chosen[0], chosen[1])

    def find_path_with_astar(self, start, goal):
        """
        Compute a path from 'start' to 'goal' using the A* algorithm on the global grid.

        Parameters:
        -----------
        start : (int, int)
            The start cell.
        goal : (int, int)
            The target cell.

        Returns:
        --------
        path : list of (int, int)
            The sequence of grid cells from start to goal. None if no path found.
        """
        if not goal:
            print("No emergency location set for A*!")
            return None

        print(f"Finding path from {start} to {goal} using A*...")
        import heapq

        open_set = []
        closed_set = set()

        # g_score = cost so far from start
        # f_score = g_score + heuristic
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        heapq.heappush(open_set, (f_score[start], start))

        came_from = {}

        # Movement directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            closed_set.add(current)

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                # Validate
                if (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    if self.global_grid[neighbor] == SAFE and neighbor not in closed_set:
                        tentative_g = g_score[current] + 1
                        if (neighbor not in g_score) or (tentative_g < g_score[neighbor]):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                            # If neighbor not in open_set, push it
                            if neighbor not in [pos for _, pos in open_set]:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        # No path found
        return None

    def heuristic(self, a, b):
        """Manhattan distance heuristic for A* pathfinding."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path_to_emergency(self):
        """
        Compute a path from (0, 0) to a chosen emergency location.
        If no emergency location is set, pick one from safe cells.
        """
        if not self.emergency_location:
            self.emergency_location = self.find_emergency_location()
            if not self.emergency_location:
                print("No valid emergency location was chosen!")
                return
        start = (0, 0)
        path = self.find_path_with_astar(start, self.emergency_location)
        if path:
            self.path = path
            self.path_found = True
            print(f"Path found with length {len(path)} cells.")
        else:
            print("No path found to the emergency location.")
            self.path_found = False

    def place_next_drone(self):
        """
        Transition to the next drone in the list of drone_positions, if available.
        Update the global grid with that drone's coverage.

        Returns:
        --------
        bool
            True if a new drone was placed, False if no more drones are left.
        """
        if self.current_drone + 1 < len(self.drone_positions):
            self.current_drone += 1
            self.expanding = True
            self.expansion_timer = 0.0
            self.current_expanded.clear()

            position = self.drone_positions[self.current_drone]
            size = self.drone_sizes[self.current_drone]

            local_info, top_left = self.drone_scan(position, size)
            self.stitch_information(local_info, top_left)
            return True
        return False

    def update_expansion(self, dt):
        """
        Handle the "expanding" animation for a newly placed drone.

        Parameters:
        -----------
        dt : float
            The time delta in seconds since the last frame.
        """
        if not self.expanding:
            return

        self.expansion_timer += dt
        progress = min(1.0, self.expansion_timer / self.animation_duration)

        # If the animation is complete
        if progress >= 1.0:
            self.expanding = False
            coverage = np.sum(self.global_grid != UNEXPLORED)
            self.coverage_history.append(coverage)

            # If we haven't set an emergency location, do so when sufficiently explored
            if not self.emergency_location:
                unknown_frac = np.sum(self.global_grid == UNEXPLORED) / (self.grid_size * self.grid_size)
                if unknown_frac < 0.3:
                    self.emergency_location = self.find_emergency_location()
                    if self.emergency_location:
                        print(f"Emergency location chosen at {self.emergency_location}.")

    def reset_animation(self):
        """
        Completely reset the animation state, clearing the global map,
        coverage history, and pathfinding data.
        """
        self.global_grid = np.full((self.grid_size, self.grid_size), UNEXPLORED, dtype=np.int32)
        self.global_grid[0, 0] = SAFE
        self.coverage_history = [0]
        self.current_drone = -1
        self.animation_active = False
        self.expanding = False
        self.expansion_timer = 0.0
        self.current_expanded.clear()

        # Reset pathfinding
        self.emergency_location = None
        self.path = None
        self.path_found = False

    def draw_scene(self):
        """
        Draw the environment grid, the coverage map, the path, and the drones.
        """
        self.screen.fill(WHITE)

        # 1) Draw each cell in global_grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                cell_val = self.global_grid[x, y]
                if cell_val == UNEXPLORED:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)  # dark gray
                elif cell_val == OBSTACLE:
                    pygame.draw.rect(self.screen, (180, 0, 0), rect)  # red
                else:
                    pygame.draw.rect(self.screen, (200, 200, 255), rect)  # light blue

                pygame.draw.rect(self.screen, BLACK, rect, 1)  # Grid line

        # 2) Draw the path (if found)
        if self.path:
            for i, (px, py) in enumerate(self.path):
                if i == 0 or i == len(self.path) - 1:
                    continue
                path_rect = pygame.Rect(py * self.cell_size, px * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, YELLOW, path_rect)
                if len(self.path) > 10 and i % 5 == 0:
                    text = self.font.render(str(i), True, BLACK)
                    text_rect = text.get_rect(
                        center=(py * self.cell_size + self.cell_size // 2, px * self.cell_size + self.cell_size // 2)
                    )
                    self.screen.blit(text, text_rect)

        # 3) Highlight start (0,0)
        start_rect = pygame.Rect(0, 0, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, GREEN, start_rect, 4)

        # 4) Highlight emergency location if set
        if self.emergency_location:
            ex, ey = self.emergency_location
            em_rect = pygame.Rect(ey * self.cell_size, ex * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, RED, em_rect, 4)

        # 5) Draw the drones and their coverage bounding boxes
        for i in range(self.current_drone + 1):
            if i >= len(self.drone_positions):
                break
            dx, dy = self.drone_positions[i]
            size = self.drone_sizes[i]
            color = DRONE_COLORS[i % len(DRONE_COLORS)]
            half = (size - 1) // 2

            left = max(0, dy - half)
            top = max(0, dx - half)
            width = min(size, self.grid_size - left)
            height = min(size, self.grid_size - top)

            drone_rect = pygame.Rect(left * self.cell_size, top * self.cell_size, width * self.cell_size, height * self.cell_size)
            drone_surf = pygame.Surface((width * self.cell_size, height * self.cell_size), pygame.SRCALPHA)
            drone_surf.fill((color[0], color[1], color[2], 80))
            self.screen.blit(drone_surf, (drone_rect.x, drone_rect.y))

            # Drone center
            center_x = dy * self.cell_size + self.cell_size // 2
            center_y = dx * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), self.cell_size // 4)

            # Drone label
            label = self.font.render(str(i + 1), True, WHITE)
            label_rect = label.get_rect(center=(center_x, center_y))
            self.screen.blit(label, label_rect)

    def draw_info_panel(self):
        """
        Draw the side panel with coverage stats, path info, instructions, etc.
        """
        px = self.grid_size * self.cell_size + 10
        py = 10
        pw = 380
        ph = self.grid_size * self.cell_size

        pygame.draw.rect(self.screen, GREY, (px, py, pw, ph))
        pygame.draw.rect(self.screen, BLACK, (px, py, pw, ph), 2)

        # Title
        title = self.title_font.render("Drone Coverage & Pathfinding", True, BLACK)
        self.screen.blit(title, (px + 10, py + 10))

        total_cells = self.grid_size * self.grid_size
        explored = np.sum(self.global_grid != UNEXPLORED)
        safe_cells = np.sum(self.global_grid == SAFE)
        obstacle_cells = np.sum(self.global_grid == OBSTACLE)
        unexplored = total_cells - explored

        lines = [
            f"Grid Size: {self.grid_size}x{self.grid_size}",
            f"Total Cells: {total_cells}",
            f"Explored: {explored} ({explored * 100 / total_cells:.1f}%)",
            f"Safe Path Cells: {safe_cells}",
            f"Obstacle Cells: {obstacle_cells}",
            f"Unexplored: {unexplored}",
            f"Drones: {self.current_drone + 1}/{len(self.drone_positions)}"
        ]

        # If we have an emergency location
        if self.emergency_location:
            lines.append("")
            lines.append(f"Emergency at: {self.emergency_location}")

            if self.path:
                lines.append(f"Path Found: {len(self.path)} steps")

        y_offset = 60
        for line in lines:
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (px + 10, py + y_offset))
            y_offset += 25

        # Coverage chart
        chart_x = px + 20
        chart_y = py + 270
        chart_w = pw - 40
        chart_h = 150

        pygame.draw.rect(self.screen, WHITE, (chart_x, chart_y, chart_w, chart_h))
        pygame.draw.rect(self.screen, BLACK, (chart_x, chart_y, chart_w, chart_h), 1)

        chart_title = self.font.render("Coverage Progress", True, BLACK)
        self.screen.blit(chart_title, (chart_x, chart_y - 25))

        # Plot coverage_history
        if len(self.coverage_history) > 1:
            max_cov = total_cells
            points = []
            for i, cov_val in enumerate(self.coverage_history):
                pxp = chart_x + (i * chart_w / (len(self.coverage_history) - 1))
                pyp = chart_y + chart_h - (cov_val * chart_h / max_cov)
                points.append((pxp, pyp))

            # Connect points
            if len(points) > 1:
                pygame.draw.lines(self.screen, RED, False, points, 2)

            # Draw points
            for point in points:
                pygame.draw.circle(self.screen, BLUE, (int(point[0]), int(point[1])), 3)

        # Instructions
        instructions = [
            "Space: Play/pause animation",
            "R: Reset animation",
            "P: Find path to emergency",
            "+/-: Speed up/slow down",
            "Esc: Exit"
        ]
        y_offset = py + ph - 140
        for instr in instructions:
            t = self.font.render(instr, True, BLACK)
            self.screen.blit(t, (px + 10, y_offset))
            y_offset += 22

    def run(self):
        """
        The main event loop:
          - Handles user inputs (keys for controlling the animation/pathfinding).
          - Updates expansions for drone coverage or resets as needed.
          - Once all drones are placed, attempts pathfinding if not already done.
        """
        running = True
        time_acc = 0.0

        while running:
            dt = self.clock.tick(30) / 1000.0

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.animation_active = not self.animation_active
                    elif event.key == pygame.K_r:
                        self.reset_animation()
                    elif event.key == pygame.K_p:
                        self.find_path_to_emergency()
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        self.animation_speed = max(0.05, self.animation_speed - 0.1)
                    elif event.key in [pygame.K_MINUS, pygame.K_UNDERSCORE]:
                        self.animation_speed = min(2.0, self.animation_speed + 0.1)

            # If no drones are placed yet, place the first on idle
            if self.current_drone < 0 and not self.expanding and not self.animation_active:
                if self.drone_positions:
                    self.place_next_drone()

            # Update expansion
            self.update_expansion(dt)

            # Auto-advance to next drone
            if not self.expanding and self.animation_active:
                time_acc += dt
                if time_acc > self.animation_speed:
                    time_acc = 0.0
                    advanced = self.place_next_drone()
                    if not advanced:  # No more drones
                        self.animation_active = False
                        # Check coverage
                        unexplored = np.sum(self.global_grid == UNEXPLORED)
                        if unexplored == 0 and not self.path_found:
                            print("Fully explored! Attempting path to emergency...")
                            self.find_path_to_emergency()
                        elif not self.path_found:
                            perc = 100 - (unexplored * 100 / (self.grid_size ** 2))
                            print(f"Exploration progress: {perc:.1f}%")

            # Draw everything
            self.draw_scene()
            self.draw_info_panel()
            pygame.display.flip()

        pygame.quit()


def run_visualization(grid_size=20, json_file=None):
    """
    Entry point to run the drone coverage + pathfinding simulator. It:
    1) Instantiates DronePathfinder,
    2) Tries to load drone data from a given JSON file,
    3) Launches the `.run()` method for PyGame visualization.

    Parameters:
    -----------
    grid_size : int
        Size of the NxN grid to visualize.
    json_file : str
        Path to a JSON file containing 'drone_positions' and 'drone_radii'.
    """
    simulator = DronePathfinder(grid_size=grid_size)

    # If JSON data is provided, load it
    if json_file and os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            positions = []
            sizes = []
            for pos in data.get("drone_positions", []):
                positions.append(tuple(pos))
            for radius in data.get("drone_radii", []):
                sizes.append(radius)
            if len(positions) != len(sizes):
                print("Warning: Mismatched drone positions and sizes in JSON.")
            simulator.drone_positions = positions
            simulator.drone_sizes = sizes
            print(f"Loaded {len(positions)} drones from {json_file}")
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return
    else:
        print(f"Error: JSON file '{json_file}' not found.")
        print("Cannot continue without drone data.")
        return

    # Precompute coverage cells for each drone if needed
    simulator.drone_coverage_cells = []
    for i, (dx, dy) in enumerate(simulator.drone_positions):
        sz = simulator.drone_sizes[i]
        half = (sz - 1) // 2
        cells = []
        for dxp in range(-half, half + 1):
            for dyp in range(-half, half + 1):
                gx, gy = dx + dxp, dy + dyp
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    cells.append((gx, gy))
        simulator.drone_coverage_cells.append(cells)

    print(f"Using {len(simulator.drone_positions)} drones for a {grid_size}x{grid_size} grid.")
    print("Starting simulation...")

    simulator.run()


if __name__ == "__main__":
    # Default grid size
    grid_size = 7

    # Attempt to parse grid size from sys.argv
    if len(sys.argv) > 1:
        try:
            grid_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid grid size argument '{sys.argv[1]}'. Using default {grid_size}x{grid_size}.")

    # The script expects a JSON file 'drone_coverage_results_{grid_size}.json'
    json_file = f"drone_coverage_results_{grid_size}.json"

    if os.path.exists(json_file):
        print(f"Using drone positions from: {json_file}")
        run_visualization(grid_size, json_file)
    else:
        print(f"Error: Required JSON file '{json_file}' not found.")
        print("Please ensure the drone position data is available and matches the grid size.")
