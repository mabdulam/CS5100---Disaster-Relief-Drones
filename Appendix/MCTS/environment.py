import numpy as np


class DroneGridEnvironment:
    """Environment for drone placement with mixed drone sizes (3x3 and 5x5).

    This class manages a grid environment where drones with different coverage areas
    can be placed to achieve complete coverage with minimal overlap and using the
    fewest number of drones possible.
    """

    def __init__(self, grid_size=10):
        """Initialize the environment with given grid size.

        Args:
            grid_size: Size of the grid (grid_size x grid_size)
        """
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        # 0: uncovered, 1: covered
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.drone_positions = []
        self.drone_sizes = []  # 1 for small (3x3), 2 for large (5x5)
        self.coverage_count = 0
        self.total_cells = self.grid_size * self.grid_size
        return self._get_state()

    def _get_state(self):
        """Return current state representation as a tuple."""
        return (tuple(map(tuple, self.coverage_grid)), tuple(self.drone_positions), tuple(self.drone_sizes))

    def _is_valid_position(self, pos):
        """Check if position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _get_coverage_area(self, pos, size):
        """Get all cells covered by a drone at the given position with specified size.

        Args:
            pos: (x, y) tuple position
            size: 1 for small drone (3x3), 2 for large drone (5x5)

        Returns:
            List of (x, y) tuples representing covered cells
        """
        x, y = pos
        coverage_area = []
        radius = 2  # size 1 = radius 1 (3x3), size 2 = radius 2 (5x5)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                new_x, new_y = x + dx, y + dy
                if self._is_valid_position((new_x, new_y)):
                    coverage_area.append((new_x, new_y))

        return coverage_area

    def _count_new_coverage(self, pos, size):
        """Count how many new cells would be covered by placing a drone.

        Args:
            pos: (x, y) tuple position
            size: Drone size (1=small, 2=large)

        Returns:
            Number of currently uncovered cells that would be newly covered
        """
        coverage_area = self._get_coverage_area(pos, size)
        return sum(1 for x, y in coverage_area if self.coverage_grid[x, y] == 0)

    def _count_overlap(self, pos, size):
        """Count how many already covered cells would overlap.

        Args:
            pos: (x, y) tuple position
            size: Drone size (1=small, 2=large)

        Returns:
            Number of already covered cells that would be overlapped
        """
        coverage_area = self._get_coverage_area(pos, size)
        return sum(1 for x, y in coverage_area if self.coverage_grid[x, y] == 1)

    def place_drone(self, pos, size):
        """Place a drone at the specified position with given size.

        Args:
            pos: (x, y) tuple position
            size: 1 for small drone (3x3), 2 for large drone (5x5)

        Returns:
            tuple: (reward, new_state, done)
        """
        # Check if position is already occupied
        if pos in self.drone_positions:
            # Return large negative reward if trying to place on existing drone
            return -100, self._get_state(), False

        # Calculate reward components
        new_coverage = self._count_new_coverage(pos, size)
        overlap = self._count_overlap(pos, size)

        # Update the grid
        coverage_area = self._get_coverage_area(pos, size)
        for cx, cy in coverage_area:
            self.coverage_grid[cx, cy] = 1

        # Add drone position and size
        self.drone_positions.append(pos)
        self.drone_sizes.append(size)

        # Update coverage count
        self.coverage_count = np.sum(self.coverage_grid)

        # Calculate reward - larger drones have higher cost but more coverage
        drone_cost = -10 * size  # Substantial penalty for using drones
        overlap_penalty = -1 * overlap
        coverage_reward = 3 * new_coverage  # Increased reward for coverage

        # Check if complete coverage
        done = self.is_grid_filled()
        completion_reward = 200 if done else 0  # Bonus for complete coverage

        # Discourage placements with no new coverage
        if new_coverage == 0:
            total_reward = -20  # Strong penalty for useless placements
        else:
            total_reward = drone_cost + overlap_penalty + coverage_reward + completion_reward

        return total_reward, self._get_state(), done

    def get_valid_actions(self, size):
        """Return list of valid positions to place a drone of the given size.

        Args:
            size: 1 for small drone (3x3), 2 for large drone (5x5)

        Returns:
            List of valid (x, y) positions
        """
        # Only consider positions that provide significant new coverage and aren't occupied
        valid_actions = []

        # First identify all positions that provide new coverage
        coverage_by_pos = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Skip positions that already have a drone
                if (x, y) in self.drone_positions:
                    continue

                new_coverage = self._count_new_coverage((x, y), size)
                if new_coverage > 0:
                    coverage_by_pos[(x, y)] = new_coverage

        # If no positions provide coverage, we're done
        if not coverage_by_pos:
            return []

        # Find positions with at least 50% of the maximum possible new coverage
        max_coverage = max(coverage_by_pos.values()) if coverage_by_pos else 0
        threshold = max(1, max_coverage * 0.5)  # At least 50% of max, but minimum 1

        for pos, coverage in coverage_by_pos.items():
            if coverage >= threshold:
                valid_actions.append(pos)

        # If filter is too restrictive, return all positions with any coverage
        if not valid_actions and coverage_by_pos:
            valid_actions = list(coverage_by_pos.keys())

        return valid_actions

    def is_grid_filled(self):
        """Check if the grid is completely filled."""
        return self.coverage_count >= self.total_cells

    def get_coverage_percentage(self):
        """Return the percentage of the grid that is covered."""
        return 100.0 * self.coverage_count / self.total_cells if self.total_cells > 0 else 0

    def count_drones_by_type(self):
        """Count the number of small and large drones."""
        small_count = self.drone_sizes.count(1)
        large_count = self.drone_sizes.count(2)
        return small_count, large_count

    def calculate_theoretical_max_coverage(self):
        """Calculate the theoretical maximum coverage of all drones without overlap."""
        small_count, large_count = self.count_drones_by_type()
        small_area = 9  # 3x3 grid
        large_area = 25  # 5x5 grid
        return small_count * small_area + large_count * large_area

    def render_text(self):
        """Visualize the current state of the grid as text."""
        grid_display = np.zeros((self.grid_size, self.grid_size), dtype=str)

        # Fill with coverage information
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.coverage_grid[x, y] == 1:
                    grid_display[x, y] = '.'  # covered cell
                else:
                    grid_display[x, y] = ' '  # uncovered cell

        # Add drone positions with different markers for sizes
        for i, (x, y) in enumerate(self.drone_positions):
            if i < len(self.drone_sizes):
                if self.drone_sizes[i] == 1:
                    grid_display[x, y] = 'S'  # Small drone (3x3)
                else:
                    grid_display[x, y] = 'L'  # Large drone (5x5)
            else:
                grid_display[x, y] = 'D'  # Default drone

        # Print the grid
        print('=' * (2 * self.grid_size + 1))
        for row in grid_display:
            print('|' + '|'.join(row) + '|')
        print('=' * (2 * self.grid_size + 1))

        # Print statistics
        small_count, large_count = self.count_drones_by_type()
        print(f"Drones placed: {len(self.drone_positions)}")
        print(f"  Small drones (3x3): {small_count}")
        print(f"  Large drones (5x5): {large_count}")
        print(f"Coverage: {self.coverage_count}/{self.total_cells} cells ({self.get_coverage_percentage():.1f}%)")

        # If we have drones placed, calculate efficiency
        if self.drone_positions:
            theoretical_max = self.calculate_theoretical_max_coverage()
            efficiency = self.coverage_count / theoretical_max if theoretical_max > 0 else 0
            print(f"Coverage efficiency: {efficiency:.2f} (accounting for overlap)")