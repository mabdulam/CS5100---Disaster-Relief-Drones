"""
drone_env_limited.py

This module defines the DroneCoverageEnvAdaptiveLimited class, which models a
2D grid environment for a drone coverage task. The agent can spawn drones or
remove them, seeking to cover as many free cells as possible while minimizing
overlap and respecting constraints such as stall thresholds and maximum steps.

Key Modifications:
- Ability to force the environment to continue until nearly 100% coverage is reached.
- Smarter drone placement in test mode to preferentially cover new cells.
- Keeping a coverage history for debugging or analysis.
"""

import numpy as np
import random

class DroneCoverageEnvAdaptiveLimited:
    """
    A center-based squares environment for placing drones on a grid to achieve
    high coverage.

    The environment is parameterized by a configuration dictionary (config),
    which includes:

      - N, M: grid dimensions.
      - available_sizes: list of possible drone square sizes (e.g., 3, 5).
      - max_drones: maximum number of drones allowed at once.
      - obstacle_percent: fraction of cells that become obstacles randomly.
      - coverage_multiplier: scalar for how strongly coverage is rewarded.
      - alpha_env, beta_env, gamma_penalty_env: penalty factors for overlap,
        uncovered fraction, and per-drone active cost, respectively.
      - stall_threshold_env: if coverage does not improve for this many steps,
        the episode can end (unless we force coverage).
      - max_steps_env: absolute max steps in an episode (unless coverage forced).
      - test_mode: boolean for special behavior in test scenarios (e.g., more
        deterministic spawns).
      - force_complete_coverage: boolean to continue the episode until coverage
        is near 100%, overriding normal stall or step-limit rules if coverage
        is still under 99%.

    The environment implements standard reset() and step() methods for RL:
    reset() => fresh environment, random obstacles, empty drone list.
    step(action) => modifies environment state, returns (obs, reward, done, info).
    """

    def __init__(self, config):
        """
        Initializes the environment using settings from 'config'.
        
        Args:
            config (dict): Configuration dictionary with keys including:
                - "N", "M": Grid dimensions (int).
                - "available_sizes": List of allowed drone square sizes.
                - "max_drones": Max number of drones allowed.
                - "obstacle_percent": Fraction of cells to become obstacles.
                - "coverage_multiplier": Coverage scalar reward factor.
                - "alpha_env": Overlap penalty factor.
                - "beta_env": Uncovered fraction penalty factor.
                - "gamma_penalty_env": Cost per active drone.
                - "stall_threshold_env": Steps allowed without coverage improvement.
                - "max_steps_env": Max environment steps per episode.
                - "test_mode": If true, environment may behave differently (e.g., 
                               smarter spawns).
                - "force_complete_coverage": If true, continue episode until coverage 
                                             is nearly 100%.
        """
        # Grid dimensions
        self.N = config["N"]
        self.M = config["M"]

        # Drone parameters
        self.available_sizes = config["available_sizes"]
        self.max_drones = config["max_drones"]

        # Obstacles
        self.obstacle_percent = config["obstacle_percent"]

        # Coverage rewards/penalties
        self.coverage_multiplier = config.get("coverage_multiplier", 5.0)
        self.alpha = config["alpha_env"]
        self.beta = config["beta_env"]
        self.gamma_penalty = config["gamma_penalty_env"]
        self.stall_threshold = config["stall_threshold_env"]
        self.max_steps = config["max_steps_env"]

        # Mode flags
        self.test_mode = config.get("test_mode", False)
        self.force_complete_coverage = config.get("force_complete_coverage", True)

        # Internal states
        self.done = False
        self.obstacles = set()  # Set of (x, y) obstacle cells
        self.num_free_cells = self.N * self.M
        self.drones = []        # List of dicts: {"cx":..., "cy":..., "size":..., "active":...}

        # Coverage improvement tracking
        self.previous_coverage = 0
        self.stall_counter = 0
        self.steps_taken = 0

        # Optional coverage history for analysis
        self.coverage_history = []

    def reset(self):
        """
        Resets the environment to a fresh state:
          - Clears done, coverage counters, and step counters.
          - Generates new random obstacles.
          - Empties the drone list.
          - Returns the initial observation dictionary.
        
        Returns:
            dict: A dictionary with "drones" and "obstacles" describing the new state.
        """
        self.done = False
        self._generate_obstacles()
        self.drones = []
        self.previous_coverage = 0
        self.stall_counter = 0
        self.steps_taken = 0
        self.coverage_history = []
        return self._get_observation()

    def _generate_obstacles(self):
        """
        Creates a random set of obstacles in the grid according to obstacle_percent.
        This is called on reset() to randomize environment obstacles each episode.
        """
        total_cells = self.N * self.M
        num_obs = int(self.obstacle_percent * total_cells)
        all_cells = [(x, y) for x in range(self.N) for y in range(self.M)]
        chosen = random.sample(all_cells, num_obs)
        self.obstacles = set(chosen)
        self.num_free_cells = total_cells - num_obs

    def _get_observation(self):
        """
        Constructs an observation dictionary reflecting current environment state.

        Returns:
            dict: {
                "drones": [(cx, cy, size, active_bool), ...],
                "obstacles": [(ox, oy), ...]
            }
        """
        drone_list = []
        for d in self.drones:
            drone_list.append((d["cx"], d["cy"], d["size"], d["active"]))
        return {
            "drones": drone_list,
            "obstacles": list(self.obstacles)
        }

    def step(self, action):
        """
        Advances the environment by one time step based on the given 'action'.
        Valid actions:
          - type="SPAWN_RANDOM": Spawn a new drone at a random or selected location.
          - type="ACT": With "move" in {"REMOVE", "STAY"} for an existing drone index.
          - type="NOOP": Do nothing.

        Coverage, overlaps, and coverage fraction are recalculated. The environment
        may end (done=True) if coverage saturates, stalls exceed threshold, or
        steps exceed max_steps (unless coverage is forced to continue).

        Args:
            action (dict): e.g., {"type": "SPAWN_RANDOM", "size": 5} or
                                    {"type": "ACT", "drone_index": i, "move": "REMOVE"}.

        Returns:
            obs (dict): Updated observation after the action.
            reward (float): Reward from coverage fraction minus penalties.
            done (bool): Whether the episode has ended.
            info (dict): Additional info (currently empty).
        """
        # 1) Process the action
        if action["type"] == "SPAWN_RANDOM":
            self._spawn_random_drone(action.get("size", 3))
        elif action["type"] == "ACT":
            idx = action.get("drone_index", -1)
            mv = action.get("move", None)
            self._act_on_drone(idx, mv)
        # "NOOP" => do nothing

        # 2) Compute coverage and overlap
        coverage_count, overlap_count = self._compute_coverage_and_overlap()
        self.coverage_history.append(coverage_count)

        coverage_fraction = coverage_count / float(self.num_free_cells) if self.num_free_cells > 0 else 1.0
        overlap_fraction  = overlap_count / float(self.num_free_cells) if self.num_free_cells > 0 else 0.0
        uncovered_fraction = 1.0 - coverage_fraction
        num_active = sum(d["active"] for d in self.drones)

        # 3) Reward function
        reward = (
            self.coverage_multiplier * coverage_fraction
            - self.alpha * overlap_fraction
            - self.beta * uncovered_fraction
            - self.gamma_penalty * num_active
        )

        # Extra bonus if coverage is complete
        if coverage_count >= self.num_free_cells:
            reward += 100.0
            self.done = True

        # 4) Stall detection
        if coverage_count > self.previous_coverage:
            self.previous_coverage = coverage_count
            self.stall_counter = 0
        else:
            self.stall_counter += 1
            # Only end due to stall if we are NOT forcing coverage,
            # or if coverage fraction is high enough
            if (self.stall_counter >= self.stall_threshold
                    and not (self.force_complete_coverage and coverage_fraction < 0.99)):
                self.done = True

        # 5) Step increments & max step check
        self.steps_taken += 1
        if (self.steps_taken >= self.max_steps
                and not (self.force_complete_coverage and coverage_fraction < 0.99)):
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _spawn_random_drone(self, size):
        """
        Spawns a new drone of the specified 'size' at a random location (in training mode)
        or a more calculated location (in test mode), provided we haven't reached max_drones.

        Args:
            size (int): The side length of the square drone (e.g., 3, 5).
        """
        if len(self.drones) >= self.max_drones:
            return
        if size not in self.available_sizes:
            size = random.choice(self.available_sizes)

        if self.test_mode:
            # Smarter placement: tries to maximize new coverage
            best_coverage = -1
            best_positions = []

            # For efficiency, we only sample up to 100 positions in the grid
            sample_size = min(100, self.N * self.M)
            positions = [(x, y) for x in range(self.N) for y in range(self.M)]
            sample_positions = random.sample(positions, sample_size)

            for rx, ry in sample_positions:
                new_coverage = self._calculate_new_coverage(rx, ry, size)
                if new_coverage > best_coverage:
                    best_coverage = new_coverage
                    best_positions = [(rx, ry)]
                elif new_coverage == best_coverage:
                    best_positions.append((rx, ry))

            if best_positions:
                rx, ry = random.choice(best_positions)
            else:
                # Fallback if no positions are better
                rx = random.randint(0, self.N - 1)
                ry = random.randint(0, self.M - 1)
        else:
            # Training mode: purely random
            rx = random.randint(0, self.N - 1)
            ry = random.randint(0, self.M - 1)

        self.drones.append({"cx": rx, "cy": ry, "size": size, "active": True})

    def _calculate_new_coverage(self, cx, cy, size):
        """
        Calculates how many new (previously uncovered) cells would be covered by
        placing a drone of given 'size' at coordinate (cx, cy). Used in test_mode
        to pick spawn locations that add the most coverage.

        Args:
            cx (int): x-coordinate (row index).
            cy (int): y-coordinate (column index).
            size (int): side length of the drone.

        Returns:
            int: number of new cells that would be covered, ignoring overlap.
        """
        half = (size - 1) // 2

        # Collect all currently covered cells by existing drones
        current_coverage = set()
        for d in self.drones:
            if not d["active"]:
                continue
            dcx, dcy, ds = d["cx"], d["cy"], d["size"]
            dhalf = (ds - 1) // 2
            for dx in range(-dhalf, dhalf + 1):
                for dy in range(-dhalf, dhalf + 1):
                    gx = dcx + dx
                    gy = dcy + dy
                    if 0 <= gx < self.N and 0 <= gy < self.M and (gx, gy) not in self.obstacles:
                        current_coverage.add((gx, gy))

        # Now see how many new cells would be covered by this new drone
        new_coverage = 0
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                gx = cx + dx
                gy = cy + dy
                if (0 <= gx < self.N and 0 <= gy < self.M
                        and (gx, gy) not in self.obstacles
                        and (gx, gy) not in current_coverage):
                    new_coverage += 1

        return new_coverage

    def _act_on_drone(self, idx, move):
        """
        Processes an action ("REMOVE" or "STAY") for a drone at index 'idx'.
        If 'REMOVE', the drone is removed from the list. If 'STAY', we do nothing.

        Args:
            idx (int): index of the drone in self.drones list.
            move (str): "REMOVE" or "STAY".
        """
        if idx < 0 or idx >= len(self.drones):
            return
        d = self.drones[idx]
        if move == "REMOVE":
            self.drones.pop(idx)
        elif move == "STAY":
            # do nothing
            pass
        # No other move types are allowed in this environment

    def _compute_coverage_and_overlap(self):
        """
        Counts how many cells are covered at least once (coverage) and how many
        are covered at least twice (overlap).

        Returns:
            (int, int): (coverage_count, overlap_count)
        """
        cover_count = {}
        for d in self.drones:
            if not d["active"]:
                continue
            cx, cy, s = d["cx"], d["cy"], d["size"]
            half = (s - 1) // 2
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    gx = cx + dx
                    gy = cy + dy
                    if 0 <= gx < self.N and 0 <= gy < self.M:
                        if (gx, gy) not in self.obstacles:
                            cover_count[(gx, gy)] = cover_count.get((gx, gy), 0) + 1

        coverage_count = sum(1 for v in cover_count.values() if v >= 1)
        overlap_count  = sum(1 for v in cover_count.values() if v >= 2)
        return coverage_count, overlap_count

    def get_coverage_fraction(self):
        """
        Returns the current fraction of free cells covered by at least one drone.

        Returns:
            float: coverage fraction in [0.0, 1.0].
        """
        coverage_count, _ = self._compute_coverage_and_overlap()
        return coverage_count / float(self.num_free_cells) if self.num_free_cells > 0 else 1.0
