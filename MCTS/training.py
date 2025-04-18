import numpy as np
import random
import math
import time
import json
import os
from collections import defaultdict
from environment import DroneGridEnvironment


class MCTSNode:
    """Node for Monte Carlo Tree Search with mixed drone sizes."""

    def __init__(self, state, parent=None, action=None, action_size=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action position that led to this state
        self.action_size = action_size  # Size of the drone placed (1=small, 2=large)
        self.children = {}  # (action, size) -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.untried_actions = None  # Lazy initialization, will contain action-size pairs

    def is_fully_expanded(self, env):
        """Check if all possible actions from this state have been tried."""
        if self.untried_actions is None:
            # Initialize state in environment
            env.coverage_grid = np.array(self.state[0])
            env.drone_positions = list(self.state[1])
            env.drone_sizes = list(self.state[2]) if len(self.state) > 2 else []
            env.coverage_count = np.sum(env.coverage_grid)

            # Get valid actions for both drone sizes
            small_actions = [(pos, 1) for pos in env.get_valid_actions(size=1)]
            large_actions = [(pos, 2) for pos in env.get_valid_actions(size=2)]

            # Combine all valid actions with their sizes
            self.untried_actions = large_actions + small_actions  # Prioritize large drones by putting them first

        return len(self.untried_actions) == 0

    def select_action(self, c_param=1.414):
        """Select action according to UCB1 formula."""
        # UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        best_value = float("-inf")
        best_action_pair = None

        for action_pair, child in self.children.items():
            # Exploit term
            exploit = child.value_sum / child.visit_count if child.visit_count > 0 else 0
            # Explore term
            explore = c_param * math.sqrt(
                math.log(self.visit_count) / child.visit_count) if child.visit_count > 0 else float("inf")

            ucb_value = exploit + explore

            if ucb_value > best_value:
                best_value = ucb_value
                best_action_pair = action_pair

        return best_action_pair

    def expand(self, env, action_pair):
        """Add a new child node for the given action pair (position, size)."""
        action, size = action_pair

        # Set environment to current state
        env.coverage_grid = np.array(self.state[0])
        env.drone_positions = list(self.state[1])
        env.drone_sizes = list(self.state[2]) if len(self.state) > 2 else []
        env.coverage_count = np.sum(env.coverage_grid)

        # Execute action
        _, next_state, _ = env.place_drone(action, size)

        # Create child node
        child = MCTSNode(next_state, parent=self, action=action, action_size=size)

        # Remove action from untried actions
        if self.untried_actions is not None and action_pair in self.untried_actions:
            self.untried_actions.remove(action_pair)

        # Add child to children dictionary
        self.children[action_pair] = child

        return child

    def update(self, reward):
        """Update node statistics with reward."""
        self.visit_count += 1
        self.value_sum += reward


def mcts_search(env, iterations=1000, max_rollout_steps=10, exploration_weight=1.414):
    """Perform Monte Carlo Tree Search to find best drone placement."""
    # First check if we already have complete coverage
    if env.coverage_count >= env.total_cells:
        print("Grid is already completely filled. No more drones needed.")
        return None

    original_state = env.reset()
    root = MCTSNode(original_state)
    existing_positions = set(env.drone_positions)

    # Progress tracking
    if iterations >= 100:
        progress_interval = iterations // 10
    else:
        progress_interval = 10

    for i in range(iterations):
        # Occasionally show progress
        if (i + 1) % progress_interval == 0:
            print(f"MCTS Progress: {i + 1}/{iterations} iterations completed")

        # Selection
        node = root
        env.reset()  # Reset to original state
        current_positions = set(existing_positions)  # Keep track of occupied positions

        # Path to selected node
        selected_path = []

        # Select until we reach a leaf node
        done = False
        while node.is_fully_expanded(env) and node.children:
            action_pair = node.select_action(c_param=exploration_weight)
            if action_pair is None:
                break

            action, size = action_pair

            # Skip if position is already occupied
            if action in current_positions:
                # Try another action if available
                alternative_pairs = [pair for pair in node.children.keys() if pair[0] not in current_positions]
                if alternative_pairs:
                    action_pair = random.choice(alternative_pairs)
                    action, size = action_pair
                else:
                    break  # No valid actions left

            node = node.children[action_pair]
            selected_path.append(action_pair)
            current_positions.add(action)  # Mark position as occupied
            reward, _, done = env.place_drone(action, size)

            if done:  # Terminal state reached
                break

        # Expansion if the node is not terminal and not fully expanded
        if not done and not node.is_fully_expanded(env):
            # Parse state to get occupied positions
            if isinstance(node.state, tuple) and len(node.state) > 1:
                current_positions = set(node.state[1])
            else:
                current_positions = set()

            if node.untried_actions is None or len(node.untried_actions) == 0:
                # Initialize untried actions if needed
                env.coverage_grid = np.array(node.state[0])
                env.drone_positions = list(node.state[1])
                env.drone_sizes = list(node.state[2]) if len(node.state) > 2 else []
                env.coverage_count = np.sum(env.coverage_grid)

                # Get valid actions for both drone sizes
                small_actions = [(pos, 1) for pos in env.get_valid_actions(size=1)]
                large_actions = [(pos, 2) for pos in env.get_valid_actions(size=2)]

                # Prioritize large drones
                node.untried_actions = large_actions + small_actions

            if node.untried_actions:
                # Select action with highest potential for new coverage efficiency
                best_action_pairs = []
                best_efficiency = -1

                for action_pair in node.untried_actions[:]:
                    action, size = action_pair
                    new_coverage = env._count_new_coverage(action, size)

                    # Calculate efficiency as coverage per drone area
                    # We use size^2.5 to give even more preference to large drones
                    efficiency = new_coverage / (size ** 2.5) if size > 0 else 0

                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_action_pairs = [action_pair]
                    elif efficiency == best_efficiency:
                        best_action_pairs.append(action_pair)

                # Choose randomly among the best actions
                action_pair = random.choice(best_action_pairs) if best_action_pairs else random.choice(
                    node.untried_actions)
                action, size = action_pair
                reward, _, done = env.place_drone(action, size)
                child = node.expand(env, action_pair)
                node = child

        # Simulation (rollout)
        total_reward = 0
        done = False
        step = 0
        drone_count = len(env.drone_positions)

        while not done and step < max_rollout_steps:
            # Get current positions of drones
            current_positions = set(env.drone_positions)

            # Get valid actions for both drone sizes (filtering out occupied positions)
            small_actions = [(pos, 1) for pos in env.get_valid_actions(size=1) if pos not in current_positions]
            large_actions = [(pos, 2) for pos in env.get_valid_actions(size=2) if pos not in current_positions]

            # Prioritize large drones
            valid_actions = large_actions + small_actions

            # If no valid actions or the grid is filled, stop the simulation
            if not valid_actions or env.is_grid_filled():
                break

            # Semi-greedy policy for rollout - prioritize high coverage positions
            if random.random() < 0.8:  # 80% chance of greedy selection
                # Find action that maximizes coverage efficiency
                best_efficiency = -1
                best_action_pairs = []

                for action_pair in valid_actions:
                    action, size = action_pair
                    new_coverage = env._count_new_coverage(action, size)

                    # Skip actions with very little new coverage
                    if new_coverage < size * 2:
                        continue

                    # Calculate efficiency as coverage per drone area
                    # Increased power to strongly prefer large drones
                    efficiency = new_coverage / (size ** 2.5) if size > 0 else 0

                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_action_pairs = [action_pair]
                    elif efficiency == best_efficiency:
                        best_action_pairs.append(action_pair)

                # If we found good actions, choose among them
                if best_action_pairs:
                    action_pair = random.choice(best_action_pairs)
                else:
                    # Otherwise choose randomly from all valid actions
                    # But with heavy bias towards large drones
                    if large_actions and random.random() < 0.7:
                        action_pair = random.choice(large_actions)
                    else:
                        action_pair = random.choice(valid_actions)
            else:
                # Random exploration with bias towards large drones
                if large_actions and random.random() < 0.7:
                    action_pair = random.choice(large_actions)
                else:
                    action_pair = random.choice(valid_actions)

            action, size = action_pair
            reward, _, done = env.place_drone(action, size)
            total_reward += reward
            step += 1

        # Huge bonus for completing with fewer drones
        if done:
            new_drone_count = len(env.drone_positions) - drone_count
            fewer_drone_bonus = 500 / (new_drone_count + 1)  # Higher reward for fewer additional drones
            total_reward += fewer_drone_bonus

        # Backpropagation
        while node is not None:
            node.update(total_reward)
            node = node.parent

    # Select best action from root
    best_action_pair = None
    best_value = float("-inf")

    # Print all possible actions and their values
    print("\nPossible next drone placements:")
    for action_pair, child in root.children.items():
        value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
        action, size = action_pair
        size_name = "Large (5x5)" if size == 2 else "Small (3x3)"
        print(f"  Position {action}, {size_name}: Value = {value:.2f}, Visits = {child.visit_count}")

        if value > best_value:
            best_value = value
            best_action_pair = action_pair

    if best_action_pair:
        action, size = best_action_pair
        size_name = "Large (5x5)" if size == 2 else "Small (3x3)"
        print(f"\nSelected: Position {action}, {size_name} with value {best_value:.2f}")

    return best_action_pair


def solve_drone_placement(grid_size=10, max_drones=10, mcts_iterations=2000):
    """Solve the drone placement problem using MCTS."""
    env = DroneGridEnvironment(grid_size)
    state = env.reset()

    print(f"Starting drone placement for {grid_size}x{grid_size} grid")
    print(f"Using MCTS with {mcts_iterations} iterations per decision")
    print(f"Maximum drones: {max_drones}")
    env.render_text()

    done = False
    drone_count = 0
    total_reward = 0
    placements = []
    drone_sizes = []

    # Track coverage history
    coverage_history = [0]

    while not done and drone_count < max_drones:
        # Check if we've already achieved complete coverage
        if env.coverage_count >= env.total_cells:
            print("\nComplete coverage achieved! Stopping drone placement.")
            done = True
            break

        # Check if there are any uncovered cells left at all
        uncovered_cells = env.total_cells - env.coverage_count
        if uncovered_cells <= 0:
            print(f"\nAll {env.total_cells} cells are covered! Stopping drone placement.")
            done = True
            break

        # Use MCTS to find the best action
        print(f"\nPlanning drone placement {drone_count + 1}... ({uncovered_cells} cells still uncovered)")
        action_pair = mcts_search(env, iterations=mcts_iterations,
                                  max_rollout_steps=max_drones - drone_count,
                                  exploration_weight=1.5)

        # Break immediately if no action was found or we've reached complete coverage
        if action_pair is None:
            print("No valid actions remaining or grid is completely filled.")
            done = True
            break

        action, size = action_pair
        print(f"Placing {'small (3x3)' if size == 1 else 'large (5x5)'} drone at position {action}")

        # Calculate coverage before placement
        coverage_before = env.coverage_count

        # Place the drone
        reward, state, done = env.place_drone(action, size)

        # Calculate new coverage gained
        new_coverage = env.coverage_count - coverage_before
        coverage_history.append(env.coverage_count)

        total_reward += reward
        drone_count += 1
        placements.append(action)
        drone_sizes.append(size)

        print(f"New cells covered: {new_coverage}")
        print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        print(
            f"Current coverage: {env.coverage_count}/{env.total_cells} cells ({100 * env.coverage_count / env.total_cells:.2f}%)")
        env.render_text()

        # Check for complete coverage again
        if env.coverage_count >= env.total_cells:
            print("\nComplete coverage achieved!")
            done = True
            break

        # If no new coverage from last placement, something is wrong
        if new_coverage <= 0 and drone_count > 0:
            print("WARNING: Last placement did not increase coverage! Removing this drone.")
            # Remove the last drone placement
            if len(placements) > 0:
                placements.pop()
            if len(drone_sizes) > 0:
                drone_sizes.pop()
            if len(env.drone_positions) > 0:
                env.drone_positions.pop()
            if len(env.drone_sizes) > 0:
                env.drone_sizes.pop()

            # Recalculate coverage grid
            env.coverage_grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
            for i, pos in enumerate(env.drone_positions):
                size = env.drone_sizes[i]
                coverage_area = env._get_coverage_area(pos, size)
                for cx, cy in coverage_area:
                    env.coverage_grid[cx, cy] = 1

            # Update coverage count
            env.coverage_count = np.sum(env.coverage_grid)
            drone_count -= 1
            print(
                f"Updated coverage: {env.coverage_count}/{env.total_cells} cells ({100 * env.coverage_count / env.total_cells:.2f}%)")
            env.render_text()

    print(f"\nFinal result: {drone_count} drones placed ({drone_sizes.count(1)} small, {drone_sizes.count(2)} large)")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Coverage: {env.coverage_count}/{env.total_cells} cells ({100 * env.coverage_count / env.total_cells:.2f}%)")

    # Report coverage efficiency
    if drone_count > 0:
        print(f"Average cells covered per drone: {env.coverage_count / drone_count:.2f}")

        # Calculate coverage statistics by drone type
        small_count = drone_sizes.count(1)
        large_count = drone_sizes.count(2)
        theoretical_max_coverage = small_count * 9 + large_count * 25

        print(f"Theoretical maximum coverage: {theoretical_max_coverage} cells")
        if theoretical_max_coverage > 0:
            print(f"Coverage efficiency: {env.coverage_count / theoretical_max_coverage:.2f} (accounting for overlap)")

    # Report coverage history
    print("\nCoverage progression:")
    for i, coverage in enumerate(coverage_history):
        if i > 0:
            gain = coverage - coverage_history[i - 1]
            print(
                f"Drone {i} ({drone_sizes[i - 1]}x{drone_sizes[i - 1]}): Coverage = {coverage}/{env.total_cells} cells (+{gain} from previous)")

    # Save results to file for visualization
    try:
        results = {
            "grid_size": grid_size,
            "drone_positions": env.drone_positions,  # Use environment's drone positions
            "drone_sizes": env.drone_sizes  # Use environment's drone sizes
        }

        results_file = os.path.join(os.getcwd(), "drone_placement_results.json")

        print(f"Saving {len(env.drone_positions)} drones to results file")
        print(f"  Small drones: {env.drone_sizes.count(1)}")
        print(f"  Large drones: {env.drone_sizes.count(2)}")

        with open(results_file, "w") as f:
            json.dump(results, f)

        print(f"\nResults saved to: {results_file}")
    except Exception as e:
        print(f"\nError saving results: {e}")

    return env.drone_positions, env.drone_sizes  # Return the actual drone positions and sizes from environment


if __name__ == "__main__":
    # Parameters
    grid_size = 10
    max_drones = 8  # Reduced maximum drones
    mcts_iterations = 3000  # Increased iterations for better optimization

    print("Drone Placement Optimization with Mixed Drone Sizes")
    print("=" * 50)
    print("Optimizing for MINIMUM number of drones")
    print("=" * 50)

    # Solve the problem
    start_time = time.time()
    placements, drone_sizes = solve_drone_placement(grid_size, max_drones, mcts_iterations)
    end_time = time.time()

    print(f"\nComputation time: {end_time - start_time:.2f} seconds")