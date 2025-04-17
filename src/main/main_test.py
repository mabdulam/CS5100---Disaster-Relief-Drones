#!/usr/bin/env python
"""
Drone Coverage with Q-Learning (Center-based squares): MULTI-GRID TEST RUN

This file (main_test.py) orchestrates multiple training loops for grid sizes in the
range 7×7 through 25×25. For each grid dimension `grid_size`, we dynamically update
the configuration (CONFIG) to:

  1. Set `N` and `M` to `grid_size`.
  2. Set `max_drones` to at least `2 * grid_size` (and possibly higher if needed).
  3. Use `num_episodes = 20000`.

After training with Q-learning on each grid size, the resulting Q-table is saved to
`Q_table_{grid_size}.pickle`. The final coverage arrangement is stored in a
`drone_coverage_results_{grid_size}.json` file. We also forcibly ensure nearly 100%
coverage (≥ 99.99%) by optionally adding additional drones in a post-processing step.

MODIFICATIONS:
--------------
1) force_complete_coverage: The environment setting ensures episodes aim for 100%
   coverage. The training or final evaluation won't terminate due to stalls or step
   limits if coverage is below ~99%.
2) Additional check after evaluate_policy() ensures coverage is actually at 100%,
   and if not, spawns more drones to fill remaining gaps.

Usage:
------
Simply run this script (e.g., `python main_test.py`) to iterate over selected grid
sizes. Each size triggers Q-learning for 20,000 episodes, and saves both the Q-table
and coverage results JSON.

Note: This script depends on:
  - `train_agent.py` for Q_learning_adaptive_limited, evaluate_policy, and
    get_coverage_fraction (if used).
  - `testing.py` for run_visualization (not mandatory to run in this script,
    unless you want interactive coverage display).
"""

import os
import json
import pickle
import time
import numpy as np

# Import the necessary training code and environment references
from src.training.train_agent import CONFIG, Q_learning_adaptive_limited, evaluate_policy
from src.testing.testing import run_visualization
from src.env.drone_env_limited import DroneCoverageEnvAdaptiveLimited


def main():
    """
    Main loop to train and evaluate Q-learning for each grid size in a specified range.

    Procedure:
      1) For each grid_size in [22..25] (example range given):
         - Update CONFIG to set N=M=grid_size.
         - Adjust max_drones and num_episodes.
         - Force coverage completion in environment (force_complete_coverage=True).
         - Train using Q_learning_adaptive_limited().
         - Save the resulting Q-table to "Q_table_{grid_size}.pickle".
         - Evaluate the final policy with evaluate_policy().
         - Check if coverage is < 99.99% and add extra drones to fill coverage gaps.
         - Save final coverage arrangement to "drone_coverage_results_{grid_size}.json".

    Returns:
        None
    """
    # Loop through the desired grid sizes.
    # (In the given code, it starts from 22..25, but you can adjust as needed.)
    for grid_size in range(22, 26):
        print("=" * 60)
        print(f"Running Q-learning training for grid {grid_size}x{grid_size}")
        print(f"  Setting max_drones = {2 * grid_size}")
        print("  Setting episodes   = 20000")
        print("=" * 60)

        # 1) Dynamically update the CONFIG to match the current grid_size
        CONFIG["N"] = grid_size
        CONFIG["M"] = grid_size
        # Ensure we can place enough drones to reliably reach complete coverage
        CONFIG["max_drones"] = max(2 * grid_size, (grid_size * grid_size) // 4)
        CONFIG["num_episodes"] = 20000
        CONFIG["force_complete_coverage"] = True  # Make sure coverage is forced

        # 2) Create a unique results file name for the coverage JSON
        results_file = os.path.join(os.getcwd(), f"drone_coverage_results_{grid_size}.json")

        # Record the training start time for informational logging
        start_time = time.time()

        # 3) Train the Q-learning agent
        Q_table = Q_learning_adaptive_limited(CONFIG)

        # 4) Save Q-table to a pickle file for future reuse
        qtable_filename = f"Q_table_{grid_size}.pickle"
        with open(qtable_filename, "wb") as handle:
            pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved best Q-table => {qtable_filename}")

        # Log the elapsed training time
        training_time = time.time() - start_time
        print(f"[INFO] Training completed in {training_time:.2f} seconds")

        # 5) Evaluate the final learned policy => ensures coverage
        print("\n[INFO] Evaluating policy with guaranteed 100% coverage...")
        eval_start_time = time.time()
        total_reward, final_obs = evaluate_policy(Q_table, CONFIG)
        final_drones = final_obs["drones"]
        obstacles = final_obs.get("obstacles", [])
        eval_time = time.time() - eval_start_time

        print(f"\n[INFO] Final Q-policy => reward: {total_reward:.3f} for grid {grid_size}x{grid_size}")
        print(f"[INFO] Evaluation completed in {eval_time:.2f} seconds")
        print(f"[INFO] Total drones placed: {len(final_drones)}")

        # 6) Double-check final coverage using a fresh environment instance
        test_env = DroneCoverageEnvAdaptiveLimited(CONFIG)
        test_env.reset()

        # Populate the environment with the final drone arrangement
        for (cx, cy, sz, act) in final_drones:
            test_env.drones.append({"cx": cx, "cy": cy, "size": sz, "active": act})

        # Compute coverage in this final arrangement
        coverage_count, overlap_count = test_env._compute_coverage_and_overlap()
        coverage_percentage = (100.0 * coverage_count / test_env.num_free_cells
                               if test_env.num_free_cells > 0 else 100.0)

        print(f"[INFO] Final coverage: {coverage_count}/{test_env.num_free_cells} "
              f"cells ({coverage_percentage:.2f}%)")

        # 7) If coverage is not at ~100%, forcibly add more drones to fill gaps
        if coverage_percentage < 99.99:
            print("[WARNING] Coverage not 100%, forcing complete coverage...")

            max_additional_drones = 20
            additional_drones = 0

            while coverage_percentage < 99.99 and additional_drones < max_additional_drones:
                # Try placing the largest drone to maximize coverage quickly
                size = max(CONFIG["available_sizes"])

                best_new_coverage = -1
                best_position = None

                # Attempt a brute-force check of every (x,y) to see which yields best coverage
                for x in range(test_env.N):
                    for y in range(test_env.M):
                        new_coverage = test_env._calculate_new_coverage(x, y, size)
                        if new_coverage > best_new_coverage:
                            best_new_coverage = new_coverage
                            best_position = (x, y)

                if best_position and best_new_coverage > 0:
                    x, y = best_position
                    test_env.drones.append({"cx": x, "cy": y, "size": size, "active": True})
                    final_drones.append((x, y, size, True))
                    additional_drones += 1

                    # Recompute coverage
                    coverage_count, overlap_count = test_env._compute_coverage_and_overlap()
                    coverage_percentage = (100.0 * coverage_count / test_env.num_free_cells
                                           if test_env.num_free_cells > 0 else 100.0)
                    print(f"  Added drone at {best_position}, new coverage: {coverage_percentage:.2f}%")
                else:
                    print("  Could not find position to improve coverage!")
                    break

            print(f"[INFO] Added {additional_drones} additional drones "
                  f"to reach {coverage_percentage:.2f}% coverage")

        # 8) Construct and save the final coverage data to JSON
        coverage_data = {
            "grid_size": grid_size,
            "final_reward": total_reward,
            "drone_positions": [],
            "drone_radii": [],
            "obstacles": obstacles,
            "coverage_percentage": coverage_percentage
        }
        for (cx, cy, sz, act) in final_drones:
            coverage_data["drone_positions"].append((cx, cy))
            coverage_data["drone_radii"].append(sz)

        try:
            with open(results_file, "w") as f:
                json.dump(coverage_data, f, indent=4)
            print(f"[INFO] Coverage results saved => {results_file}")
        except Exception as e:
            print(f"[WARNING] Could not save => {e}")

        print("\n")  # Blank line separating different grid size runs


if __name__ == "__main__":
    main()
