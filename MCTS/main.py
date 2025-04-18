#!/usr/bin/env python
"""
Drone Placement with Monte Carlo Tree Search

This program solves the problem of placing the minimum number of drones
to ensure complete coverage of a grid with minimal overlap.
Supports mixed drone sizes: small (3x3) and large (5x5).

Usage:
    python main.py [--train] [--visualize] [--grid-size N] [--drones M] [--iterations I]

Options:
    --train             Run the training phase
    --visualize         Run the visualization
    --grid-size N       Set the grid size to NxN (default: 10)
    --drones M          Maximum number of drones to place (default: 15)
    --iterations I      Number of MCTS iterations (default: 1000)
"""

import argparse
import sys
import os
from training import solve_drone_placement
from testing import run_visualization


def parse_arguments():
    parser = argparse.ArgumentParser(description='Drone Placement Optimization with Mixed Drone Sizes')
    parser.add_argument('--train', action='store_true', help='Run the training phase')
    parser.add_argument('--visualize', action='store_true', help='Run the visualization')
    parser.add_argument('--grid-size', type=int, default=10, help='Grid size (NxN)')
    parser.add_argument('--drones', type=int, default=15, help='Maximum number of drones')
    parser.add_argument('--iterations', type=int, default=1000, help='MCTS iterations')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Default behavior is to run both training and visualization
    if not args.train and not args.visualize:
        args.train = True
        args.visualize = True

    import os
    results_file = os.path.join(os.getcwd(), "drone_placement_results.json")

    if args.train:
        print("=" * 50)
        print(f"Running training with parameters:")
        print(f"  Grid Size: {args.grid_size}x{args.grid_size}")
        print(f"  Max Drones: {args.drones}")
        print(f"  MCTS Iterations: {args.iterations}")
        print(f"  Drone Types: Small (3x3) and Large (5x5)")
        print("=" * 50)

        # Run training
        solve_drone_placement(
            grid_size=args.grid_size,
            max_drones=args.drones,
            mcts_iterations=args.iterations
        )

        # Double-check if results file was created
        if not os.path.exists(results_file):
            print(f"WARNING: Results file was not created at {results_file}")
            if args.visualize:
                print("Visualization will not be possible without results file.")
                args.visualize = False

    if args.visualize:
        # Make sure the results file exists if we're only visualizing
        if not os.path.exists(results_file):
            print(f"Error: Results file '{results_file}' not found.")
            print("Run with --train first or ensure the results file exists.")
            return

        # Verify file content before visualization
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results['drone_positions'])} drone positions for visualization")
            print(f"  {results['drone_sizes'].count(1)} small drones (3x3)")
            print(f"  {results['drone_sizes'].count(2)} large drones (5x5)")
        except Exception as e:
            print(f"WARNING: Issue with results file: {e}")

        print("\nStarting visualization...")
        run_visualization(
            results_file=results_file,
            grid_size=args.grid_size
        )


if __name__ == "__main__":
    main()