# Disaster Relief Drones – Detailed README

This repository contains code for a **Drone Coverage and Pathfinding** project using **tabular Q-learning** in a 2D grid environment. The goal is to spawn and position drones in such a way that they cover (scan) as many free cells as possible, while also demonstrating an emergency pathfinding step (A*). Below is a comprehensive guide to:

- Creating or recreating the conda/virtual environment
- Installing necessary packages
- Understanding the folder structure
- Running the training script (`main_test.py`) for multiple grid sizes
- (Optionally) Visualizing the coverage with the `testing.py` script

## 1. Environment Setup

### 1.1. Recommended: Using Anaconda / Miniconda

1. **Install** [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. **Create** a new conda environment:
   ```bash
   conda create -n drone_demos python=3.9
   ```
   This creates an environment named `drone_demos` with Python 3.9.

3. **Activate** the environment:
   ```bash
   conda activate drone_demos
   ```

4. **Install** required packages:
   - numpy (for array operations)
   - matplotlib (for plotting training curves)
   - pygame (for visualization in testing.py)
   - pip (already included, but used to install from PyPI)

   Example:
   ```bash
   conda install numpy matplotlib
   conda install -c conda-forge pygame
   ```

   If you prefer pip inside the conda environment, you can do:
   ```bash
   pip install numpy matplotlib pygame
   ```

### 1.2. Using pip in a Virtual Environment

Alternatively, if you are not using Anaconda:

1. Install Python 3.9+ from [python.org](https://www.python.org/) if not already installed.

2. Create a venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install numpy matplotlib pygame
   ```

4. Confirm you can import the modules (no errors) by launching Python:
   ```bash
   python
   >>> import numpy, matplotlib, pygame
   >>> exit()
   ```

### 1.3. Demo

https://m.youtube.com/watch?v=1LlGsLSzpQA&pp=ygUOZ2FtaW5na2luZzA5NDE%3D

## 2. Directory Structure

Below is a typical layout (yours may vary slightly, but the essential structure is):

```
CS5100---Disaster-Relief-Drones
│
├─ README.md
└─ src
   ├─ env
   │   └─ drone_env_limited.py
   ├─ main
   │   └─ main_test.py
   ├─ testing
   │   └─ testing.py
   └─ training
       └─ train_agent.py
```

- **src/env**: Contains the environment code (`drone_env_limited.py`).
- **src/training**: Contains the Q-learning script (`train_agent.py`).
- **src/main**: Contains the higher-level orchestration script (`main_test.py`) that loops over multiple grid sizes.
- **src/testing**: Contains a `testing.py` (in this case named `testing_3.py` or `testing.py`) for visualization with PyGame.

## 3. Running the Multi-Grid Training

### 3.1. Script Overview

`main_test.py`: Trains a tabular Q-learning agent across multiple grid sizes (default: 22×22 to 25×25).

For each `grid_size`, it dynamically updates configuration (`CONFIG`) to:
- Set N = M = grid_size
- Set max_drones to at least 2 × grid_size
- Use num_episodes = 20000
- Performs Q-learning (`Q_learning_adaptive_limited` from `train_agent.py`)
- Saves each resulting Q-table to `Q_table_{grid_size}.pickle`
- Evaluates final coverage with `evaluate_policy`, saves coverage data in `drone_coverage_results_{grid_size}.json`
- If coverage isn't near 100%, forces coverage by adding additional large drones.

### 3.2. Steps to Run

1. Ensure the environment is activated (conda or venv) with the required packages installed.
2. Navigate (via terminal) to the top-level folder (`CS5100---Disaster-Relief-Drones`) or wherever you cloned this repo.
3. Run:
   ```bash
   python src/main/main_test.py
   ```
   or (Windows):
   ```bash
   python .\src\main\main_test.py
   ```
4. Observe the console output. You will see logs indicating:
   - The current grid size
   - Episode progress for training
   - Reward/coverage each episode
   - A final coverage result
   - Possibly forcing additional drone placements if coverage < 99.99%.

### 3.3. Generated Outputs

- **Pickle files**:
  - `Q_table_{grid_size}.pickle` for each grid size in [22..25] (by default).
  - Contains the learned Q-table as a Python dictionary (tabular RL approach).

- **JSON coverage results**:
  - `drone_coverage_results_{grid_size}.json` summarizing:
    - The final coverage fraction
    - The final drone positions and sizes
    - Any obstacles
    - The final reward achieved.

- **training_progress.png**:
  - A Matplotlib plot of training rewards and coverage fraction per episode, saved after Q-learning completes for each training session. (Located in the current working directory.)

- **training_output.txt**:
  - Logs each episode's steps, reward, coverage, etc., appended after each run.

**Important**: The files will be stored in the directory from which Python is run (the "current working directory"). If you run `python src\main\main_test.py` from the top-level directory, they appear there. If you change to `src\main\` and run `python main_test.py`, they appear in `src\main\`.

## 4. (Optional) Coverage Visualization with testing.py

If you want a PyGame-based visualization of a final result:

1. After training a particular grid size (say 20×20), you will have a JSON file named `drone_coverage_results_20.json`.

2. Run the testing script with the matching grid size:
   ```bash
   python src/testing/testing.py 20
   ```
   or if it's named testing_3.py:
   ```bash
   python src/testing/testing_3.py 20
   ```

This script looks for `drone_coverage_results_20.json`. If found, it loads the drone positions and sizes, initializes a random environment (for pathfinding demonstration), and animates the scanning of drones across the grid.

**PyGame controls**:
- **Space** = Toggle play/pause of drone placement.
- **R** = Reset the animation.
- **P** = Attempt pathfinding to an "emergency location."
- **+** / **-** = Increase or decrease animation speed.
- **Esc** = Quit the window.

## 5. Additional Notes

- **Force complete coverage**:
  By default, the environment can continue beyond typical stall or step limits if coverage is under 99%. This ensures the agent always tries for nearly 100% coverage, which can significantly increase training time if the agent takes a while to discover good placements.

- **Performance**:
  - Because tabular Q-learning enumerates states by a string representation of all drones, the state space can grow large if max_drones or grid size is high. Expect slower performance for bigger grids or many drones.
  - You can reduce num_episodes or grid size to test quickly.

- **Windows vs. macOS/Linux**:
  The instructions are essentially the same. Just be aware that the path separators differ (backslash vs. forward slash).

## 6. Common Troubleshooting

- **ModuleNotFoundError** (e.g. No module named 'src.env'):
  - Ensure you are running Python from the project's top-level directory or you have the correct package structure.
  - E.g., `python src/main/main_test.py` or set your PYTHONPATH to the project root.

- **No coverage or poor coverage**:
  - Check that `force_complete_coverage` is set to `True` in CONFIG.
  - You may need more episodes or a bigger max_drones.

- **PyGame window won't open** or "no display name and no DISPLAY environment variable":
  - If you are on a headless server, you can't run PyGame's GUI. You'll need a local machine or X11 forwarding.

- **Large overlaps or negative rewards**:
  - The reward function heavily penalizes overlap. The agent sometimes picks fewer drones or tries to place them more spread out. Possibly adjust `alpha_env` or `beta_env`.

## 7. Contributing

If you wish to modify or extend the environment (e.g., add movement or more complex drone shapes), see `drone_env_limited.py`.

For deeper RL changes (like different exploration strategies), see `train_agent.py`.

Pull requests are welcome if you have improvements or bug fixes.

## 8. License

```
MIT License

Copyright (c) 2025 Mohammed Abdul-Ameer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 9. Contact

If you have questions or encounter issues, you can open an issue on the repository or contact the maintainers at:
<abdulameer.m@northeastern.edu>
<kotti.s@northeastern.edu>
<sonawane.at@northeastern.edu>
<lin.zihan@northeastern.edu>

Happy Drone Coverage & Pathfinding!