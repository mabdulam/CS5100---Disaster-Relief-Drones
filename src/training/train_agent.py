"""
train_agent.py

This script implements a tabular Q-learning algorithm for a drone coverage problem,
using an environment defined in drone_env_limited.py. The Q-learning agent spawns
and removes drones on a grid to maximize coverage while minimizing overlap and
drone costs.
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. FIX SEEDS GLOBALLY
# ----------------------------------------------------------------------
# We set these seeds to ensure reproducible behavior in random draws.
random.seed(1234)
np.random.seed(1234)

from src.env.drone_env_limited import DroneCoverageEnvAdaptiveLimited

###############################################################################
# ALL CONFIGURATION IN ONE PLACE
###############################################################################
CONFIG = {
    # Grid Dimensions
    # ---------------
    # N and M define the row and column dimensions, respectively, of the 2D grid.
    "N": 10,
    "M": 10,

    # Drone Parameters
    # ----------------
    # available_sizes : a list of possible drone sizes (e.g., 3x3, 5x5).
    # max_drones      : maximum number of drones allowed at once.
    "available_sizes": [3, 5],
    "max_drones": 25,

    # Obstacle Parameters
    # -------------------
    # obstacle_percent : fraction of grid cells that become obstacles (randomly selected).
    "obstacle_percent": 0.0,

    # Coverage Reward Multipliers
    # ---------------------------
    # coverage_multiplier  : how strongly to reward coverage fraction.
    # alpha_env            : penalty scaling factor for overlapping coverage.
    # beta_env             : penalty scaling factor for uncovered fraction.
    # gamma_penalty_env    : penalty for each active drone (e.g., cost to keep a drone).
    "coverage_multiplier": 200.0,
    "alpha_env": 15.0,
    "beta_env": 4.0,
    "gamma_penalty_env": 0.005,

    # Episode Termination Parameters
    # ------------------------------
    # stall_threshold_env : number of consecutive steps allowed without coverage improvement.
    # max_steps_env       : maximum steps per episode, regardless of coverage.
    "stall_threshold_env": 800,
    "max_steps_env": 1500,

    # Q-Learning Hyperparameters
    # --------------------------
    # num_episodes : total training episodes.
    # gamma_rl     : discount factor for future rewards.
    # alpha_rl     : learning rate for Q-value updates.
    # epsilon_rl   : initial epsilon for epsilon-greedy exploration.
    # epsilon_decay: rate at which epsilon decays each episode.
    # epsilon_min  : minimum epsilon (stops decaying once below this).
    "num_episodes": 2000,
    "gamma_rl": 0.95,
    "alpha_rl": 0.1,
    "epsilon_rl": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,

    # Execution Modes
    # ---------------
    # test_mode              : if True, environment uses special logic for deterministic behavior.
    # force_complete_coverage: if True, tries to ensure near 100% coverage.
    "test_mode": False,
    "force_complete_coverage": True
}


def state_to_str(obs):
    """
    Convert the environment's observation into a canonical string form.

    The environment state (observation) is a dictionary of:
      {
        "drones"    : [(cx, cy, size, active), ...],
        "obstacles" : [list_of_obstacle_coords]
      }

    We take the 'drones' list, transform each drone to (size, cx, cy, active_bit),
    sort that list (for consistency), and then convert it to a string.

    Args:
        obs (dict): A dictionary with "drones" and "obstacles" entries from the environment.

    Returns:
        str: A canonical, string-based representation of the drone configuration.
    """
    drones = obs["drones"]  # list of tuples (cx, cy, size, active)
    canon = []
    for (cx, cy, sz, act) in drones:
        a_bit = 1 if act else 0
        canon.append((sz, cx, cy, a_bit))
    canon.sort()
    return str(canon)


def possible_actions(env, random_spawns=True):
    """
    Generate a list of all possible actions given the current environment state.

    The environment tracks existing drones in env.drones and also a maximum
    drone capacity (env.max_drones).

    Actions:
      1. NOOP : Do nothing.
      2. SPAWN_RANDOM : For each available size, if we have not reached env.max_drones,
                        spawn a drone in a random cell (within the environment).
      3. ACT : For each existing drone, either "REMOVE" it or let it "STAY" (do nothing).

    Args:
        env (DroneCoverageEnvAdaptiveLimited): The environment instance.
        random_spawns (bool): Indicates whether we use random spawns for the coverage,
                              though here it is mostly for consistency.

    Returns:
        List[dict]: A list of valid action dictionaries. Each dictionary can have:
            {
              "type": "NOOP"
            }
            or
            {
              "type": "SPAWN_RANDOM",
              "size": <int_size>
            }
            or
            {
              "type": "ACT",
              "drone_index": <int_index>,
              "move": "REMOVE" or "STAY"
            }
    """
    acts = [{"type": "NOOP"}]

    # If we can still add drones, offer spawn actions
    if len(env.drones) < env.max_drones:
        for s in env.available_sizes:
            acts.append({"type": "SPAWN_RANDOM", "size": s})

    # For each existing drone => remove it or let it stay
    for i in range(len(env.drones)):
        acts.append({"type": "ACT", "drone_index": i, "move": "REMOVE"})
        acts.append({"type": "ACT", "drone_index": i, "move": "STAY"})

    return acts


def get_coverage_fraction(env):
    """
    Compute the fraction of free (non-obstacle) cells currently covered by drones.

    This calls the environment's internal _compute_coverage_and_overlap() to
    get coverage_count. The fraction is coverage_count / env.num_free_cells.

    Args:
        env (DroneCoverageEnvAdaptiveLimited): The environment instance.

    Returns:
        float: A coverage fraction in [0.0, 1.0].
    """
    coverage_count, _ = env._compute_coverage_and_overlap()
    if env.num_free_cells > 0:
        return coverage_count / float(env.num_free_cells)
    else:
        return 1.0


def safe_q(Q_table, s, a):
    """
    Safely retrieve or initialize a Q-value from the Q_table.

    If s not in Q_table, we create Q_table[s] = {}
    If a not in Q_table[s], we set Q_table[s][a] = 0.0

    Args:
        Q_table (dict): The Q-table, a nested dict: Q_table[state_str][action_str] = float
        s (str): State string representation (from state_to_str).
        a (str): Action string representation (JSON-like, from dict turned into str).

    Returns:
        float: The existing or newly initialized Q-value for (s,a).
    """
    if s not in Q_table:
        Q_table[s] = {}
    if a not in Q_table[s]:
        Q_table[s][a] = 0.0
    return Q_table[s][a]


def Q_learning_adaptive_limited(config):
    """
    Train a tabular Q-learning agent in the DroneCoverageEnvAdaptiveLimited environment.

    We track coverage fraction, reward, and store the best Q-table
    whenever we find a new maximum coverage fraction.

    Training Loop:
      - For each episode:
        1. Reset environment => get initial state (obs).
        2. Convert obs to s_str (via state_to_str).
        3. While not done:
           a. possible_actions => gather valid actions
           b. choose action (epsilon-greedy)
           c. step environment => next_obs, reward, done
           d. update Q-value using old_q and best_next
           e. move s_str to new state
        4. Decay epsilon
        5. Record coverage fraction, compare and possibly store best Q-table snapshot

    Args:
        config (dict): A dictionary of hyperparameters and environment settings.

    Returns:
        dict: The best Q-table found, keyed by state_str, then action_str => Q-value.
    """
    env = DroneCoverageEnvAdaptiveLimited(config)
    Q_table = {}

    best_Q_table = {}
    best_coverage_fraction = -1.0

    num_episodes = config["num_episodes"]
    gamma = config["gamma_rl"]
    alpha = config["alpha_rl"]
    epsilon = config["epsilon_rl"]
    eps_decay = config["epsilon_decay"]
    eps_min = config["epsilon_min"]
    force_complete = config.get("force_complete_coverage", True)

    ep_rewards = []
    ep_coverages = []
    complete_coverage_count = 0  # How many episodes achieve >= ~100% coverage

    # Track how often the agent visits "[]" => empty state
    empty_visits = 0

    with open("training_output.txt", "w") as log_file:
        for ep in range(num_episodes):
            obs = env.reset()
            s_str = state_to_str(obs)
            done = False
            ep_reward = 0.0
            steps = 0

            # Track coverage improvements in the episode
            max_coverage_this_episode = 0.0

            if s_str == "[]":
                empty_visits += 1

            while not done:
                acts = possible_actions(env, random_spawns=True)

                # Current coverage fraction (for decisions regarding forced coverage)
                coverage_fraction = get_coverage_fraction(env)
                max_coverage_this_episode = max(max_coverage_this_episode, coverage_fraction)

                # If we want to strongly encourage coverage and haven't reached it yet
                if force_complete and coverage_fraction < 0.99 and len(env.drones) < env.max_drones:
                    # 80% chance to spawn if coverage < 99%
                    if random.random() < 0.8:
                        # Filter all spawn actions
                        spawn_actions = [a for a in acts if a["type"] == "SPAWN_RANDOM"]
                        if spawn_actions:
                            # 70% chance to pick a large drone if available
                            large_spawns = [a for a in spawn_actions if a.get("size", 3) >= 5]
                            if large_spawns and random.random() < 0.7:
                                act = random.choice(large_spawns)
                            else:
                                act = random.choice(spawn_actions)
                        else:
                            # fallback to normal epsilon-greedy
                            if random.random() < epsilon:
                                act = random.choice(acts)
                            else:
                                best_val = float("-inf")
                                chosen = None
                                for a in acts:
                                    val = safe_q(Q_table, s_str, str(a))
                                    if val > best_val:
                                        best_val = val
                                        chosen = a
                                act = chosen
                    else:
                        # normal epsilon-greedy
                        if random.random() < epsilon:
                            act = random.choice(acts)
                        else:
                            best_val = float("-inf")
                            chosen = None
                            for a in acts:
                                val = safe_q(Q_table, s_str, str(a))
                                if val > best_val:
                                    best_val = val
                                    chosen = a
                            act = chosen
                else:
                    # standard epsilon-greedy
                    if random.random() < epsilon:
                        act = random.choice(acts)
                    else:
                        best_val = float("-inf")
                        chosen = None
                        for a in acts:
                            val = safe_q(Q_table, s_str, str(a))
                            if val > best_val:
                                best_val = val
                                chosen = a
                        act = chosen

                next_obs, reward, done, info = env.step(act)
                ep_reward += reward

                sp_str = state_to_str(next_obs)
                old_q = safe_q(Q_table, s_str, str(act))

                # If next state is new to Q_table, initialize it
                if sp_str not in Q_table:
                    Q_table[sp_str] = {}

                # Q-learning update
                if not done:
                    nxt_acts = possible_actions(env, random_spawns=True)
                    best_next = float("-inf")
                    for na in nxt_acts:
                        v = safe_q(Q_table, sp_str, str(na))
                        if v > best_next:
                            best_next = v
                    td_target = reward + gamma * best_next
                else:
                    td_target = reward

                new_q = old_q + alpha * (td_target - old_q)
                Q_table[s_str][str(act)] = new_q

                s_str = sp_str
                steps += 1

                if s_str == "[]":
                    empty_visits += 1

                # Check coverage again; if >= 99%, optional early end
                current_coverage = get_coverage_fraction(env)
                if current_coverage >= 0.99:
                    if force_complete:
                        ep_reward += 1000  # reward bonus for success
                        done = True

            # End of episode, decay epsilon if above min
            if epsilon > eps_min:
                epsilon *= eps_decay

            ep_rewards.append(ep_reward)

            # Compute coverage fraction for final state
            coverage_fraction_episode = 0.0
            if env.num_free_cells > 0:
                coverage_fraction_episode = env.previous_coverage / float(env.num_free_cells)
            ep_coverages.append(coverage_fraction_episode)

            # Count it if we achieved near-complete coverage
            if coverage_fraction_episode >= 0.99:
                complete_coverage_count += 1

            # If better coverage => copy entire Q-table
            if coverage_fraction_episode > best_coverage_fraction:
                best_coverage_fraction = coverage_fraction_episode
                best_Q_table.clear()
                for st in Q_table:
                    best_Q_table[st] = {}
                    for ac in Q_table[st]:
                        best_Q_table[st][ac] = Q_table[st][ac]
                print(f"  --> Found new best coverage: {100.0 * coverage_fraction_episode:.1f}%")

            line_str = (f"Episode {ep + 1}/{num_episodes} => "
                        f"steps={steps}, reward={ep_reward:.3f}, "
                        f"coverage={env.previous_coverage}/{env.num_free_cells} "
                        f"({coverage_fraction_episode * 100:.1f}%)")
            print(line_str)
            log_file.write(line_str + "\n")

    # Plot training curves for reward and coverage
    plt.figure(figsize=(10, 5))

    # Subplot 1: Rewards
    plt.subplot(1, 2, 1)
    plt.plot(ep_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()

    # Subplot 2: Coverage Fraction
    plt.subplot(1, 2, 2)
    plt.plot(ep_coverages, label="Coverage Fraction")
    plt.xlabel("Episode")
    plt.ylabel("Coverage Fraction")
    plt.title("Coverage Fraction Over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=100)
    plt.close()

    print(f"\n[INFO] Visited the empty state '[]' {empty_visits} times in training!")
    percentage_100 = (complete_coverage_count / num_episodes) * 100.0
    print(f"[INFO] Achieved >=99% coverage in {complete_coverage_count} out of {num_episodes} episodes "
          f"({percentage_100:.1f}%)")
    print(f"[INFO] Best coverage achieved: {best_coverage_fraction * 100:.1f}%")

    # Return the best Q-table (snapshot) found
    return best_Q_table


def evaluate_policy(Q_table, config):
    """
    Evaluate a Q-table policy in the DroneCoverageEnvAdaptiveLimited environment.
    This function runs in a test_mode scenario, optionally forcing coverage
    to near-complete if needed.

    Steps:
      1. Initialize environment in test mode with a higher max_steps_env.
      2. Spawn multiple drones up front, mostly preferring large spawns.
      3. Follow Q-values greedily to place or remove drones until coverage is near 99%
         or we exhaust steps.
      4. Optionally force coverage completion if we still haven't reached 99% coverage.

    Args:
        Q_table (dict): The Q-table mapping state_str => (action_str => Q-value).
        config (dict): Configuration dictionary, including test_mode toggles.

    Returns:
        tuple: (total_reward, final_observation_dictionary)
    """
    random.seed(1234)
    np.random.seed(1234)

    # Create a local copy of config, forced into test_mode = True
    config = dict(config)
    config["test_mode"] = True
    config["max_steps_env"] = 1000  # Extended max steps for testing

    env = DroneCoverageEnvAdaptiveLimited(config)
    obs = env.reset()
    s_str = state_to_str(obs)

    done = False
    total_r = 0.0
    steps = 0
    max_steps = config["max_steps_env"]

    # We'll consider 99% coverage effectively complete
    target_coverage = 0.99

    # 1. Pre-spawn loop: attempt up to 10 spawns, preferring large drones
    for i in range(10):
        if done:
            break
        acts = possible_actions(env, random_spawns=True)
        spawn_actions = [a for a in acts if a["type"] == "SPAWN_RANDOM"]
        large_spawns = [a for a in spawn_actions if a.get("size", 3) >= 5]

        # If we can spawn large drones, do so randomly;
        # else, pick any spawn action; else pick best Q-value fallback
        if large_spawns:
            act = random.choice(large_spawns)
        elif spawn_actions:
            act = random.choice(spawn_actions)
        else:
            best_val = float("-inf")
            chosen = acts[0]
            for a in acts:
                val = safe_q(Q_table, s_str, str(a))
                if val > best_val:
                    best_val = val
                    chosen = a
            act = chosen

        next_obs, r, done, _ = env.step(act)
        total_r += r
        s_str = state_to_str(next_obs)
        steps += 1

    # 2. Main loop following Q-values
    while not done and steps < max_steps:
        acts = possible_actions(env, random_spawns=True)
        current_coverage = get_coverage_fraction(env)

        # If coverage below target, try more spawns if available
        if current_coverage < target_coverage and len(env.drones) < env.max_drones:
            spawn_actions = [a for a in acts if a["type"] == "SPAWN_RANDOM"]
            if spawn_actions:
                # Prefer large spawns if possible
                large_spawns = [a for a in spawn_actions if a.get("size", 3) >= 5]
                chosen = random.choice(large_spawns) if large_spawns else random.choice(spawn_actions)
            else:
                # Use Q-values if no spawn action is available
                best_val = float("-inf")
                chosen = acts[0]
                for a in acts:
                    val = safe_q(Q_table, s_str, str(a))
                    if val > best_val:
                        best_val = val
                        chosen = a
        else:
            # purely greedy from Q-values
            best_val = float("-inf")
            chosen = acts[0]
            for a in acts:
                val = safe_q(Q_table, s_str, str(a))
                if val > best_val:
                    best_val = val
                    chosen = a

        next_obs, r, done, _ = env.step(chosen)
        total_r += r
        s_str = state_to_str(next_obs)
        steps += 1

        # If we've reached 99% coverage, we can stop
        if get_coverage_fraction(env) >= target_coverage:
            print("[INFO] Achieved complete coverage!")
            break

    # 3. If coverage is still below 99%, force spawns
    final_coverage = get_coverage_fraction(env)
    if final_coverage < target_coverage:
        print("[WARNING] Coverage not complete after standard evaluation, forcing completion...")

        while final_coverage < target_coverage and len(env.drones) < env.max_drones and steps < max_steps:
            # Spawn largest drone type
            size = max(env.available_sizes)
            action = {"type": "SPAWN_RANDOM", "size": size}
            next_obs, r, done, _ = env.step(action)
            total_r += r
            steps += 1

            final_coverage = get_coverage_fraction(env)
            print(f"  Added drone: coverage now {final_coverage * 100:.1f}%")

            if final_coverage >= target_coverage:
                print("[INFO] Achieved complete coverage after forced completion!")
                break

    # Final coverage stats
    print(f"[INFO] Final coverage: {final_coverage * 100:.1f}%")
    print(f"[INFO] Total drones: {len(env.drones)}")
    print(f"[INFO] Steps taken: {steps}")

    return total_r, env._get_observation()
