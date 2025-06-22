from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import copy
import numpy as np
# from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import torch, hydra

# from scripts.reactive_tamp import REACTIVE_TAMP
# from src.m3p2i_aip.config.config_store import ExampleConfig
# import learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
# from sim1 import run_sim1

import random
import time

import os
import json

from torch.utils.tensorboard import SummaryWriter


def load_subgoal_data(npz_file, device):
    data = np.load(npz_file)
    subgoal_data = {
        'obs': torch.tensor(data['obs'], dtype=torch.float32).to(device),
        'goal': torch.tensor(data['goal'], dtype=torch.float32).to(device),
        'subgoal_target': torch.tensor(data['subgoal_target'], dtype=torch.float32).to(device)
    }
    return subgoal_data


class SubgoalDataSampler:
    def __init__(self, subgoal_data):
        self.subgoal_data = subgoal_data
        self.size = len(subgoal_data['obs'])

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        obs_batch = self.subgoal_data['obs'][idxs]
        subgoal_target_batch = self.subgoal_data['subgoal_target'][idxs]

        return obs_batch, subgoal_target_batch


class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

    def clear(self):
        """Clear all trajectories and states"""
        self.pool.clear()
        self.pool_init_state.clear()
        self.counter = 0


class OfflineDataAccumulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }
        if os.path.exists(filepath):
            self.load()

    def load(self):
        loaded = np.load(self.filepath)
        for k in self.data:
            self.data[k] = list(loaded[k])

    def add(self, new_data):
        for k in self.data:
            self.data[k].extend(new_data[k])

    def save(self):
        data_to_save = {k: np.array([np.ravel(item).astype(np.float32) for item in self.data[k]]) for k in self.data}
        np.savez(self.filepath, **data_to_save)


class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.goal_distance = get_goal_distance(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_accumulator = OfflineDataAccumulator("offline_dataset4.npz")

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)

        self.sampler = None
        self.reactive_tamp = None

        # Subgoal learning related parameters
        self.subgoal_dataset = []  # Stores (state, goal, subgoal) tuples for training
        self.subgoal_dataset_capacity = 10000  # Capacity of the subgoal dataset
        self.use_direct_subgoal = True  # Use direct subgoal generation instead of offset
        self.subgoal_hindsight = True  # Use hindsight method to construct subgoal training data

        # Training state tracking
        self.training_state = {"total_episodes": 0}
        self.env = None
        self.env_test = None
        self.env_type = 'simpler'
        self.planner = None
        self.agent = None
        self.buffer = None

        # Return and performance tracking
        self.running_return_history = []  # Record of return history
        self.running_return_avg = 0.0  # Average return
        self.running_loss_history = []  # Record of loss history
        self.running_average_history = []  # Record of additional performance metrics
        self.progress_window_size = 30  # Window size for computing moving average

        # Successful trajectory history
        self.success_history = []  # Stores successful trajectories

        # History of all trajectories (successful or not)
        self.all_trajectories = []  # Stores all historical trajectories
        self.all_trajectories_capacity = 100  # Maximum number of stored trajectories
        self.all_episode_trajectories = []

        # Learning rate and early stopping related
        self.best_return = -np.inf  # Best return value
        self.episodes_since_improvement = 0  # Number of episodes without improvement
        self.early_stop_patience = 100  # Early stopping patience
        self.ema_return = None  # Exponential moving average of return
        self.ema_alpha = 0.1  # EMA coefficient
        self.save_best_model = True  # Whether to save the best model
        self.episodes = args.episodes  # Number of episodes per cycle, will be set in the learn() method
        self.cycles = 0  # Number of update cycles per cycle, will be set in the learn() method

        # Subgoal optimization settings
        self.use_subgoal_network = True  # Whether to use the subgoal network

        # Progress window size and EMA parameter
        self.progress_window_size = 30  # Moving window for recent episodes
        self.ema_alpha = 0.1  # Alpha value for exponential moving average

        # Training phase distinction
        self.advanced_stage = False  # Whether to enter the advanced training stage
        self.min_success_trajectories = 20  # Minimum number of successful trajectories required to enter advanced stage

        self.subgoal_data = load_subgoal_data(self.args.subgoal_data_path, self.device)

        self.global_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }

    def learn(self, args, env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, planner,
              training_state=None, sampler=None):
        """Main learning loop

        Args:
            args: Configuration parameters
            env: Training environment
            env_test: Testing environment
            agent: Agent
            buffer: Replay buffer
            planner: MPPI planner
            training_state: Optional training state dictionary

        Returns:
            tuple: Average return and return delta
        """

        self.initial_goals = []
        self.desired_goals = []
        self.explore_goals = []
        self.achieved_trajectories = []
        self.success_trajectories = []
        self.achieved_rewards = []
        self.episode_return = 0

        self.episode_trajectory = []

        self.env = env
        self.env_test = env_test
        self.agent = agent

        self.bc_diffusion_agent = bc_diffusion_agent
        self.ql_diffusion_agent = ql_diffusion_agent
        self.buffer = buffer
        self.planner = planner

        self.saver = False

        if hasattr(args, 'episodes'):
            self.episodes = args.episodes
        elif hasattr(args, 'n_episodes'):
            self.episodes = args.n_episodes
        else:
            self.episodes = 1

        if hasattr(args, 'n_cycles'):
            self.cycles = args.n_cycles
        else:
            self.cycles = 10

        # Calculate total number of epochs
        if hasattr(args, 'n_epochs'):
            total_epochs = args.n_epochs
        else:
            total_epochs = 1

        # Calculate total number of cycles
        total_cycles = total_epochs * self.cycles

        # Update training state
        if training_state is not None:
            self.training_state = training_state

        # Record return and other info before this cycle, used to evaluate performance at the end
        pre_return_avg = self.running_return_avg
        total_episodes = self.training_state.get("total_episodes", 0)

        # Get current epoch info
        current_epoch = self.training_state.get("current_epoch", 0)
        total_epochs = self.training_state.get("total_epochs", 1)

        # Define training phases based on epoch (20% exploration - 50% training - 50% fine-tuning)
        exploration_epochs = int(total_epochs * 0.2)
        training_epochs = int(total_epochs * 0.5)  # 0.3 + 0.4 = 0.7

        # Determine current training stage
        if current_epoch < exploration_epochs:
            self.stage = "Exploration Stage"
            stage_progress = current_epoch / exploration_epochs
        elif current_epoch < training_epochs:
            self.stage = "Training Stage"
            stage_progress = (current_epoch - exploration_epochs) / (training_epochs - exploration_epochs)
        else:
            self.stage = "Fine-tuning Stage"
            stage_progress = (current_epoch - training_epochs) / (total_epochs - training_epochs)

        # Output current stage and progress
        check_interval = getattr(args, 'check_interval', 10)  # Default: check every 10 episodes
        if total_episodes % check_interval == 0:
            print(
                f"[Epoch {current_epoch + 1}/{total_epochs}] Current stage: {self.stage} (Progress: {stage_progress:.2f})")

        # Determine if in warmup stage - based on epoch
        # is_warmup = current_epoch < max(1, int(total_epochs * 0.25))  # First 25% of epochs as warmup
        is_warmup = False
        self.is_warmup = is_warmup
        if is_warmup and total_episodes % check_interval == 0:
            print(f"[Warmup Stage] Using random actions with noise for exploration")

        # Reset environment and get initial state
        obs = self.env.reset()
        self.prev_position = obs['achieved_goal'].copy()
        goal_a = obs['achieved_goal'].copy()
        goal_d = obs['desired_goal'].copy()
        self.initial_goals.append(goal_a.copy())
        self.desired_goals.append(goal_d.copy())

        self.current = Trajectory(obs)

        all_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }

        # Initialize cycle info
        for episode in range(self.episodes):
            self.training_state["total_episodes"] = total_episodes + episode + 1

            # Get current state and goal state
            achieved_goal = self.env._get_obs()['achieved_goal'].copy()
            desired_goal = self.env._get_obs()['desired_goal'].copy()

            self.use_waypoints = False

            # Generate subgoal
            if is_warmup:
                # In warmup stage, use random subgoal with noise
                subgoal = achieved_goal + np.random.uniform(-0.08, 0.08, size=achieved_goal.shape)
                subgoal = np.clip(subgoal, -1.5, 1.5)
                print(f"[Warmup Stage] Using random subgoal {subgoal}")
            else:
                # Normally generate subgoal
                obs = self.env._get_obs()
                subgoal = self.generate_subgoal(obs, pretrain=is_warmup)

            # Execute one episode
            timesteps = getattr(args, 'episode_duration', 50)  # Default 50 steps
            current, episode_reward, trajectory, final_distance, success_reached = self.rollout(timesteps,
                                                                                                subgoal=subgoal)
            self.explore_goals.append(subgoal)
            self.episode_trajectory.append(trajectory)

            self.episode_return += episode_reward

            # Record return
            self.record_return(episode_reward, episode)

            # Save to buffer
            ep_obs = current.ep['obs']  # list of dicts: [{'observation':..., 'achieved_goal':..., 'desired_goal':...}]
            ep_actions = current.ep['hgg_acts']
            ep_rewards = current.ep['rews']
            ep_dones = current.ep['done']

            for t in range(len(ep_obs) - 1):
                obs_t = np.concatenate([
                    ep_obs[t]['observation'].squeeze(),
                    ep_obs[t]['desired_goal'].squeeze()
                ], axis=0)
                next_obs_t = np.concatenate([
                    ep_obs[t + 1]['observation'].squeeze(),
                    ep_obs[t + 1]['desired_goal'].squeeze()
                ], axis=0)

                all_data['observations'].append(obs_t)
                all_data['actions'].append(ep_actions[t])
                all_data['rewards'].append(ep_rewards[t])
                all_data['terminals'].append(ep_dones[t])
                all_data['next_observations'].append(next_obs_t)

            if success_reached:
                self.saver = True
                break

        # Add trajectory to buffer
        self.buffer.store_trajectory(current)

        self.update_network()

        # Save buffer
        self.data_accumulator.add(all_data)
        self.data_accumulator.save()

        # final_trajectory is assumed to be collected in a single rollout
        final_trajectory = np.concatenate(self.episode_trajectory)

        self.achieved_trajectories.append(final_trajectory)
        self.all_episode_trajectories.append(final_trajectory)

        # Add trajectory to success history if close to goal
        final_distance = np.linalg.norm(final_trajectory[-1] - obs['desired_goal'])
        if final_distance < 0.05:  # Considered successful if close to goal
            self.success_history.append(final_trajectory)
            self.success_trajectories.append(final_trajectory)
            print(f"[Training Stage] Collected {len(self.success_history)} successful trajectories.")
            # Limit history size
            if len(self.success_history) > 200:
                self.success_history.pop(0)

            # Check if we should enter the advanced training stage
            if len(self.success_history) >= self.min_success_trajectories and not self.advanced_stage:
                self.advanced_stage = True
                print(
                    f"[Training Stage] Collected {len(self.success_history)} successful trajectories. Entering advanced training stage!")

        # End of cycle: update path pool with sampled subgoals
        if hasattr(args, 'save_acc') and args.save_acc:
            self.update_path_pool()

        # End of cycle: update training statistics
        self.training_state["episodes"] = self.training_state.get("episodes", 0) + self.episodes
        return_delta = self.running_return_avg - pre_return_avg

        # Print training result of this cycle
        print(
            f"\nEnd of cycle: Return change: {return_delta:.4f}, Current average return: {self.running_return_avg:.4f}")

        # Compute recent success rate
        recent_success_rate = np.mean([1 if ret > 0 else 0 for ret in self.running_return_history[-20:]])
        print(f"Recent success rate: {recent_success_rate:.2f}")

        # Return current training results
        return self.running_return_avg, return_delta

    # Add function to train the subgoal generation network
    def train_subgoal_network(self):
        """Train the subgoal generation network

        Two-stage training strategy:
        1. Early stage: Sample from all historical trajectories, select points closer to the goal as subgoals
        2. Later stage: Use value function to optimize after enough successful trajectories are collected
        """
        if not hasattr(self.agent, 'use_direct_subgoal') or not self.agent.use_direct_subgoal:
            return

        # Check whether we are in the advanced stage (sufficient successful trajectories)
        condition = len(self.all_episode_trajectories) > 400 or len(
            self.success_history) >= self.min_success_trajectories
        advanced_stage = condition and not self.is_warmup

        if advanced_stage and not self.advanced_stage:
            self.advanced_stage = True
            print(f"[Subgoal Training] Entering advanced training stage!")

        if advanced_stage:
            print(
                f"[Subgoal Training] In advanced stage, using {len(self.success_history)} successful trajectories for training")
        else:
            print(
                f"[Subgoal Training] In basic stage, sampling subgoals from {len(self.all_episode_trajectories)} historical trajectories")

        # Construct subgoal training data dictionary
        subgoal_data = {
            'obs': [],  # Current state
            'goal': [],  # Final goal
            'subgoal_target': []  # Target subgoal
        }

        # Early stage: Sample from all historical trajectories
        if not advanced_stage and len(self.all_episode_trajectories) > 0:
            valid_samples = 0
            desired_goal = self.env._get_obs()['desired_goal'].copy()
            trajectories = self.all_episode_trajectories

            # Select top 50 trajectories closest to the desired_goal
            if len(trajectories) > 150:
                distances = []
                for traj in trajectories:
                    if len(traj) < 8:
                        distances.append(np.inf)
                    else:
                        traj_end = traj[-1]
                        dist = np.linalg.norm(traj_end - desired_goal)
                        distances.append(dist)
                top_indices = np.argsort(distances)[:50]
                selected_trajectories = [trajectories[i] for i in top_indices]
            else:
                selected_trajectories = trajectories

            for traj_data in selected_trajectories:
                traj_obs = traj_data
                traj_length = len(traj_obs)

                if traj_length < 5:
                    continue

                for i in range(0, traj_length - 6, 2):
                    current_state = traj_obs[i].copy()
                    final_goal = self.env._get_obs()['desired_goal'].copy()
                    current_to_goal_dist = np.linalg.norm(current_state - final_goal)

                    for j in range(1, min(6, traj_length - i)):
                        future_idx = i + j
                        future_state = traj_obs[future_idx].copy()
                        future_to_goal_dist = np.linalg.norm(future_state - final_goal)

                        if 0.02 < current_to_goal_dist - future_to_goal_dist:
                            full_observation = np.zeros(13)
                            full_observation[:3] = current_state
                            complete_obs = np.concatenate([full_observation, final_goal])

                            subgoal_data['obs'].append(complete_obs)
                            subgoal_data['goal'].append(final_goal)
                            subgoal_data['subgoal_target'].append(future_state)
                            valid_samples += 1
                            break

            print(f"[Subgoal Training] Extracted {valid_samples} valid subgoal samples from historical trajectories")

        # Advanced stage: Sample from successful trajectories
        if advanced_stage:
            n_samples = min(200, len(self.success_history))
            if n_samples > 0:
                sampled_trajs = random.sample(self.success_history, n_samples)
                success_samples = 0

                for traj in sampled_trajs:
                    traj_length = len(traj)
                    if traj_length < 5:
                        continue

                    for i in range(0, traj_length - 10, 2):
                        current_state = traj[i].copy()
                        final_goal = self.env._get_obs()['desired_goal'].copy()
                        current_to_goal_dist = np.linalg.norm(current_state - final_goal)
                        max_j = min(20, traj_length - i)

                        if max_j > 0:
                            for j in reversed(range(1, max_j)):
                                future_idx = i + j
                                future_state = traj[future_idx].copy()
                                future_to_goal_dist = np.linalg.norm(future_state - final_goal)

                                if 0.04 < current_to_goal_dist - future_to_goal_dist < 0.07:
                                    subgoal_target = future_state

                                    full_observation = np.zeros(13)
                                    full_observation[:3] = current_state
                                    complete_obs = np.concatenate([full_observation, final_goal])

                                    subgoal_data['obs'].append(complete_obs)
                                    subgoal_data['goal'].append(final_goal)
                                    subgoal_data['subgoal_target'].append(subgoal_target)
                                    success_samples += 1
                                    break

                print(f"[Subgoal Training] Extracted {success_samples} subgoal samples from successful trajectories")

        # Check if there is enough training data
        if len(subgoal_data['obs']) < 10:
            print("[Subgoal Training] Not enough valid training data, skipping training")
            return

        # Convert to numpy arrays
        for key in subgoal_data:
            subgoal_data[key] = np.array(subgoal_data[key])

        print(f"[Subgoal Training] Prepared {len(subgoal_data['obs'])} training samples in total")

        if len(subgoal_data['obs']) > 3000:
            # Train using only the most recent N samples
            recent_N = 1500
            batch_size = min(64, recent_N)
            start_idx = max(0, len(subgoal_data['obs']) - recent_N)
            recent_range = np.arange(start_idx, len(subgoal_data['obs']))
            n_batches = len(recent_range) // batch_size

            total_loss = 0
            for _ in range(n_batches):
                idxs = np.random.choice(recent_range, batch_size, replace=False)
                batch = {
                    'obs': subgoal_data['obs'][idxs],
                    'goal': subgoal_data['goal'][idxs],
                    'subgoal_target': subgoal_data['subgoal_target'][idxs]
                }

                loss = self.agent.train_subgoal(batch)
                if loss is not None:
                    total_loss += loss
        else:
            batch_size = min(64, len(subgoal_data['obs']))
            n_batches = len(subgoal_data['obs']) // batch_size

            total_loss = 0
            for _ in range(n_batches):
                idxs = np.random.randint(0, len(subgoal_data['obs']), batch_size)
                batch = {
                    'obs': subgoal_data['obs'][idxs],
                    'goal': subgoal_data['goal'][idxs],
                    'subgoal_target': subgoal_data['subgoal_target'][idxs]
                }

                loss = self.agent.train_subgoal(batch)
                if loss is not None:
                    total_loss += loss

        avg_loss = total_loss / max(1, n_batches)
        print(f"[Subgoal Training] Training complete, average loss: {avg_loss:.4f}")

        return {"subgoal_loss": avg_loss}

    def update_network(self):
        # Get current training stage and subgoal parameters

        # if self.advanced_stage == True:
        # Sample from the replay buffer
        # print("self.args.batch_size", self.args.batch_size)
        transitions = self.buffer.sample_batch(self.args.batch_size)
        # print("transitions", transitions)

        # Check if transitions is None
        if transitions is None:
            print("Warning: Failed to sample from buffer, skipping this update")
            return

        # # Directly load subgoal data
        # self.subgoal_data = load_subgoal_data(self.args.subgoal_data_path, self.device)
        # data_sampler = SubgoalDataSampler(self.subgoal_data)

        # Train control policy network
        # info = self.agent.train(transitions)
        # Instantiate writer
        writer = SummaryWriter(log_dir='logs/ql_diffusion')

        for _ in range(100):
            # if self.stage in ["Training Stage", "Fine-tuning Stage"]:
            transitions = self.buffer.sample_batch(self.args.batch_size)
            info = self.ql_diffusion_agent.train(transitions, iterations=100, batch_size=100, log_writer=writer)

        # If subgoal training is enabled, train subgoal network
        # if hasattr(self, 'use_subgoal_network') and self.use_subgoal_network:
        #     subgoal_info = self.train_subgoal_network()
        #     if subgoal_info is not None:
        #         for k, v in subgoal_info.items():
        #             info[k] = v

        # Update training state counter
        if 'total_episodes' not in self.training_state:
            self.training_state['total_episodes'] = 0
        self.training_state['total_episodes'] += 1

        # Compute average accumulated reward
        check_interval = getattr(self.args, 'check_interval', 10)  # Default value is 10
        # if self.training_state.get("total_episodes", 0) % check_interval == 0:
        #     self.calculate_running_avg_return()

    def generate_subgoal(self, obs, pretrain=False):
        """Generate subgoal (directly generate a full subgoal, not an offset)

        Multi-stage strategy:
        1. Early stage: Use simple subgoal generation
        2. Advanced stage: Use value-function-based subgoal optimization

        Args:
            achieved_goal: Current state
            desired_goal: Target state
            pretrain: Whether it is in pretraining stage

        Returns:
            subgoal: Generated subgoal
        """

        # QL_diffusion
        # state = np.concatenate((obs['observation'], obs['desired_goal'].reshape(1, -1)), axis=1)
        # optimized_subgoal = self.ql_diffusion_agent.sample_action(np.array(state))
        # print("Subgoal generated by ql_diffusion model", optimized_subgoal)
        # return optimized_subgoal

        # bc_diffusion
        full_observation = np.zeros(13)
        full_observation[:3] = obs['achieved_goal']

        observation = torch.tensor(np.concatenate([
            np.array(full_observation).reshape(-1),
            np.array(obs['desired_goal']).reshape(-1)
        ]), dtype=torch.float32)

        # Convert state to tensor
        if not isinstance(observation, torch.Tensor):
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation_tensor = observation.to(self.device)

        # Ensure correct input shape
        if observation_tensor.dim() == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        # optimized_subgoal = self.bc_diffusion_agent.sample_action(np.array(observation_tensor))
        optimized_subgoal = self.bc_diffusion_agent.sample_action(observation_tensor)

        print("Subgoal generated by bc_diffusion model", optimized_subgoal)

        return optimized_subgoal

    def rollout(self, timesteps, subgoal=None):
        """Execute a subgoal-driven trajectory

        Args:
            timesteps: Maximum number of steps
            subgoal: Optional subgoal

        Returns:
            episode_experience: Collected trajectory experience
            episode_reward: Accumulated reward
        """

        success_reached = False
        self.env.goal = subgoal
        # Get current observation
        obs = self.env._get_obs()  # Get current observation from environment
        # current = Trajectory(obs)  # Create trajectory object
        trajectory = [obs['achieved_goal'].copy()]  # Record position trajectory

        episode_reward = 0  # Accumulated reward

        # If subgoal is provided, set it in the environment
        if subgoal is not None:
            self.env.subgoal = torch.tensor(subgoal, dtype=torch.float32)

        # Record initial position and goal
        initial_position = obs['achieved_goal'].copy()
        desired_goal = obs['desired_goal'].copy()
        direct_distance_to_goal = np.linalg.norm(initial_position - desired_goal)

        # Track path length and efficiency
        total_path_length = 0.0
        # prev_position = initial_position

        # Execute for the given number of steps
        for t in range(timesteps):
            # Get current achieved goal
            achieved_goal = obs['achieved_goal'].copy()

            # Generate action using MPPI planner
            action_mppi = bytes_to_torch(
                self.planner.run_tamp(
                    torch_to_bytes(self.env._dof_state),
                    torch_to_bytes(self.env._root_state),
                    subgoal.tolist() if subgoal is not None else desired_goal.tolist())
            )

            # Execute the action
            obs, reward, done, info, distance, dis_subgoal = self.env.step(action_mppi)

            current_pos = obs['achieved_goal'].copy()
            curr_distance = np.linalg.norm(current_pos - obs['desired_goal'].copy())

            # 1. Subgoal distance reward
            curr_subgoal_distance = np.linalg.norm(subgoal - obs['desired_goal'].copy())
            reward = curr_subgoal_distance * (-1)

            # 2. Bonus reward
            if distance < 0.05:
                print(f"Goal reached, terminated at step {t}, distance {distance:.4f}")

                # Extra reward for successfully reaching the goal
                success_bonus = 10
                reward += success_bonus
                print(f"Success bonus added: {success_bonus}")
                success_reached = True

            # Compute step size and total path length
            step_distance = np.linalg.norm(current_pos - self.prev_position)
            total_path_length += step_distance
            self.prev_position = current_pos

            episode_reward += reward

            if subgoal is not None and isinstance(subgoal, np.ndarray):
                subgoal = torch.tensor(subgoal, dtype=torch.float32)

            # Store the current step in the trajectory
            self.current.store_step(action_mppi, obs, reward, done, subgoal)
            trajectory.append(current_pos)

            if distance < 0.05:
                break

            if dis_subgoal < 0.005:
                print("----------------------Subgoal reached-----------------", subgoal)
                break

        # Compute final path efficiency metric
        final_direct_distance = np.linalg.norm(trajectory[-1] - initial_position)
        final_efficiency = final_direct_distance / (total_path_length + 1e-6)

        print(
            f"Trajectory completed: path length = {total_path_length:.4f}, direct distance = {final_direct_distance:.4f}, efficiency = {final_efficiency:.4f}")

        # Create trajectory data dictionary
        trajectory_data = {
            'obs': self.current.ep['obs'],
            'path': np.array(trajectory),
            'efficiency': final_efficiency,
            'path_length': total_path_length,
            'reward': episode_reward,
            'success': False
        }

        # Add to success history if close to the goal
        final_distance = np.linalg.norm(trajectory[-1] - desired_goal)

        # Save to all trajectory history regardless of success
        self.all_trajectories.append(trajectory_data)
        # Limit history size
        if len(self.all_trajectories) > self.all_trajectories_capacity:
            self.all_trajectories.pop(0)

        return self.current, episode_reward, trajectory, final_distance, success_reached

    def record_return(self, episode_reward, episode_idx):
        """Record and update return statistics

        Args:
            episode_reward: Accumulated reward of the current episode
            episode_idx: Index of the episode
        """
        # Add to return history
        self.running_return_history.append(episode_reward)
        if len(self.running_return_history) > self.progress_window_size:
            self.running_return_history.pop(0)

        # Compute moving average
        self.running_return_avg = np.mean(self.running_return_history)

        # Update exponential moving average
        if self.ema_return is None:
            self.ema_return = episode_reward
        else:
            self.ema_return = self.ema_alpha * episode_reward + (1 - self.ema_alpha) * self.ema_return

        # Update best return value
        if self.running_return_avg > self.best_return:
            self.best_return = self.running_return_avg
            self.episodes_since_improvement = 0

            # Save best model
            # if episode_idx > 20 and hasattr(self, 'save_best_model') and self.save_best_model:
            #     try:
            #         import os
            #         os.makedirs("saved_models", exist_ok=True)
            #         torch.save(self.agent.subgoal_network.state_dict(), "saved_models/subgoal_network.pth")
            #         torch.save(self.agent.policy.state_dict(), "saved_models/best_policy.pth")
            #         torch.save(self.agent.critic.state_dict(), "saved_models/best_critic.pth")
            #         print(f"New best model saved, average return: {self.best_return:.4f}")
            #     except Exception as e:
            #         print(f"Error saving model: {e}")
        else:
            self.episodes_since_improvement += 1

    def update_path_pool(self):
        """Update the path pool by storing successful trajectories"""
        # Extract trajectories from success_history and add to the path pool
        if not hasattr(self, 'success_history') or not self.success_history:
            return

        # Sample trajectories from success history and add to the path pool
        for success_traj in self.success_history[-10:]:  # Add the most recent 10 successful trajectories
            if 'path' in success_traj:
                self.achieved_trajectory_pool.insert(
                    success_traj['path'],
                    success_traj['obs'][0]['observation'] if len(success_traj['obs']) > 0 else None
                )

        print(f"Path pool updated, current size: {self.achieved_trajectory_pool.counter}")

    def evaluate_subgoal_value(self, obs, subgoal, final_goal):
        """
        Inputs:
            obs: [13] numpy array, current observation of the robot
            subgoal: [3] numpy array, candidate subgoal

        Output:
            Q value given by the critic
        """
        device = next(self.agent.critic.parameters()).device

        if isinstance(final_goal, np.ndarray):
            final_goal = torch.tensor(final_goal, dtype=torch.float32, device=device)
        if final_goal.ndim == 1:
            final_goal = final_goal.unsqueeze(0)

        # Construct 16-dimensional state vector from obs + goal
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 13]
        subgoal = torch.tensor(subgoal, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3]

        if obs.dim() == 3:
            obs = obs.view(obs.size(0), -1)

        # state vector: [obs, final_goal]
        state_final_goal = torch.cat([obs, final_goal], dim=1)
        state = state_final_goal
        # state = torch.cat([state_final_goal, subgoal], dim=1)

        # Preprocess the state (e.g., normalization)
        state = self.agent._state_preprocessor(state, train=True)

        # Use actor policy to generate action for the subgoal
        # action, _, _ = self.agent.policy.act({"states": state.T}, role="policy")
        action = subgoal

        # Use critic network to evaluate Q value
        q1, _, _ = self.agent.critic.act({"states": state.T, "taken_actions": action}, role="critic")
        q2, _, _ = self.agent.critic2.act({"states": state.T, "taken_actions": action}, role="critic")

        # Take the minimum Q value (to prevent overestimation)
        value = torch.min(q1, q2)

        return value.item()

    def optimize_subgoal_with_noise(self, current_state, predicted_subgoal, final_goal, n_samples=15,
                                    noise_scale=0.008):
        """Perform noisy search around predicted subgoal to find a locally optimal one

        Args:
            current_state: Current state
            predicted_subgoal: Initially predicted subgoal by the model
            final_goal: Final goal
            n_samples: Number of noisy samples
            noise_scale: Noise magnitude

        Returns:
            optimized_subgoal: The optimized subgoal
        """

        candidates = []
        values = []

        # Generate noisy candidate subgoals
        for i in range(len(predicted_subgoal)):
            value = self.evaluate_subgoal_value(current_state, predicted_subgoal[i], final_goal)

            candidates.append(predicted_subgoal[i])
            values.append(value)

        # Find the candidate subgoal with highest value
        best_idx = np.argmax(values)
        best_subgoal = candidates[best_idx]
        best_value = values[best_idx]

        return best_subgoal



