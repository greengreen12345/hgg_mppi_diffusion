from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
from packaging import version

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from algorithm.replay_buffer import goal_based_process

from utils.tf_utils import Normalizer_torch


# Add function for generating action distributions
def create_action_distributions(actions, log_std=-2.0, min_log_std=-20, max_log_std=2):
    """Create Gaussian distribution for actions to compute entropy and log_prob"""
    log_std = torch.ones_like(actions) * log_std
    log_std = torch.clamp(log_std, min_log_std, max_log_std)

    # Use reparameterization trick to construct distribution
    distribution = torch.distributions.Normal(actions, log_std.exp())

    return distribution, log_std


def compute_log_prob(distribution, actions):
    """Compute log-probability density of actions"""
    log_prob = distribution.log_prob(actions)
    # Sum over all action dimensions
    log_prob = log_prob.sum(dim=-1, keepdim=True)
    return log_prob


# Add function for computing n-step returns
def compute_n_step_returns(rewards, next_q_values, terminated, gamma, n_steps=3):
    """Compute n-step returns to accelerate convergence"""
    returns = next_q_values.clone()
    for i in reversed(range(rewards.shape[0])):
        if terminated[i]:
            returns[i] = 0.0  # No future return for terminal state
        returns[i] = rewards[i] + gamma * returns[i]
    return returns


# Add runtime state normalizer
class RunningStatsNorm:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # Reinitialize statistics if shapes mismatch
        if batch_mean.shape != self.mean.shape:
            print(f"Warning: shape mismatch, reinitializing. Original shape: {self.mean.shape}, New shape: {batch_mean.shape}")
            self.mean = np.zeros_like(batch_mean, dtype=np.float32)
            self.var = np.ones_like(batch_mean, dtype=np.float32)
            self.count = 1e-4

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        # Ensure shape compatibility
        if x.shape != self.mean.shape and len(x.shape) == 1 and len(self.mean.shape) == 1:
            if x.shape[0] > self.mean.shape[0]:
                # Truncate input if its dimension exceeds statistics
                print(f"Warning: input shape ({x.shape}) exceeds stats shape ({self.mean.shape}), truncating")
                x = x[:self.mean.shape[0]]
            else:
                # Zero pad input if its dimension is smaller
                print(f"Warning: input shape ({x.shape}) less than stats shape ({self.mean.shape}), padding")
                padded_x = np.zeros_like(self.mean)
                padded_x[:x.shape[0]] = x
                x = padded_x

        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# fmt: off
# [start-config-dict-torch]
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,  # Number of gradient steps
    "batch_size": 256,  # Increase batch size to 256 for better stability

    "discount_factor": 0.99,  # Standard discount factor of 0.99
    "polyak": 0.005,  # Increase to 0.005 to accelerate target network updates

    "actor_learning_rate": 3e-4,  # Adjust actor learning rate to a common 3e-4
    "critic_learning_rate": 3e-4,  # Adjust critic learning rate to a common 3e-4
    "learning_rate_scheduler": None,  # Learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {"patience": 500, "factor": 0.7, "threshold": 0.01},  # Scheduler parameters

    "state_preprocessor": None,  # State preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},  # Preprocessor kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 5000,  # Significantly increase random exploration steps to 5000
    "learning_starts": 5000,  # Increase learning start steps to 5000

    "grad_norm_clip": 1.0,  # Adjust gradient clipping threshold

    "exploration": {
        "noise": None,  # Exploration noise
        "initial_scale": 0.5,  # Increase initial noise scale
        "final_scale": 0.05,  # Increase final noise scale
        "timesteps": 100000,  # Increase decay duration
    },

    "rewards_shaper": None,  # Rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,  # Enable automatic mixed precision for better performance

    # Add advanced configurations
    "advanced": {
        "n_step_returns": 1,  # Use standard single-step returns
        "use_entropy_reg": False,  # Disable entropy regularization
        "entropy_coef": 0.01,  # Entropy regularization coefficient
        "use_prioritized_replay": True,  # Keep prioritized experience replay
        "use_residual_connections": True,  # Add residual connections
        "use_layer_normalization": True,  # Add layer normalization
        "use_huber_loss": True,  # Use Huber loss
        "huber_delta": 10.0,  # Delta value for Huber loss
        "critic_l2_reg": 1e-4,  # L2 regularization for critic
        "use_recurrent": False,  # Whether to use recurrent networks
        "use_noisy_networks": True,  # Use NoisyNet for parameter noise exploration
        "use_direct_path": True,  # Whether to use direct path
        "use_prediction": True,  # Whether to use predictive action generation
        "prediction_horizon": 5,  # Prediction steps
    },

    "experiment": {
        "directory": "",  # Parent directory of the experiment
        "experiment_name": "",  # Experiment name
        "write_interval": "auto",  # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",  # Interval for checkpoints (timesteps)
        "store_separately": False,  # Whether to store checkpoints separately

        "wandb": False,  # Whether to use Weights & Biases
        "wandb_kwargs": {}  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}


# [end-config-dict-torch]
# fmt: on


class DDPG(Agent):
    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
            device: Optional[Union[str, torch.device]] = None,
            cfg: Optional[dict] = None,
    ) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)
        _cfg.update(vars(cfg) if cfg is not None else {})

        self.args = _cfg

        print("models：", models)

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)
        self.critic2 = self.models.get("critic2", self.critic)  # Use second critic network or reuse the existing one
        self.target_critic2 = self.models.get("target_critic2", self.target_critic)  # Second target critic network

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic
        self.checkpoint_modules["critic2"] = self.critic2
        self.checkpoint_modules["target_critic2"] = self.target_critic2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        if self.target_policy is not None and self.target_critic is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic.update_parameters(self.critic, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),
                                                     lr=self._actor_learning_rate,
                                                     weight_decay=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                     lr=self._critic_learning_rate,
                                                     weight_decay=1e-5)
            self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(),
                                                      lr=self._critic_learning_rate,
                                                      weight_decay=1e-5)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic2_scheduler = self._learning_rate_scheduler(
                    self.critic2_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer
            self.checkpoint_modules["critic2_optimizer"] = self.critic2_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # Normalizer
        self.obs_normalizer = Normalizer_torch(16, self.device)

        # Add state normalizer
        if observation_space is not None:
            if isinstance(observation_space, gymnasium.Space):
                state_dim = observation_space.shape[0]
            else:
                state_dim = observation_space
            self.state_normalizer = RunningStatsNorm(shape=state_dim)
        else:
            self.state_normalizer = None

        # Add update counter
        self.update_counter = 0

        # Initialize subgoal network
        self.use_direct_subgoal = True  # Default to using direct offset
        if True:
            self.use_direct_subgoal = True
            print("Initializing subgoal generation network...")

            # Compute correct input dimension: observation_space + goal_space
            if hasattr(cfg, 'goal_dims') and cfg.goal_dims:
                goal_dims = cfg.goal_dims[0] if isinstance(cfg.goal_dims, list) else cfg.goal_dims
            else:
                goal_dims = 3  # Goal position is usually 3D (x, y, z)

            # Compute correct input dimension
            subgoal_input_dim = 16  # 13 + 3 = 16

            # Subgoal network - directly output subgoal position instead of offset
            self.subgoal_network = nn.Sequential(
                nn.Linear(subgoal_input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, goal_dims)  # Output subgoal of goal_dims
            ).to(self.device)

            # Subgoal network optimizer
            self.subgoal_optimizer = torch.optim.Adam(
                self.subgoal_network.parameters(),
                lr=self._actor_learning_rate,
                weight_decay=1e-5
            )

            # Subgoal learning hyperparameters
            self.subgoal_loss_weight = 1.0
            self.subgoal_l2_reg = 0.001
            self.subgoal_batch_size = 128

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

        # Reset training step counter
        self.timestep = 0

        # Ensure target networks match the main networks (hard update)
        if self.target_policy is not None and self.target_critic is not None:
            self.target_policy.update_parameters(self.policy, polyak=1.0)
            self.target_critic.update_parameters(self.critic, polyak=1.0)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # Sample random actions
        if self.timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # Sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

            # Add log_prob computation for SAC-style entropy regularization
            if self.args['advanced'].get('use_entropy_reg', False):
                # Create action distribution
                distribution, _ = create_action_distributions(actions, log_std=-2.0)  # Use smaller std
                # Compute log_prob
                log_prob = compute_log_prob(distribution, actions)
                # Update outputs
                outputs["log_prob"] = log_prob

        # Add exploration noise
        if self._exploration_noise is not None:
            # Sample noise
            noises = self._exploration_noise.sample(actions.shape)

            # Define exploration decay
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # Apply decaying noise
            if self.timestep <= self._exploration_timesteps:
                scale = (1 - self.timestep / self._exploration_timesteps) * (
                        self._exploration_initial_scale - self._exploration_final_scale
                ) + self._exploration_final_scale
                noises.mul_(scale)

                # Modify actions
                actions.add_(noises)
                actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # Log noise stats
                self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

            else:
                # Log no noise stats
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)

        self.timestep = self.timestep + 1

        return actions, outputs.get("log_prob", None), outputs

    def step(self, states: torch.Tensor, explore=False, goal_based=False):
        """Select actions based on current state

        Args:
            states: Current state
            explore: Whether to add exploration noise
            goal_based: Whether to use goal-based state

        Returns:
            actions: Action (or subgoal, depending on use_direct_subgoal)
        """

        obs = states  # More semantically clear

        # Use direct subgoal generation
        with torch.no_grad():
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

            # Ensure input has correct dimensions
            if observation_tensor.dim() == 1:
                observation_tensor = observation_tensor.unsqueeze(0)

            # Input subgoal network to directly generate subgoal position
            subgoal = self.subgoal_network(observation_tensor)

            # Remove batch dimension if necessary
            if subgoal.dim() > 1 and subgoal.shape[0] == 1:
                subgoal = subgoal.squeeze(0)

            # Adjust the range according to actual environment
            goal_range = 1.5

            if explore:
                # Add exploration noise
                subgoal_np = subgoal.cpu().numpy()
                # Add smaller noise to subgoal
                noise = np.random.normal(0, 0.02, size=subgoal_np.shape)
                subgoal_np = subgoal_np + noise
                subgoal = torch.tensor(subgoal_np, device=self.device)

            # Ensure subgoal is within a reasonable range
            subgoal = torch.clamp(subgoal, -goal_range, goal_range)

            return subgoal

    def record_transition(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            terminated: torch.Tensor,
            truncated: torch.Tensor,
            infos: Any,
            timestep: int,
            timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Total number of timesteps
        :type timesteps: int
        """

        # Sample batch data from replay buffer
        batch = self.args.buffer.sample_batch_ddpg()
        if batch is None:
            return  # Not enough samples for training

        # Use the new training method to handle batch data, training both control actions and HGG actions
        metrics = self.train(batch)

        # Update target networks using soft update
        self.target_policy.update_parameters(self.policy, polyak=self._polyak)
        self.target_critic.update_parameters(self.critic, polyak=self._polyak)
        if self.target_critic2 is not None and self.critic2 is not None:
            self.target_critic2.update_parameters(self.critic2, polyak=self._polyak)

        # Log training metrics
        for key, value in metrics.items():
            self.track_data(f"Loss / {key}", value)

    def target_update(self):

        # soft update
        self.target_policy.update_parameters(self.policy, polyak=self._polyak)
        self.target_critic.update_parameters(self.critic, polyak=self._polyak)

    def normalizer_update(self, batch):

        self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def train(self, batch):
        if not batch or 'obs' not in batch or len(batch['obs']) == 0:
            print("Warning: empty batch, skipping training")
            return {}

        # Add batch size check
        if len(batch['obs']) < self._batch_size / 4:
            print(f"Warning: batch size too small ({len(batch['obs'])})")

        # Update normalizer
        if hasattr(self, 's_norm'):
            self.normalizer_update(batch)

        device = next(self.policy.parameters()).device

        # Unpack batch data
        states = batch["obs"]
        actions = batch["acts"]  # Control actions (MPPI)
        hgg_actions = batch["hgg_acts"]  # HGG-generated actions used for subgoal generation
        next_states = batch["obs_next"]
        rewards = torch.tensor(batch["rews"], dtype=torch.float32).to(device).view(-1)
        terminated = torch.tensor(batch["done"], dtype=torch.bool).to(device).view(-1)

        # Check whether prioritized replay is used and whether weights are valid
        use_prioritized = ("weights" in batch and "tree_idxs" in batch and
                           len(batch["weights"]) > 0 and len(batch["tree_idxs"]) > 0)

        # Set weights and tree indices
        if use_prioritized:
            tree_idxs = batch["tree_idxs"]
            weights = torch.tensor(batch["weights"], dtype=torch.float32).to(device).view(-1)
        else:
            weights = torch.ones_like(rewards)
            tree_idxs = None

        if isinstance(next_states, list):
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
        elif isinstance(next_states, np.ndarray):
            states = torch.from_numpy(states).float()
            next_states = torch.from_numpy(next_states).float()

        states = states.to(device)
        next_states = next_states.to(device)

        if isinstance(terminated, list):
            terminated = torch.tensor(terminated, dtype=torch.bool)
        elif isinstance(terminated, np.ndarray):
            terminated = torch.from_numpy(terminated).bool()
        terminated = terminated.to(device)

        # Process control actions
        if isinstance(actions, list):
            if isinstance(actions[0], torch.Tensor):
                actions = torch.stack(actions)
            else:
                actions = torch.tensor(actions, dtype=torch.float32)
        actions = actions.to(device)

        # Process HGG actions
        if isinstance(hgg_actions, list):
            if isinstance(hgg_actions[0], torch.Tensor):
                hgg_actions = torch.stack(hgg_actions)
            else:
                hgg_actions = torch.tensor(hgg_actions, dtype=torch.float32)
        hgg_actions = hgg_actions.to(device)

        # Process states
        states = self._state_preprocessor(states, train=True)
        next_states = self._state_preprocessor(next_states, train=True)

        # Add Gaussian noise to states to improve generalization
        if hasattr(self.args, 'advanced') and self.args['advanced'].get('use_noisy_networks', False):
            state_noise = torch.randn_like(states) * 0.01
            next_state_noise = torch.randn_like(next_states) * 0.01
            states_noisy = states + state_noise
            next_states_noisy = next_states + next_state_noise
        else:
            states_noisy = states
            next_states_noisy = next_states

        # Compute target values using target networks
        with torch.no_grad():
            # Get next actions from target policy
            next_actions, next_actions_log_prob, _ = self.target_policy.act({"states": next_states_noisy.T},
                                                                            role="target_policy")

            # Compute Q-values using target critic 1
            target_q_values1, _, _ = self.target_critic.act(
                {"states": next_states_noisy.T, "taken_actions": next_actions}, role="target_critic"
            )
            target_q_values1 = target_q_values1.view(-1)

            # Compute Q-values using target critic 2
            target_q_values2, _, _ = self.target_critic2.act(
                {"states": next_states_noisy.T, "taken_actions": next_actions}, role="target_critic"
            )
            target_q_values2 = target_q_values2.view(-1)

            # Use the minimum of the two Q-values to reduce overestimation
            target_q_values = torch.min(target_q_values1, target_q_values2)

            # Add SAC-style entropy regularization
            if hasattr(self.args, 'advanced') and self.args['advanced']['use_entropy_reg']:
                entropy_coef = self.args['advanced']['entropy_coef']
                target_q_values = target_q_values - entropy_coef * next_actions_log_prob

            # Compute n-step returns
            if hasattr(self.args, 'advanced') and self.args['advanced']['n_step_returns'] > 1:
                n_steps = self.args['advanced']['n_step_returns']
                target_values = compute_n_step_returns(rewards, target_q_values, terminated, self._discount_factor,
                                                       n_steps)
            else:
                # Compute standard 1-step return
                target_values = rewards + self._discount_factor * (~terminated) * target_q_values

        """
        Part 1: Train critic network using both control and HGG-generated subgoal actions
        """
        # Compute critic1 loss for control actions
        critic_values_control, _, _ = self.critic.act(
            {"states": states_noisy.T, "taken_actions": actions}, role="critic"
        )
        td_errors_control = critic_values_control.view(-1) - target_values

        # Use Huber loss instead of MSE for better stability
        if hasattr(self.args, 'advanced') and self.args['advanced']['use_huber_loss']:
            huber_delta = self.args['advanced']['huber_delta']
            critic_loss_control = F.huber_loss(critic_values_control.view(-1), target_values, delta=huber_delta,
                                               reduction='none')
        else:
            critic_loss_control = F.mse_loss(critic_values_control.view(-1), target_values, reduction='none')

        # Compute critic1 loss for HGG actions
        critic_values_hgg, _, _ = self.critic.act(
            {"states": states_noisy.T, "taken_actions": hgg_actions}, role="critic"
        )
        td_errors_hgg = critic_values_hgg.view(-1) - target_values

        # Use Huber loss
        if hasattr(self.args, 'advanced') and self.args['advanced']['use_huber_loss']:
            huber_delta = self.args['advanced']['huber_delta']
            critic_loss_hgg = F.huber_loss(critic_values_hgg.view(-1), target_values, delta=huber_delta,
                                           reduction='none')
        else:
            critic_loss_hgg = F.mse_loss(critic_values_hgg.view(-1), target_values, reduction='none')

        # Combine both losses and apply weights
        critic_loss = (critic_loss_control * weights).mean() * 0.5 + (critic_loss_hgg * weights).mean() * 0.5

        # Apply L2 regularization
        if hasattr(self.args, 'advanced') and self.args['advanced']['critic_l2_reg'] > 0:
            l2_reg = 0
            for param in self.critic.parameters():
                l2_reg += torch.norm(param, 2)
            critic_loss += self.args['advanced']['critic_l2_reg'] * l2_reg

        # Optimize critic1
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self._grad_norm_clip)
        self.scaler.step(self.critic_optimizer)

        """
        Part 2: Train critic2 network using both control and HGG-generated subgoal actions
        """
        # Compute critic2 loss for control actions
        critic_values2_control, _, _ = self.critic2.act(
            {"states": states_noisy.T, "taken_actions": actions}, role="critic"
        )
        td_errors2_control = critic_values2_control.view(-1) - target_values

        # Use Huber loss
        if hasattr(self.args, 'advanced') and self.args['advanced']['use_huber_loss']:
            huber_delta = self.args['advanced']['huber_delta']
            critic2_loss_control = F.huber_loss(critic_values2_control.view(-1), target_values, delta=huber_delta,
                                                reduction='none')
        else:
            critic2_loss_control = F.mse_loss(critic_values2_control.view(-1), target_values, reduction='none')

        # Compute critic2 loss for HGG actions
        critic_values2_hgg, _, _ = self.critic2.act(
            {"states": states_noisy.T, "taken_actions": hgg_actions}, role="critic"
        )
        td_errors2_hgg = critic_values2_hgg.view(-1) - target_values

        # Use Huber loss
        if hasattr(self.args, 'advanced') and self.args['advanced']['use_huber_loss']:
            huber_delta = self.args['advanced']['huber_delta']
            critic2_loss_hgg = F.huber_loss(critic_values2_hgg.view(-1), target_values, delta=huber_delta,
                                            reduction='none')
        else:
            critic2_loss_hgg = F.mse_loss(critic_values2_hgg.view(-1), target_values, reduction='none')

        # Combine both losses and apply weights
        critic2_loss = (critic2_loss_control * weights).mean() * 0.5 + (critic2_loss_hgg * weights).mean() * 0.5

        # Apply L2 regularization
        if hasattr(self.args, 'advanced') and self.args['advanced']['critic_l2_reg'] > 0:
            l2_reg = 0
            for param in self.critic2.parameters():
                l2_reg += torch.norm(param, 2)
            critic2_loss += self.args['advanced']['critic_l2_reg'] * l2_reg

        # Optimize critic2
        self.critic2_optimizer.zero_grad()
        self.scaler.scale(critic2_loss).backward()
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.critic2_optimizer)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self._grad_norm_clip)
        self.scaler.step(self.critic2_optimizer)

        # Delay actor (policy) updates – update every few critic steps
        actor_update_freq = 2
        if not hasattr(self, 'actor_update_counter'):
            self.actor_update_counter = 0

        self.actor_update_counter += 1
        if self.actor_update_counter % actor_update_freq == 0:
            # Compute policy loss using critic1 evaluation

            # Ensure states are correctly preprocessed – simplified usage
            processed_states = self._state_preprocessor(states, train=True)

            # Forward pass through policy
            new_actions, actions_log_prob, _ = self.policy.act({"states": processed_states.T}, role="policy")

            # Compute Q-values
            policy_q_values, _, _ = self.critic.act(
                {"states": processed_states.T, "taken_actions": new_actions}, role="critic"
            )

            # Compute policy loss, add entropy regularization if enabled
            if hasattr(self.args, 'advanced') and self.args['advanced']['use_entropy_reg']:
                entropy_coef = self.args['advanced']['entropy_coef']
                policy_loss = -(policy_q_values - entropy_coef * actions_log_prob).mean()
            else:
                policy_loss = -policy_q_values.mean()

            # Add L2 regularization on actions to prevent excessively large outputs
            action_reg = torch.mean(torch.square(new_actions))
            policy_loss += 1e-3 * action_reg  # Small-weight L2 regularization

            # Optimize policy
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

        self.scaler.update()

        # Update priorities if prioritized replay is enabled
        if use_prioritized and tree_idxs is not None:
            # Calculate new priorities using average absolute TD errors
            new_priorities = (td_errors_control.abs() + td_errors_hgg.abs() +
                              td_errors2_control.abs() + td_errors2_hgg.abs()) / 4
            new_priorities = new_priorities.detach().cpu().numpy() + 1e-6  # Add small value to avoid zero priority

            # Update priorities in memory
            for idx, priority in zip(tree_idxs, new_priorities):
                if hasattr(self.memory, 'update_priorities'):
                    self.memory.update_priorities(idx, priority)

        # Log training statistics
        metrics = {
            "control_critic_loss": critic_loss_control.mean().item(),
            "hgg_critic_loss": critic_loss_hgg.mean().item(),
            "control_critic2_loss": critic2_loss_control.mean().item(),
            "hgg_critic2_loss": critic2_loss_hgg.mean().item(),
        }

        if self.actor_update_counter % actor_update_freq == 0:
            metrics["policy_loss"] = policy_loss.item()

        return metrics

    def get_q_value(self, obs: torch.Tensor) -> torch.Tensor:

        """Compute Q value for given observations using the current policy"""
        obs = self._state_preprocessor(obs, train=False)

        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        device = next(self.policy.parameters()).device
        obs = obs.to(device)

        with torch.no_grad():
            actions, _, _ = self.policy.act({"states": obs}, role="policy")
            q_values, _, _ = self.critic.act({"states": obs, "taken_actions": actions}, role="critic")
        return q_values[:, 0]

    def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=True):
        if hasattr(self.args, 'buffer') and hasattr(self.args.buffer, 'sample_batch_ddpg'):
            return self.args.buffer.sample_batch_ddpg(batch_size, normalizer, plain)
        else:
            return None

    def train_subgoal(self, batch):
        """Train the subgoal generation network

        Args:
            batch: A dictionary containing state, goal, and subgoal data, possibly with 'subgoal' or 'subgoal_target'

        Returns:
            loss: Training loss
        """
        if not hasattr(self, 'subgoal_network') or not self.use_direct_subgoal:
            return 0.0

        # Prepare training data
        batch_obs = batch['obs']  # From buffer sampling, 16D (13D observation + 3D goal)
        goals = batch['goal']  # 3D goal

        subgoal_key = 'subgoal' if 'subgoal' in batch else 'subgoal_target'
        subgoal_targets = batch[subgoal_key]

        # Ensure subgoal_targets is tensor type
        if not isinstance(subgoal_targets, torch.Tensor):
            subgoal_targets = torch.tensor(subgoal_targets, dtype=torch.float32, device=self.device)

        # Fix dimensions: from [64, 1, 3] -> [64, 3]
        if subgoal_targets.dim() == 3 and subgoal_targets.shape[1] == 1:
            subgoal_targets = subgoal_targets.squeeze(1)

        if not isinstance(subgoal_targets, torch.Tensor):
            subgoal_targets = torch.tensor(subgoal_targets, dtype=torch.float32, device=self.device)

        # Ensure inputs are tensors
        if not isinstance(batch_obs, torch.Tensor):
            batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.device)
        if not isinstance(goals, torch.Tensor):
            goals = torch.tensor(goals, dtype=torch.float32, device=self.device)
        if not isinstance(subgoal_targets, torch.Tensor):
            subgoal_targets = torch.tensor(subgoal_targets, dtype=torch.float32, device=self.device)

        # batch_obs is already the 16D concatenated observation (as prepared by HGG)
        # Use batch_obs directly as network input
        inputs = batch_obs

        # Use weights if provided
        weights = None
        if 'value' in batch:
            weights = batch['value']
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Normalize inputs
        if self.state_normalizer is not None:
            inputs_np = inputs.cpu().numpy() if isinstance(inputs, torch.Tensor) else inputs
            inputs = self.state_normalizer.normalize(inputs_np)
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        # Forward pass
        predicted_subgoals = self.subgoal_network(inputs)

        # Compute MSE loss, possibly weighted
        if weights is not None:
            # Weighted MSE loss
            mse_loss = F.mse_loss(predicted_subgoals, subgoal_targets, reduction='none')
            mse_loss = mse_loss.mean(dim=1)  # Per-sample average error
            loss = (mse_loss * weights).mean()  # Weighted average
        else:
            # Standard MSE loss
            loss = F.mse_loss(predicted_subgoals, subgoal_targets)

        # Add L2 regularization
        l2_reg = 0.0
        for param in self.subgoal_network.parameters():
            l2_reg += torch.norm(param, 2)

        # Total loss
        total_loss = loss + self.subgoal_l2_reg * l2_reg

        # Backward and optimize
        self.subgoal_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.subgoal_network.parameters(), 1.0)

        # Update parameters
        self.subgoal_optimizer.step()

        return total_loss.item()

    def add_noise(self, action):
        """Add exploration noise to actions

        Args:
            action: Original action

        Returns:
            noisy_action: Action with noise added
        """
        # Default noise parameters
        noise_scale = 0.1

        # Use config-defined noise if available
        if hasattr(self, 'args') and 'noise_eps' in self.args:
            noise_scale = self.args['noise_eps']

        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = action + noise

        return action
