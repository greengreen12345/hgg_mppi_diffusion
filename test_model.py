import learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
import numpy as np
from algorithm.ddpg.ddpg import DDPG
import torch
from algorithm import create_agent, create_ql_diffusion_agent

from types import SimpleNamespace
import torch, hydra, zerorpc
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes

from common import get_args, experiment_setup

planner = zerorpc.Client()
planner.connect("tcp://127.0.0.1:4242")
print("Server found and waiting for the viewer")

def optimize_subgoal_with_noise(current_state, predicted_subgoal, final_goal, n_samples, noise_scale, policy, critic1, critic2):
    candidates = [predicted_subgoal]
    values = [evaluate_subgoal_value(current_state, predicted_subgoal, final_goal, policy, critic1, critic2)]

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_scale, size=predicted_subgoal.shape)
        noisy_subgoal = np.clip(predicted_subgoal + noise, -1.5, 1.5)
        value = evaluate_subgoal_value(current_state, noisy_subgoal, final_goal, policy, critic1, critic2)
        candidates.append(noisy_subgoal)
        values.append(value)

    best_idx = np.argmax(values)
    best_subgoal = candidates[best_idx]
    best_value = values[best_idx]

    if best_idx > 0:
        improvement = best_value - values[0]
        print(f"Noise optimization improved: {improvement:.4f}, before: {values[0]:.4f}, after: {best_value:.4f}")

    return best_subgoal

def evaluate_subgoal_value(obs, subgoal, final_goal, policy, critic1, critic2):
    if isinstance(final_goal, np.ndarray):
        final_goal = torch.tensor(final_goal, dtype=torch.float32, device=device)
    if final_goal.ndim == 1:
        final_goal = final_goal.unsqueeze(0)

    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    subgoal = torch.tensor(subgoal, dtype=torch.float32, device=device).unsqueeze(0)

    if obs.dim() == 3:
        obs = obs.view(obs.size(0), -1)

    subgoal = torch.tensor(subgoal, dtype=torch.float32, device=device)
    if subgoal.ndim == 1:
        subgoal = subgoal.unsqueeze(0)
    elif subgoal.ndim == 3:
        subgoal = subgoal.squeeze(1)

    state_final_goal = torch.cat([obs, final_goal], dim=1)
    state = torch.cat([state_final_goal, subgoal], dim=1)

    action, _, _ = policy.act({"states": state.T}, role="policy")
    q1, _, _ = critic1.act({"states": state.T, "taken_actions": action}, role="critic")
    q2, _, _ = critic2.act({"states": state.T, "taken_actions": action}, role="critic")

    value = torch.min(q1, q2)
    return value.item()

# Build test args (simulate config)
test_cfg = SimpleNamespace(
    goal_dims=3,
    obs_dims=13,
    use_direct_subgoal=True
)

args = get_args()

# Create agent using create_agent (includes policy/critic networks)
agent = create_agent(test_cfg)
ql_diffusion_agent = create_ql_diffusion_agent(args)

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = wrapper.IsaacGymWrapper(
    "panda", "panda_env", num_envs=1, viewer=False, device="cuda:0", cube_on_shelf=False)
env.reset()
obs = env._get_obs()

# Build test config
test_cfg = type("TestArgs", (), {})()
setattr(test_cfg, "goal_dims", 3)
setattr(test_cfg, "obs_dims", 13)
setattr(test_cfg, "use_direct_subgoal", True)

# Load model parameters
# agent.subgoal_network.load_state_dict(torch.load("saved_models/subgoal_network.pth", map_location=device))
# agent.policy.load_state_dict(torch.load("saved_models/best_policy.pth", map_location=device))
# agent.critic.load_state_dict(torch.load("saved_models/best_critic.pth", map_location=device))
# agent.policy.eval()
# agent.critic.eval()
# agent.subgoal_network.eval()
# agent.critic2 = agent.critic  # If there is no critic2, reuse critic1

# diffusion_agent.actor.load_state_dict(torch.load("saved_models_diffusion_BC/actor_epoch_17_cycle_8.pth", map_location=device))
ql_diffusion_agent.actor.load_state_dict(torch.load("saved_models_diffusion_QL4/actor_loop_1004.pth", map_location=device))

episode_reward = []

for t in range(1000):
    obs = env._get_obs()
    
    # Concatenate observation and goal as input
    obs = np.concatenate([
        np.ravel(obs['observation']),
        np.ravel(obs['desired_goal'])
    ], axis=0)
    input_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    # Inference subgoal and apply noise optimization
    with torch.no_grad():
        diffusion_goal = ql_diffusion_agent.actor(input_tensor.detach())
        env.goal = diffusion_goal
        print("diffusion_goal", diffusion_goal)

    for t in range(20):
        # Generate action using MPPI planner
        action_mppi = bytes_to_torch(
            planner.run_tamp(
                torch_to_bytes(env._dof_state),
                torch_to_bytes(env._root_state),
                diffusion_goal.tolist())
        )

        # Execute action
        obs, reward, done, info, distance, dis_subgoal = env.step(action_mppi)

        # 1. Subgoal distance reward
        curr_subgoal_distance = np.linalg.norm(diffusion_goal.detach().cpu().numpy() - obs['desired_goal'].copy())
        reward = curr_subgoal_distance * (-1)

        # 2. Bonus reward
        if distance < 0.05:
            print(f"Goal reached, terminating at step {t}, distance {distance:.4f}")
            success_bonus = 10
            reward += success_bonus

        episode_reward.append(reward)

        if dis_subgoal < 0.005:
            break
    print("distance:", distance)

with open("episode_reward1507.txt", "w") as f:
    for r in episode_reward:
        f.write(f"{r}\n")

