from .ddpg import DDPG
from skrl.utils.model_instantiators.torch import deterministic_model
import gymnasium
from skrl.memories.torch import RandomMemory

from agents.ql_diffusion import Diffusion_QL as QL_Agent
from agents.bc_diffusion import Diffusion_BC as BC_Agent


def create_agent(args):
    
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(16,))  
    action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(3,))
    device = "cuda:0"
    # models
    network = [
        {
            "name": "net",
            "input": "STATES",
            "input_size": None,
            "layers": [64, 64],
            "activations": "elu",
        }
    ]
    models = {}
    models["policy"] = deterministic_model(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        input_size = 55,
        network=network,
        output="ACTIONS",
    )
    models["target_policy"] = deterministic_model(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        input_size=55,
        network=network,
        output="ACTIONS",
    )
    models["critic"] = deterministic_model(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        input_size=None,
        network=network,
        output="ONE",
    )
    models["target_critic"] = deterministic_model(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        input_size=None,
        network=network,
        output="ONE",
    )
    memory = RandomMemory(memory_size=50, num_envs=1, device=device)

    return DDPG(
        models=models,
        memory=memory,
        observation_space=observation_space,
        action_space=action_space,
        device="cuda:0",
        cfg=args,
    )



def create_ql_diffusion_agent(args):
    
    state_dim = 16
    action_dim = 3
    device = "cuda:0"
    max_action = 1.5

    
    return QL_Agent(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=device,
                    discount=0.98,
                    tau=0.002,
                    max_q_backup=False,
                    beta_schedule='linear',
                    n_timesteps=100,
                    eta=0.1,
                    lr=1e-4,
                    lr_decay=args.lr_decay,
                    lr_maxt=2000,
                    grad_norm=1.0)

def create_bc_diffusion_agent():
    
    state_dim = 16
    action_dim = 3
    device = "cuda:0"
    max_action = 1.5

    return BC_Agent(state_dim=state_dim,
				  action_dim=action_dim,
				  max_action=max_action,
				  device=device,
				  discount=0.99,
				  tau=0.005,

				  beta_schedule='linear',
				  n_timesteps=100,

				  lr=2e-4,
				  )

    



