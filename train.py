import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
# from scripts.reactive_tamp import REACTIVE_TAMP
# from scripts.sim import run_sim
from m3p2i_aip.config.config_store import ExampleConfig

import json
import logging
import numpy as np

import torch, hydra, zerorpc

import os

from algorithm.data_sampler import Data_Sampler

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list) and all(isinstance(i, torch.Tensor) for i in x):
        return [i.tolist() for i in obj]

    return obj


def main():
    args = get_args()

    
    env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, learner, tester = experiment_setup(args)

    bc_diffusion_agent.actor.load_state_dict(
        torch.load("saved_models_diffusion_BC/actor_epoch_17_cycle_8.pth", map_location=agent.device))

    

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")

    
    total_episodes = args.episodes
    exploration_phase = int(total_episodes * 0.3)  
    training_phase = int(total_episodes * 0.6)  
    fine_tuning_phase = total_episodes  

    
    training_state = {
        "total_episodes": total_episodes,
        "exploration_phase": exploration_phase,
        "training_phase": training_phase,
        "fine_tuning_phase": fine_tuning_phase,
        "total_epochs": args.epochs,  
    }
    
    

    # 1. load .npz file
    data_npz = np.load("offline_dataset3.npz")

    # 2. Convert into dictionary
    data = {
        'observations': data_npz['observations'],
        'actions': data_npz['actions'],
        'rewards': data_npz['rewards'],
        'terminals': data_npz['terminals'],
        'next_observations': data_npz['next_observations']
    }

    # 3. initialize Data_Sampler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = Data_Sampler(data, device, reward_tune='no')

    for i in range(1015):
        info = ql_diffusion_agent.train(sampler, iterations=100, batch_size=100, log_writer=None, offline=True)
        if (i >100 and i < 110) or (i >200 and i < 210) or (i >300 and i < 310) or (i >400 and i < 410) or (i >500 and i < 510) or (i >600 and i < 610) or (i >700 and i < 710) or (i >800 and i < 810) or (i >900 and i < 910) or (i >1000 and i < 1010):
            if hasattr(ql_diffusion_agent, "actor"):
                # >>> save actor model <<<
                model_save_path = f"saved_models_diffusion_QL4/actor_loop_{i}.pth"
                os.makedirs("saved_models_diffusion_QL4", exist_ok=True)
                torch.save(ql_diffusion_agent.actor.state_dict(), model_save_path)
                print(f"save model successful: {model_save_path}")

            if hasattr(ql_diffusion_agent, "critic"):
                # >>> save critic model <<<
                model_save_path = f"saved_models_diffusion_QL4/critic_loop_{i}.pth"
                os.makedirs("saved_models_diffusion_QL4", exist_ok=True)
                torch.save(ql_diffusion_agent.critic.state_dict(), model_save_path)
                print(f"save model successful: {model_save_path}")

        

    # ql_diffusion_agent.actor.load_state_dict(
    #     torch.load("saved_models_diffusion_QL/actor_epoch_0_cycle_5.pth", map_location=agent.device))

    for epoch in range(args.epochs):
        # print("*************************epoch***********************", epoch, args.epochs)
        
        training_state["current_epoch"] = epoch
        # for cycle in range(args.cycles+5):
        for cycle in range(args.cycles):
            print("*************************epoch***********************", epoch, args.epochs)
            print("*********************************cycle*******************************", cycle, args.cycles)
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, bc_diffusion_agent, ql_diffusion_agent, buffer, planner, training_state)

            log_entry = {
                "epoch": epoch,
                "cycle": cycle,
                "initial_goals": convert_ndarray(learner.initial_goals),
                                                                         "desired_goals": convert_ndarray(
                    learner.desired_goals),
                "explore_goals": convert_ndarray(learner.explore_goals),
                "trajectories": convert_ndarray(learner.achieved_trajectories),
                "success_trajectories": convert_ndarray(learner.success_trajectories),
                "episode_return": convert_ndarray(learner.episode_return),
            }
            with open("explore_goals33.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            

        #tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

