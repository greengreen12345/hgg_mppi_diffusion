from isaacgym import gymapi, gymtorch
import torch, numpy as np
from typing import List
from dataclasses import dataclass, field
import  m3p2i_aip.utils.isaacgym_utils.actor_utils as actor_utils

@dataclass
class IsaacGymConfig():
    dt: float = 0.05 # 0.01
    substeps: int = 2
    use_gpu_pipeline: bool = True
    num_threads: int = 8
    viewer: bool = False
    spacing: float = 10 # !! 2.0
    camera_pos: List[float] = field(default_factory=lambda: [1.5, 6, 8])
    camera_target: List[float] = field(default_factory=lambda: [1.5, 0, 0])

def parse_isaacgym_config(cfg: IsaacGymConfig, device: str = "cuda:0") -> gymapi.SimParams:
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.01
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = device == "cuda:0"
    # sim_params.num_client_threads = cfg.num_client_threads
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.num_threads = 8
    sim_params.physx.use_gpu = True
    # sim_params.physx.friction_offset_threshold = 0.01
    # sim_params.physx.friction_correlation_distance = 0.001

    return sim_params

class IsaacGymWrapper:
    def __init__(
        self,
        cfg: IsaacGymConfig,
        # env_type: str = "point_env",
        env_type: str = "panda_env",
        # actors: List[str],
        # init_positions: List[List[float]] = None,
        num_envs: int = 1,
        viewer: bool = False,
        device: str = "cuda:0",
        cube_on_shelf: bool = False,
        # interactive_goal = True
    ):
        self._gym = gymapi.acquire_gym()
        self.env_type = env_type
        self.env_cfg = actor_utils.load_env_cfgs(env_type)
        self.device = device
        self.robot_indices = torch.tensor([i for i, a in enumerate(self.env_cfg) if a.type == "robot"], device=self.device)
        self.robot_per_env = len(self.robot_indices)

        self.cfg = cfg
        self.has_object = False
        self.target_range = 0.15
        if viewer:
            self.cfg.viewer = viewer
        # self.interactive_goal = interactive_goal
        self.num_envs = num_envs
        #self.num_envs = 200
        self.cube_on_shelf = cube_on_shelf
        # self.restarted = 1
        #self.goal = self._sample_goal()
        self.distance_threshold = 0.02
        self.reward_type = 'sparse'

        self.start_sim()


    def start_sim(self):
        self._sim = self._gym.create_sim(
            compute_device = 0,
            graphics_device = 0,
            type = gymapi.SIM_PHYSX,
            params = parse_isaacgym_config(self.cfg, self.device),
        )

        # if self.cfg.viewer:
        #     self.viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
        #     self.initialize_keyboard_listeners()
        # else:
        #     self.viewer = None
        self.viewer = None

        self.add_ground_plane()

        self.creat_env()

        self._gym.prepare_sim(self._sim)

        self.set_initial_joint_pose()

        self.acquire_states()

    def acquire_states(self):
        self.num_dofs = self._gym.get_sim_dof_count(self._sim)
        self.dofs_per_robot = int(self.num_dofs/(self.num_envs*self.robot_per_env))
        self.num_bodies = self._gym.get_sim_rigid_body_count(self._sim)
        self.bodies_per_env = int(self.num_bodies/self.num_envs)

        self._dof_state = gymtorch.wrap_tensor(
            self._gym.acquire_dof_state_tensor(self._sim)
        ).view(self.num_envs, -1)
        #print("dof_state.shape:", self._dof_state.shape)

        self._root_state = gymtorch.wrap_tensor(
            self._gym.acquire_actor_root_state_tensor(self._sim)
        ).view(self.num_envs, -1, 13)

        self._rigid_body_state = gymtorch.wrap_tensor(
            self._gym.acquire_rigid_body_state_tensor(self._sim)
        ).view(self.num_envs, -1, 13)

        self._net_contact_force = gymtorch.wrap_tensor(
            self._gym.acquire_net_contact_force_tensor(self._sim)
        ).view(self.num_envs, -1, 3)

        self.initial_root_state = self._root_state.clone().to(self.device)
        self.initial_dof_state = self._dof_state.clone().to(self.device)

        # Important!!!
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

    @property
    def robot_pos(self):
        return torch.index_select(self._dof_state, 1, torch.tensor([0, 2], device=self.device))

    @property
    def robot_vel(self):
        return torch.index_select(self._dof_state, 1, torch.tensor([1, 3], device=self.device))

    def _get_actor_index_by_name(self, name: str):
        return torch.tensor([a.name for a in self.env_cfg].index(name), device=self.device)

    def _get_actor_index_by_robot_index(self, robot_idx: int):
        return self.robot_indices[robot_idx]

    def get_actor_position_by_actor_index(self, actor_idx: int):
        return torch.index_select(self._root_state, 1, actor_idx)[:, 0, 0:3]

    def get_actor_position_by_name(self, name: str):
        actor_idx = self._get_actor_index_by_name(name)
        return self.get_actor_position_by_actor_index(actor_idx)

    def get_actor_position_by_robot_index(self, robot_idx: int):
        actor_idx = self._get_actor_index_by_robot_index(robot_idx)
        return self.get_actor_position_by_actor_index(actor_idx)

    def get_actor_velocity_by_actor_index(self, idx: int):
        return torch.index_select(self._root_state, 1, idx)[:, 0, 7:10]

    def get_actor_velocity_by_name(self, name: str):
        actor_idx = self._get_actor_index_by_name(name)
        return self.get_actor_velocity_by_actor_index(actor_idx)

    def get_actor_velocity_by_robot_index(self, robot_idx: int):
        actor_idx = self._get_actor_index_by_robot_index(robot_idx)
        return self.get_actor_velocity_by_actor_index(actor_idx)

    def get_actor_orientation_by_actor_index(self, idx: int):
        return torch.index_select(self._root_state, 1, idx)[:, 0, 3:7]

    def get_actor_orientation_by_name(self, name: str):
        actor_idx = self._get_actor_index_by_name(name)
        return self.get_actor_orientation_by_actor_index(actor_idx)

    def get_actor_orientation_by_robot_index(self, robot_idx: int):
        actor_idx = self._get_actor_index_by_robot_index(robot_idx)
        return self.get_actor_orientation_by_actor_index(actor_idx)

    def get_rigid_body_by_rigid_body_index(self, rigid_body_idx: int):
        return torch.index_select(self._rigid_body_state, 1, rigid_body_idx)[:, 0, :]

    def get_actor_link_by_name(self, actor_name: str, link_name: str):
        actor_idx = self._get_actor_index_by_name(actor_name)
        rigid_body_idx = torch.tensor(
            self._gym.find_actor_rigid_body_index(
                self.envs[0], actor_idx, link_name, gymapi.IndexDomain.DOMAIN_ENV
            ),
            device=self.device,
        )
        return self.get_rigid_body_by_rigid_body_index(rigid_body_idx)

    def get_actor_contact_forces_by_name(self, actor_name: str, link_name: str):
        actor_idx = self._get_actor_index_by_name(actor_name)
        rigid_body_idx = torch.tensor(
            self._gym.find_actor_rigid_body_index(
                self.envs[0], actor_idx, link_name, gymapi.IndexDomain.DOMAIN_ENV
            ),
            device=self.device,
        )
        return self._net_contact_force[:, rigid_body_idx]
    
    def set_dof_state_tensor(self, u):
        self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(u))

    def set_actor_root_state_tensor(self, u):
        self._gym.set_actor_root_state_tensor(self._sim, gymtorch.unwrap_tensor(u))

    def set_dof_velocity_target_tensor(self, u):
        self._gym.set_dof_velocity_target_tensor(self._sim, gymtorch.unwrap_tensor(u))

    def set_dof_actuation_force_tensor(self, u):
        self._gym.set_dof_actuation_force_tensor(self._sim, gymtorch.unwrap_tensor(u))

    def apply_rigid_body_force_tensors(self, u):
        self._gym.apply_rigid_body_force_tensors(self._sim, gymtorch.unwrap_tensor(u.view(-1, 3)))
    
    def update_dyn_obs(self, i, period=100):
        dyn_obs_id = self._get_actor_index_by_name("dyn-obs")
        dyn_obs_pos = self._root_state[:, dyn_obs_id, :3]

        if self.env_type == "point_env":
            offsets = torch.tensor([0.01, 0.01, 0], dtype=torch.float32, device=self.device)
        else:
            offsets = torch.tensor([0, 0.0, 0], dtype=torch.float32, device=self.device)

        if i % period > period/4 and i % period < period/4*3:
            dyn_obs_pos += offsets
        else:
            dyn_obs_pos -= offsets
        self._gym.set_actor_root_state_tensor(
            self._sim, gymtorch.unwrap_tensor(self._root_state)
        )

    def set_initial_joint_pose(self):
        # Set initial joint poses
        robots = [a for a in self.env_cfg if a.type == "robot"]
        for robot in robots:
            dof_state = []
            if robot.init_joint_pose:
                dof_state += robot.init_joint_pose
                print(dof_state)
            else:
                dof_state += (
                    [0] * 2 * self._gym.get_actor_dof_count(self.envs[0], robot.handle)
                )
        dof_state = (
            torch.tensor(dof_state, device=self.device)
            .type(torch.float32)
            .repeat(self.num_envs, 1)
        )
        self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(dof_state))
        self._gym.refresh_dof_state_tensor(self._sim)

    def creat_env(self):
        # Load / create assets for all actors in the envs
        env_actor_assets = []
        for actor_cfg in self.env_cfg:
            asset = actor_utils.load_asset(self._gym, self._sim, actor_cfg)
            env_actor_assets.append(asset)

        self.camera_pos = [0, 1.5, 2.8]
        self.camera_target = [0, 0, 1]
        self._gym.viewer_camera_look_at(self.viewer, None,
                                        gymapi.Vec3(*self.camera_pos),
                                        gymapi.Vec3(*self.camera_target))

        # Create envs and fill with assets
        self.envs = []
        self.spacing = 2
        for env_idx in range(self.num_envs):
            env = self._gym.create_env(
                self._sim,
                gymapi.Vec3(-self.spacing, 0.0, -self.spacing),
                gymapi.Vec3(self.spacing, self.spacing, self.spacing),
                int(self.num_envs**0.5),
            )

            for actor_asset, actor_cfg in zip(env_actor_assets, self.env_cfg):
                actor_cfg.handle = self._create_actor(
                    env, env_idx, actor_asset, actor_cfg
                )
            self.envs.append(env)

    def _create_actor(self, env, env_idx, asset, actor: actor_utils.ActorWrapper) -> int:
        if actor.noise_sigma_size is not None:
            asset = actor_utils.load_asset(self._gym, self._sim, actor)

        pose = gymapi.Transform()
        if actor.name == "cubeA":
            pose.p = gymapi.Vec3(*actor.init_pos_on_shelf) if self.cube_on_shelf else gymapi.Vec3(*actor.init_pos_on_table)
        else:
            pose.p = gymapi.Vec3(*actor.init_pos)
        pose.r = gymapi.Quat(*actor.init_ori)
        handle = self._gym.create_actor(
            env=env,
            asset=asset,
            pose=pose,
            name=actor.name,
            group=env_idx if actor.collision else -2, #  env_idx + self.num_envs, #
        )

        if actor.noise_sigma_size:
            actor.color = np.random.rand(3)

        self._gym.set_rigid_body_color(
            env, handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*actor.color)
        )
        props = self._gym.get_actor_rigid_body_properties(env, handle)
        # actor_mass_noise = np.random.uniform(
        #     -actor.noise_percentage_mass * actor.mass,
        #     actor.noise_percentage_mass * actor.mass,
        # )
        # props[0].mass = actor.mass + actor_mass_noise
        # self._gym.set_actor_rigid_body_properties(env, handle, props)
        # print("mass", props[0].mass)

        body_names = self._gym.get_actor_rigid_body_names(env, handle)
        body_to_shape = self._gym.get_actor_rigid_body_shape_indices(env, handle)
        caster_shapes = [
            b.start
            for body_idx, b in enumerate(body_to_shape)
            if actor.caster_links is not None
            and body_names[body_idx] in actor.caster_links
        ]

        props = self._gym.get_actor_rigid_shape_properties(env, handle)
        for i, p in enumerate(props):
            actor_friction_noise = np.random.uniform(
                -actor.noise_percentage_friction * actor.friction,
                actor.noise_percentage_friction * actor.friction,
            )
            p.friction = actor.friction + actor_friction_noise #
            p.torsion_friction = np.random.uniform(0.001, 0.01)
            p.rolling_friction = actor.friction + actor_friction_noise

            if i in caster_shapes:
                p.friction = 0
                p.torsion_friction = 0
                p.rolling_friction = 0

        self._gym.set_actor_rigid_shape_properties(env, handle, props)

        if actor.type == "robot":
            # TODO: Currently the robot_rigid_body_viz_idx is only supported for a single robot case.
            if actor.visualize_link:
                self.robot_rigid_body_viz_idx = self._gym.find_actor_rigid_body_index(
                    env, handle, actor.visualize_link, gymapi.IndexDomain.DOMAIN_ENV
                )

            props = self._gym.get_asset_dof_properties(asset)
            if actor.dof_mode == "effort":
                props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
                props["stiffness"].fill(0.0)
                props["armature"].fill(0.0)
                props["damping"].fill(10.0)
            elif actor.dof_mode == "velocity":
                props["driveMode"].fill(gymapi.DOF_MODE_VEL)
                props["stiffness"].fill(0.0)
                props["damping"].fill(600.0)
            elif actor.dof_mode == "position":
                props["driveMode"].fill(gymapi.DOF_MODE_POS)
                props["stiffness"].fill(80.0)
                props["damping"].fill(0.0)
            else:
                raise ValueError("Invalid dof_mode")
            self._gym.set_actor_dof_properties(env, handle, props)
        return handle
    
    def step1(self):
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)

        if self.viewer is not None:
            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self.viewer, self._sim, False)

    def step(self, action):

        self.set_dof_velocity_target_tensor(action)
        self.step1()

        obs = self._get_obs()
        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.desired_goal),
        }
        #print(self.goal)
        # distance between achieved_goal and explore_goal
        dis_subgoal = self.goal_distance(obs['achieved_goal'], self.goal)

        
        reward = self.compute_reward(obs['achieved_goal'], self.desired_goal)
        dis_goal = -reward

        if dis_goal < 0.05:
            done = True

        return obs, reward, done, info, dis_goal, dis_subgoal

    def _is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def compute_reward(self, achieved_goal, goal):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        return -d

    def goal_distance(self, goal_a, goal_b):
        if isinstance(goal_a, torch.Tensor):
            goal_a = goal_a.detach().cpu().numpy()
        if isinstance(goal_b, torch.Tensor):
            goal_b = goal_b.detach().cpu().numpy()

        goal_a = np.squeeze(goal_a)
        goal_b = np.squeeze(goal_b)

        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def stop_sim(self):
        print("Done! Stop sim!")
        if self.viewer:
            self._gym.destroy_viewer(self.viewer)
        # for env_idx in range(self.num_envs):
        #     self._gym.destroy_env(self.envs[env_idx])
        self._gym.destroy_sim(self._sim)
    
    def visualize_trajs(self, trajs):
        trajs = trajs.cpu().clone().numpy()
        n_traj, t_horizon = trajs.shape[0], trajs.shape[1]-1
        line_array = np.zeros((t_horizon, 6), dtype=np.float32)
        color_array = np.zeros((t_horizon, 3), dtype=np.float32)
        color_array[:, 1] = 255 
        
        self._gym.clear_lines(self.viewer)
        for i in range(n_traj):
            for j in range(t_horizon):
                if self.env_type == "point_env":
                    pos = [trajs[i, j, 0], trajs[i, j, 1], 0.1, 
                           trajs[i, j+1, 0], trajs[i, j+1, 1], 0.1]
                else:
                    pos = [trajs[i, j, 0], trajs[i, j, 1], trajs[i, j, 2], 
                           trajs[i, j+1, 0], trajs[i, j+1, 1], trajs[i, j+1, 2]]
                line_array[j, :] = pos
            self._gym.add_lines(self.viewer, self.envs[0], t_horizon, line_array, color_array)

    def initialize_keyboard_listeners(self):
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "left")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "down")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "right")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "up")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_1, "1")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_2, "2")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_3, "3")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_4, "4")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_5, "5")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_6, "6")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_7, "7")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_8, "8")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_9, "9")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "key_left")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_DOWN, "key_down")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "key_right")
        self._gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_UP, "key_up")

    def play_with_cube(self):
        if self.env_type != "panda_env":
            return 0
        x_pos = torch.tensor([0.03, 0, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs)
        y_pos = torch.tensor([0, 0.03, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs)
        z_pos = torch.tensor([0, 0, 0.03], dtype=torch.float32, device=self.device).repeat(self.num_envs)
        cube_targets = {'key_up':-y_pos, 'key_down':y_pos, 'key_left':x_pos, 'key_right':-x_pos}
        goal_targets = {'up':-y_pos, 'down':y_pos, 'left':x_pos, 'right':-x_pos}
        obs_targets = {'1':x_pos, '2':-x_pos, '3':-y_pos, '4':y_pos, '5':z_pos, '6':-z_pos}
        cubeA_index = self._get_actor_index_by_name("cubeA")
        cubeB_index = self._get_actor_index_by_name("cubeB")
        obs_index = self._get_actor_index_by_name("dyn-obs")
        for evt in self._gym.query_viewer_action_events(self.viewer):
            # Press WASD and up,left,right,down to interact with the cubes
            if evt.value > 0:
                if evt.action in ['key_up', 'key_down', 'key_left', 'key_right']:
                    self._root_state[:, cubeA_index, :3] += cube_targets[evt.action]
                if evt.action in ['up', 'down', 'left', 'right']:
                    self._root_state[:, cubeB_index, :3] += goal_targets[evt.action]
                if evt.action in ['1', '2', '3', '4', '5', '6']:
                    self._root_state[:, obs_index, :3] += obs_targets[evt.action]

                self._gym.set_actor_root_state_tensor(
                    self._sim, gymtorch.unwrap_tensor(self._root_state)
                )

    def keyboard_control(self):
        # Set targets for different robots
        vel_targets = {}
        zero_vel = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        if self.env_type == "point_env":
            up_vel = torch.tensor([0, -2], dtype=torch.float32, device=self.device).repeat(self.num_envs)
            down_vel = torch.tensor([0, 2], dtype=torch.float32, device=self.device).repeat(self.num_envs)
            left_vel = torch.tensor([2, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs)
            right_vel = torch.tensor([-2, 0], dtype=torch.float32, device=self.device).repeat(self.num_envs)
            vel_targets = {"up":up_vel, "down":down_vel, "left":left_vel, "right":right_vel}
        elif self.env_type == "panda_env":
            for i in range(self.num_dofs):
                joint_i = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
                joint_i[i] = 1
                vel_targets[str(i+1)] = joint_i

        # Respond the keyboard (velocity control)
        for evt in self._gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                self._gym.set_dof_velocity_target_tensor(self._sim, gymtorch.unwrap_tensor(vel_targets[evt.action]))
            else:
                self._gym.set_dof_velocity_target_tensor(self._sim, gymtorch.unwrap_tensor(zero_vel))

    def add_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1
        plane_params.dynamic_friction = 1
        plane_params.restitution = 0
        self._gym.add_ground(self._sim, plane_params)

    def get_actor_position_by_name(self, name: str):
        actor_idx = self._get_actor_index_by_name(name)
        return self.get_actor_position_by_actor_index(actor_idx)

    def get_actor_position_by_actor_index(self, actor_idx: int):
        return torch.index_select(self._root_state, 1, actor_idx)[:, 0, 0:3]

    def get_actor_size(self, actor_name: str):
        for actor_cfg in self.env_cfg:
            name = actor_cfg.get("name") if isinstance(actor_cfg, dict) else getattr(actor_cfg, "name", None)
            if name == actor_name:
                return actor_cfg.get("size") if isinstance(actor_cfg, dict) else getattr(actor_cfg, "size", None)
        raise ValueError(f"Actor {actor_name} not found in configuration.")

    def reset(self):
        self.set_initial_joint_pose()
        self.acquire_states()
        self.step1()

        cube_state = self.get_actor_link_by_name("cubeA", "box")
        pre_pick_goal = cube_state[0, :3].clone()
        self.pre_height_diff = 0.05
        pre_pick_goal[2] += self.pre_height_diff
        self.goal = pre_pick_goal
        
        self.desired_goal = [0.2, -0.2, 1.1085299]
        
        self.sampled_goal = pre_pick_goal  
        self.subgoal = pre_pick_goal  
        obs = self._get_obs()
        return obs

    def validate_simulation_state(self):

        expected_position = torch.tensor([0, 0, 0], device=self.device)
        actual_position = self.get_actor_position_by_name("some_actor_name")
        if torch.allclose(expected_position, actual_position, atol=0.1):
            return True
        else:
            return False

    def _get_obs(self):
        """
        Obtain obs information about the current environment:
        obs includes end-effector position, rotation (quaternion), linear velocity, angular velocity;
        achieved_goal is the position of the end-effector.
        """

        left_finger = self.get_actor_link_by_name("panda", "panda_leftfinger")
        right_finger = self.get_actor_link_by_name("panda", "panda_rightfinger")
        self.ee_state = ((left_finger + right_finger) / 2)[0]

        position = self.ee_state[0:3].unsqueeze(0)
        orientation = self.ee_state[3:7].unsqueeze(0)
        linear_velocity = self.ee_state[7:10].unsqueeze(0)
        angular_velocity = self.ee_state[10:13].unsqueeze(0)

        # Construct Observation
        obs = torch.cat([position, orientation, linear_velocity, angular_velocity], dim=1)  # shape: [num_envs, 13]

        # Set achieved_goal as the end-effector position
        achieved_goal = position.clone()

        #self.goal = torch.tensor(self.goal, device=self.device, dtype=torch.float32)
        #self.goal = self.goal.clone().detach().to(self.device, dtype=torch.float32)

        if not isinstance(self.goal, torch.Tensor):
            #self.goal = torch.from_numpy(self.goal).to(self.device, dtype=torch.float32)
            self.goal = torch.tensor(self.goal, device=self.device, dtype=torch.float32)

        else:
            self.goal = self.goal.clone().detach().to(self.device, dtype=torch.float32)

        # return {
        #     'observation': obs.cpu().numpy().copy(),
        #     'achieved_goal': achieved_goal.cpu().numpy().copy(),
        #     'desired_goal': self.desired_goal.cpu().numpy().copy()
        # }

        return {
            'observation': obs.cpu().numpy().copy(),
            'achieved_goal': achieved_goal.cpu().numpy().copy(),
            'desired_goal': torch.as_tensor(self.desired_goal, device=self.device, dtype=torch.float32).cpu().numpy().copy()
        }




