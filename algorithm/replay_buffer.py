import numpy as np
import copy
from envs.utils import quaternion_to_euler_angle
import torch

def goal_concat(obs, goal):

	obs = np.squeeze(obs)
	goal = np.squeeze(goal)

	return np.concatenate([obs, goal], axis=0)



def goal_based_process(obs):
	return goal_concat(obs['observation'], obs['desired_goal'])

class Trajectory:
	def __init__(self, init_obs):
		self.ep = {
			'obs': [copy.deepcopy(init_obs)],
			'rews': [],
			'acts': [],
			'hgg_acts': [],
			'done': []
		}
		self.length = 0

	def store_step(self, action, obs, reward, done, hgg_action=None):
		if isinstance(hgg_action, (float, np.float32, np.float64)):
			hgg_action = torch.tensor([hgg_action], dtype=torch.float32)
		elif isinstance(hgg_action, np.ndarray):
			hgg_action = torch.tensor(hgg_action, dtype=torch.float32)
		elif not isinstance(hgg_action, torch.Tensor):
			raise TypeError(f"Unsupported type for hgg_action: {type(hgg_action)}")

		self.ep['acts'].append(action.detach().cpu().clone())

		if hgg_action is not None:
			self.ep['hgg_acts'].append(hgg_action.detach().cpu().clone())
		else:

			self.ep['hgg_acts'].append(action.detach().cpu().clone())
			
		self.ep['obs'].append(copy.deepcopy(obs))
		self.ep['rews'].append(copy.deepcopy([reward]))
		self.ep['done'].append(copy.deepcopy([np.float32(done)]))
		self.length += 1

	def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
		# from "Energy-Based Hindsight Experience Prioritization"
		if env_id[:5]=='Fetch':
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['achieved_goal'])
			# obj = np.array([obj])
			obj = np.array(obj)

			clip_energy = 0.5
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			g, m, delta_t = 9.81, 1, 0.04
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)
		else:
			assert env_id[:4]=='Hand'
			obj = []
			for i in range(len(self.ep['obs'])):
				obj.append(self.ep['obs'][i]['observation'][-7:])
			obj = np.array([obj])

			clip_energy = 2.5
			g, m, delta_t, inertia  = 9.81, 1, 0.04, 1
			quaternion = obj[:,:,3:].copy()
			angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
			diff_angle = np.diff(angle, axis=1)
			angular_velocity = diff_angle / delta_t
			rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
			rotational_energy = np.sum(rotational_energy, axis=2)
			obj = obj[:,:,:3]
			height = obj[:, :, 2]
			height_0 = np.repeat(height[:,0].reshape(-1,1), height[:,1::].shape[1], axis=1)
			height = height[:,1::] - height_0
			potential_energy = g*m*height
			diff = np.diff(obj, axis=1)
			velocity = diff / delta_t
			kinetic_energy = 0.5 * m * np.power(velocity, 2)
			kinetic_energy = np.sum(kinetic_energy, axis=2)
			energy_totoal = w_potential*potential_energy + w_linear*kinetic_energy + w_rotational*rotational_energy
			energy_diff = np.diff(energy_totoal, axis=1)
			energy_transition = energy_totoal.copy()
			energy_transition[:,1::] = energy_diff.copy()
			energy_transition = np.clip(energy_transition, 0, clip_energy)
			energy_transition_total = np.sum(energy_transition, axis=1)
			energy_final = energy_transition_total.reshape(-1,1)
			return np.sum(energy_final)

class ReplayBuffer_Episodic:
	def __init__(self, args):
		self.args = args
		if args.buffer_type=='energy':
			self.energy = True
			self.energy_sum = 0.0
			self.energy_offset = 0.0
			self.energy_max = 1.0
		else:
			self.energy = False

		# Add support for prioritized experience replay
		self.use_prioritized = False
		self.alpha = 0.6  # priority exponent
		self.beta = 0.4  # importance sampling coefficient
		self.beta_increment = 0.001  # beta increment during training
		self.epsilon = 0.01  # avoid zero priority

		# Check if prioritized replay is enabled
		if hasattr(args, 'advanced') and args.advanced and isinstance(args.advanced, dict):
			self.use_prioritized = args.advanced.get('use_prioritized_replay', False)

		self.buffer = {}
		self.steps = []
		self.length = 0
		self.counter = 0
		self.steps_counter = 0
		self.sample_methods = {
			'ddpg': self.sample_batch_ddpg
		}
		self.sample_batch = self.sample_methods[args.alg]
		self.args.goal_based = True

	def store_trajectory(self, trajectory):
		episode = trajectory.ep
		if self.energy:
			energy = trajectory.energy(self.args.env)
			self.energy_sum += energy

		if self.counter==0:
			for key in episode.keys():
				self.buffer[key] = []
			if self.energy:
				self.buffer_energy = []
				self.buffer_energy_sum = []

		# Handle prioritized experience replay
		if self.use_prioritized and hasattr(self, 'priority_tree'):
			max_priority = 1.0
			if self.counter > 0:
				tree_indices = range(self.priority_tree.capacity - 1, 2 * self.priority_tree.capacity - 1)
				valid_indices = [i for i in tree_indices if i < len(self.priority_tree.tree)]
				if valid_indices:
					max_priority = max([self.priority_tree.tree[i] for i in valid_indices])
			self.priority_tree.add(max_priority, episode)

		if self.counter<self.args.buffer_size:
			for key in self.buffer.keys():
				self.buffer[key].append(episode[key])
			if self.energy:
				self.buffer_energy.append(copy.deepcopy(energy))
				self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
			self.length += 1
			self.steps.append(trajectory.length)
		else:
			idx = self.counter%self.args.buffer_size
			for key in self.buffer.keys():
				self.buffer[key][idx] = episode[key]
			if self.energy:
				self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
				self.buffer_energy[idx] = copy.deepcopy(energy)
				self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
			self.steps[idx] = trajectory.length
		self.counter += 1
		self.steps_counter += trajectory.length

	def energy_sample(self):
		t = self.energy_offset + np.random.uniform(0,1)*(self.energy_sum-self.energy_offset)
		if self.counter>self.args.buffer_size:
			if self.buffer_energy_sum[-1]>=t:
				return self.energy_search(t, self.counter%self.length, self.length-1)
			else:
				return self.energy_search(t, 0, self.counter%self.length-1)
		else:
			return self.energy_search(t, 0, self.length-1)

	def energy_search(self, t, l, r):
		if l==r: return l
		mid = (l+r)//2
		if self.buffer_energy_sum[mid]>=t:
			return self.energy_search(t, l, mid)
		else:
			return self.energy_search(t, mid+1, r)

	def update_priorities(self, tree_idxs, priorities):
		"""Update priorities of samples"""
		if self.use_prioritized and hasattr(self, 'priority_tree') and self.priority_tree is not None:
			for idx, priority in zip(tree_idxs, priorities):
				priority = (abs(priority) + self.epsilon) ** self.alpha
				self.priority_tree.update(idx, priority)

	def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=True):
		batch = {'obs': [], 'obs_next': [], 'acts': [], 'hgg_acts': [], 'rews': [], 'done': [], 'weights': [],
				 'tree_idxs': []}

		# Check if the buffer is empty
		if self.counter == 0 or self.length == 0:
			print("Warning: Buffer is empty, cannot sample")
			return None

		if batch_size < 1:
			batch_size = self.args.batch_size if hasattr(self.args, 'batch_size') else 256

		batch_size = min(batch_size, self.length)

		if self.use_prioritized and hasattr(self,
											'priority_tree') and self.priority_tree is not None and self.priority_tree.size > 0:
			self.beta = min(1.0, self.beta + self.beta_increment)

			weights = np.zeros(batch_size, dtype=np.float32)
			tree_idxs = np.zeros(batch_size, dtype=np.int32)

			priority_segment = self.priority_tree.total_priority() / batch_size

			for i in range(batch_size):
				a = priority_segment * i
				b = priority_segment * (i + 1)
				v = np.random.uniform(a, b)

				tree_idx, priority, episode = self.priority_tree.get_leaf(v)
				tree_idxs[i] = tree_idx

				sample_probability = priority / self.priority_tree.total_priority()
				weights[i] = (self.priority_tree.size * sample_probability) ** (-self.beta)

				step = np.random.randint(len(episode['obs']) - 1)

				if self.args.goal_based:
					print("+++++++++++self.args.goal_based+++++++++++++")
					if plain:
						print("+++++++++++plain+++++++++++++")
						goal = episode['obs'][step]['desired_goal']
					elif normalizer:
						goal = episode['obs'][step]['achieved_goal']
					else:
						if (self.args.her != 'none') and (np.random.uniform() <= self.args.her_ratio):
							if self.args.her == 'match':
								goal = self.args.goal_sampler.sample()
								goal_pool = np.array([obs['achieved_goal'] for obs in episode['obs'][step + 1:]])
								step_her = (step + 1) + np.argmin(np.sum(np.square(goal_pool - goal), axis=1))
								goal = episode['obs'][step_her]['achieved_goal']
							else:
								step_her = {
									'final': len(episode['obs']) - 1,
									'future': np.random.randint(step + 1, len(episode['obs']))
								}[self.args.her]
								goal = episode['obs'][step_her]['achieved_goal']
						else:
							goal = episode['obs'][step]['desired_goal']

					desired_goal = episode['obs'][step]['desired_goal']
					achieved = episode['obs'][step + 1]['achieved_goal']
					achieved_old = episode['obs'][step]['achieved_goal']
					obs = goal_concat(episode['obs'][step]['observation'], goal)
					obs_next = goal_concat(episode['obs'][step + 1]['observation'], goal)
					act = episode['acts'][step]
					rew = episode['rews'][step]
					done = episode['done'][step]

					batch['obs'].append(copy.deepcopy(obs))
					batch['obs_next'].append(copy.deepcopy(obs_next))
					batch['acts'].append(copy.deepcopy(act))
					batch['hgg_acts'].append(copy.deepcopy(episode['hgg_acts'][step]))
					batch['rews'].append(copy.deepcopy(rew))
					batch['done'].append(copy.deepcopy(done))
				else:
					print("+++++++++++not self.args.goal_based+++++++++++++")
					for key in ['obs', 'acts', 'rews', 'done']:
						if key == 'obs':
							batch['obs'].append(copy.deepcopy(episode[key][step]))
							batch['obs_next'].append(copy.deepcopy(episode[key][step + 1]))
						else:
							batch[key].append(copy.deepcopy(episode[key][step]))

			return batch

		# Normal sampling
		for i in range(batch_size):
			if self.energy:
				idx = self.energy_sample()
			else:
				idx = np.random.randint(self.length)
			step = np.random.randint(self.steps[idx])

			if self.args.goal_based:


				if plain:

					# no additional tricks
					goal = self.buffer['obs'][idx][step]['desired_goal']
					#goal = self.buffer['hgg_acts'][idx][step]
				elif normalizer:
					# uniform sampling for normalizer update
					goal = self.buffer['obs'][idx][step]['achieved_goal']
				else:
					#print("*****************upsampling by HER trick***********")
					# upsampling by HER trick
					if (self.args.her!='none') and (np.random.uniform()<=self.args.her_ratio):
						if self.args.her=='match':
							goal = self.args.goal_sampler.sample()
							goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step+1:]])
							step_her = (step+1) + np.argmin(np.sum(np.square(goal_pool-goal),axis=1))
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
						else:
							step_her = {
								'final': self.steps[idx],
								'future': np.random.randint(step+1, self.steps[idx]+1)
							}[self.args.her]
							goal = self.buffer['obs'][idx][step_her]['achieved_goal']
							#print("*****************upsampling by HER trick future***********")
					else:
						goal = self.buffer['obs'][idx][step]['desired_goal']

				desired_goal = self.buffer['obs'][idx][step]['desired_goal']
				achieved = self.buffer['obs'][idx][step+1]['achieved_goal']
				achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
				state = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
				#obs = state
				obs = goal_concat(state, self.buffer['hgg_acts'][idx][step])
				state_next = goal_concat(self.buffer['obs'][idx][step+1]['observation'], goal)
				#obs_next = state_next
				obs_next = goal_concat(state_next, self.buffer['hgg_acts'][idx][step])
				act = self.buffer['acts'][idx][step]
				rew = self.buffer['rews'][idx][step]
				# rew = self.args.compute_reward((achieved, achieved_old), goal)
				#rew = self.args.compute_reward(achieved, goal)
				#rew = self.args.compute_reward(goal, desired_goal)
				# print("**********rew**********", rew)
				# if 0 < abs(reward) < 0.03:
				# 	reward_goal = 300
				# else:
				# 	reward_goal = 0
				# rew = reward + reward_goal
				done = self.buffer['done'][idx][step]

				batch['obs'].append(copy.deepcopy(obs))
				batch['obs_next'].append(copy.deepcopy(obs_next))
				batch['acts'].append(copy.deepcopy(act))
				batch['hgg_acts'].append(copy.deepcopy(self.buffer['hgg_acts'][idx][step]))
				# batch['rews'].append(copy.deepcopy([rew]))
				batch['rews'].append(copy.deepcopy(rew))
				batch['done'].append(copy.deepcopy(done))
			else:

				for key in ['obs', 'acts', 'rews', 'done']:
					if key=='obs':
						batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
						batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step+1]))
					else:
						batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))

		return batch
