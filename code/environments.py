from abc import abstractmethod, ABC

import gym
import numpy as np


class Environment(ABC):
	@property
	@abstractmethod
	def actions(self):
		...

	@property
	@abstractmethod
	def n(self):
		...

	@abstractmethod
	def sample_action(self):
		...

	@abstractmethod
	def sample_observation(self):
		...

	@abstractmethod
	def reset(self):
		...

	@abstractmethod
	def step(self, a):
		...

	@abstractmethod
	def render(self, close):
		...


class DiscreteGymEnv(Environment):
	def __init__(self, name):
		self.env = gym.make(name)
		self._actions = [i for i in range(self.env.action_space.n)]
		self.it = 0

	@property
	def actions(self):
		return self._actions

	@property
	def n(self):
		return self.env.action_space.n

	def sample_action(self):
		return self.env.action_space.sample()

	def sample_observation(self):
		return self.env.observation_space.sample()

	def reset(self):
		self.it = 0
		return self.env.reset()

	def step(self, a):
		self.it += 1
		return self.env.step(a)

	def render(self, close=False):
		self.env.render(close=close)


class MountainCar(DiscreteGymEnv):
	def __init__(self):
		super().__init__('MountainCar-v0')


class CartPole(DiscreteGymEnv):
	def __init__(self):
		super().__init__('CartPole-v0')

	def step(self, a):
		observation, r, done, info = super(CartPole, self).step(a)
		if done and self.it < 200:
			r = -3000
		return observation, r, done, info


class ArraySorting(Environment):
	def __init__(self, sz, array=None):
		self.initial_array = np.random.permutation(sz) if array is None else np.array(array)
		self.cur = self.initial_array.copy()
		self.sz = sz

		actions = []
		for i in range(sz):
			for j in range(i + 1, sz):
				actions.append((i, j))

		self._actions = actions
		self._n = len(actions)
		self._identity = np.arange(sz)
		self._render = False

	@property
	def actions(self):
		return self._actions

	@property
	def n(self):
		return self._n

	def sample_action(self):
		return np.random.choice(self._n)

	def sample_observation(self):
		return np.random.permutation(self.sz)

	def reset(self, array=None):
		if array is not None:
			self.initial_array = np.array(array)
		self.cur = self.initial_array.copy()
		return self.cur

	def step(self, a):
		a, b = self._actions[a]
		self.cur[a], self.cur[b] = self.cur[b], self.cur[a]
		done = np.array_equal(self.cur, self._identity)
		if self._render:
			print(self.cur)
			self._render = False
		return self.cur, -1, done, None

	def render(self, close=False):
		self._render = True
