import gym
import numpy as np
from gym import spaces

from util import reverse_subarray, count_breakpoints_reduced, breakpoints

REV_OP = 'rev'
TRANS_OP = 'trans'


class PermutationSpace(gym.Space):
	def contains(self, x):
		raise Exception('Env does not support contains call')

	def __init__(self, n):
		self.n = n
		gym.Space.__init__(self, (n,), np.int)

	def sample(self):
		return np.random.permutation(self.n)


class PermutationSorting(gym.Env):
	def __init__(
			self, base, reversals=True, transpositions=False):

		if isinstance(base, int):
			n = base
			base = None
			state = np.random.permutation(n)
		else:
			base = np.array(base)
			n = base.shape[0]
			state = base.copy()

		actions = []

		if reversals:
			for i in range(n - 1):
				for j in range(i + 1, n):
					actions.append((REV_OP, (i, j)))

		if transpositions:
			for i in range(n - 1):
				for k in range(i + 1, n):
					for j in range(i, k):
						actions.append((TRANS_OP, (i, j, k)))

		self._identity = np.arange(n)
		self.actions = actions
		self.observation_space = PermutationSpace(n)
		self.action_space = spaces.Discrete(len(actions))
		self._base = base
		self._state = state
		self._render = False
		self._n = n
		self._breakpoints = 0
		self._reversals = reversals
		self._transpositions = transpositions

	def reset(self):
		if self._base is None:
			self._state = np.random.permutation(self._n)
		else:
			self._state = self._base.copy()

		self._breakpoints = breakpoints(self._state)
		if self._render:
			print(self._state)
			self._render = False

		return np.array(self._state)

	def render(self, mode='human'):
		self._render = True

	def step(self, action):
		type_, indices = self.actions[action]
		state = self._state

		if type_ == REV_OP:
			i, j = indices
			self._breakpoints -= count_breakpoints_reduced(state, i, j)
			reverse_subarray(state, i, j)
		else:
			i, j, k = indices
			state = np.concatenate((state[:i], state[j + 1: k + 1], state[i:j + 1], state[k + 1:]))

		if self._reversals and not self._transpositions:
			is_final = self._breakpoints == 0
		else:
			is_final = np.array_equal(state, self._identity)

		done = is_final
		reward = -1

		if self._render:
			print(state)
			self._render = False

		self._state = state
		return np.array(state), reward, done, {}
