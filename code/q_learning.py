from collections import deque

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from util import plot_running_avg


class FeatureTransformer:
	def __init__(self, n):
		self.maxv = 1
		# self.maxv = n - 1

	def transform(self, obs):
		return np.atleast_2d(obs[0] / self.maxv)


class Model:
	def __init__(self, env, feature_transformer, lr):
		self.env = env
		self.lr = lr
		self.feature_transformer = feature_transformer
		self.dnn = self._build_dnn()

	def _build_dnn(self):
		model = Sequential()

		model.add(Dense(256, input_dim=self.env.sample_observation().shape[0], activation='relu'))
		# model.add(Dropout(0.2))
		model.add(Dense(128, activation='relu'))

		model.add(Dense(32, activation='relu'))

		# model.add(Dropout(0.2))
		# model.add(Dense(32, activation='relu'))
		# model.add(Dropout(0.2))
		model.add(Dense(self.env.n, activation='linear'))

		model.compile(loss='mse', optimizer=Adam(lr=self.lr))
		return model

	def _transform(self, s):
		return self.feature_transformer.transform([s])

	def predict(self, s):
		x = self._transform(s)
		self.dnn.predict(x)
		y = self.dnn.predict(x)
		return y

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return self.env.sample_action()
		else:
			return np.argmax(self.predict(s)[0])

	def update(self, s, a, G):
		target_f = self.predict(s)
		target_f[0][a] = G
		self.dnn.fit(self._transform(s), target_f, epochs=1, verbose=0)


class DQLAgent:
	def __init__(
			self, env,
			gamma=0.95,
			lr=1e-3,
			memory_len=2000,
			max_it=10000,
			batch_size=32):

		ft = FeatureTransformer(env.sample_observation().shape[0])
		self.model = Model(env, ft, lr)
		self.memory = deque(maxlen=memory_len)
		self.gamma = gamma
		self.env = env
		self.max_it = max_it
		self.batch_size = batch_size

	def _remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	# noinspection PyPep8Naming,PyShadowingBuiltins
	def _play_one(self, eps):
		observation = self.env.reset()
		done = False
		total_reward = 0
		it = 0
		while not done and it < self.max_it:
			action = self.model.sample_action(observation, eps)
			prev_observation = observation
			observation, reward, done, info = self.env.step(action)
			next = self.model.predict(observation)
			assert (next.shape == (1, self.env.n))
			self._remember(prev_observation, action, reward, observation, done)
			total_reward += reward
			it += 1
		return total_reward

	# noinspection PyPep8Naming
	def replay(self):
		minibatch = np.random.choice(len(self.memory), self.batch_size)
		for idx in minibatch:
			s, a, r, next_s, done = self.memory[idx]
			G = r
			if not done:
				G = r + self.gamma * np.amax(self.model.predict(next_s)[0])
			self.model.update(s, a, G)

	def train(self, n, f_eps, plot_rewards=False):
		total_rewards = np.empty(n)
		for i in range(n):
			eps = f_eps(i)
			total_reward = self._play_one(eps)
			self.replay()
			total_rewards[i] = total_reward
			if i % 25 == 0:
				print("Episode:", i, "Reward:", total_reward, "Eps:", eps)

		if plot_rewards:
			plot_running_avg(total_rewards)

	def play(self):
		observation = self.env.reset()
		done = False
		max_it = 1000
		while not done and max_it > 0:
			self.env.render()
			action = self.model.sample_action(observation, 0)
			observation, _, done, _ = self.env.step(action)
			max_it -= 1
