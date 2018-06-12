import random
from collections import deque
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from util import v_upperbound, reverse_subarray, AtomicInteger, plot_running_avg, plot, \
	greedy_reversal_sort

PRETRAIN_WEIGHTS_PATH = './saved_models/pretrain_weights.h5'
FINAL_WEIGHTS_PATH = './saved_models/final_weights.h5'


# Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DDQNAgent:
	def __init__(self, env, state_transformer, initial_epsilon=0.2):
		self.render = False
		self.env = env
		self.state_size = state_transformer.dimensions
		self.action_size = self.env.action_space.n
		self.state_transformer = state_transformer

		# these is hyper parameters for the Double DQN
		self.discount_factor = 0.99
		self.learning_rate = 0.001
		self.epsilon = initial_epsilon
		self.epsilon_decay = 0.993
		self.epsilon_min = 0.01
		self.batch_size = 32
		self.train_start = 1000
		# create replay memory using deque
		self.memory = deque(maxlen=2000)

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()

		# initialize target model
		self.update_target_model()

	# approximate Q function using Neural Network
	# state is input and Q Value of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Dense(48, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(12, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(state)
			return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)

		update_input = np.zeros((batch_size, self.state_size))
		update_target = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(batch_size):
			update_input[i] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			update_target[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		target = self.model.predict(update_input)
		target_next = self.model.predict(update_target)
		target_val = self.target_model.predict(update_target)

		for i in range(self.batch_size):
			# like Q Learning, get maximum Q value at s'
			# But from target model
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				# the key point of Double DQN
				# selection of action is from model
				# update is from target model
				a = np.argmax(target_next[i])
				target[i][action[i]] = reward[i] + self.discount_factor * (
					target_val[i][a])

		# make minibatch which includes target q value and predicted q value
		# and do the model fit!
		self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

	def fill_target(self, state, target):
		for idx, (_, (i, j)) in enumerate(self.env.actions):
			reverse_subarray(state, i, j)
			target[idx] = -1 + self.discount_factor * v_upperbound(state, self.discount_factor)
			reverse_subarray(state, i, j)

	def serial_pretrain(self, rows=10000, epochs=10):
		targets = np.empty((rows, self.action_size))
		states = np.empty((rows, self.state_size))
		for i in range(rows):
			p = self.env.observation_space.sample()
			states[i] = self.state_transformer.transform(p)
			self.fill_target(p, targets[i])
			if i % 100 == 0:
				print("%.1f %%" % (i / rows * 100))
		self.model.fit(states, targets, batch_size=self.batch_size, epochs=epochs, verbose=1, validation_split=0.1)
		self.update_target_model()
		self.model.save_weights(PRETRAIN_WEIGHTS_PATH)

	def parallel_pretrain(self, rows=10000, epochs=10, n_threads=8):
		def f(i):
			p = self.env.observation_space.sample()
			states[i] = self.state_transformer.transform(p)
			self.fill_target(p, targets[i])
			cur = progress.inc()
			if cur % 100 == 0:
				print("%.1f %%" % (cur / rows * 100))

		progress = AtomicInteger()
		targets = np.empty((rows, self.action_size))
		states = np.empty((rows, self.state_size))
		pool = ThreadPool(n_threads)
		pool.map(f, range(rows))
		self.model.fit(states, targets, batch_size=self.batch_size, epochs=epochs, verbose=1, validation_split=0.1)
		self.update_target_model()
		self.model.save_weights(PRETRAIN_WEIGHTS_PATH)

	def load_pretrain_weights(self):
		self.model.load_weights(PRETRAIN_WEIGHTS_PATH)
		self.update_target_model()

	def load_final_weights(self):
		self.model.load_weights(FINAL_WEIGHTS_PATH)
		self.update_target_model()

	@staticmethod
	def _is_identity(p):
		for i in range(len(p)):
			if p[i] != i:
				return False
		return True

	def run_episode(self, max_steps, forced=None):
		done = False
		score = 0
		state = self.env.reset(forced=forced)

		if self._is_identity(state):
			return 0

		state = self.state_transformer.transform(state)
		rem_steps = max_steps

		while not done and rem_steps > 0:
			if self.render:
				self.env.render()

			rem_steps -= 1

			# get action for the current state and go one step in environment
			action = self.get_action(state)
			next_state, reward, done, info = self.env.step(action)
			next_state = self.state_transformer.transform(next_state)

			# save the sample <s, a, r, s'> to the replay memory
			self.append_sample(state, action, reward, next_state, done)
			# every time step do the training
			self.train_model()
			score += reward
			state = next_state

		# every episode update the target model to be same with model
		self.update_target_model()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return score

	def train(self, episodes=1000, max_steps=1000, plot_rewards=True):
		scores = np.empty(episodes)
		for e in range(episodes):
			score = self.run_episode(max_steps)
			scores[e] = score
			print("Episode:", e, "  score:", score, "  epsilon:", self.epsilon)

		self.model.save_weights(FINAL_WEIGHTS_PATH)

		if plot_rewards:
			plot(scores)
			plot_running_avg(scores)

	def train_exploiting_greedy(self, episodes=1000, max_steps=1000, plot_rewards=True):
		scores = []
		e = 0
		for _ in range(episodes):
			trace = []
			greedy_reversal_sort(self.env.observation_space.sample(), trace)
			for __ in range(3):
				for permutation in trace[::-1]:
					score = self.run_episode(max_steps, forced=permutation)
					scores.append(score)
					print("Episode:", e, "  score:", score, "  epsilon:", self.epsilon)
					e += 1
				print()
			print()

		self.model.save_weights(FINAL_WEIGHTS_PATH)

		scores = np.array(scores)
		if plot_rewards:
			plot(scores)
			plot_running_avg(scores)

	def solve(self, permutation, its=100, max_steps=100, exploit_greedy_trace=False):
		ans = None
		if exploit_greedy_trace:
			trace = []
			greedy_reversal_sort(permutation, trace)
		for _ in range(its):
			if exploit_greedy_trace:
				last_ans = None
				# noinspection PyUnboundLocalVariable
				for p in trace[::-1]:
					last_ans = self.run_episode(max_steps=max_steps, forced=p)
				if ans is None or last_ans > ans:
					ans = last_ans
			else:
				pans = self.run_episode(max_steps=max_steps, forced=permutation)
				if ans is None or pans > ans:
					ans = pans
		return -ans
