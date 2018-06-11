import numpy as np

from util import plot_running_avg, plot, v_bound, reverse_subarray


# noinspection PyPep8Naming
class SGDRegressor:
	def __init__(self, D):
		self.w = np.random.randn(D) / np.sqrt(D)

	def partial_fit(self, input_, target, eligibility, lr):
		self.w += lr * (target - input_.dot(self.w)) * eligibility

	def predict(self, X):
		X = np.array(X)
		return X.dot(self.w)


# noinspection PyShadowingNames,PyPep8Naming
class Model:
	def __init__(self, env, feature_transformer, lr):
		self.env = env
		self.models = []
		self.feature_transformer = feature_transformer
		self.lr = lr

		D = feature_transformer.dimensions
		self.eligibilities = np.zeros((env.action_space.n, D))
		for i in range(env.action_space.n):
			model = SGDRegressor(D)
			self.models.append(model)

	def _transform(self, s):
		return [self.feature_transformer.transform(s)]

	def predict(self, s):
		X = self._transform(s)
		result = np.stack([m.predict(X) for m in self.models]).T
		return result

	def update(self, s, a, G, gamma, lambda_):
		X = self._transform(s)
		self.eligibilities *= gamma * lambda_
		self.eligibilities[a] += X[0]
		self.models[a].partial_fit(X[0], G, self.eligibilities[a], self.lr)

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			return np.argmax(self.predict(s))


class TDLambdaAgent:
	def __init__(
			self, env, state_transformer,
			lr=1e-2,
			gamma=0.999,
			lambda_=0.25,
			max_its=1000):

		self.model = Model(env, state_transformer, lr)
		self.gamma = gamma
		self.lambda_ = lambda_
		self.env = env
		self.max_its = max_its

	# noinspection PyPep8Naming,PyShadowingBuiltins
	def _play_one(self, eps):
		observation = self.env.reset()
		done = False
		total_reward = 0
		it = self.max_its
		while not done and it > 0:
			action = self.model.sample_action(observation, eps)
			prev_observation = observation
			observation, reward, done, info = self.env.step(action)
			next = self.model.predict(observation)
			G = reward + self.gamma * np.max(next[0])
			self.model.update(prev_observation, action, G, self.gamma, self.lambda_)
			total_reward += reward
			it -= 1

		return total_reward

	def train(self, n, f_eps, plot_rewards=False, plot_best=False):
		total_rewards = np.empty(n)
		bests = np.empty(n)
		best = None
		for i in range(n):
			eps = f_eps(i)
			total_reward = self._play_one(eps)
			if best is None or total_reward > best:
				best = total_reward
			bests[i] = best
			total_rewards[i] = total_reward
			if i % 1 == 0:
				print("Episode:", i, "Reward:", total_reward, "Best:", best, "Eps:", eps)

		if plot_rewards:
			plot_running_avg(total_rewards)

		if plot_best:
			plot(bests, 'Bests')

	def pretrain(self, its=1000):
		actions = self.env.actions
		for _ in range(its):
			state = self.env.observation_space.sample()
			for a, (_, (i, j)) in enumerate(actions):
				reverse_subarray(state, i, j)
				next_bound = v_bound(state, self.gamma)
				reverse_subarray(state, i, j)
				G = -1 + self.gamma * next_bound
				self.model.update(state, a, G, self.gamma, self.lambda_)

	def play(self, eps=0.01, max_it=10000):
		observation = self.env.reset()
		done = False
		while not done and max_it > 0:
			self.env.render()
			action = self.model.sample_action(observation, eps)
			observation, _, done, _ = self.env.step(action)
			max_it -= 1
