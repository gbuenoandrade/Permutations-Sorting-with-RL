import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from util import plot_running_avg


# noinspection PyShadowingNames
class FeatureTransformer:
	def __init__(self, env, gammas, n_components, n_examples):
		examples = np.array([env.sample_observation() for _ in range(n_examples)])
		scaler = StandardScaler()
		scaler.fit(examples)

		featurizer = FeatureUnion(
			[('rbf%d' % idx, RBFSampler(gamma=gamma, n_components=n_components)) for idx, gamma in enumerate(gammas)])
		example_features = featurizer.fit_transform(scaler.transform(examples))

		self.dimensions = example_features.shape[1]
		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, obs):
		scaled = self.scaler.transform(obs)
		return self.featurizer.transform(scaled)


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
		self.eligibilities = np.zeros((env.n, D))
		for i in range(env.n):
			model = SGDRegressor(D)
			self.models.append(model)

	def _transform(self, s):
		return self.feature_transformer.transform([s])

	def predict(self, s):
		X = self._transform(s)
		assert (len(X.shape) == 2)
		result = np.stack([m.predict(X) for m in self.models]).T
		assert (len(result.shape) == 2)
		return result

	def update(self, s, a, G, gamma, lambda_):
		X = self._transform(s)
		assert (len(X.shape) == 2)
		self.eligibilities *= gamma * lambda_
		self.eligibilities[a] += X[0]
		self.models[a].partial_fit(X[0], G, self.eligibilities[a], self.lr)

	def sample_action(self, s, eps):
		if np.random.random() < eps:
			return self.env.sample_action()
		else:
			return np.argmax(self.predict(s))


class TDLambdaAgent:
	def __init__(
			self, env,
			gammas=(5.0, 2.0, 1.0, 0.5),
			n_components=500,
			n_examples=10000,
			lr=1e-2,
			gamma=0.999,
			lambda_=0.2,
			max_it=10000):
		ft = FeatureTransformer(env, gammas, n_components, n_examples)
		model = Model(env, ft, lr)

		self.gamma = gamma
		self.lambda_ = lambda_
		self.env = env
		self.model = model
		self.max_it = max_it

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
			G = reward + self.gamma * np.max(next[0])
			self.model.update(prev_observation, action, G, self.gamma, self.lambda_)
			total_reward += reward
			it += 1
		return total_reward

	def train(self, n, f_eps, plot_rewards=False):
		total_rewards = np.empty(n)
		for i in range(n):
			eps = f_eps(i)
			total_reward = self._play_one(eps)
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
