import numpy as np

from state_transformers import RBFStateTransformer, IdentityStateTransformer, MaxStateTransformer, OneHotStateTransformer, OneHotRBFStateTransformer
from td_lambda import TDLambdaAgent

from permutation_sorting import PermutationSorting


class Eps1:
	def __init__(self):
		self._eps = 0.5
		self._min = 0.005
		self._decay = 0.99

	def eps(self, i):
		if self._eps > self._min:
			self._eps *= self._decay
		return self._eps


def eps2(i):
	return 1.5 / np.sqrt(i + 1)


def eps3(i):
	return 1 * (0.9 ** i)


def main():
	# v = [3, 0, 1, 2, 4]
	# v = [2, 4, 1, 3, 0, 6, 5]
	# n = len(v)
	n = 7
	env = PermutationSorting(base=n)

	# st = RBFStateTransformer(env, IdentityStateTransformer(n))
	# st = MaxStateTransformer(n)
	# st = OneHotStateTransformer(n)
	# st = OneHotRBFStateTransformer(env)
	st = RBFStateTransformer(env, MaxStateTransformer(n))

	agent = TDLambdaAgent(env, state_transformer=st)
	agent.pretrain(1000)
	agent.train(1000, Eps1().eps, True, True)


if __name__ == '__main__':
	main()
