import numpy as np

from environments import CartPole, MountainCar, ArraySorting, PermutationSorting
from q_learning import DQLAgent
from td_lambda import TDLambdaAgent


def eps1(i):
	return max(0.005, 0.993 ** i)


def eps2(i):
	return 1.5 / np.sqrt(i + 1)


def eps3(i):
	return 1 * (0.9 ** i)


def main():
	env = PermutationSorting(5)
	agent = TDLambdaAgent(env)
	agent.train(500, eps1, True)
	agent.play()


if __name__ == '__main__':
	main()
