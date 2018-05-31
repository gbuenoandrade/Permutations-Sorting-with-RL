import numpy as np

from environments import CartPole, MountainCar, ArraySorting
from q_learning import DQLAgent


def eps1(i):
	return max(0.005, 0.993 ** i)


def eps2(i):
	return 1.5 / np.sqrt(i + 1)


def eps3(i):
	return 1 * (0.9 ** i)


def main():
	env = MountainCar()
	agent = DQLAgent(env)
	agent.train(1000, eps1)
	agent.play()


if __name__ == '__main__':
	main()
