import numpy as np

from ddqn import DDQNAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import PermutationExactSolver


def main():
	np.random.seed(12345678)

	n = 10
	env = PermutationSorting(n)
	state_transformer = OneHotStateTransformer(n)
	agent = DDQNAgent(env, state_transformer)
	agent.parallel_pretrain(rows=10000, epochs=30)
	# agent.load_pretrain_weights()
	agent.train(episodes=10000, max_steps=250)
	# agent.load_final_weights()

	for _ in range(10):
		p = np.random.permutation(n)
		rl_ans = agent.solve(p)
		exact_ans = PermutationExactSolver(n).solve(p)
		print(p, '-', 'RL:', rl_ans, ' Exact:', exact_ans)


if __name__ == '__main__':
	main()
