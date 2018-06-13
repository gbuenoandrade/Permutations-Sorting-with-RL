import numpy as np

from ddqn import DDQNAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import greedy_reversal_sort, plot, plot_running_avg, EPS, PermutationExactSolver


def compare_agent_to_solver(agent, solver, its=100, solve_its=100, plot_result=True, exploit_greedy_trace=False):
	scores = np.empty(its)
	for i in range(its):
		permutation = agent.env.observation_space.sample()
		base_result = solver(permutation)
		rl = agent.solve(permutation, exploit_greedy_trace=exploit_greedy_trace, its=solve_its)
		scores[i] = rl / (base_result + EPS)
		print('It:', i, ' Ratio: %.3f' % scores[i])
	if plot_result:
		plot(scores)
		plot_running_avg(scores)
	scores_mean = scores.mean()
	print('Mean = %.3f' % scores_mean)
	return scores_mean


def main():
	n = 10
	env = PermutationSorting(n)
	state_transformer = OneHotStateTransformer(n)
	agent = DDQNAgent(env, state_transformer)

	# agent.parallel_pretrain(rows=20000)

	# agent.load_pretrain_weights()
	# agent.epsilon = 0.01
	#
	# agent.train_exploiting_greedy(episodes=200, plot_rewards=True, max_steps=250)
	# agent.train(episodes=3000, plot_rewards=True, max_steps=250)

	agent.load_final_weights()

	solver = greedy_reversal_sort
	# solver = PermutationExactSolver(n).solve
	compare_agent_to_solver(agent, solver, plot_result=False, exploit_greedy_trace=False, solve_its=50)


if __name__ == '__main__':
	main()
