import numpy as np

from ddqn import DDQNAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import greedy_reversal_sort, plot, plot_running_avg, EPS


def compare_with_greedy(agent, its=100, plot_result=True, exploit_greedy_trace=False):
	scores = np.empty(its)
	for i in range(its):
		permutation = agent.env.observation_space.sample()
		greedy = greedy_reversal_sort(permutation)
		rl = agent.solve(permutation, exploit_greedy_trace=exploit_greedy_trace)
		scores[i] = greedy / (rl + EPS)
		print('It:', i, ' Ratio: %.3f' % scores[i])
	if plot_result:
		plot(scores)
		plot_running_avg(scores)
	scores_mean = scores.mean()
	print('Mean = %.3f' % scores_mean)
	return scores_mean


def main():
	n = 5
	env = PermutationSorting(n)
	state_transformer = OneHotStateTransformer(n)
	agent = DDQNAgent(env, state_transformer)
	# agent.serial_pretrain()
	agent.parallel_pretrain(rows=100000)
	# agent.load_pretrain_weights()
	# agent.train(plot_rewards=True)
	# compare_with_greedy(agent, plot_result=True, exploit_greedy_trace=False)


if __name__ == '__main__':
	main()
