import numpy as np

from ddqn import DDQNAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import plot, plot_running_avg, EPS, to_file, from_file, \
	plot_dashed, PermutationExactSolver, greedy_reversal_sort


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


def generate_fixed(m, n=10):
	fixed = []
	for i in range(m):
		fixed.append(list(np.random.permutation(n)))
	to_file('fixed', fixed)


def save_ans(solve, label):
	fixed = from_file('fixed')
	fixed = fixed[100:200]
	ans = []
	for i, perm in enumerate(fixed):
		perm = np.array(perm)
		if i % 10 == 0:
			print('%.2f %%' % (i / len(fixed) * 100))
		ans.append(solve(perm))
	to_file('%s_ans' % label, ans)


def compare(labels):
	fixed = from_file('fixed')
	xs = [range(len(fixed))] * 3
	ys = []
	for label in labels:
		y = from_file(label + '_ans')
		ys.append(y)
	plot_dashed(xs, ys, labels)


def main():
	n = 10

	# generate_fixed(1000)

	# solver = PermutationExactSolver(n)
	# save_ans(solver.solve, 'exact')

	# solver = greedy_reversal_sort
	# save_ans(solver, 'greedy')

	compare(('greedy', 'exact', 'rl'))
	# compare(('rl_0.05', 'rl_0.1', 'rl_0.2'))

	# env = PermutationSorting(n)
	# state_transformer = OneHotStateTransformer(n)
	# agent = DDQNAgent(env, state_transformer)
	# agent.load_final_weights()
	# agent.epsilon = 0.1
	# save_ans(lambda perm: agent.solve(
	# 	perm, 100, 30, update_eps=False, update_model=False), label='rl_test')


if __name__ == '__main__':
	main()
