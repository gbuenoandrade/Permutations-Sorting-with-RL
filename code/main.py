import numpy as np

from ddqn import DDQNAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import plot, plot_running_avg, EPS, to_file, from_file, \
	plot_dashed


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
	fixed = fixed[:100]
	ans = []
	for i, perm in enumerate(fixed):
		perm = np.array(perm)
		if i % 10 == 0:
			print('%.2f %%' % (i / len(fixed) * 100))
		ans.append(solve(perm))
	to_file('%s_ans' % label, ans)


def compare(labels):
	fixed = from_file('fixed')
	# fixed = fixed[:100]
	xs = [range(len(fixed))] * len(labels)
	ys = []
	for label in labels:
		y = from_file(label + '_ans')
		ys.append(y)
	plot_dashed(xs, ys, labels)


def plot_type1():
	ygreedy = np.array(from_file('greedy' + '_ans'))
	yrl = np.array(from_file('rl' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))
	ygreedy = ygreedy / yexact
	yrl = yrl / yexact

	print(ygreedy.mean())
	print(yrl.mean())

	xs = [range(len(yrl))] * 2
	plot_dashed(
		xs, (ygreedy, yrl), ('Kececioglu and Sankoff (1995)', 'RL'),
		xlabel='episodes', ylabel='performance ratio', file='rlandgreedy_vs_exact2')


def plot_type2():
	flavio = np.array(from_file('flaviostate' + '_ans'))
	onehot = np.array(from_file('onehot' + '_ans'))
	maxstate = np.array(from_file('maxstate' + '_ans'))
	exact = np.array(from_file('exact' + '_ans'))

	flavio = flavio / exact
	onehot = onehot / exact
	maxstate = maxstate / exact

	xs = [range(len(flavio))] * 3
	plot_dashed(
		xs, (flavio, onehot, maxstate), ('Permutation Characterization', 'One-Hot Encoding', 'Min-Max Normalization'),
		xlabel='episodes', ylabel='performance ratio', file='states_comp')


def plot_type3():
	ylambda = np.array(from_file('tdlambda' + '_ans'))
	ydnr = np.array(from_file('dnr' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))

	for i in range(len(yexact)):
		if abs(yexact[i]) < EPS:
			ylambda[i] = 1
			ydnr[i] = 1
		else:
			ylambda[i] = ylambda[i] / yexact[i]
			ydnr[i] = ydnr[i] / yexact[i]

	xs = [range(len(yexact))] * 2
	plot_dashed(
		xs, (ylambda, ydnr), ('TD-Lambda', 'DDQN'),
		xlabel='episodes', ylabel='performance ratio', file='lambda_ddqn')


def main():
	n = 10
	env = PermutationSorting(n)
	state_transformer = OneHotStateTransformer(n)
	agent = DDQNAgent(env, state_transformer)
	agent.parallel_pretrain(1000)
	# agent.load_pretrain_weights()
	agent.train(max_steps=200)
	# agent.load_final_weights()
	p = np.random.permutation(n)
	ans = agent.solve(p)


if __name__ == '__main__':
	main()
