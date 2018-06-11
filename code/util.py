import threading

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-7


class AtomicInteger:
	def __init__(self):
		self._value = 0
		self._lock = threading.Lock()

	def inc(self):
		with self._lock:
			self._value += 1
			return self._value

	def dec(self):
		with self._lock:
			self._value -= 1
			return self._value

	def reset(self):
		self._value = 0

	@property
	def value(self):
		return self._value


def get_running_avg(x):
	n = len(x)
	running_avg = np.empty(n)
	for t in range(n):
		running_avg[t] = x[max(0, t - 100):(t + 1)].mean()
	return running_avg


def plot_running_avg(x, title='Running Average'):
	running_avg = get_running_avg(x)
	plt.plot(running_avg)
	plt.title(title)
	plt.show()


def plot(x, title=''):
	plt.plot(x)
	plt.title(title)
	plt.show()


def plot_x_and_avg(x, title='X and Running Average'):
	plt.plot(x)
	plt.plot(get_running_avg(x))
	plt.title(title)
	plt.show()


def reverse_subarray(v, i, j):
	while i < j:
		v[i], v[j] = v[j], v[i]
		i += 1
		j -= 1


def count_breakpoints_reduced(v, i, j):
	n = len(v)
	prev = v[i - 1] if i > 0 else -1
	next = v[j + 1] if j < n - 1 else n
	breakpoints_reduced = 0
	if abs(v[i] - prev) != 1:
		breakpoints_reduced += 1
	if abs(v[j] - next) != 1:
		breakpoints_reduced += 1
	if abs(v[j] - prev) != 1:
		breakpoints_reduced -= 1
	if abs(v[i] - next) != 1:
		breakpoints_reduced -= 1
	return breakpoints_reduced


def score_rev(v, i, j):
	n = len(v)
	breakpoints_reduced = count_breakpoints_reduced(v, i, j)
	for r in (range(1, i), range(j + 1, n)):
		for idx in r:
			if v[idx] == v[idx - 1] - 1:
				return breakpoints_reduced, 1
	for idx in range(i + 1, j + 1):
		if v[idx] == v[idx - 1] + 1:
			return breakpoints_reduced, 1
	return breakpoints_reduced, 1 if (i > 0 and v[j] < v[i - 1]) or (j < n - 1 and v[i] > v[j + 1]) else 0


def breakpoints(v):
	n = len(v)
	ans = 0
	for i in range(1, n):
		if abs(v[i] - v[i - 1]) != 1:
			ans += 1
	if v[0] != 0:
		ans += 1
	if v[n - 1] != n - 1:
		ans += 1
	return ans


def greedy_reversal_sort(v, trace=None):
	v = v.copy()
	if trace is not None:
		trace.append(v.copy())
	n = len(v)
	b = breakpoints(v)
	ans = 0
	while b > 0:
		max_score = None
		x, y = None, None
		for i in range(n - 1):
			for j in range(i + 1, n):
				score = score_rev(v, i, j)
				if max_score is None or score > max_score:
					max_score = score
					x, y = i, j
		b -= max_score[0]
		reverse_subarray(v, x, y)
		if trace is not None:
			trace.append(v.copy())
		ans += 1
	return ans


def v_upperbound(state, gamma):
	steps = greedy_reversal_sort(state)
	ans = (np.float_power(gamma, steps) - 1) / (gamma - 1)
	return -ans


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
