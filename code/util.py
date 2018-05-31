import matplotlib.pyplot as plt
import numpy as np


def plot_running_avg(total_rewards):
	n = len(total_rewards)
	running_avg = np.empty(n)
	for t in range(n):
		running_avg[t] = total_rewards[max(0, t - 100):(t + 1)].mean()
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()
