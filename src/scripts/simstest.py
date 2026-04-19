import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


similarities = np.load("sims.npy", "r")

print(np.percentile(similarities, 10))
print(np.percentile(similarities, 20))
print(np.percentile(similarities, 30))

density = gaussian_kde(similarities)
x = np.linspace(0, 1, 200)

plt.plot(x, density(x), color="teal", linewidth=2)
plt.fill_between(x, density(x), alpha=0.3, color="teal")

plt.show()
