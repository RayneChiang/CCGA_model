import numpy as np
from matplotlib import pyplot as plt
ga_result = np.load('Rastrigin_ga_result.npy')
ccga_result = np.load('Rastrigin_ccga_result.npy')
print(ga_result)
print(ccga_result)
fig, ax = plt.subplots()
ax.plot(range(len(ga_result)), ga_result)
ax.plot(range(len(ccga_result)), ccga_result)
ax.set_ylim(0,50)
plt.show()
