from SetCoveringProblemCreator import *
import matplotlib.pyplot as plt
import random
import numpy as np

x = np.array([50,150,250,350])
y = np.array([126.4,249.6,407,585.4])
y1 = np.array([2.4979991993593593, 9.286549412995118, 14.177446878757825, 16.63850954863446])

plt.plot(x,y1,marker='o')
plt.xlabel("Number Of Subsets", fontsize=14)
plt.ylabel("Standard deviation after 50 generations", fontsize=14)
plt.title("Std deviation (over 10 SCP) and number of subsets",fontsize=16)
plt.show()

plt.plot(x,y,marker='o')
plt.xlabel("Number Of Subsets", fontsize=14)
plt.ylabel("Best fitness value after 50 generations", fontsize=14)
plt.title("Best fitness value (avg over 10 SCP and after 50 generations) and number of subsets",fontsize=16)
plt.show()