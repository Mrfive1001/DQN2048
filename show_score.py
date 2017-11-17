import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('myscore.pkl', 'rb') as f:
    score = pickle.load(f)
plt.plot(score)
plt.show()