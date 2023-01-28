# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:05:48 2021

@author: khann
"""
import numpy as np
import random
import time
import matplotlib.pyplot as plt


bayes_acc = np.load("Data1/bayes_acc_2021-11-13"+".npy")

Ntrains=[10, 100, 500, 1000, 1500, 5000, 7500, 10000, 15000, 20000]
zhang_acc=[0.63]*len(Ntrains)

# Create figure
fig, (ax1) = plt.subplots(1,1)
#first plot bayes optimal performance
ax1.semilogx(Ntrains,[bayes_acc]*len(Ntrains)  )
ax1.semilogx(Ntrains,zhang_acc)
ax1.set(title="Correctness of Zhang's Algorithm's Implementation")
ax1.grid()
ax1.set_ylim(top=0.7)
ax1.set_ylim(bottom=0.56)
fig.tight_layout()
plt.legend(["Bayes Optimal", "Zhang's Algorithm"])
plt.show()
