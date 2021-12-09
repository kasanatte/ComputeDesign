import numpy as np

probs = [0.1, 0.2, 0.3, 0.4]
probs_cum = np.cumsum(probs)

each_rand = np.random.uniform(size=10)

for rand in each_rand:
    X = np.where(probs_cum > rand)
    print(type(X))
    print(X)
