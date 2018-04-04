import numpy as np

d = np.load("data/stock.npz")

print(d["price"][190])