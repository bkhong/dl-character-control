'''
This file was used to convert the heightmap data from the original paper into numpy format.
'''

import numpy as np

IN_FILE = 'demo/heightmaps/hmap_urban_001_smooth.txt'
OUT_FILE = 'project/data/heightmaps/hmap5'


# Load the data
with open(IN_FILE) as f:
    data = f.readlines()

arr = np.array([[float(f) for f in line.split()] for line in data], dtype=np.float32)
print(arr.shape)

# Apply vertical scale and shift (this is done in the demo code for the original paper)
arr -= arr.mean()
arr *= 3

np.save(OUT_FILE, arr)
