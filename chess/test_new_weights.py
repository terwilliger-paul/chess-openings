import numpy as np
import pandas as pd

# Constants
HALF_LIFE = 2 #days

# Curve has diameter of 3 days
# If a line falls below a certain memory threshold, it leaves `maintentance`
repetition = [0, .01, .05, 1, 1.5, 2, 3, 4.5, 7]
decay = np.log(2) / HALF_LIFE

# Load up pgn lines
# Add them to the database of lines
# Load up database of lines


# Convert dates in database into minutes since last test
dates = np.abs(np.random.randn(5, 3) + 2)
min_since = np.sort(dates, axis=1)
min_since[0, :] = 0

# How well we know each line
know_lines = np.array([np.sum([np.exp(-decay * t) for t in row])
                       for row in min_since])

print know_lines
print min_since
print np.abs(know_lines - know_lines.max())
