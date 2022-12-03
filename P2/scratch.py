import numpy as np
from scipy import stats

# your distribution:
distribution = stats.norm(loc=50, scale=5)

# percentile point, the range for the inverse cumulative distribution function:
bounds_for_range = distribution.cdf([0, 100])

# Linspace for the inverse cdf:
pp = np.linspace(*bounds_for_range, num=1000)

x = distribution.ppf(pp)

# And just to check that it makes sense you can try:
from matplotlib import pyplot as plt
plt.hist(x)
plt.show()