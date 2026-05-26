import numpy as np
import scipy.interpolate
from scipy import interpolate

x = np.array([1,2,3,4,5])
y = np.array([2,3,5,4,2])

linear_interp = interpolate.interp1d(x,y)
x_new = 2.5
y_new = linear_interp(x_new)
print(f"Interpolate value at {x_new} : {y_new}")