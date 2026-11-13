import numpy as np
def triangular(u,tau):
    if tau<=0:raise ValueError('tau must be positive')
    return np.maximum(0.,1.-np.abs(np.asarray(u)/tau-1.))/tau
