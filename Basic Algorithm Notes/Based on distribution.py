import numpy as np

def three_sigma(s):
    mu, std = np.mean(s), np.std(s)
    lower, upper = mu-3*std, mu+3*std
    return lower, upper

def z_score(s):
  z_score = (s - np.mean(s)) / np.std(s)
  return z_score

def boxplot(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return lower, upper

from outliers import smirnov_grubbs as grubbs
print(grubbs.test([8, 9, 10, 1, 9], alpha=0.05))
print(grubbs.min_test_outliers([8, 9, 10, 1, 9], alpha=0.05))
print(grubbs.max_test_outliers([8, 9, 10, 1, 9], alpha=0.05))
print(grubbs.max_test_indices([8, 9, 10, 50, 9], alpha=0.05))
